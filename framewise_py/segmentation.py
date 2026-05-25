"""Load and model a CNMF/CaImAn-style segmentation result folder.

A segmentation result is a directory laid out as::

    <root>/
      SEG/
        SEG.tiff           (N, H, W) uint16 binary footprint masks
        infer_results.mat  MATLAB v5: C (N, T) float32 activity traces
        SEG_SUM.png        (optional) summary image
      rmbg/
        rmbg_b000.tif ...  (optional) background-removed movie blocks, each
                           (block, H, W); concatenated → (T_video, H, W)

The three are index-aligned: neuron ``n`` ⟷ SEG page ``n`` ⟷ ``C`` row ``n``
(0-based; the MATLAB / footprints_colored label is ``n + 1``).

Footprints live on disk as a multi-GB BigTIFF of binary pages. On first load we
convert them once to a CaImAn-style sparse matrix ``A`` of shape ``(H*W, N)``
and cache it as ``SEG/footprints_A.npz`` beside the source, so later loads are
instant. ``C`` (a few hundred MB) is held in RAM; the rmbg movie stays lazy
(memmap + dask), since the full stack is ~tens of GB.

Coordinate note: masks are ``(row=Y, col=X)``, origin top-left, C-order.
framewise's VideoPanel transposes ``(H, W) → (W, H)`` for display
(`VideoPanel._normalize_frame_layout`); the label overlay built here is in raw
``(H, W)`` so the panel transposes it like any other frame.
"""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import scipy.sparse as sp

    from .signal_panel import Trace

# Default acquisition rate (Hz) used when no rate is encoded in the folder/file
# name. Matches framewise's video default; adjust per dataset via the panel's
# fps control (the neuron traces' time axis follows it).
DEFAULT_FPS = 20.0

# Cycling RGB palette for per-neuron coloring. Bright, distinct hues; reused for
# the label overlay and the selected-neuron trace color so the two stay linked.
_PALETTE = np.array(
    [
        (228, 26, 28),
        (55, 126, 184),
        (77, 175, 74),
        (152, 78, 163),
        (255, 127, 0),
        (255, 255, 51),
        (166, 86, 40),
        (247, 129, 191),
    ],
    dtype=np.uint8,
)

# Alpha for footprint pixels in the colored overlay (0 = transparent bg).
_LABEL_ALPHA = 170

_CACHE_NAME = "footprints_A.npz"

# Per-neuron cell-type labels. "unknown" is the default (every neuron starts
# here); "bad" keeps the special hide-from-canvas behavior (excluded from the
# colored overlay + hit-testing). The rest are cell-type classes used for
# downstream analysis. Labels are non-destructive — the footprints/traces stay
# in the source data; only this assignment is persisted (to `cell_labels.json`,
# explicitly via File → Save, never auto-saved).
CELL_LABELS = ("unknown", "PV", "CHI", "D1", "D2", "bad")
DEFAULT_LABEL = "unknown"

# Legacy file: a plain bad-ROI list, superseded by `cell_labels.json`. Still read
# once on load to migrate old marks (each listed index becomes label "bad").
_BAD_LEGACY_NAME = "bad_rois.json"
_LABELS_NAME = "cell_labels.json"


class SegmentationLoadError(Exception):
    pass


def is_segmentation_dir(path: Path) -> bool:
    """True if `path` is a segmentation result root (has SEG/infer_results.mat).

    Accepts either the project root (containing a ``SEG`` subfolder) or the
    ``SEG`` folder itself.
    """
    if not path.is_dir():
        return False
    seg = _seg_dir(path)
    return seg is not None and (seg / "infer_results.mat").exists()


def _seg_dir(path: Path) -> Path | None:
    """Resolve the SEG folder from either the project root or SEG itself."""
    if (path / "infer_results.mat").exists():
        return path
    if (path / "SEG" / "infer_results.mat").exists():
        return path / "SEG"
    return None


@dataclass
class SegmentationResult:
    """A loaded segmentation: lazy movie + sparse footprints + activity traces.

    `A` is a CaImAn-style sparse matrix (H*W, N): column n is neuron n's
    flattened (row-major) binary footprint. `C` is (N, T) activity. `video` is
    a lazy (T_video, H, W) array (rmbg movie) or None when no rmbg blocks exist.
    `label_rgba` is a precomputed (H, W, 4) colored overlay; `label_idx` is an
    (H, W) int32 where each pixel holds the topmost neuron index (-1 = none) for
    click hit-testing.
    """

    name: str
    root: Path
    A: "sp.csc_matrix"
    C: np.ndarray
    H: int
    W: int
    fps: float
    video: Any | None
    label_rgba: np.ndarray
    label_idx: np.ndarray
    colors: np.ndarray  # (N, 3) uint8, color assigned to each neuron
    # Per-neuron cell-type label (length N, indexed by neuron id), one of
    # `CELL_LABELS`; default "unknown". "bad" neurons are hidden from display +
    # hit-testing (see `bad`). Labels persist to `label_path` (SEG/cell_labels.json)
    # only on an explicit save (`save_labels()`), never automatically.
    labels: list = field(default_factory=list)
    label_path: "Path | None" = None
    # Set by `set_label` when labels diverge from disk; cleared by `save_labels`.
    labels_dirty: bool = field(default=False, repr=False)
    # Optional demixed temporal traces (from SEG_demix_validation/), index-
    # aligned with C: `C_demix[n]` is neuron n's demixed trace; `demixed[n]`
    # flags whether it was actually demixed. None when no demix data is present.
    C_demix: "np.ndarray | None" = None
    demixed: "np.ndarray | None" = None

    @property
    def n_neurons(self) -> int:
        return int(self.A.shape[1])

    @property
    def bad(self) -> set:
        """Set of neuron indices currently labeled "bad" (cached; invalidated by
        `set_label`). These are hidden from the colored overlay + hit-testing."""
        b = getattr(self, "_bad_cache", None)
        if b is None:
            b = {i for i, lab in enumerate(self.labels) if lab == "bad"}
            self._bad_cache = b
        return b

    @property
    def has_demix(self) -> bool:
        return self.C_demix is not None

    def is_demixed(self, n: int) -> bool:
        return self.demixed is not None and bool(self.demixed[int(n)])

    @property
    def n_frames(self) -> int:
        return int(self.C.shape[1])

    def footprint(self, n: int) -> np.ndarray:
        """Boolean (H, W) mask for neuron n, rebuilt from the sparse column."""
        col = self.A[:, n].toarray().reshape(self.H, self.W)
        return col.astype(bool)

    def neuron_at(self, y: int, x: int) -> int:
        """Topmost neuron index covering raw-frame pixel (row=y, col=x), or -1."""
        if 0 <= y < self.H and 0 <= x < self.W:
            return int(self.label_idx[y, x])
        return -1

    def neurons_at(self, y: int, x: int) -> list[int]:
        """All neuron indices whose footprint covers pixel (row=y, col=x),
        sorted by footprint area ascending (smallest first). Used to cycle
        through overlapping cells on repeated clicks at the same spot."""
        if not (0 <= y < self.H and 0 <= x < self.W):
            return []
        pix = y * self.W + x
        csr = self._csr()
        cols = csr.indices[csr.indptr[pix] : csr.indptr[pix + 1]]
        if cols.size == 0:
            return []
        areas = self._areas()
        return sorted(
            (int(c) for c in cols if int(c) not in self.bad),
            key=lambda n: int(areas[n]),
        )

    def neurons_in_box(self, y0: int, y1: int, x0: int, x1: int) -> list[int]:
        """Neuron indices whose footprint centroid lies in the box [y0,y1]×[x0,x1]
        (inclusive, raw row/col coords). Used by the drag-box selection tool.
        Bad ROIs are excluded."""
        cy, cx = self._centroids()
        inside = (cy >= y0) & (cy <= y1) & (cx >= x0) & (cx <= x1)
        return [int(n) for n in np.flatnonzero(inside) if int(n) not in self.bad]

    def _centroids(self) -> tuple[np.ndarray, np.ndarray]:
        """Lazily cached per-neuron footprint centroid (cy, cx) in raw coords.
        Empty footprints get NaN so they never fall inside a box."""
        c = getattr(self, "_centroid_cache", None)
        if c is None:
            coo = self.A.tocoo()
            n = self.A.shape[1]
            area = np.asarray(self._areas(), dtype=np.float64)
            sy = np.bincount(coo.col, weights=coo.row // self.W, minlength=n)
            sx = np.bincount(coo.col, weights=coo.row % self.W, minlength=n)
            with np.errstate(invalid="ignore", divide="ignore"):
                cy = np.where(area > 0, sy / area, np.nan)
                cx = np.where(area > 0, sx / area, np.nan)
            c = (cy, cx)
            self._centroid_cache = c
        return c

    def _csr(self):
        """Lazily cached CSR view of A for fast single-pixel (row) lookups."""
        csr = getattr(self, "_csr_cache", None)
        if csr is None:
            csr = self.A.tocsr()
            self._csr_cache = csr
        return csr

    def _areas(self) -> np.ndarray:
        """Lazily cached per-neuron footprint area (nnz of each A column)."""
        areas = getattr(self, "_area_cache", None)
        if areas is None:
            areas = np.diff(self.A.indptr)  # A is CSC: indptr deltas = column nnz
            self._area_cache = areas
        return areas

    def neuron_color(self, n: int) -> tuple[int, int, int]:
        r, g, b = self.colors[n]
        return int(r), int(g), int(b)

    # ----- cell-type labels (non-destructive, persisted to label_path) -----

    def label_of(self, n: int) -> str:
        """The cell-type label of neuron n (one of `CELL_LABELS`)."""
        return self.labels[int(n)]

    def set_label(self, n: int, label: str) -> bool:
        """Set neuron n's cell-type label (in memory). Returns True if it changed.
        Invalidates the bad-set + display-image caches so overlays rebuild.
        Persist explicitly via `save_labels()` — labels are never auto-saved."""
        if label not in CELL_LABELS:
            raise ValueError(f"Unknown cell-type label: {label!r}")
        n = int(n)
        changed = self.labels[n] != label
        if changed:
            self.labels[n] = label
            self.labels_dirty = True
            self._bad_cache = None  # bad membership may have changed
            self._label_mode_cache = {}  # overlays rebuild without/with this cell
        return changed

    def neurons_with_label(self, label: str) -> list[int]:
        """0-based indices of every neuron carrying `label` (ascending)."""
        return [i for i, lab in enumerate(self.labels) if lab == label]

    def label_counts(self) -> dict:
        """{label: [neuron indices]} for each label that has at least one neuron."""
        out: dict[str, list[int]] = {}
        for i, lab in enumerate(self.labels):
            out.setdefault(lab, []).append(i)
        return out

    def save_labels(self) -> bool:
        """Persist per-neuron labels to SEG/cell_labels.json, writing an entry for
        every neuron (default "unknown"). Returns True on success. Created on the
        first save; non-fatal on failure (e.g. a read-only archive volume)."""
        if self.label_path is None:
            return False
        try:
            self.label_path.write_text(
                json.dumps(
                    {"labels": {str(i): lab for i, lab in enumerate(self.labels)}},
                    indent=0,
                )
            )
            self.labels_dirty = False
            return True
        except OSError as exc:
            print(f"segmentation: could not save {self.label_path} ({exc})")
            return False

    # ----- bad-ROI helpers (bad is just the "bad" label) -----

    def is_bad(self, n: int) -> bool:
        return self.labels[int(n)] == "bad"

    def bad_neurons(self) -> list[int]:
        return sorted(self.bad)

    def set_bad(self, n: int, bad: bool = True) -> bool:
        """Convenience: mark/unmark neuron n bad via its label. Unmarking resets
        the label to "unknown". Returns True if it changed."""
        return self.set_label(int(n), "bad" if bad else DEFAULT_LABEL)

    def _good_coo(self) -> tuple[np.ndarray, np.ndarray]:
        """(pixel_row, neuron_col) of A's nonzeros, excluding bad neurons,
        ordered so higher neuron index wins on overlap."""
        coo = self.A.tocoo()
        rows, cols = coo.row, coo.col
        if self.bad:
            keep = ~np.isin(cols, np.fromiter(self.bad, dtype=np.int64))
            rows, cols = rows[keep], cols[keep]
        order = np.argsort(cols, kind="stable")
        return rows[order], cols[order]

    def label_image(self, mode: str) -> np.ndarray:
        """Colored (H, W, 4) overlay of the *good* footprints in the requested
        render mode: ``"fill"`` (solid translucent), ``"outline"`` (contour), or
        ``"center"`` (a dot per neuron). Bad ROIs are excluded. Cached per mode;
        the cache is cleared by `set_label`."""
        cache = getattr(self, "_label_mode_cache", None)
        if cache is None:
            cache = {}
            self._label_mode_cache = cache
        if mode not in cache:
            builder = {
                "fill": self._build_fill_rgba,
                "outline": self._build_outline_rgba,
                "center": self._build_center_rgba,
            }.get(mode)
            if builder is None:
                raise ValueError(f"Unknown footprint mode: {mode!r}")
            cache[mode] = builder()
        return cache[mode]

    def _build_fill_rgba(self) -> np.ndarray:
        rows, cols = self._good_coo()
        rgba = np.zeros((self.H * self.W, 4), dtype=np.uint8)
        rgba[rows, :3] = self.colors[cols]
        rgba[rows, 3] = _LABEL_ALPHA
        return rgba.reshape(self.H, self.W, 4)

    def _good_idx(self) -> np.ndarray:
        """(H, W) topmost good-neuron index per pixel (-1 = none/bad)."""
        rows, cols = self._good_coo()
        idx = np.full(self.H * self.W, -1, dtype=np.int32)
        idx[rows] = cols
        return idx.reshape(self.H, self.W)

    def _build_outline_rgba(self) -> np.ndarray:
        """Contour of each good footprint: pixels whose 4-neighborhood crosses a
        label boundary, colored by neuron (opaque)."""
        idx = self._good_idx()
        H, W = idx.shape
        nb = np.zeros_like(idx, dtype=bool)
        nb[:-1, :] |= idx[:-1, :] != idx[1:, :]
        nb[1:, :] |= idx[1:, :] != idx[:-1, :]
        nb[:, :-1] |= idx[:, :-1] != idx[:, 1:]
        nb[:, 1:] |= idx[:, 1:] != idx[:, :-1]
        boundary = (idx >= 0) & nb
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[boundary, :3] = self.colors[idx[boundary]]
        rgba[boundary, 3] = 255
        return rgba

    def _build_center_rgba(self, radius: int = 2) -> np.ndarray:
        """A small filled square at each good footprint's centroid."""
        cy, cx = self._centroids()
        H, W = self.H, self.W
        valid = ~np.isnan(cy)
        if self.bad:
            valid = valid & ~np.isin(np.arange(len(cy)), np.fromiter(self.bad, np.int64))
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        ys = np.round(cy[valid]).astype(int)
        xs = np.round(cx[valid]).astype(int)
        cols = self.colors[np.flatnonzero(valid)]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                yy = np.clip(ys + dy, 0, H - 1)
                xx = np.clip(xs + dx, 0, W - 1)
                rgba[yy, xx, :3] = cols
                rgba[yy, xx, 3] = 255
        return rgba

    def bad_image(self) -> np.ndarray:
        """(H, W, 4) gray overlay of the bad footprints, for the 'Show bad'
        toggle. Empty when nothing is marked bad."""
        rgba = np.zeros((self.H * self.W, 4), dtype=np.uint8)
        if self.bad:
            coo = self.A.tocoo()
            keep = np.isin(coo.col, np.fromiter(self.bad, dtype=np.int64))
            rgba[coo.row[keep], :3] = (128, 128, 128)
            rgba[coo.row[keep], 3] = 150
        return rgba.reshape(self.H, self.W, 4)

    def trace(self, n: int) -> "Trace":
        """Activity trace for neuron n as a SignalPanel Trace (0-based index,
        matching C row n and the SEG.tiff page n)."""
        from .signal_panel import Trace

        return Trace(
            name=f"neuron {n}",
            data=np.asarray(self.C[n], dtype=np.float32),
            sampling_rate=self.fps,
        )

    def trace_demix(self, n: int) -> "Trace":
        """Demixed activity trace for neuron n (requires `has_demix`)."""
        from .signal_panel import Trace

        return Trace(
            name=f"neuron {n}",
            data=np.asarray(self.C_demix[n], dtype=np.float32),
            sampling_rate=self.fps,
        )


def load_segmentation(path: str | Path, fps: float | None = None) -> SegmentationResult:
    """Load a segmentation result folder into a `SegmentationResult`."""
    import scipy.io as sio
    import scipy.sparse as sp

    root = Path(path)
    seg = _seg_dir(root)
    if seg is None:
        raise SegmentationLoadError(
            f"Not a segmentation folder (no SEG/infer_results.mat): {root}"
        )
    # Name from the project root (parent of SEG), not the SEG folder itself.
    project_root = seg.parent if seg.name.upper() == "SEG" else seg
    name = project_root.name or seg.name

    # --- traces: C (N, T) float32 ---
    mat = sio.loadmat(str(seg / "infer_results.mat"))
    if "C" not in mat:
        raise SegmentationLoadError(f"infer_results.mat has no 'C' variable: {seg}")
    C = np.ascontiguousarray(mat["C"], dtype=np.float32)
    if C.ndim != 2:
        raise SegmentationLoadError(f"Expected 2D C, got shape {C.shape}")

    # --- footprints: sparse A (H*W, N), cached ---
    A, H, W = _load_or_build_A(seg / "SEG.tiff", seg / _CACHE_NAME, sp)
    if A.shape[1] != C.shape[0]:
        raise SegmentationLoadError(
            f"Neuron count mismatch: A has {A.shape[1]} footprints, C has {C.shape[0]} rows"
        )

    # --- per-neuron colors + colored/index label images ---
    colors = _PALETTE[np.arange(A.shape[1]) % len(_PALETTE)]
    label_rgba, label_idx = _build_label_images(A, H, W, colors)

    # --- optional rmbg movie (lazy concatenated blocks) ---
    video = _load_rmbg_movie(project_root)

    # --- cell-type labels (persisted beside the segmentation; legacy bad list
    # migrated on first load) ---
    label_path = seg / _LABELS_NAME
    labels = _load_labels(label_path, seg / _BAD_LEGACY_NAME, A.shape[1])

    # --- optional demixed traces (SEG_demix_validation/) ---
    C_demix, demixed = _load_demix(
        project_root / "SEG_demix_validation" / "infer_results_demix.mat",
        A.shape[1],
        sio,
    )

    return SegmentationResult(
        name=name,
        root=project_root,
        A=A,
        C=C,
        H=H,
        W=W,
        fps=float(fps) if fps else DEFAULT_FPS,
        video=video,
        label_rgba=label_rgba,
        label_idx=label_idx,
        colors=colors,
        labels=labels,
        label_path=label_path,
        C_demix=C_demix,
        demixed=demixed,
    )


def _load_demix(path: Path, n_neurons: int, sio) -> tuple:
    """Load demixed traces from infer_results_demix.mat → (C_demix, demixed).

    Returns (None, None) when absent or malformed. `C_demix` is (N, T) float32
    index-aligned with C; `demixed` is an (N,) bool mask (or None)."""
    if not path.exists():
        return None, None
    try:
        m = sio.loadmat(str(path))
        if "C_demix" not in m:
            print(f"segmentation: {path} has no 'C_demix'; ignoring demix")
            return None, None
        cd = np.ascontiguousarray(m["C_demix"], dtype=np.float32)
        if cd.ndim != 2 or cd.shape[0] != n_neurons:
            print(f"segmentation: C_demix shape {cd.shape} != {n_neurons} neurons; ignoring")
            return None, None
        dm = None
        if "demixed" in m:
            flags = np.asarray(m["demixed"]).reshape(-1).astype(bool)
            if flags.size == n_neurons:
                dm = flags
        return cd, dm
    except (OSError, ValueError, KeyError) as exc:
        print(f"segmentation: could not read {path} ({exc}); ignoring demix")
        return None, None


def _load_labels(label_path: Path, bad_legacy: Path, n_neurons: int) -> list:
    """Per-neuron cell-type labels (length n_neurons, default "unknown").

    Reads SEG/cell_labels.json when present. Otherwise migrates a legacy
    SEG/bad_rois.json (each listed index becomes label "bad"). Every neuron gets
    a label; unknown/out-of-range entries fall back to the default."""
    labels = [DEFAULT_LABEL] * n_neurons
    if label_path.exists():
        try:
            data = json.loads(label_path.read_text())
            raw = data.get("labels", {}) if isinstance(data, dict) else {}
            for k, v in raw.items():
                i = int(k)
                if 0 <= i < n_neurons and v in CELL_LABELS:
                    labels[i] = v
        except (OSError, ValueError, TypeError) as exc:
            print(f"segmentation: could not read {label_path} ({exc}); ignoring")
        return labels
    # No labels file yet — migrate any legacy bad-ROI list into "bad" labels.
    for i in _load_bad(bad_legacy, n_neurons):
        labels[i] = "bad"
    return labels


def _load_bad(path: Path, n_neurons: int) -> set:
    """Read SEG/bad_rois.json → set of valid 0-based indices (empty if absent)."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
        ids = data.get("bad_neurons", []) if isinstance(data, dict) else data
        return {int(i) for i in ids if 0 <= int(i) < n_neurons}
    except (OSError, ValueError, TypeError) as exc:
        print(f"segmentation: could not read {path} ({exc}); ignoring")
        return set()


def _load_or_build_A(
    seg_tiff: Path, cache: Path, sp
) -> tuple["sp.csc_matrix", int, int]:
    """Return (A, H, W). Build the sparse footprint matrix from SEG.tiff and
    cache it; reuse the cache when it is newer than the source TIFF."""
    if cache.exists() and cache.stat().st_mtime >= seg_tiff.stat().st_mtime:
        npz = np.load(cache, allow_pickle=False)
        A = sp.csc_matrix(
            (npz["data"], npz["indices"], npz["indptr"]), shape=tuple(npz["shape"])
        )
        return A, int(npz["H"]), int(npz["W"])

    import tifffile

    if not seg_tiff.exists():
        raise SegmentationLoadError(f"Missing SEG.tiff: {seg_tiff}")

    arr = tifffile.memmap(seg_tiff, mode="r")  # (N, H, W) uint16 binary
    if arr.ndim != 3:
        raise SegmentationLoadError(f"Expected (N, H, W) SEG.tiff, got {arr.shape}")
    N, H, W = (int(d) for d in arr.shape)

    # Build (H*W, N) sparse: gather each footprint's flat pixel indices per page
    # so peak memory is one (H, W) page, not the whole stack.
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for n in range(N):
        flat = np.flatnonzero(np.asarray(arr[n]).reshape(-1))
        if flat.size:
            rows.append(flat)
            cols.append(np.full(flat.size, n, dtype=np.int32))
    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.empty(0, dtype=np.int32)
    data = np.ones(row.size, dtype=np.uint8)
    A = sp.csc_matrix((data, (row, col)), shape=(H * W, N))

    try:
        np.savez(
            cache,
            data=A.data,
            indices=A.indices,
            indptr=A.indptr,
            shape=np.asarray(A.shape),
            H=H,
            W=W,
        )
    except OSError as exc:  # read-only archive volume etc. — non-fatal
        print(f"segmentation: could not cache {cache} ({exc}); will rebuild next time")
    return A, H, W


def _build_label_images(
    A: "sp.csc_matrix", H: int, W: int, colors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """From sparse A build (label_rgba (H,W,4) uint8, label_idx (H,W) int32).

    Overlapping footprints resolve by last-writer-wins (higher neuron index on
    top), which is fine for display and topmost-hit click selection.
    """
    coo = A.tocoo()
    order = np.argsort(coo.col, kind="stable")  # ensure higher col overwrites lower
    rows = coo.row[order]
    cols = coo.col[order]

    rgba = np.zeros((H * W, 4), dtype=np.uint8)
    rgba[rows, :3] = colors[cols]
    rgba[rows, 3] = _LABEL_ALPHA

    idx = np.full(H * W, -1, dtype=np.int32)
    idx[rows] = cols
    return rgba.reshape(H, W, 4), idx.reshape(H, W)


class _BlockSeries:
    """Lazy (T, H, W) array-like over contiguous, uncompressed TIFF blocks
    concatenated along the time axis.

    Frames are read by byte offset through buffered file handles (seek + read),
    NOT memory-mapped. This is deliberate: memmap is ~0 ms/frame but every
    touched page is charged to the process working set, so scrubbing a ~95 GB
    movie inflates RSS into the GB range. Buffered reads (~2 ms/frame, plenty
    for playback) land in the shared OS file cache instead, so RSS stays flat
    (~tens of MB) regardless of how much of the movie you play.

    Exposes the minimal interface framewise needs: ``shape``/``dtype``/``ndim``/
    ``len`` plus integer- and slice-based ``__getitem__`` (including the
    ``[t0:t1, y0:y1, x0:x1]`` form the ROI extractor uses). A lock guards the
    shared file cursors so the GUI playback thread and a background reader
    (ROI extraction / console) can read concurrently.

    Earlier alternatives and why they were dropped: ``da.from_array(memmap)``
    copies the whole 2.5 GB block (dask copies array-likes exposing ``.copy()``);
    ``da.from_zarr`` + tifffile's zarr store was ~600 ms/frame (graph rebuild +
    zarr decode each access), making playback unusable.
    """

    def __init__(
        self,
        paths: list[str],
        offsets: list[int],
        counts: list[int],
        hw: tuple[int, int],
        dtype: np.dtype,
    ) -> None:
        import threading

        self._paths = list(paths)
        self._offsets = [int(o) for o in offsets]  # byte offset of frame 0 / block
        self._counts = [int(c) for c in counts]  # frames per block
        self._h, self._w = int(hw[0]), int(hw[1])
        self.dtype = np.dtype(dtype)
        self._fb = self._h * self._w * self.dtype.itemsize  # bytes per frame
        self._starts = np.cumsum([0, *self._counts])  # global start of each block
        self.shape = (int(self._starts[-1]), self._h, self._w)
        self.ndim = 3
        self._lock = threading.Lock()
        self._handles: dict[int, Any] = {}

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype=None):  # guard: the full movie is ~tens of GB
        raise RuntimeError(
            "_BlockSeries is lazy; index frames/slices rather than converting "
            "the whole movie to an array"
        )

    def _block_of(self, f: int) -> tuple[int, int]:
        """(block index, local frame) for a global frame index."""
        bi = int(np.searchsorted(self._starts, f, side="right") - 1)
        return bi, f - int(self._starts[bi])

    def _read(self, bi: int, local: int, n: int) -> np.ndarray:
        """Read `n` contiguous frames starting at `local` within block `bi`."""
        with self._lock:
            h = self._handles.get(bi)
            if h is None:
                h = open(self._paths[bi], "rb", buffering=0)
                self._handles[bi] = h
            h.seek(self._offsets[bi] + local * self._fb)
            buf = h.read(n * self._fb)
        return np.frombuffer(buf, dtype=self.dtype).reshape(n, self._h, self._w)

    def __getitem__(self, idx):
        tsel, rest = (idx[0], idx[1:]) if isinstance(idx, tuple) else (idx, ())

        if isinstance(tsel, (int, np.integer)):
            f = int(tsel)
            if f < 0:
                f += self.shape[0]
            bi, local = self._block_of(f)
            frame = self._read(bi, local, 1)[0]
            return frame[rest] if rest else frame

        if isinstance(tsel, slice):
            start, stop, step = tsel.indices(self.shape[0])
            if step == 1:
                parts = []
                pos = start
                while pos < stop:
                    bi, local = self._block_of(pos)
                    take = min(self._counts[bi] - local, stop - pos)
                    chunk = self._read(bi, local, take)
                    parts.append(chunk[(slice(None), *rest)] if rest else chunk)
                    pos += take
                if not parts:
                    empty = np.empty((0, *self.shape[1:]), dtype=self.dtype)
                    return empty[(slice(None), *rest)] if rest else empty
                return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
            # Uncommon stepped slice: gather frame by frame.
            frames = [self[i] for i in range(start, stop, step)]
            arr = (
                np.stack(frames, axis=0)
                if frames
                else np.empty((0, *self.shape[1:]), dtype=self.dtype)
            )
            return arr[(slice(None), *rest)] if rest else arr

        raise TypeError(f"_BlockSeries index must be int or slice, got {type(tsel)}")

    def __del__(self):
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass


def _load_rmbg_movie(root: Path) -> Any | None:
    """Concatenate rmbg_b*.tif blocks into one lazy (T, H, W) movie.

    Returns None when no rmbg folder/blocks exist (movie is optional). The
    blocks are uncompressed contiguous TIFFs; we record each one's data offset
    and frame count (via a transient memmap, which is fast and exposes
    ``.offset``) and read frames by offset through `_BlockSeries`."""
    rmbg = root / "rmbg"
    if not rmbg.is_dir():
        return None
    blocks = sorted(
        glob.glob(str(rmbg / "rmbg_b*.tif")),
        key=lambda p: _block_index(Path(p).name),
    )
    if not blocks:
        return None

    import tifffile

    offsets: list[int] = []
    counts: list[int] = []
    hw: tuple[int, int] | None = None
    dtype: np.dtype | None = None
    for b in blocks:
        mm = tifffile.memmap(b, mode="r")  # transient: just for offset/shape
        offsets.append(int(mm.offset))
        counts.append(int(mm.shape[0]))
        hw = (int(mm.shape[1]), int(mm.shape[2]))
        dtype = mm.dtype
        del mm  # release the mapping; we read via buffered handles instead
    return _BlockSeries(blocks, offsets, counts, hw, dtype)


def _block_index(name: str) -> int:
    """Sort key for rmbg_bNNN.tif by numeric block index."""
    m = re.search(r"rmbg_b(\d+)", name)
    return int(m.group(1)) if m else 0
