"""ROI model + ΔF/F trace extraction.

Ported from the standalone napari script (`hybrid_viewer_avi.py`) into framewise's
pyqtgraph world. This module is UI-free: it holds the per-ROI data model
(`RoiItem`), the geometry → mask rasterizer (`ellipse_mask`), a grayscale helper,
and a background `TraceExtractWorker` that walks the panel's lazy frame array and
computes ΔF/F.

Coordinate note: framewise displays each (H, W) frame transposed to (W, H) (see
`VideoPanel._normalize_frame_layout`). An ROI drawn on the ViewBox therefore lives
in displayed (W, H) pixel coords; `ellipse_mask` builds the mask in those coords
and transposes back to (H, W) so it indexes the raw frame directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .signal_panel import Trace

# Luminance weights for RGB → gray, matching the source script.
_LUMA = (0.21, 0.72, 0.07)


@dataclass
class RoiItem:
    """One ROI: a pyqtgraph ROI item on a VideoPanel's ViewBox plus its metadata.

    `roi` is the live `pg.EllipseROI` (typed Any to keep this module UI-free).
    `color` is the pen color used for the canvas item and reused as the ΔF/F
    trace color so the two stay visually linked.
    """

    id: str
    name: str
    roi: Any
    color: tuple = (255, 215, 0)  # gold default
    line_width: float = 1.5
    fill: bool = False
    fill_alpha: int = 60  # 0-255, used only when `fill` is True


def frame_to_gray(arr: np.ndarray) -> np.ndarray:
    """Collapse a single frame to 2D. RGB(A) → luminance; (H,W,1|2) → channel 0."""
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return _LUMA[0] * arr[..., 0] + _LUMA[1] * arr[..., 1] + _LUMA[2] * arr[..., 2]
    if arr.ndim == 3:
        return arr[..., 0]
    return arr


def ellipse_mask(
    pos: Any, size: Any, angle_deg: float, frame_hw: tuple[int, int]
) -> np.ndarray:
    """Rasterize a pyqtgraph EllipseROI to a boolean mask of shape (H, W).

    `pos` = bbox-corner (x, y) and `size` = (w, h) are in the displayed (W, H)
    coordinate frame (ViewBox == ImageItem pixels). `angle_deg` rotates about
    `pos` (the EllipseROI origin), matching pyqtgraph's transform. The mask is
    built in displayed (W, H) coords, then transposed to the raw (H, W) frame.
    """
    H, W = int(frame_hw[0]), int(frame_hw[1])
    px, py = float(pos[0]), float(pos[1])
    sw, sh = float(size[0]), float(size[1])
    if sw <= 0 or sh <= 0:
        return np.zeros((H, W), dtype=bool)

    theta = math.radians(float(angle_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    ax, ay = sw / 2.0, sh / 2.0

    # Displayed array is (W, H): axis 0 = x (cols), axis 1 = y (rows).
    x = np.arange(W, dtype=np.float64).reshape(W, 1) + 0.5
    y = np.arange(H, dtype=np.float64).reshape(1, H) + 0.5
    dx, dy = x - px, y - py

    # Rotate into the ROI's local frame (inverse rotation R(-theta)).
    lx = cos_t * dx + sin_t * dy
    ly = -sin_t * dx + cos_t * dy

    inside = ((lx - ax) / ax) ** 2 + ((ly - ay) / ay) ** 2 <= 1.0  # (W, H)
    return inside.T  # (H, W)


class TraceExtractWorker(QThread):
    """Extract ΔF/F traces for one or more ROI masks off the GUI thread.

    Reads each frame of the (lazy) array exactly once and applies every mask, so
    cost scales with the video length, not the ROI count. ΔF/F uses the 20th
    percentile as F0 (ported from the source script).

    Note: the custom completion signal is `traces_ready`, NOT `finished` — QThread
    already defines `finished`, so reusing that name would clash.
    """

    progress = pyqtSignal(str)
    traces_ready = pyqtSignal(object)  # list[Trace]

    def __init__(
        self,
        array: Any,
        masks_with_labels: list[tuple[str, np.ndarray]],
        n_frames: int,
        fps: float,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._array = array
        self._masks_with_labels = masks_with_labels
        self._n_frames = int(n_frames)
        self._fps = float(fps)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        import time

        labels = [lbl for lbl, _ in self._masks_with_labels]
        masks = [m for _, m in self._masks_with_labels]
        has_pixels = [bool(m.any()) for m in masks]
        traces = [np.zeros(self._n_frames, dtype=np.float64) for _ in masks]
        luma = np.array(_LUMA, dtype=np.float64)

        t0 = time.monotonic()
        if not any(has_pixels):
            actual = self._n_frames  # nothing to read; traces stay zero
        elif self._supports_slicing():
            actual = self._fill_blockwise(masks, has_pixels, traces, luma)
        else:
            flat_idx = [np.flatnonzero(m.reshape(-1)) for m in masks]
            actual = self._fill_per_frame(flat_idx, has_pixels, traces, luma)

        if actual is None:  # cancelled — don't emit during shutdown/cancel
            return
        dt = time.monotonic() - t0
        fps = actual / dt if dt > 0 else 0.0
        print(f"ROI extract: {actual} frames in {dt:.2f}s ({fps:.0f} fps)")

        results: list[Trace] = []
        for label, raw, ok in zip(labels, traces, has_pixels):
            tr = raw[:actual]
            if not ok or tr.size == 0:
                dff = tr.astype(np.float32)
            else:
                f0 = float(np.percentile(tr, 20))
                dff = ((tr - f0) / f0 if f0 != 0 else tr * 0.0).astype(np.float32)
            results.append(Trace(name=label, data=dff, sampling_rate=self._fps))

        self.traces_ready.emit(results)

    def _supports_slicing(self) -> bool:
        """Sliceable arrays (dask / numpy / memmap) take the fast block path;
        the imageio video wrapper supports only integer indexing → per-frame."""
        try:
            self._array[0:1]
            return True
        except TypeError:
            return False

    @staticmethod
    def _combined_bbox(masks) -> tuple[int, int, int, int]:
        """Tight (y0, y1, x0, x1) box (half-open) covering every mask's pixels."""
        any_mask = masks[0].copy()
        for m in masks[1:]:
            any_mask |= m
        rows = np.any(any_mask, axis=1)
        cols = np.any(any_mask, axis=0)
        y0 = int(np.argmax(rows))
        y1 = int(len(rows) - np.argmax(rows[::-1]))
        x0 = int(np.argmax(cols))
        x1 = int(len(cols) - np.argmax(cols[::-1]))
        return y0, y1, x0, x1

    def _open_h5_dataset(self):
        """If the source is an HDF5-backed array, open our OWN read-only handle
        so we can read big bbox hyperslabs in one call (dask reads per-frame
        chunks). Returns (file, dataset) or (None, None). A private handle keeps
        us off the GUI thread's shared handle."""
        info = getattr(self._array, "framewise_h5", None)
        if info is None:
            return None, None
        try:
            import h5py

            f = h5py.File(info[0], "r")
            return f, f[info[1]]
        except Exception as exc:
            print(f"ROI extract: direct HDF5 open failed ({exc}); using dask path")
            return None, None

    def _fill_blockwise(self, masks, has_pixels, traces, luma) -> int | None:
        """Read RAM-bounded blocks, cropped to the masks' combined bounding box,
        and gather masked pixels for the whole block at once. Reads HDF5 bbox
        hyperslabs directly when possible. Returns frames processed (or None)."""
        n = self._n_frames
        y0, y1, x0, x1 = self._combined_bbox(masks)
        # Mask indices within the cropped (bbox) frame, row-major.
        cropped_idx = [
            np.flatnonzero(m[y0:y1, x0:x1].reshape(-1)) for m in masks
        ]

        h5f, dset = self._open_h5_dataset()
        try:
            read = (
                (lambda s, e: np.asarray(dset[s:e, y0:y1, x0:x1]))
                if dset is not None
                else (lambda s, e: np.asarray(self._array[s:e, y0:y1, x0:x1]))
            )

            sample = read(0, 1)
            color = sample.ndim == 4 and sample.shape[3] >= 3
            per_frame_bytes = sample.itemsize * int(np.prod(sample.shape[1:]) or 1)
            # ~256 MB/block, capped so each read stays short enough for
            # responsive cancel/close AND so progress updates often enough.
            block = max(1, int(256 * 1024 * 1024 // max(1, per_frame_bytes)))
            block = min(block, 1024)

            actual = 0
            for start in range(0, n, block):
                if self._cancelled:
                    return None
                end = min(start + block, n)
                # Emit BEFORE the (blocking) read so the bar updates as each
                # block begins, not only after it finishes.
                pct = int(start / n * 100) if n else 0
                self.progress.emit(f"Extracting ΔF/F: {pct}% ({start}/{n})")
                try:
                    blk = read(start, end)
                except Exception as exc:
                    print(f"ROI extract: block {start}:{end} read failed ({exc})")
                    break
                b = blk.shape[0]
                if color:
                    flat = blk.reshape(b, -1, blk.shape[3])  # (B, bh*bw, C)
                    for k, idx in enumerate(cropped_idx):
                        if has_pixels[k]:
                            g = flat[:, idx, :3].astype(np.float64) @ luma  # (B, n)
                            traces[k][start : start + b] = g.mean(axis=1)
                else:
                    if blk.ndim == 4:  # (B, bh, bw, 1|2) — use channel 0
                        blk = blk[..., 0]
                    flat = blk.reshape(b, -1)  # (B, bh*bw)
                    for k, idx in enumerate(cropped_idx):
                        if has_pixels[k]:
                            traces[k][start : start + b] = (
                                flat[:, idx].astype(np.float64).mean(axis=1)
                            )
                actual += b
                pct = int(end / n * 100) if n else 0
                self.progress.emit(f"Extracting ΔF/F: {pct}% ({end}/{n})")
            return actual
        finally:
            if h5f is not None:
                h5f.close()

    def _fill_per_frame(self, flat_idx, has_pixels, traces, luma) -> int | None:
        """Per-frame fallback for integer-indexing-only sources (video)."""
        n = self._n_frames
        actual = 0
        try:
            for i in range(n):
                if self._cancelled:
                    return None
                try:
                    f = np.asarray(self._array[i])
                except Exception as exc:  # video may report more frames than it has
                    print(f"ROI extract: frame {i} read failed ({exc}); stopping")
                    break
                if f.ndim == 3 and f.shape[2] >= 3:
                    flat = f.reshape(-1, f.shape[2])
                    for k, idx in enumerate(flat_idx):
                        if has_pixels[k]:
                            traces[k][i] = float(
                                (flat[idx, :3].astype(np.float64) @ luma).mean()
                            )
                else:
                    if f.ndim == 3:
                        f = f[..., 0]
                    flat = f.reshape(-1)
                    for k, idx in enumerate(flat_idx):
                        if has_pixels[k]:
                            traces[k][i] = float(flat[idx].mean())
                actual += 1
                if i % 200 == 0:
                    pct = int(i / n * 100) if n else 0
                    self.progress.emit(f"Extracting ΔF/F: {pct}% ({i}/{n})")
        except Exception:
            import traceback

            traceback.print_exc()
        return actual
