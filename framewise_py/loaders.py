"""Data loaders for HDF5, TIFF, video, and TDT block files.

Video-style loaders return an array-like with shape (T, H, W), (T, H, W, C),
or (T, C, H, W). The first axis is time/frame. Lazy access (per-frame
indexing) is preferred so large files don't load entirely into memory.

TDT loader returns a list of named 1D signal traces.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask.array as da
import h5py
import numpy as np

if TYPE_CHECKING:
    from .signal_panel import Trace


class LoadError(Exception):
    pass


def is_tdt_path(path: Path) -> bool:
    """A TDT block is a directory containing at least one .Tbk file."""
    return path.is_dir() and any(path.glob("*.Tbk"))


def load(path: str | Path) -> tuple[Any, str]:
    """Load a video-style file and return (array, name).

    The returned array has shape (T, ...) where T is the frame axis.
    For TDT blocks (signal data, not video), use `load_tdt()` instead.
    """
    path = Path(path)
    if not path.exists():
        raise LoadError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        return _load_hdf5(path), path.stem
    if suffix in (".tif", ".tiff"):
        return _load_tiff(path), path.stem
    if suffix in (".mp4", ".avi", ".mov", ".mkv"):
        return _load_video(path), path.stem
    raise LoadError(f"Unsupported file extension: {suffix}")


def load_tdt(path: str | Path) -> tuple[list["Trace"], str]:
    """Load a TDT block directory and return (traces, name).

    All 1D streams in the block are returned as traces. Multi-channel streams
    (e.g. neural recordings) are split into one trace per channel.
    """
    from .signal_panel import Trace

    path = Path(path)
    if not is_tdt_path(path):
        raise LoadError(f"Not a TDT block directory: {path}")

    import tdt

    block = tdt.read_block(str(path))
    traces: list[Trace] = []
    streams = getattr(block, "streams", None)
    if streams is None:
        raise LoadError(f"TDT block has no streams: {path}")

    for stream_name in streams.keys():
        stream = streams[stream_name]
        data = np.asarray(stream.data)
        fs = float(stream.fs)
        if data.ndim == 1:
            traces.append(Trace(name=stream_name, data=data, sampling_rate=fs))
        elif data.ndim == 2:
            # Multi-channel: split into one trace per channel
            for ch in range(data.shape[0]):
                traces.append(
                    Trace(
                        name=f"{stream_name}[{ch}]",
                        data=data[ch],
                        sampling_rate=fs,
                    )
                )
        # Skip higher-dim streams for now

    if not traces:
        raise LoadError(f"No 1D/2D streams found in TDT block: {path}")

    return traces, path.name


def _load_hdf5(path: Path) -> da.Array:
    """Open HDF5 file and return the first suitable dataset as a dask array.

    A "suitable" dataset is 3D (T, H, W) or 4D (T, C, H, W) with T >= 2.
    The file handle stays open for the lifetime of the dask array.
    """
    f = h5py.File(path, "r")
    dataset = _find_video_dataset(f)
    if dataset is None:
        f.close()
        raise LoadError(f"No 3D/4D dataset found in {path}")

    chunks = (1,) + dataset.shape[1:]
    arr = da.from_array(dataset, chunks=chunks)
    # Record the source so heavy readers (ROI extraction) can open their own
    # read-only handle and pull big contiguous/bbox hyperslabs in one call,
    # instead of dask's per-frame (1,H,W) chunk reads. Guarded: if the dask
    # Array forbids attribute assignment, callers just fall back to dask.
    try:
        arr.framewise_h5 = (str(path), dataset.name)
    except Exception:
        pass
    return arr


def _find_video_dataset(group: h5py.Group, path: str = "/") -> h5py.Dataset | None:
    """Recursively find the first dataset with shape (T, H, W) or (T, C, H, W)."""
    for name, item in group.items():
        full_path = f"{path.rstrip('/')}/{name}"
        if isinstance(item, h5py.Dataset):
            if item.ndim in (3, 4) and item.shape[0] >= 2:
                return item
        elif isinstance(item, h5py.Group):
            found = _find_video_dataset(item, full_path)
            if found is not None:
                return found
    return None


def _find_image_dataset(group: h5py.Group) -> h5py.Dataset | None:
    """Recursively find the first 2D dataset (single image), or 3D with T==1."""
    for _name, item in group.items():
        if isinstance(item, h5py.Dataset):
            if item.ndim == 2:
                return item
            if item.ndim == 3 and item.shape[0] == 1:
                return item
            if item.ndim == 3 and item.shape[-1] in (3, 4):
                return item  # (H, W, C)
        elif isinstance(item, h5py.Group):
            found = _find_image_dataset(item)
            if found is not None:
                return found
    return None


# Extensions we recognize as "video-like" (multi-frame) for overlay routing.
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
# Static-image-only extensions.
_STATIC_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
# TIFF/HDF5 can be either — inspected at load time.
_AMBIGUOUS_EXTS = {".tif", ".tiff", ".h5", ".hdf5"}


def load_overlay(path: str | Path) -> tuple[Any, str, str]:
    """Load a file for use as an overlay layer.

    Returns (data, kind, name) where:
      - kind == "static": data is an ndarray of shape (H, W) or (H, W, C)
      - kind == "video":  data is a lazy array of shape (T, H, W[, C]) with T >= 2

    Routes by extension first, then inspects content for TIFF/HDF5.
    """
    path = Path(path)
    if not path.exists():
        raise LoadError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in _STATIC_IMAGE_EXTS:
        return _load_static_image(path), "static", path.stem

    if suffix in _VIDEO_EXTS:
        return _load_video(path), "video", path.stem

    if suffix in (".tif", ".tiff"):
        arr = _load_tiff_any(path)
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] not in (3, 4)):
            return arr, "static", path.stem
        return arr, "video", path.stem

    if suffix in (".h5", ".hdf5"):
        return _load_hdf5_any(path, path.stem)

    raise LoadError(f"Unsupported overlay file extension: {suffix}")


def _load_static_image(path: Path) -> np.ndarray:
    """Load a single 2D image via imageio."""
    import imageio.v3 as iio

    arr = np.asarray(iio.imread(str(path)))
    if arr.ndim not in (2, 3):
        raise LoadError(f"Unsupported image shape {arr.shape} in {path}")
    return arr


def _load_tiff_any(path: Path) -> np.ndarray:
    """Read TIFF as-is (no T>=2 constraint), preferring memmap for large files."""
    import tifffile

    try:
        return tifffile.memmap(path, mode="r")
    except (ValueError, OSError):
        return tifffile.imread(path)


def _load_hdf5_any(path: Path, name: str) -> tuple[Any, str, str]:
    """Try video dataset first; fall back to single-image dataset."""
    f = h5py.File(path, "r")
    video_ds = _find_video_dataset(f)
    if video_ds is not None:
        chunks = (1,) + video_ds.shape[1:]
        return da.from_array(video_ds, chunks=chunks), "video", name

    image_ds = _find_image_dataset(f)
    if image_ds is not None:
        arr = np.asarray(image_ds[:])
        f.close()
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[-1] not in (3, 4):
            arr = arr[0]
        return arr, "static", name

    f.close()
    raise LoadError(f"No image/video dataset found in {path}")


def _load_tiff(path: Path) -> Any:
    """Load a multi-page TIFF stack. Uses memory-mapping when possible so
    huge files (>>RAM) don't load entirely into memory."""
    import tifffile

    try:
        arr = tifffile.memmap(path, mode="r")
    except (ValueError, OSError):
        # Compressed or otherwise non-mmapable — fall back to full read.
        arr = tifffile.imread(path)

    if arr.ndim < 3 or arr.shape[0] < 2:
        raise LoadError(
            f"TIFF must be a multi-frame stack with T >= 2, got shape {arr.shape}"
        )
    return arr


def _load_video(path: Path) -> "_VideoArray":
    """Load a compressed video file via imageio + ffmpeg.

    Frames are decoded on demand. Best performance comes from videos encoded
    with every frame as a keyframe (e.g. `ffmpeg -i in.mp4 -g 1 out.mp4`).
    """
    return _VideoArray(path)


class _VideoArray:
    """Lazy random-access wrapper around an imageio video reader.

    Exposes shape, dtype, ndim, and __getitem__(frame_idx) returning a numpy
    array, matching the minimal interface the rest of the app expects.
    """

    def __init__(self, path: Path) -> None:
        import imageio

        self._reader = imageio.get_reader(str(path))

        # Frame count: try count_frames(), fall back to get_length().
        try:
            n = self._reader.count_frames()
        except Exception:
            n = self._reader.get_length()
        if n is None or n < 1 or n == float("inf"):
            raise LoadError(f"Could not determine frame count for {path}")

        # Probe first frame for shape & dtype.
        frame0 = self._reader.get_data(0)
        self._frame_shape = frame0.shape
        self.dtype = frame0.dtype
        self.shape = (int(n),) + tuple(frame0.shape)
        self.ndim = len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(f"VideoArray supports only integer indexing, got {type(idx)}")
        return self._reader.get_data(int(idx))

    def __del__(self) -> None:
        try:
            self._reader.close()
        except Exception:
            pass
