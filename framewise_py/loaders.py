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
    return da.from_array(dataset, chunks=chunks)


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
