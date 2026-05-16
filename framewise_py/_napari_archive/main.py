"""Entry point: open one napari viewer per file (CLI args or file dialog)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import napari
from qtpy.QtWidgets import QApplication, QFileDialog

from .viewer_pool import ViewerPool

FILE_FILTER = (
    "All supported (*.h5 *.hdf5 *.tif *.tiff *.mp4 *.avi *.mov *.mkv);;"
    "HDF5 (*.h5 *.hdf5);;"
    "TIFF (*.tif *.tiff);;"
    "Video (*.mp4 *.avi *.mov *.mkv);;"
    "All files (*)"
)


def pick_files() -> list[Path]:
    """Show a file picker and return the selected paths (empty if cancelled)."""
    app = QApplication.instance() or QApplication(sys.argv)
    paths, _ = QFileDialog.getOpenFileNames(
        None,
        "Open video files",
        "",
        FILE_FILTER,
    )
    # Keep the app alive — napari.run() will reuse this instance.
    _ = app
    return [Path(p) for p in paths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="framewise",
        description="Multi-video synchronized viewer for calcium imaging.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Video files to open. If omitted, a file dialog appears.",
    )
    args = parser.parse_args(argv)

    files: list[Path] = args.files or pick_files()
    if not files:
        print("No files selected.", file=sys.stderr)
        return 1

    pool = ViewerPool()
    for path in files:
        try:
            entry = pool.open(path)
            print(f"Opened: {entry.name} ({path})")
        except Exception as exc:
            print(f"Failed to open {path}: {exc}", file=sys.stderr)

    if not pool.entries:
        print("No files were opened.", file=sys.stderr)
        return 1

    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
