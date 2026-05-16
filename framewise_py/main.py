"""Entry point: QApplication + MainWindow with optional CLI files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="framewise",
        description="Multi-video viewer with hybrid synchronized playback.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Video files to open at startup (optional; File → Open also works).",
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    for path in args.files:
        window.add_video(path)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
