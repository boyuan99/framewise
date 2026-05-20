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

    # QtWebEngine (the embedded Jupyter Lab dock) requires shared OpenGL
    # contexts to be enabled before the QApplication is created. Harmless when
    # WebEngine isn't installed.
    try:
        from PyQt6.QtCore import Qt

        QApplication.setAttribute(
            Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True
        )
    except Exception:
        pass

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    for path in args.files:
        window.add_video(path)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
