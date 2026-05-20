"""Framewise — napari-based multi-video viewer for calcium imaging."""

import os

# Lock the Qt binding to PyQt6 before anything imports qtpy/qtconsole. qtpy
# resolves QT_API on first import and caches it; if qtconsole loads after some
# other library already pulled in PyQt5/PySide, it would bind to the wrong one
# and crash. Setting it here (the package root, imported before any submodule)
# guarantees the right binding. setdefault so an explicit override still wins.
os.environ.setdefault("QT_API", "pyqt6")

__version__ = "0.1.0"
