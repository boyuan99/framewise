"""Manages a collection of napari.Viewer instances, one per opened file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import napari

from .loaders import load


@dataclass
class ViewerEntry:
    viewer: napari.Viewer
    name: str
    path: Path


@dataclass
class ViewerPool:
    entries: list[ViewerEntry] = field(default_factory=list)

    def open(self, path: str | Path) -> ViewerEntry:
        path = Path(path)
        array, name = load(path)

        viewer = napari.Viewer(title=f"Framewise — {name}")
        viewer.add_image(array, name=name)

        entry = ViewerEntry(viewer=viewer, name=name, path=path)
        self.entries.append(entry)

        # Drop from pool when the window closes so sync logic doesn't dangle.
        viewer.window._qt_window.destroyed.connect(lambda *_: self._remove(entry))
        return entry

    def _remove(self, entry: ViewerEntry) -> None:
        if entry in self.entries:
            self.entries.remove(entry)
