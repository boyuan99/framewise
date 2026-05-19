"""Manages VideoPanel and SignalPanel dock widgets in the main window."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDockWidget, QMainWindow, QWidget

from .loaders import is_tdt_path, load, load_tdt
from .master_clock import MasterClock
from .signal_panel import SignalPanel
from .video_panel import VideoPanel

AnyPanel = Union[VideoPanel, SignalPanel]


@dataclass
class PanelEntry:
    panel: AnyPanel
    dock: QDockWidget
    path: Path
    kind: str  # "video" or "signal"


class PanelManager:
    """Owns the list of dock widgets and handles grid layout.

    If a MasterClock is passed at construction, every new panel auto-binds
    to it for frame-lock.
    """

    panel_added = pyqtSignal  # placeholder; callbacks used instead

    def __init__(
        self,
        main_window: QMainWindow,
        master_clock: MasterClock | None = None,
    ) -> None:
        self._main_window = main_window
        self._master_clock = master_clock
        self._entries: list[PanelEntry] = []
        self._on_added_callbacks: list = []
        self._on_removed_callbacks: list = []

    @property
    def entries(self) -> list[PanelEntry]:
        return list(self._entries)

    @property
    def video_entries(self) -> list[PanelEntry]:
        return [e for e in self._entries if e.kind == "video"]

    def on_added(self, callback) -> None:
        self._on_added_callbacks.append(callback)

    def on_removed(self, callback) -> None:
        self._on_removed_callbacks.append(callback)

    def add(self, path: str | Path) -> PanelEntry:
        """Dispatch by path type: TDT block dir → SignalPanel; else VideoPanel."""
        path = Path(path)
        if is_tdt_path(path):
            return self.add_signals(path)
        return self.add_video(path)

    def add_video(self, path: str | Path) -> PanelEntry:
        path = Path(path)
        array, name = load(path)
        panel = VideoPanel(array, name=name)
        return self._register(panel, name, path, "video")

    def add_signals(self, path: str | Path) -> PanelEntry:
        path = Path(path)
        traces, name = load_tdt(path)
        panel = SignalPanel(name=name, traces=traces)
        return self._register(panel, name, path, "signal")

    def _register(
        self,
        panel: AnyPanel,
        name: str,
        path: Path,
        kind: str,
    ) -> PanelEntry:
        if self._master_clock is not None:
            panel.bind_master_clock(self._master_clock)

        dock = QDockWidget(name, self._main_window)
        dock.setWidget(panel)
        dock.setObjectName(f"{kind}_{len(self._entries)}_{name}")
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        # X button truly deletes the dock (default would only hide it), so the
        # destroyed-signal chain below fires and SyncController + UI clean up.
        dock.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        entry = PanelEntry(panel=panel, dock=dock, path=path, kind=kind)
        self._entries.append(entry)
        self._place_in_grid(entry)

        dock.destroyed.connect(lambda *_: self._remove_entry(entry))

        for cb in self._on_added_callbacks:
            cb(entry)
        return entry

    def _place_in_grid(self, new_entry: PanelEntry) -> None:
        """Place the new dock so all open panels form a roughly square grid."""
        mw = self._main_window
        if len(self._entries) == 1:
            mw.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, new_entry.dock)
            return

        n = len(self._entries)
        cols = math.ceil(math.sqrt(n))
        idx = n - 1
        row, col = divmod(idx, cols)

        if row == 0:
            mw.splitDockWidget(
                self._entries[idx - 1].dock,
                new_entry.dock,
                Qt.Orientation.Horizontal,
            )
        else:
            above_idx = (row - 1) * cols + col
            anchor_idx = above_idx if above_idx < idx else idx - 1
            mw.splitDockWidget(
                self._entries[anchor_idx].dock,
                new_entry.dock,
                Qt.Orientation.Vertical,
            )

    def _remove_entry(self, entry: PanelEntry) -> None:
        if entry not in self._entries:
            return
        self._entries.remove(entry)
        for cb in self._on_removed_callbacks:
            cb(entry)
