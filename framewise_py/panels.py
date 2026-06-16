"""Central video/signal grid + the manager that owns the loaded panels.

Panels live in a central tiled grid (nested QSplitters) rather than as floating
dock widgets, so the video area fills the window and resizes freely, and so the
video grid can be one page of the main window's workspace stack (the other page
being the embedded notebook) — letting videos and notebook coexist while a tab
switches which is in view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QMdiArea,
    QMdiSubWindow,
    QStackedLayout,
    QWidget,
)

from .loaders import is_tdt_path, load, load_tdt
from .master_clock import MasterClock
from .segmentation import is_segmentation_dir, load_segmentation
from .segmentation_panel import SegmentationPanel
from .signal_panel import SignalPanel
from .video_panel import VideoPanel

AnyPanel = Union[VideoPanel, SignalPanel]


@dataclass
class PanelEntry:
    panel: AnyPanel
    path: Path
    kind: str  # "video" or "signal"


class _VideoSubWindow(QMdiSubWindow):
    """A loaded panel as a free-floating MDI sub-window: drag the title bar to
    move, drag edges to resize, native min/max/close — all bounded to the MDI
    area. Closing it routes through `on_close` (the manager) like the old dock
    X button did."""

    def __init__(
        self, entry: PanelEntry, on_close: Callable[[PanelEntry], None]
    ) -> None:
        super().__init__()
        self.entry = entry
        self.setWidget(entry.panel)
        self.setWindowTitle(f"[{entry.kind}] {entry.panel.name}")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(480, 360)
        # destroyed (after the X deletes it) drives manager cleanup, mirroring
        # the previous dock.destroyed wiring.
        self.destroyed.connect(lambda *_: on_close(entry))


class VideoGrid(QWidget):
    """Central widget hosting loaded panels as free-floating MDI sub-windows
    inside a bounded area. Switching the workspace away and back never destroys
    these (they live here permanently until closed), so videos persist."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._subs: list[_VideoSubWindow] = []

        self._stack = QStackedLayout(self)

        self._placeholder = QLabel("Drop videos here or use File → Open")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888;")

        self._mdi = QMdiArea()

        self._stack.addWidget(self._placeholder)  # index 0 — empty state
        self._stack.addWidget(self._mdi)  # index 1 — MDI area
        self._update_view()

    def add_panel(
        self, entry: PanelEntry, on_close: Callable[[PanelEntry], None]
    ) -> None:
        sub = _VideoSubWindow(entry, on_close)
        self._subs.append(sub)
        self._mdi.addSubWindow(sub)
        sub.show()
        self._update_view()

    def remove_panel(self, entry: PanelEntry) -> None:
        # The sub-window self-deletes (WA_DeleteOnClose); just drop bookkeeping.
        self._subs = [s for s in self._subs if s.entry is not entry]
        self._update_view()

    def tile(self) -> None:
        self._mdi.tileSubWindows()

    def cascade(self) -> None:
        self._mdi.cascadeSubWindows()

    def _update_view(self) -> None:
        # On app shutdown a sub-window's destroyed signal can fire after the
        # central QStackedLayout's C++ object is already gone; ignore that race
        # (mirrors ResourceManagerPanel._refresh_tree's RuntimeError guard).
        try:
            self._stack.setCurrentIndex(1 if self._subs else 0)
        except RuntimeError:
            return


class PanelManager:
    """Owns the list of loaded panels and routes them into the VideoGrid.

    If a MasterClock is passed at construction, every new panel auto-binds
    to it for frame-lock.
    """

    def __init__(
        self,
        grid: VideoGrid,
        master_clock: MasterClock | None = None,
    ) -> None:
        self._grid = grid
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
        """Dispatch by path type: segmentation result dir → SegmentationPanel;
        TDT block dir → SignalPanel; else VideoPanel."""
        path = Path(path)
        if is_segmentation_dir(path):
            return self.add_segmentation(path)
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

    def add_segmentation(self, path: str | Path) -> PanelEntry:
        path = Path(path)
        seg = load_segmentation(path)
        panel = SegmentationPanel(seg)
        return self._register(panel, seg.name, path, "segmentation")

    def register_signal_panel(self, panel: SignalPanel, name: str) -> PanelEntry:
        """Register a pre-built SignalPanel (e.g. ROI ΔF/F) with no source file.

        Routes it into the grid and binds the master clock like any other panel.
        """
        return self._register(panel, name, Path(""), "signal")

    def register_spectrogram_panel(self, panel, name: str) -> PanelEntry:
        """Register a pre-built SpectrogramPanel (e.g. CWT heatmap from a notebook).

        Same wiring as register_signal_panel; panel type-checked via duck typing
        (must expose `bind_master_clock` and `name`) to avoid an import cycle.
        """
        return self._register(panel, name, Path(""), "spectrogram")

    def _register(
        self,
        panel: AnyPanel,
        name: str,
        path: Path,
        kind: str,
    ) -> PanelEntry:
        if self._master_clock is not None:
            panel.bind_master_clock(self._master_clock)

        entry = PanelEntry(panel=panel, path=path, kind=kind)
        self._entries.append(entry)
        self._grid.add_panel(entry, self._remove_entry)

        for cb in self._on_added_callbacks:
            cb(entry)
        return entry

    def _remove_entry(self, entry: PanelEntry) -> None:
        if entry not in self._entries:
            return
        self._entries.remove(entry)
        for cb in self._on_removed_callbacks:
            cb(entry)
        self._grid.remove_panel(entry)
