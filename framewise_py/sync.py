"""Sync group controller + dock widget UI.

Each sync group is a set of VideoPanels that should scrub together. When any
panel in a group emits frame_changed, the others are updated proportionally
(by normalized time, so videos with different frame counts align by position).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .video_panel import VideoPanel

if TYPE_CHECKING:
    from .panels import PanelEntry, PanelManager

FILE_FILTER = (
    "All supported (*.h5 *.hdf5 *.tif *.tiff *.mp4 *.avi *.mov *.mkv);;"
    "HDF5 (*.h5 *.hdf5);;"
    "TIFF (*.tif *.tiff);;"
    "Video (*.mp4 *.avi *.mov *.mkv);;"
    "All files (*)"
)


class SyncController:
    """Tracks which VideoPanels belong to which sync groups and routes
    frame_changed events between group members."""

    def __init__(self) -> None:
        # Each group is a set of VideoPanel instances. Index = group id.
        self._groups: list[set[VideoPanel]] = [set()]
        self._panels: list[VideoPanel] = []
        self._dispatching = False

    @property
    def n_groups(self) -> int:
        return len(self._groups)

    @property
    def panels(self) -> list[VideoPanel]:
        return list(self._panels)

    def add_panel(self, panel: VideoPanel) -> None:
        if panel in self._panels:
            return
        self._panels.append(panel)
        panel.frame_changed.connect(lambda f, p=panel: self._on_panel_scrubbed(p, f))

    def remove_panel(self, panel: VideoPanel) -> None:
        if panel not in self._panels:
            return
        self._panels.remove(panel)
        for group in self._groups:
            group.discard(panel)

    def add_group(self) -> int:
        self._groups.append(set())
        return len(self._groups) - 1

    def remove_group(self, group_id: int) -> None:
        if 0 <= group_id < len(self._groups) and len(self._groups) > 1:
            self._groups.pop(group_id)

    def set_membership(self, panel: VideoPanel, group_id: int, member: bool) -> None:
        if not (0 <= group_id < len(self._groups)):
            return
        if member:
            self._groups[group_id].add(panel)
        else:
            self._groups[group_id].discard(panel)

    def is_member(self, panel: VideoPanel, group_id: int) -> bool:
        return 0 <= group_id < len(self._groups) and panel in self._groups[group_id]

    def _on_panel_scrubbed(self, source: VideoPanel, frame: int) -> None:
        if self._dispatching:
            return
        # Find groups containing source, sync all other members proportionally.
        self._dispatching = True
        try:
            t_norm = frame / max(1, source.n_frames - 1)
            for group in self._groups:
                if source not in group:
                    continue
                for member in group:
                    if member is source:
                        continue
                    target = int(round(t_norm * (member.n_frames - 1)))
                    member.set_frame_from_sync(target)
        finally:
            self._dispatching = False


class SyncManagerPanel(QWidget):
    """Dock widget UI showing the panel × group membership table."""

    add_video_requested = pyqtSignal(list)  # list[Path] from file dialog

    def __init__(
        self,
        controller: SyncController,
        panel_manager: "PanelManager",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._panel_manager = panel_manager
        self._build_ui()

        panel_manager.on_added(self._on_panel_added)
        panel_manager.on_removed(self._on_panel_removed)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.table = QTableWidget(0, 1)
        self.table.setHorizontalHeaderLabels(["Group 1"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, stretch=1)

        btn_row = QHBoxLayout()
        btn_add_group = QPushButton("+ Group")
        btn_add_group.clicked.connect(self._add_group)
        btn_row.addWidget(btn_add_group)

        btn_remove_group = QPushButton("− Group")
        btn_remove_group.clicked.connect(self._remove_last_group)
        btn_row.addWidget(btn_remove_group)

        btn_row.addStretch()

        btn_add_video = QPushButton("Add Video…")
        btn_add_video.clicked.connect(self._add_video)
        btn_row.addWidget(btn_add_video)

        btn_add_tdt = QPushButton("Add TDT…")
        btn_add_tdt.clicked.connect(self._add_tdt)
        btn_row.addWidget(btn_add_tdt)

        layout.addLayout(btn_row)

    def _refresh_table(self) -> None:
        # During shutdown, dock destroyed signals can fire after this widget's
        # children are already gone — skip silently in that case.
        try:
            entries = self._panel_manager.entries
            groups = self._controller.n_groups
            self.table.setRowCount(len(entries))
            self.table.setColumnCount(groups)
            self.table.setHorizontalHeaderLabels(
                [f"Group {i+1}" for i in range(groups)]
            )
            self.table.setVerticalHeaderLabels(
                [f"[{e.kind}] {e.panel.name}" for e in entries]
            )

            for row, entry in enumerate(entries):
                for col in range(groups):
                    if not isinstance(entry.panel, VideoPanel):
                        self.table.setCellWidget(row, col, QLabel(""))
                        continue
                    cb = QCheckBox()
                    cb.setChecked(self._controller.is_member(entry.panel, col))
                    cb.stateChanged.connect(
                        lambda state, p=entry.panel, g=col: self._controller.set_membership(
                            p, g, state == Qt.CheckState.Checked.value
                        )
                    )
                    container = QWidget()
                    h = QHBoxLayout(container)
                    h.setContentsMargins(0, 0, 0, 0)
                    h.addWidget(cb, alignment=Qt.AlignmentFlag.AlignCenter)
                    self.table.setCellWidget(row, col, container)
        except RuntimeError:
            return

    def _on_panel_added(self, entry: "PanelEntry") -> None:
        # Sync grouping currently only applies to VideoPanels.
        if isinstance(entry.panel, VideoPanel):
            self._controller.add_panel(entry.panel)
        self._refresh_table()

    def _on_panel_removed(self, entry: "PanelEntry") -> None:
        if isinstance(entry.panel, VideoPanel):
            self._controller.remove_panel(entry.panel)
        self._refresh_table()

    def _add_group(self) -> None:
        self._controller.add_group()
        self._refresh_table()

    def _remove_last_group(self) -> None:
        if self._controller.n_groups > 1:
            self._controller.remove_group(self._controller.n_groups - 1)
            self._refresh_table()

    def _add_video(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add video files",
            "",
            FILE_FILTER,
        )
        if paths:
            self.add_video_requested.emit([str(p) for p in paths])

    def _add_tdt(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Add TDT block (select block directory)",
            "",
        )
        if directory:
            self.add_video_requested.emit([directory])
