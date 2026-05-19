"""Sync group controller + Resource Manager dock UI.

`SyncController` owns the cross-panel sync groups: when any base panel's frame
changes, all other panels in the same group are scrubbed proportionally
(normalized 0–1 position). Overlays within a panel do NOT participate in sync —
they have their own independent playheads (see VideoPanel).

`ResourceManagerPanel` is the dock UI: a hierarchical tree showing each panel
and its overlays as children. Sync-group checkboxes appear only on top-level
video panel rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .settings import get_last_dir, set_last_dir_from_path
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
    frame_changed events between group members.

    Note: only the base panel's frame_changed signal triggers sync. Overlay
    scrubs intentionally do not fire frame_changed, so they stay isolated.
    """

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


class ResourceManagerPanel(QWidget):
    """Hierarchical resource tree: panels at the top level with overlays as
    children. Sync-group membership checkboxes live on the top-level video
    rows only; overlays don't participate in sync.
    """

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

        self.tree = QTreeWidget()
        self.tree.setColumnCount(1 + self._controller.n_groups)
        self.tree.setHeaderLabels(self._header_labels())
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, self.tree.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setRootIsDecorated(True)
        layout.addWidget(self.tree, stretch=1)

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

    def _header_labels(self) -> list[str]:
        return ["Resource"] + [f"Group {i + 1}" for i in range(self._controller.n_groups)]

    def _refresh_tree(self) -> None:
        # During shutdown, dock destroyed signals can fire after this widget's
        # children are already gone — skip silently in that case.
        try:
            n_groups = self._controller.n_groups
            self.tree.setColumnCount(1 + n_groups)
            self.tree.setHeaderLabels(self._header_labels())

            self.tree.clear()
            for entry in self._panel_manager.entries:
                top = QTreeWidgetItem([f"[{entry.kind}] {entry.panel.name}"])
                self.tree.addTopLevelItem(top)

                if isinstance(entry.panel, VideoPanel):
                    for col in range(n_groups):
                        self.tree.setItemWidget(
                            top, col + 1, self._make_group_checkbox(entry.panel, col)
                        )
                    for ov in entry.panel._overlays:  # noqa: SLF001
                        child = QTreeWidgetItem([f"[overlay {ov.kind}] {ov.name}"])
                        top.addChild(child)
                    top.setExpanded(True)
        except RuntimeError:
            return

    def _make_group_checkbox(self, panel: VideoPanel, group_id: int) -> QWidget:
        cb = QCheckBox()
        cb.setChecked(self._controller.is_member(panel, group_id))
        cb.stateChanged.connect(
            lambda state, p=panel, g=group_id: self._controller.set_membership(
                p, g, state == Qt.CheckState.Checked.value
            )
        )
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(cb, alignment=Qt.AlignmentFlag.AlignCenter)
        return container

    def _on_panel_added(self, entry: "PanelEntry") -> None:
        # Sync grouping currently only applies to VideoPanels. Subscribe to
        # overlay changes so we can refresh the tree without polling.
        if isinstance(entry.panel, VideoPanel):
            self._controller.add_panel(entry.panel)
            entry.panel.overlay_added.connect(lambda *_: self._refresh_tree())
            entry.panel.overlay_removed.connect(lambda *_: self._refresh_tree())
        self._refresh_tree()

    def _on_panel_removed(self, entry: "PanelEntry") -> None:
        if isinstance(entry.panel, VideoPanel):
            self._controller.remove_panel(entry.panel)
        self._refresh_tree()

    def _add_group(self) -> None:
        self._controller.add_group()
        self._refresh_tree()

    def _remove_last_group(self) -> None:
        if self._controller.n_groups > 1:
            self._controller.remove_group(self._controller.n_groups - 1)
            self._refresh_tree()

    def _add_video(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add video files",
            get_last_dir(),
            FILE_FILTER,
        )
        if paths:
            set_last_dir_from_path(paths[0])
            self.add_video_requested.emit([str(p) for p in paths])

    def _add_tdt(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Add TDT block (select block directory)",
            get_last_dir(),
        )
        if directory:
            set_last_dir_from_path(directory)
            self.add_video_requested.emit([directory])
