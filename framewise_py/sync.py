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
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QMenu,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .settings import get_last_dir, set_last_dir_from_path
from .signal_panel import SignalPanel
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


class _RoiPropertiesDialog(QDialog):
    """Edit an ROI's visual properties: color, line width, fill + fill alpha."""

    def __init__(
        self,
        parent: QWidget,
        color: tuple,
        line_width: float,
        fill: bool,
        fill_alpha: int,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("ROI properties")
        self._color = QColor(*color)

        form = QFormLayout(self)

        self._color_btn = QPushButton()
        self._color_btn.clicked.connect(self._pick_color)
        self._refresh_color_btn()
        form.addRow("Color", self._color_btn)

        self._width = QDoubleSpinBox()
        self._width.setRange(0.5, 10.0)
        self._width.setSingleStep(0.5)
        self._width.setValue(float(line_width))
        form.addRow("Line width", self._width)

        self._fill = QCheckBox()
        self._fill.setChecked(bool(fill))
        form.addRow("Fill", self._fill)

        self._alpha = QSpinBox()
        self._alpha.setRange(0, 255)
        self._alpha.setValue(int(fill_alpha))
        self._alpha.setEnabled(bool(fill))
        self._fill.toggled.connect(self._alpha.setEnabled)
        form.addRow("Fill alpha", self._alpha)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(self._color, self, "ROI color")
        if c.isValid():
            self._color = c
            self._refresh_color_btn()

    def _refresh_color_btn(self) -> None:
        self._color_btn.setText(self._color.name())
        self._color_btn.setStyleSheet(f"background:{self._color.name()};")

    def values(self) -> tuple:
        """Return (color_rgb, line_width, fill, fill_alpha)."""
        rgb = (self._color.red(), self._color.green(), self._color.blue())
        return rgb, self._width.value(), self._fill.isChecked(), self._alpha.value()


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
        # ROI management lives here: multi-select nodes + right-click actions.
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        # Trace-visibility checkboxes on signal-panel children flow through here.
        self.tree.itemChanged.connect(self._on_item_changed)
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

            # Block itemChanged: setCheckState below would otherwise fire the
            # trace-visibility handler during the rebuild.
            self.tree.blockSignals(True)
            self.tree.clear()
            for entry in self._panel_manager.entries:
                top = QTreeWidgetItem([f"[{entry.kind}] {entry.panel.name}"])
                self.tree.addTopLevelItem(top)

                if isinstance(entry.panel, VideoPanel):
                    top.setData(0, Qt.ItemDataRole.UserRole, ("video", entry.panel))
                    for col in range(n_groups):
                        self.tree.setItemWidget(
                            top, col + 1, self._make_group_checkbox(entry.panel, col)
                        )
                    for ov in entry.panel._overlays:  # noqa: SLF001
                        child = QTreeWidgetItem([f"[overlay {ov.kind}] {ov.name}"])
                        top.addChild(child)

                    # ROIs grouped under a folder node; each ROI is its own child.
                    folder = QTreeWidgetItem(["ROIs"])
                    folder.setData(
                        0, Qt.ItemDataRole.UserRole, ("roi_folder", entry.panel)
                    )
                    top.addChild(folder)
                    for r in entry.panel.rois:
                        node = QTreeWidgetItem([f"[ellipse] {r.name}"])
                        node.setData(
                            0, Qt.ItemDataRole.UserRole, ("roi", entry.panel, r.id)
                        )
                        folder.addChild(node)
                    folder.setExpanded(True)
                    top.setExpanded(True)

                elif isinstance(entry.panel, SignalPanel):
                    # Each trace is a checkable child: checkbox = visibility,
                    # text colored like the plotted line (tree doubles as legend).
                    for name in entry.panel.trace_names():
                        node = QTreeWidgetItem([name])
                        node.setData(
                            0, Qt.ItemDataRole.UserRole, ("trace", entry.panel, name)
                        )
                        node.setFlags(
                            node.flags() | Qt.ItemFlag.ItemIsUserCheckable
                        )
                        node.setCheckState(
                            0,
                            Qt.CheckState.Checked
                            if entry.panel.is_trace_visible(name)
                            else Qt.CheckState.Unchecked,
                        )
                        color = entry.panel.trace_color(name)
                        if color:
                            node.setForeground(0, QColor(color))
                        top.addChild(node)
                    top.setExpanded(True)
        except RuntimeError:
            return
        finally:
            try:
                self.tree.blockSignals(False)
            except RuntimeError:
                pass

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
            entry.panel.roi_added.connect(lambda *_: self._refresh_tree())
            entry.panel.roi_removed.connect(lambda *_: self._refresh_tree())
        elif isinstance(entry.panel, SignalPanel):
            entry.panel.traces_changed.connect(lambda *_: self._refresh_tree())
        self._refresh_tree()

    def _on_item_changed(self, item, column) -> None:
        """A trace node's checkbox toggled → show/hide that trace."""
        tag = item.data(0, Qt.ItemDataRole.UserRole)
        if tag and tag[0] == "trace":
            _, panel, name = tag
            panel.set_trace_visible(
                name, item.checkState(0) == Qt.CheckState.Checked
            )

    def _on_panel_removed(self, entry: "PanelEntry") -> None:
        if isinstance(entry.panel, VideoPanel):
            self._controller.remove_panel(entry.panel)
        self._refresh_tree()

    # ----- ROI actions (this tree is the ROI management surface) -----

    def _selected_roi_pairs(self) -> list[tuple]:
        """(panel, roi_id) for every selected ROI node."""
        pairs = []
        for item in self.tree.selectedItems():
            tag = item.data(0, Qt.ItemDataRole.UserRole)
            if tag and tag[0] == "roi":
                pairs.append((tag[1], tag[2]))
        return pairs

    def _on_selection_changed(self) -> None:
        # Highlight selected ROIs on each video's canvas; clear the rest.
        by_panel: dict = {}
        for panel, rid in self._selected_roi_pairs():
            by_panel.setdefault(panel, set()).add(rid)
        for entry in self._panel_manager.entries:
            if isinstance(entry.panel, VideoPanel):
                entry.panel.set_roi_highlight(by_panel.get(entry.panel, set()))

    def _on_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        tag = item.data(0, Qt.ItemDataRole.UserRole)
        if not tag:
            return

        menu = QMenu(self)
        if tag[0] in ("video", "roi_folder"):
            panel = tag[1]
            menu.addAction("Add Ellipse ROI", lambda: panel.add_ellipse_roi())
            all_ids = [r.id for r in panel.rois]
            if all_ids:
                # Extract every ROI on this video in one pass into its ΔF/F panel.
                menu.addAction(
                    f"Extract ΔF/F (all {len(all_ids)})",
                    lambda ids=all_ids, p=panel: p.extract_dff(ids),
                )
        elif tag[0] == "roi":
            # Right-clicking one node of a multi-selection acts on the whole
            # selection; right-clicking an unselected node acts on just it.
            pairs = self._selected_roi_pairs()
            if (tag[1], tag[2]) not in pairs:
                pairs = [(tag[1], tag[2])]
            menu.addAction(
                f"Extract ΔF/F ({len(pairs)})", lambda: self._extract_selected(pairs)
            )
            menu.addAction("Delete", lambda: self._delete_selected(pairs))
            menu.addAction("Properties…", lambda: self._edit_roi_properties(pairs))
            if len(pairs) == 1:
                menu.addAction("Rename…", lambda: self._rename_roi(pairs[0]))

        if menu.actions():
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _extract_selected(self, pairs: list[tuple]) -> None:
        by_panel: dict = {}
        for panel, rid in pairs:
            by_panel.setdefault(panel, []).append(rid)
        for panel, ids in by_panel.items():
            panel.extract_dff(ids)

    def _delete_selected(self, pairs: list[tuple]) -> None:
        for panel, rid in pairs:
            panel.remove_roi(rid)

    def _rename_roi(self, pair: tuple) -> None:
        panel, rid = pair
        item = next((r for r in panel.rois if r.id == rid), None)
        current = item.name if item else rid
        text, ok = QInputDialog.getText(self, "Rename ROI", "Name:", text=current)
        if ok and text.strip():
            panel.rename_roi(rid, text.strip())
            self._refresh_tree()

    def _edit_roi_properties(self, pairs: list[tuple]) -> None:
        """Open the properties dialog seeded from the first selected ROI; apply
        the chosen color / width / fill to every selected ROI."""
        panel0, rid0 = pairs[0]
        item0 = next((r for r in panel0.rois if r.id == rid0), None)
        if item0 is None:
            return
        dlg = _RoiPropertiesDialog(
            self, item0.color, item0.line_width, item0.fill, item0.fill_alpha
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        color, width, fill, alpha = dlg.values()
        for panel, rid in pairs:
            panel.set_roi_properties(
                rid, color=color, line_width=width, fill=fill, fill_alpha=alpha
            )
        # Re-apply the selection highlight (set_roi_properties resets pen width).
        self._on_selection_changed()

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
