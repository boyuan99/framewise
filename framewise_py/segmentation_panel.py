"""Segmentation viewer panel: rmbg movie + clickable footprint overlay.

`SegmentationPanel` subclasses `VideoPanel` so it inherits frame scrubbing,
MasterClock frame-lock, contrast, and the overlay machinery for free. On top it
adds the segmentation-specific pieces:

  * the base "video" is the lazy rmbg movie (or a black 1-frame fallback when no
    rmbg blocks exist), so footprints have something to sit on;
  * a built-in colored **footprint label overlay** (one RGBA ImageItem drawn
    above the base — not 1724 separate ROI items, which would be unusably slow);
  * **click-to-select**: clicking a footprint toggles that neuron and emits
    `neuron_toggled(index, selected)`; MainWindow routes it into a companion
    SignalPanel that plots `C[n]`. Selected neurons are drawn opaque on a
    highlight layer in their own color, matching the trace color.

Coordinate note: the panel displays each (H, W) frame transposed to (W, H), so a
ViewBox point (x, y) maps to the raw pixel (row=y, col=x) — see
`SegmentationResult.neuron_at`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QWidget,
)

from .segmentation import CELL_LABELS, SegmentationResult
from .video_panel import VideoPanel

# Alpha applied to a selected neuron's footprint on the highlight layer.
_HIGHLIGHT_ALPHA = 255

# Box-select auto-plots companion traces only up to this many neurons; larger
# boxes select/group without plotting (the user plots specific ones on demand).
_AUTOPLOT_LIMIT = 5

# How many label operations the per-panel undo stack remembers (each entry is a
# small {neuron: old_label} diff, so this is cheap; the cap just bounds memory).
_UNDO_DEPTH = 100


class SegmentationPanel(VideoPanel):
    """A VideoPanel specialized for CNMF/CaImAn segmentation results."""

    # (neuron index, now-selected?) — MainWindow adds/removes that neuron's C
    # trace from the companion SignalPanel.
    neuron_toggled = pyqtSignal(int, bool)

    # Emitted when the user clicks an empty area of the video, asking listeners
    # (the Resource Manager) to clear the tree selection that drives emphasis.
    emphasis_cleared = pyqtSignal()

    # Emitted when the selection grouping changes (box-select makes a new group,
    # groups are pruned) so the Resource Manager refreshes its group folders.
    groups_changed = pyqtSignal()

    # Emitted when any neuron's cell-type label changes (incl. bad) so the
    # Resource Manager refreshes its "Cell types" classification folders.
    labels_changed = pyqtSignal()

    def __init__(
        self,
        seg: SegmentationResult,
        parent: QWidget | None = None,
    ) -> None:
        self._seg = seg
        # Selection is organized into groups shown in the Resource Manager.
        # Group 0 is the default "Clicked" group (single clicks / ID picks);
        # each drag-box adds a new "Box N" group. A neuron lives in one group.
        self._groups: list[dict] = [{"name": "Clicked", "neurons": set()}]
        self._box_counter = 0
        # Neurons emphasized from the Resource Manager (tree selection) — drawn
        # bright white on top so you can tell which tree row maps to which cell.
        self._emph: set[int] = set()
        # Undo stack for label edits: each entry is a {neuron: previous_label}
        # diff, restored (newest first) by `undo_last_label` / Ctrl+Z.
        self._undo_stack: list[dict] = []
        # Drag-box selection state (active while the "▭ Box" button is toggled on).
        self._box_mode = False
        self._box_start: tuple[float, float] | None = None
        self._box_rect_item: QGraphicsRectItem | None = None
        # Click-cycle state: repeated clicks at the same pixel rotate through the
        # overlapping neurons there (plus a final "none" slot), so occluded cells
        # are reachable and a single cell toggles off on the second click.
        self._cycle_pixel: tuple[int, int] | None = None
        self._cycle_list: list[int] = []
        self._cycle_pos: int = 0
        self._cycle_active: int | None = None
        base = seg.video if seg.video is not None else self._fallback_base(seg)
        super().__init__(base, name=seg.name, fps=seg.fps, parent=parent)

        # The drawn-ellipse workflow doesn't apply here; selection is by click.
        self.btn_draw_roi.hide()

        self._build_footprint_layers()
        self._add_footprint_controls()

        # Click (not drag/pan) selects a neuron under the cursor; moving the
        # mouse over a footprint shows its ID(s) live.
        scene = self.image_view.getView().scene()
        scene.sigMouseClicked.connect(self._on_scene_click)
        scene.sigMouseMoved.connect(self._on_hover)

    @property
    def seg(self) -> SegmentationResult:
        return self._seg

    @property
    def selected_neurons(self) -> list[int]:
        return sorted(self._selected)

    @property
    def _selected(self) -> set[int]:
        """Union of all groups — the set of currently selected neurons."""
        s: set[int] = set()
        for g in self._groups:
            s |= g["neurons"]
        return s

    def _is_selected(self, n: int) -> bool:
        return any(n in g["neurons"] for g in self._groups)

    def selection_groups(self) -> list[dict]:
        """Selection groups (each {'name', 'neurons'}); group 0 is the default."""
        return self._groups

    def _on_fps_changed(self, fps: float) -> None:
        # Frames and C columns are 1:1, so the neuron traces' time axis must
        # follow this panel's fps. Update seg.fps first, then let the base class
        # re-anchor the clock and emit fps_changed (which MainWindow uses to
        # rebuild the companion traces at the new sampling rate).
        if fps > 0:
            self._seg.fps = float(fps)
        super()._on_fps_changed(fps)

    @staticmethod
    def _fallback_base(seg: SegmentationResult) -> np.ndarray:
        """A single black frame so footprints render even without an rmbg movie."""
        return np.zeros((1, seg.H, seg.W), dtype=np.uint8)

    # ----- footprint + highlight layers -----

    def _build_footprint_layers(self) -> None:
        """Add the colored footprint overlay and the (initially empty) highlight
        layer above the base image, both transposed to display coords."""
        view = self.image_view.getView()

        # A bright border around the image bounds, so a dark movie isn't "lost"
        # when zoomed — the rectangle marks where the frame is at any zoom.
        self.image_view.imageItem.setBorder(pg.mkPen(color=(255, 200, 0), width=1))

        self._fp_item = pg.ImageItem()
        self._fp_item.setZValue(50)
        self._fp_item.setImage(
            self._normalize_frame_layout(self._seg.label_image("fill")),
            autoLevels=False,
        )
        view.addItem(self._fp_item)

        # Gray overlay of bad ROIs, hidden until "Show bad" is toggled on.
        self._bad_item = pg.ImageItem()
        self._bad_item.setZValue(55)
        self._bad_item.setVisible(False)
        view.addItem(self._bad_item)

        self._hl_item = pg.ImageItem()
        self._hl_item.setZValue(60)  # above footprints, below ROI (1000)
        view.addItem(self._hl_item)
        self._refresh_highlight()

        # Emphasis layer for the neuron currently selected in the Resource
        # Manager tree — drawn above the colored highlight.
        self._emph_item = pg.ImageItem()
        self._emph_item.setZValue(70)
        view.addItem(self._emph_item)
        self._refresh_emphasis()

    def _add_footprint_controls(self) -> None:
        """Footprint controls under the inherited row: show / only-selected
        toggles, opacity, select-by-ID, a live hover readout, and a count."""
        row = QHBoxLayout()
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(6)

        self.cb_footprints = QCheckBox("Footprints")
        self.cb_footprints.setChecked(True)
        self.cb_footprints.setToolTip("Show / hide the colored neuron footprints")
        self.cb_footprints.toggled.connect(lambda *_: self._update_layer_visibility())
        row.addWidget(self.cb_footprints)

        self.cb_only_selected = QCheckBox("Only selected")
        self.cb_only_selected.setToolTip(
            "Hide unselected footprints — show only the selected neurons"
        )
        self.cb_only_selected.toggled.connect(
            lambda *_: self._update_layer_visibility()
        )
        row.addWidget(self.cb_only_selected)

        self.cb_show_bad = QCheckBox("Show bad")
        self.cb_show_bad.setToolTip(
            "Show ROIs labeled bad (gray). They're hidden by default; label/restore "
            "via the Label dropdown or right-click in the Resource Manager."
        )
        self.cb_show_bad.toggled.connect(self._on_show_bad_toggled)
        row.addWidget(self.cb_show_bad)

        self.fp_mode = QComboBox()
        self.fp_mode.addItems(["Fill", "Outline", "Center"])
        self.fp_mode.setToolTip(
            "How to draw footprints: Fill (solid), Outline (contour), or Center "
            "(a dot per neuron)"
        )
        self.fp_mode.currentTextChanged.connect(self._set_fp_mode)
        row.addWidget(self.fp_mode)

        row.addWidget(QLabel("α"))
        self.fp_opacity = QDoubleSpinBox()
        self.fp_opacity.setRange(0.0, 1.0)
        self.fp_opacity.setSingleStep(0.05)
        self.fp_opacity.setDecimals(2)
        self.fp_opacity.setValue(0.6)
        self.fp_opacity.setMinimumWidth(72)
        self.fp_opacity.valueChanged.connect(self._on_footprints_opacity)
        row.addWidget(self.fp_opacity)
        self._on_footprints_opacity(0.6)

        # Select a specific neuron by its (0-based) ID — Enter or the button.
        row.addWidget(QLabel("ID"))
        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, self._seg.n_neurons - 1)
        self.id_spin.setMinimumWidth(76)
        self.id_spin.setToolTip(
            f"Neuron ID (0–{self._seg.n_neurons - 1}); press Enter or Select to pick it"
        )
        self.id_spin.lineEdit().returnPressed.connect(self._select_by_id)
        row.addWidget(self.id_spin)
        self.btn_select_id = QPushButton("Select")
        self.btn_select_id.setToolTip("Select that neuron and center the view on it")
        self.btn_select_id.clicked.connect(self._select_by_id)
        row.addWidget(self.btn_select_id)

        self.btn_box = QPushButton("▭ Box")
        self.btn_box.setCheckable(True)
        self.btn_box.setToolTip(
            "Drag a box on the video to select every neuron whose center is "
            "inside it. Toggle off to pan/zoom and click again."
        )
        self.btn_box.toggled.connect(self._on_box_toggled)
        row.addWidget(self.btn_box)

        # Cell-type labeling: picking a label applies it to the current
        # selection immediately. Also doubles as a live readout of the selected
        # cell's type (see _sync_label_combo). Labels persist only on File → Save.
        row.addWidget(QLabel("Label"))
        self.label_combo = QComboBox()
        self.label_combo.addItems(list(CELL_LABELS))
        self.label_combo.setToolTip(
            "Pick a cell type to apply it to the selected neurons immediately "
            "(default unknown; 'bad' hides them). Save via File → Save Labels."
        )
        # textActivated fires only on user choice, not programmatic setCurrentIndex
        # in _sync_label_combo — so the live readout never re-applies a label.
        self.label_combo.textActivated.connect(self._apply_label_to_selection)
        row.addWidget(self.label_combo)

        # Undo the last label edit (mistyped labels are the common case). Per-panel
        # stack; Ctrl+Z fires while this panel/its canvas has focus.
        self.btn_undo = QPushButton("↶ Undo")
        self.btn_undo.setToolTip("Undo the last label change (Ctrl+Z)")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.undo_last_label)
        row.addWidget(self.btn_undo)
        undo_sc = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_sc.activated.connect(self.undo_last_label)

        self.lbl_hover = QLabel("Hover: —")
        self.lbl_hover.setStyleSheet("color: #777;")
        self.lbl_hover.setMinimumWidth(120)
        row.addWidget(self.lbl_hover)

        self.lbl_selected = QLabel(self._selected_text())
        self.lbl_selected.setStyleSheet("color: #555;")
        row.addWidget(self.lbl_selected)

        row.addStretch()
        self.btn_clear_sel = QPushButton("Clear selection")
        self.btn_clear_sel.setToolTip("Deselect all neurons")
        self.btn_clear_sel.clicked.connect(self.clear_selection)
        row.addWidget(self.btn_clear_sel)

        # Wrap in a horizontally-scrollable strip so the control row's width
        # doesn't pin the panel's minimum width (same pattern as VideoPanel).
        holder = QWidget()
        holder.setLayout(row)
        scroll = QScrollArea()
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        scroll.setMinimumWidth(0)
        scroll.setFixedHeight(holder.sizeHint().height() + 12)
        # Insert directly below the slider (above the inherited control scroll).
        self._insert_top_control(scroll)
        self._update_layer_visibility()

    def _selected_text(self) -> str:
        return f"{len(self._selected)} / {self._seg.n_neurons} selected"

    def _update_layer_visibility(self) -> None:
        """Apply the Footprints / Only-selected toggles to the two layers.

        Only-selected hides the full colored overlay; the highlight layer (which
        draws only the selected neurons) stays, so just the picks remain visible.
        """
        show = self.cb_footprints.isChecked()
        only_selected = self.cb_only_selected.isChecked()
        self._fp_item.setVisible(show and not only_selected)
        self._hl_item.setVisible(show)

    def _on_footprints_opacity(self, value: float) -> None:
        self._fp_item.setOpacity(max(0.0, min(1.0, float(value))))

    def _set_fp_mode(self, label: str) -> None:
        """Swap the base footprint overlay between Fill / Outline / Center."""
        mode = {"Fill": "fill", "Outline": "outline", "Center": "center"}[label]
        img = self._seg.label_image(mode)
        self._fp_item.setImage(self._normalize_frame_layout(img), autoLevels=False)

    def _current_fp_mode(self) -> str:
        return {"Fill": "fill", "Outline": "outline", "Center": "center"}[
            self.fp_mode.currentText()
        ]

    def _rebuild_footprint_overlay(self) -> None:
        """Re-render the base footprint layer (e.g. after the bad set changes)."""
        img = self._seg.label_image(self._current_fp_mode())
        self._fp_item.setImage(self._normalize_frame_layout(img), autoLevels=False)

    # ----- cell-type labeling -----

    def _on_show_bad_toggled(self, on: bool) -> None:
        self._bad_item.setVisible(bool(on))
        if on:
            self._bad_item.setImage(
                self._normalize_frame_layout(self._seg.bad_image()), autoLevels=False
            )

    def _apply_label_to_selection(self, label: str) -> None:
        """Apply the just-picked label to every currently-selected neuron."""
        sel = self.selected_neurons
        if not sel:
            self.roi_status.emit("No neurons selected — click or box-select first.")
            return
        self.set_labels(sel, label)

    def _sync_label_combo(self, active: int | None = None) -> None:
        """Reflect the current neuron's label in the dropdown, so the combo
        doubles as a live readout of "what type is this cell?".

        `active` is the neuron just clicked/emphasized — when given, its label
        always wins (so the combo tracks the cell you're looking at, even within
        a mixed-label multi-selection). Otherwise it shows the focused neurons'
        common label, or is left unchanged when the focus is empty/mixed."""
        if not hasattr(self, "label_combo"):
            return  # controls not built yet (early _refresh_* during __init__)
        if active is not None:
            label = self._seg.label_of(int(active))
        else:
            labels = {self._seg.label_of(n) for n in (self._selected | self._emph)}
            if len(labels) != 1:
                return
            label = next(iter(labels))
        i = self.label_combo.findText(label)
        if i >= 0 and i != self.label_combo.currentIndex():
            self.label_combo.setCurrentIndex(i)

    def set_labels(self, neurons, label: str) -> None:
        """Set one cell-type label on neurons (in memory; persist via File → Save).
        Labeling neurons "bad" also drops them from the selection (and their
        companion traces), since bad neurons are hidden from the canvas.

        Batched: the selection is updated in bulk and the canvas repaints once,
        so labeling a large box-selection is O(N), not O(N²) repaints. Recorded
        on the undo stack."""
        self._apply_label_map({int(n): label for n in neurons}, record=True)

    def _apply_label_map(self, mapping: dict, record: bool) -> None:
        """Apply a {neuron: label} map in one batch, repainting once. Any neuron
        becoming "bad" is bulk-deselected. When `record`, the inverse diff (each
        neuron's previous label) is pushed onto the undo stack so it can be
        reverted; undo itself calls this with record=False."""
        before: dict[int, str] = {}
        for n, lab in mapping.items():
            n = int(n)
            old = self._seg.label_of(n)
            if old == lab:
                continue
            if self._seg.set_label(n, lab):
                before[n] = old
        if not before:
            return
        bad_now = [n for n in before if self._seg.label_of(n) == "bad"]
        if bad_now:
            self._deselect_many(bad_now)
        if record:
            self._undo_stack.append(before)
            del self._undo_stack[:-_UNDO_DEPTH]  # bound the stack
        self._after_label_change()

    def undo_last_label(self) -> None:
        """Revert the most recent label edit (restoring each neuron's prior
        label). Selection is not restored — only the persisted labels."""
        if not self._undo_stack:
            self.roi_status.emit("Nothing to undo.")
            return
        diff = self._undo_stack.pop()
        self._apply_label_map(diff, record=False)
        self.roi_status.emit(f"Undid label change ({len(diff)} neuron(s)).")

    def _deselect_many(self, neurons) -> None:
        """Remove `neurons` from every selection group + emphasis in one pass.
        Detaches each one's companion trace (cheap; a no-op when not plotted) and
        repaints the tree once. Does NOT repaint the highlight layer — the caller
        does that once afterwards."""
        drop = {n for n in neurons if self._is_selected(n)}
        emph_drop = self._emph & set(neurons)
        if not drop and not emph_drop:
            return
        for g in self._groups:
            g["neurons"].difference_update(drop)
        self._prune_groups()
        if emph_drop:
            self._emph.difference_update(emph_drop)
            self._refresh_emphasis()
        for n in drop:  # detach companion traces (per-neuron by contract)
            self.neuron_toggled.emit(n, False)
        self.lbl_selected.setText(self._selected_text())
        self.groups_changed.emit()

    def mark_bad(self, neurons) -> None:
        """Label neurons bad (deselect + hide). Persist via File → Save."""
        self.set_labels(neurons, "bad")

    def mark_good(self, neurons) -> None:
        """Restore neurons to "unknown" and show them again."""
        self.set_labels(neurons, "unknown")

    def _after_label_change(self) -> None:
        # No disk write — labels persist only on File → Save (save_labels()).
        self._rebuild_footprint_overlay()
        if self.cb_show_bad.isChecked():
            self._bad_item.setImage(
                self._normalize_frame_layout(self._seg.bad_image()), autoLevels=False
            )
        self._refresh_highlight()
        self._sync_label_combo()
        if hasattr(self, "btn_undo"):
            self.btn_undo.setEnabled(bool(self._undo_stack))
        self.labels_changed.emit()

    # ----- hover + select-by-id -----

    def _on_hover(self, scene_pos: Any) -> None:
        """Show the ID(s) of the footprint(s) under the cursor (0-based)."""
        view = self.image_view.getView()
        if not view.sceneBoundingRect().contains(scene_pos):
            self.lbl_hover.setText("Hover: —")
            return
        p = view.mapSceneToView(scene_pos)
        ns = self._seg.neurons_at(int(round(p.y())), int(round(p.x())))
        if not ns:
            self.lbl_hover.setText("Hover: —")
        else:
            self.lbl_hover.setText("Hover: " + ", ".join(f"#{n}" for n in ns))

    def _select_by_id(self) -> None:
        """Select the neuron whose 0-based ID is in the spinbox, and center on it."""
        n = self.id_spin.value()
        if not (0 <= n < self._seg.n_neurons):
            return
        self._apply_select(n, True)
        self._center_on_neuron(n)
        # A subsequent canvas click should start a fresh cycle, not extend this.
        self._cycle_pixel = None

    def _center_on_neuron(self, n: int) -> None:
        """Pan/zoom the view to frame neuron n's footprint (display coords:
        x = column, y = row)."""
        ys, xs = np.where(self._seg.footprint(n))
        if ys.size == 0:
            return
        pad = 30
        self.image_view.getView().setRange(
            xRange=(int(xs.min()) - pad, int(xs.max()) + pad),
            yRange=(int(ys.min()) - pad, int(ys.max()) + pad),
            padding=0,
        )

    # ----- drag-box selection -----

    def _on_box_toggled(self, on: bool) -> None:
        self._box_mode = bool(on)
        view = self.image_view.getView()
        view.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        if not on:  # leaving box mode mid-drag — clean up any rubber band
            self._clear_box_rubberband()
            self._box_start = None

    def eventFilter(self, obj, event) -> bool:
        """In box mode, turn a left-drag into a rubber-band that selects every
        neuron whose centroid falls inside. Otherwise defer to VideoPanel."""
        if not self._box_mode:
            return super().eventFilter(obj, event)

        et = event.type()
        view = self.image_view.getView()
        if (
            et == QEvent.Type.GraphicsSceneMousePress
            and event.button() == Qt.MouseButton.LeftButton
        ):
            p = view.mapSceneToView(event.scenePos())
            self._box_start = (p.x(), p.y())
            self._box_rect_item = QGraphicsRectItem()
            self._box_rect_item.setPen(
                pg.mkPen(color=(0, 200, 255), width=1, style=Qt.PenStyle.DashLine)
            )
            self._box_rect_item.setBrush(pg.mkBrush(0, 200, 255, 40))
            self._box_rect_item.setZValue(2000)
            view.addItem(self._box_rect_item)
            return True
        if et == QEvent.Type.GraphicsSceneMouseMove and self._box_start is not None:
            p = view.mapSceneToView(event.scenePos())
            x0, y0 = self._box_start
            self._box_rect_item.setRect(
                QRectF(min(x0, p.x()), min(y0, p.y()), abs(p.x() - x0), abs(p.y() - y0))
            )
            return True
        if et == QEvent.Type.GraphicsSceneMouseRelease and self._box_start is not None:
            p = view.mapSceneToView(event.scenePos())
            x0, y0 = self._box_start
            self._box_start = None
            self._clear_box_rubberband()
            # Display coords (x=col, y=row) → raw box; select centroids inside.
            self._select_box(
                int(round(min(y0, p.y()))), int(round(max(y0, p.y()))),
                int(round(min(x0, p.x()))), int(round(max(x0, p.x()))),
            )
            return True
        return super().eventFilter(obj, event)

    def _clear_box_rubberband(self) -> None:
        if self._box_rect_item is not None:
            self.image_view.getView().removeItem(self._box_rect_item)
            self._box_rect_item = None

    def _select_box(self, y0: int, y1: int, x0: int, x1: int) -> None:
        """Box-select: put every neuron whose centroid is in the box into a new
        group."""
        self._box_select(self._seg.neurons_in_box(y0, y1, x0, x1))
        self._cycle_pixel = None  # next single click starts a fresh cycle

    # ----- selection -----

    def _on_scene_click(self, event: Any) -> None:
        """Map a click to the neuron(s) under the cursor and cycle the selection.

        First click at a spot selects the smallest overlapping footprint there;
        clicking the same spot again rotates to the next overlapping neuron, then
        to a "none" slot (all deselected at that spot), then wraps. Clicks at a
        new spot leave selections made elsewhere intact (selection accumulates).
        """
        view = self.image_view.getView()
        if not view.sceneBoundingRect().contains(event.scenePos()):
            return
        p = view.mapSceneToView(event.scenePos())
        # Display coords (x, y) → raw pixel (row=y, col=x).
        y, x = int(round(p.y())), int(round(p.x()))
        neurons = self._seg.neurons_at(y, x)
        if not neurons:
            # Clicking empty space dismisses the emphasis (and the tree selection
            # that drove it). Selected neurons + their traces are kept.
            if self._emph:
                self.clear_emphasis()
                self.emphasis_cleared.emit()
            return

        if (y, x) == self._cycle_pixel and neurons == self._cycle_list:
            # Continuing the cycle here: retract the spot's current pick, advance.
            if self._cycle_active is not None:
                self._apply_select(self._cycle_active, False)
                self._cycle_active = None
            self._cycle_pos = (self._cycle_pos + 1) % (len(neurons) + 1)
        else:
            # New spot — start a fresh cycle, keep prior selections.
            self._cycle_pixel = (y, x)
            self._cycle_list = neurons
            self._cycle_pos = 0
            self._cycle_active = None

        if self._cycle_pos < len(neurons):  # else: the "none" slot
            n = neurons[self._cycle_pos]
            self._apply_select(n, True)
            self._cycle_active = n

    def toggle_neuron(self, n: int) -> None:
        """Flip selection of neuron n (public/console/test entry point)."""
        self._apply_select(n, not self._is_selected(int(n)))

    def set_neuron_selected(self, n: int, selected: bool) -> None:
        """Public: set a neuron's selection state. Used by the Resource Manager's
        right-click Deselect so you don't have to find the cell on a dark video."""
        self._apply_select(int(n), bool(selected))

    def _apply_select(self, n: int, selected: bool) -> None:
        """Set neuron n's selection to `selected`, idempotently. Selecting adds
        it to the default "Clicked" group; deselecting drops it from whatever
        group holds it. Only repaints/emits on a real change."""
        n = int(n)
        if self._is_selected(n) == selected:
            return
        if selected:
            self._groups[0]["neurons"].add(n)
        else:
            for g in self._groups:
                g["neurons"].discard(n)
            self._prune_groups()
            if n in self._emph:  # a deselected neuron can't stay emphasized
                self._emph.discard(n)
                self._refresh_emphasis()
        self._refresh_highlight()
        self.lbl_selected.setText(self._selected_text())
        # Selecting a cell → show its label; deselecting → fall back to the rest.
        self._sync_label_combo(n if selected else None)
        self.neuron_toggled.emit(n, selected)
        self.groups_changed.emit()

    def _box_select(self, neurons: list[int]) -> None:
        """Put `neurons` into a brand-new "Box N" group (moving any that were
        already selected out of their old group). Auto-plots their traces only
        for small boxes — plotting hundreds of curves is slow and unreadable, so
        beyond `_AUTOPLOT_LIMIT` we just select/group and let the user plot
        specific ones on demand (right-click → Plot)."""
        neurons = [int(n) for n in neurons]
        if not neurons:
            return
        self._box_counter += 1
        grp = {"name": f"Box {self._box_counter}", "neurons": set()}
        self._groups.append(grp)
        newly = []
        for n in neurons:
            was = self._is_selected(n)
            for g in self._groups:
                if g is not grp:
                    g["neurons"].discard(n)
            grp["neurons"].add(n)
            if not was:
                newly.append(n)
        self._prune_groups()
        if len(neurons) <= _AUTOPLOT_LIMIT:
            for n in newly:  # add each new neuron's companion trace
                self.neuron_toggled.emit(n, True)
        elif newly:
            self.roi_status.emit(
                f"Box-selected {len(neurons)} neurons — traces not auto-plotted "
                f"(> {_AUTOPLOT_LIMIT}). Right-click a neuron or the group → Plot."
            )
        self._refresh_highlight()
        self.lbl_selected.setText(self._selected_text())
        self._sync_label_combo()
        self.groups_changed.emit()

    def plot_neurons(self, neurons) -> None:
        """Force-plot the companion traces for the given (already-selected)
        neurons — the on-demand path for big box-selections."""
        for n in [int(x) for x in neurons]:
            if self._is_selected(n):
                self.neuron_toggled.emit(n, True)

    def deselect_group(self, group_index: int) -> None:
        """Deselect every neuron in the given group (Resource Manager action)."""
        if 0 <= group_index < len(self._groups):
            for n in list(self._groups[group_index]["neurons"]):
                self._apply_select(n, False)

    def _prune_groups(self) -> None:
        """Drop empty groups, but always keep the default "Clicked" group."""
        self._groups = [self._groups[0]] + [
            g for g in self._groups[1:] if g["neurons"]
        ]

    def clear_selection(self) -> None:
        sel = self._selected
        if not sel:
            return
        for n in sorted(sel):
            self.neuron_toggled.emit(int(n), False)
        self._groups = [{"name": "Clicked", "neurons": set()}]
        self._box_counter = 0
        self._cycle_pixel = None
        self._cycle_list = []
        self._cycle_pos = 0
        self._cycle_active = None
        self._emph.clear()
        self._refresh_highlight()
        self._refresh_emphasis()
        self.lbl_selected.setText(self._selected_text())
        self._sync_label_combo()
        self.groups_changed.emit()

    def set_neuron_highlight(self, indices: set[int]) -> None:
        """Emphasize the given neurons (from Resource Manager tree selection),
        drawn bright white on top of the colored highlight."""
        self._emph = {int(n) for n in indices}
        self._refresh_emphasis()
        # A single emphasized cell (one tree row) drives the combo to its label.
        self._sync_label_combo(next(iter(self._emph)) if len(self._emph) == 1 else None)

    def clear_emphasis(self) -> None:
        """Drop the white emphasis (keeps neuron selections intact)."""
        if self._emph:
            self._emph = set()
            self._refresh_emphasis()
            self._sync_label_combo()

    def _refresh_emphasis(self) -> None:
        H, W = self._seg.H, self._seg.W
        em = np.zeros((H, W, 4), dtype=np.uint8)
        for n in self._emph:
            mask = self._seg.footprint(n)
            em[mask] = (255, 255, 255, 255)  # opaque white
        self._emph_item.setImage(self._normalize_frame_layout(em), autoLevels=False)

    def _refresh_highlight(self) -> None:
        """Repaint the highlight layer: selected footprints drawn opaque in
        their own neuron color."""
        H, W = self._seg.H, self._seg.W
        hl = np.zeros((H, W, 4), dtype=np.uint8)
        for n in self._selected:
            mask = self._seg.footprint(n)
            r, g, b = self._seg.neuron_color(n)
            hl[mask, 0] = r
            hl[mask, 1] = g
            hl[mask, 2] = b
            hl[mask, 3] = _HIGHLIGHT_ALPHA
        self._hl_item.setImage(self._normalize_frame_layout(hl), autoLevels=False)
