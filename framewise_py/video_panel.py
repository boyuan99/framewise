"""Single video panel: pyqtgraph ImageView + time slider + frame controls.

Uses MasterClock for cross-panel frame-lock: when the user drags this panel's
slider, the master clock is updated; when the master clock updates from
elsewhere, this panel jumps to the corresponding frame (computed via fps).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QContextMenuEvent,
    QDragEnterEvent,
    QDropEvent,
    QPainter,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .loaders import LoadError, load_overlay
from .master_clock import MasterClock
from .roi import RoiItem, TraceExtractWorker, ellipse_mask
from .settings import get_last_dir, set_last_dir_from_path

DEFAULT_FPS = 20.0

# Cycling pen colors for ROI outlines (distinct, bright). Reused as trace colors.
_ROI_COLORS = [
    (255, 215, 0),
    (238, 102, 119),
    (34, 136, 51),
    (102, 204, 238),
    (170, 51, 119),
    (204, 187, 68),
    (68, 119, 170),
]

# Two-stop colormaps: black → tint color. Names are user-facing.
# RGB(A) images ignore these (pyqtgraph skips LUT for multi-channel data).
_TINT_STOPS: dict[str, tuple[int, int, int]] = {
    "Gray": (255, 255, 255),
    "Green": (0, 255, 0),
    "Red": (255, 0, 0),
    "Magenta": (255, 0, 255),
    "Cyan": (0, 255, 255),
    "Yellow": (255, 255, 0),
    "Blue": (50, 100, 255),
}


def _make_colormap(name: str) -> pg.ColorMap:
    r, g, b = _TINT_STOPS.get(name, _TINT_STOPS["Gray"])
    return pg.ColorMap(
        [0.0, 1.0],
        [(0, 0, 0, 255), (r, g, b, 255)],
    )


OVERLAY_FILE_FILTER = (
    "All supported (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.h5 *.hdf5 *.mp4 *.avi *.mov *.mkv);;"
    "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;"
    "HDF5 (*.h5 *.hdf5);;"
    "Video (*.mp4 *.avi *.mov *.mkv);;"
    "All files (*)"
)


@dataclass
class _Overlay:
    """One overlay layer rendered above the base video.

    `source` is either a static (H,W[,C]) ndarray or a lazy (T,H,W[,C]) array;
    `kind` disambiguates. `image_item` is the pyqtgraph item drawing this
    overlay on the base ImageView's ViewBox. `row` is the widget holding this
    overlay's controls in the Overlays section.

    Video overlays own an independent playhead (`current_frame`, `slider`,
    `frame_label`, `fps`) — they are NOT synced to the base panel's clock.
    Static overlays leave those fields at their defaults.
    """

    name: str
    source: Any
    kind: str  # "static" | "video"
    image_item: pg.ImageItem
    row: QWidget
    fps: float = 0.0
    visible: bool = True
    opacity: float = 1.0
    cmap_name: str = "Gray"
    levels: tuple[float, float] | None = None
    blend_mode: str = "alpha"  # placeholder for future blend modes
    current_frame: int = 0
    slider: QSlider | None = None
    frame_label: QLabel | None = None
    hist_dialog: QDialog | None = None  # lazy: created on first Hist click
    visible_cb: QCheckBox | None = None  # so Flip can drive the UI in sync


class _FillableEllipseROI(pg.EllipseROI):
    """An EllipseROI that can optionally fill its interior.

    pyqtgraph ROIs draw only an outline; we override paint() to add a brush so
    ROIs can be shown filled with an adjustable alpha.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fill_brush: QBrush | None = None

    def set_fill_brush(self, brush: QBrush | None) -> None:
        self._fill_brush = brush
        self.update()

    def paint(self, p, opt, widget) -> None:  # noqa: D401 - mirrors EllipseROI.paint
        r = self.boundingRect()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(self.currentPen)
        p.setBrush(self._fill_brush if self._fill_brush is not None else QBrush())
        p.scale(r.width(), r.height())
        p.drawEllipse(QRectF(0.0, 0.0, 1.0, 1.0))


class VideoPanel(QWidget):
    """One video's display + scrubbing controls.

    Frame ↔ time conversion uses self.fps. Connect a MasterClock with
    bind_master_clock() to participate in frame-lock with other panels.
    """

    # Emitted when the user changes the frame by dragging the slider or
    # clicking step buttons. Other listeners (sync controllers) may also use it.
    frame_changed = pyqtSignal(int)

    # Emitted with the new fps whenever this panel's playback rate changes, so
    # listeners (e.g. a segmentation panel's companion trace plot) can re-derive
    # any time axis that depends on this panel's frame↔time mapping.
    fps_changed = pyqtSignal(float)

    # Emitted with `self` whenever this panel's overlay list changes (add /
    # remove). Lets the Resource Manager refresh the tree without polling.
    overlay_added = pyqtSignal(object)
    overlay_removed = pyqtSignal(object)

    # Emitted with `self` whenever this panel's ROI list changes (add / remove),
    # so the Resource Manager can refresh its ROI nodes without polling.
    roi_added = pyqtSignal(object)
    roi_removed = pyqtSignal(object)

    # Emitted (panel_name, list[Trace]) when a ΔF/F extraction finishes; the
    # main window routes the traces into a dedicated ROI signal panel.
    dff_extracted = pyqtSignal(str, object)

    # Human-readable extraction status (start / progress / done), shown in the
    # main window status bar so long extractions give feedback.
    roi_status = pyqtSignal(str)

    def __init__(
        self,
        array: Any,
        name: str = "video",
        fps: float = DEFAULT_FPS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._array = array
        self._name = name
        self._n_frames = int(array.shape[0])
        self._current_frame = 0
        self._fps = float(fps)
        self._syncing = False
        self._master_clock: MasterClock | None = None
        self._overlays: list[_Overlay] = []
        self._rois: list[RoiItem] = []
        self._roi_counter = 0
        self._roi_worker: TraceExtractWorker | None = None
        # Drag-to-draw state (active while the "✏ ROI" button is toggled on).
        self._draw_mode = False
        self._draw_start: tuple[float, float] | None = None
        self._draw_item: RoiItem | None = None

        self.setAcceptDrops(True)
        self._build_ui()
        self._show_frame(0, first=True)

    @property
    def name(self) -> str:
        return self._name

    @property
    def array(self):
        """Underlying (often lazy) frame array, shape (T, H, W[, C]). Exposed
        for the embedded console so users can compute on raw frames."""
        return self._array

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def duration(self) -> float:
        return self._n_frames / self._fps

    def bind_master_clock(self, clock: MasterClock) -> None:
        """Subscribe to a MasterClock and push our own scrub events to it."""
        self._master_clock = clock
        clock.time_changed.connect(self._on_master_time_changed)
        # Sync to whatever time the clock currently holds
        self._on_master_time_changed(clock.time)

    def set_frame_from_sync(self, frame: int) -> None:
        """Legacy sync API kept for backwards compat with SyncController."""
        frame = max(0, min(self._n_frames - 1, int(frame)))
        if frame == self._current_frame:
            return
        self._syncing = True
        try:
            self.slider.setValue(frame)
        finally:
            self._syncing = False

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.image_view = pg.ImageView()
        self.image_view.ui.histogram.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        layout.addWidget(self.image_view, stretch=1)
        # Intercept scene mouse events so "✏ ROI" mode can drag out new ellipses.
        self.image_view.getView().scene().installEventFilter(self)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._n_frames - 1)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        controls = QHBoxLayout()

        self.cb_base_visible = QCheckBox()
        self.cb_base_visible.setChecked(True)
        self.cb_base_visible.setToolTip("Show / hide the base layer")
        self.cb_base_visible.toggled.connect(self._on_base_visibility_toggled)
        controls.addWidget(self.cb_base_visible)

        self.btn_flip = QPushButton("Flip")
        self.btn_flip.setFixedWidth(48)
        self.btn_flip.setToolTip(
            "Flip layer visibility — invert base and every overlay's show/hide state"
        )
        self.btn_flip.clicked.connect(self._flip_visibility)
        controls.addWidget(self.btn_flip)

        self.frame_label = QLabel(self._frame_text(0))
        controls.addWidget(self.frame_label)

        controls.addStretch()

        controls.addWidget(QLabel("fps:"))
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 1000.0)
        self.fps_spin.setDecimals(2)
        self.fps_spin.setSingleStep(1.0)
        self.fps_spin.setValue(self._fps)
        self.fps_spin.setMinimumWidth(96)
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        controls.addWidget(self.fps_spin)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(list(_TINT_STOPS.keys()))
        self.cmap_combo.setMinimumWidth(100)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        controls.addWidget(self.cmap_combo)

        btn_auto = QPushButton("Auto")
        btn_auto.setFixedWidth(48)
        btn_auto.setToolTip("Recompute contrast levels from the current frame")
        btn_auto.clicked.connect(self._auto_levels)
        controls.addWidget(btn_auto)

        btn_hist = QPushButton("Hist")
        btn_hist.setFixedWidth(48)
        btn_hist.setCheckable(True)
        btn_hist.setToolTip("Show histogram for manual contrast adjustment")
        btn_hist.toggled.connect(self._toggle_histogram)
        controls.addWidget(btn_hist)

        for label, delta in (("◀◀", -10), ("◀", -1), ("▶", 1), ("▶▶", 10)):
            btn = QPushButton(label)
            btn.setFixedWidth(36)
            btn.clicked.connect(lambda _, d=delta: self._step(d))
            controls.addWidget(btn)

        self.btn_add_overlay = QPushButton("+ Overlay")
        self.btn_add_overlay.setFixedWidth(80)
        self.btn_add_overlay.setToolTip("Add an image or video as an overlay layer")
        self.btn_add_overlay.clicked.connect(self._on_add_overlay_clicked)
        controls.addWidget(self.btn_add_overlay)

        self.btn_draw_roi = QPushButton("✏ ROI")
        self.btn_draw_roi.setCheckable(True)
        self.btn_draw_roi.setFixedWidth(64)
        self.btn_draw_roi.setToolTip(
            "Draw ellipse ROIs by dragging on the video. Stays on so you can draw "
            "several; toggle off to pan/zoom again."
        )
        self.btn_draw_roi.toggled.connect(self._on_draw_toggled)
        controls.addWidget(self.btn_draw_roi)

        # The control row's combined minimum width (~600px) would otherwise pin
        # the panel's — and therefore the MDI sub-window's — minimum width,
        # blocking narrow resizes. Put it in a horizontally-scrollable strip with
        # a zero minimum width so the panel can shrink freely; controls scroll
        # when the panel is narrower than they are.
        controls.setContentsMargins(0, 0, 0, 0)
        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_scroll = QScrollArea()
        controls_scroll.setWidget(controls_widget)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        controls_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        controls_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        controls_scroll.setMinimumWidth(0)
        controls_scroll.setFixedHeight(controls_widget.sizeHint().height() + 16)
        layout.addWidget(controls_scroll)

        # Overlays section — hidden until first overlay is added. Wrapped in a
        # scrollable strip (same reason as the controls row) so wide overlay
        # rows don't pin the panel's minimum width; tall stacks scroll instead
        # of stealing the image's vertical space.
        self._overlays_frame = QFrame()
        self._overlays_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._overlays_layout = QVBoxLayout(self._overlays_frame)
        self._overlays_layout.setContentsMargins(2, 2, 2, 2)
        self._overlays_layout.setSpacing(1)

        self._overlays_scroll = QScrollArea()
        self._overlays_scroll.setWidget(self._overlays_frame)
        self._overlays_scroll.setWidgetResizable(True)
        self._overlays_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._overlays_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._overlays_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._overlays_scroll.setMinimumWidth(0)
        self._overlays_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        self._overlays_scroll.setMaximumHeight(220)
        self._overlays_scroll.setVisible(False)
        layout.addWidget(self._overlays_scroll)

        # Apply default colormap (gray) so explicit changes via the combo box
        # always go through the same code path.
        self._on_cmap_changed("Gray")

    def _on_slider_changed(self, value: int) -> None:
        if value == self._current_frame:
            return
        self._current_frame = value
        self._show_frame(value)
        if self._syncing:
            return
        self.frame_changed.emit(value)
        if self._master_clock is not None:
            self._master_clock.set_time(value / self._fps)

    def _on_master_time_changed(self, t: float) -> None:
        frame = int(round(t * self._fps))
        frame = max(0, min(self._n_frames - 1, frame))
        if frame == self._current_frame:
            return
        # Update via slider so the UI stays consistent; suppress re-emit so we
        # don't bounce back to the master clock.
        self._syncing = True
        try:
            self.slider.setValue(frame)
        finally:
            self._syncing = False

    def _on_fps_changed(self, fps: float) -> None:
        if fps <= 0:
            return
        self._fps = float(fps)
        # Re-anchor our frame to whatever the master clock currently says, so
        # changing fps doesn't desync us. Overlays are independent — they keep
        # their own fps and current_frame.
        if self._master_clock is not None:
            self._on_master_time_changed(self._master_clock.time)
        self.fps_changed.emit(self._fps)

    def _on_cmap_changed(self, name: str) -> None:
        cmap = _make_colormap(name)
        # ImageView.setColorMap routes through the histogram widget, which
        # doesn't always propagate when the histogram is hidden. Apply the
        # LUT directly to the image item too — belt and suspenders.
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        self.image_view.imageItem.setLookupTable(lut)
        # Keep the histogram gradient in sync so it reflects the choice
        # when the user toggles "Hist" on.
        self.image_view.ui.histogram.gradient.setColorMap(cmap)

    def _auto_levels(self) -> None:
        self.image_view.autoLevels()

    def _toggle_histogram(self, on: bool) -> None:
        if on:
            self.image_view.ui.histogram.show()
        else:
            self.image_view.ui.histogram.hide()

    def _on_base_visibility_toggled(self, on: bool) -> None:
        self.image_view.imageItem.setVisible(bool(on))

    def _flip_visibility(self) -> None:
        """Invert visibility of base and every overlay. Going through the
        checkboxes (rather than setVisible directly) keeps the UI state
        consistent via the existing toggled-signal handlers."""
        self.cb_base_visible.setChecked(not self.cb_base_visible.isChecked())
        for ov in self._overlays:
            if ov.visible_cb is not None:
                ov.visible_cb.setChecked(not ov.visible_cb.isChecked())

    def _step(self, delta: int) -> None:
        self.slider.setValue(self._current_frame + delta)

    def _show_frame(self, frame: int, *, first: bool = False) -> None:
        img = np.asarray(self._array[frame])
        img = self._normalize_frame_layout(img)
        self.image_view.setImage(
            img,
            autoLevels=first,
            autoRange=first,
            autoHistogramRange=first,
        )
        self.frame_label.setText(self._frame_text(frame))

    def _render_overlay_frame(self, ov: _Overlay, frame: int) -> None:
        """Render a video overlay at a specific frame. Independent of base —
        called only from the overlay's own slider / step buttons."""
        if ov.kind != "video":
            return
        n = int(ov.source.shape[0])
        frame = max(0, min(n - 1, int(frame)))
        img = np.asarray(ov.source[frame])
        img = self._normalize_frame_layout(img)
        ov.image_item.setImage(img, autoLevels=False)
        if ov.levels is not None and not _is_rgb(img):
            ov.image_item.setLevels(ov.levels)
        ov.current_frame = frame
        if ov.frame_label is not None:
            t = frame / ov.fps if ov.fps > 0 else 0.0
            ov.frame_label.setText(
                f"Frame {frame} / {n - 1}  |  t = {t:.3f} s"
            )

    @staticmethod
    def _normalize_frame_layout(img: np.ndarray) -> np.ndarray:
        """Reshape an arbitrary frame to pyqtgraph's expected (W, H) or
        (W, H, C) layout.

        Accepts:
          - (H, W)              — grayscale, scientific images
          - (H, W, 3|4)         — RGB/RGBA video frames
          - (C, H, W) with C<=4 — multi-channel sci data: take first channel
        """
        if img.ndim == 2:
            return img.T
        if img.ndim == 3:
            if img.shape[-1] in (3, 4) and img.shape[0] not in (3, 4):
                return img.transpose(1, 0, 2)
            if img.shape[0] in (1, 2, 3, 4):
                return img[0].T
        return np.swapaxes(img, 0, 1)

    def _frame_text(self, frame: int) -> str:
        t = frame / self._fps
        return f"Frame {frame} / {self._n_frames - 1}  |  t = {t:.3f} s"

    # ----- ROI management -----
    #
    # The panel OWNS the ROI items (they live on its ViewBox) but has no ROI UI;
    # the Resource Manager tree drives these methods and listens to the signals.

    @property
    def rois(self) -> list[RoiItem]:
        return list(self._rois)

    @property
    def frame_hw(self) -> tuple[int, int]:
        """Raw frame (H, W) — the shape masks index into."""
        return int(self._array.shape[1]), int(self._array.shape[2])

    def _apply_roi_style(self, item: RoiItem, selected: bool) -> None:
        """Push an ROI item's color / width / fill onto its canvas item. Selected
        ROIs get a thicker pen so tree selection is visible."""
        width = item.line_width + (2.0 if selected else 0.0)
        item.roi.setPen(pg.mkPen(color=item.color, width=width))
        if item.fill:
            r, g, b = item.color
            item.roi.set_fill_brush(pg.mkBrush(r, g, b, item.fill_alpha))
        else:
            item.roi.set_fill_brush(None)

    def add_ellipse_roi(self, pos=None, size=None) -> RoiItem:
        """Add an ellipse ROI and return its item. With no args it is centered at
        ~1/4 frame size; `pos`/`size` (displayed (W,H) coords) come from drag-draw."""
        self._roi_counter += 1
        color = _ROI_COLORS[(self._roi_counter - 1) % len(_ROI_COLORS)]
        roi_id = f"ROI_{self._roi_counter:03d}"

        # ViewBox coords are displayed (W, H): x spans the frame width, y height.
        H, W = self.frame_hw
        if size is None:
            size = (W / 4.0, H / 4.0)
        sw, sh = float(size[0]), float(size[1])
        if pos is None:
            pos = (W / 2.0 - sw / 2.0, H / 2.0 - sh / 2.0)

        roi = _FillableEllipseROI(
            (float(pos[0]), float(pos[1])), (sw, sh), removable=False
        )
        roi.setZValue(1000)  # above base + overlays
        self.image_view.getView().addItem(roi)

        item = RoiItem(id=roi_id, name=roi_id, roi=roi, color=color)
        self._rois.append(item)
        self._apply_roi_style(item, selected=False)
        self.roi_added.emit(self)
        return item

    def remove_roi(self, roi_id: str) -> None:
        item = next((r for r in self._rois if r.id == roi_id), None)
        if item is None:
            return
        self.image_view.getView().removeItem(item.roi)
        self._rois.remove(item)
        self.roi_removed.emit(self)

    def rename_roi(self, roi_id: str, name: str) -> None:
        item = next((r for r in self._rois if r.id == roi_id), None)
        if item is not None and name:
            item.name = name

    def set_roi_properties(
        self,
        roi_id: str,
        *,
        color: tuple | None = None,
        line_width: float | None = None,
        fill: bool | None = None,
        fill_alpha: int | None = None,
    ) -> None:
        """Update an ROI's visual properties (color / line width / fill / alpha)."""
        item = next((r for r in self._rois if r.id == roi_id), None)
        if item is None:
            return
        if color is not None:
            item.color = tuple(color)
        if line_width is not None:
            item.line_width = float(line_width)
        if fill is not None:
            item.fill = bool(fill)
        if fill_alpha is not None:
            item.fill_alpha = int(fill_alpha)
        self._apply_roi_style(item, selected=False)

    def set_roi_highlight(self, selected_ids: set[str]) -> None:
        """Re-style ROIs so the tree selection shows on canvas (thicker pen)."""
        for item in self._rois:
            self._apply_roi_style(item, item.id in selected_ids)

    # ----- Drag-to-draw -----

    def _on_draw_toggled(self, on: bool) -> None:
        self._draw_mode = bool(on)
        view = self.image_view.getView()
        view.setCursor(
            Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor
        )

    def eventFilter(self, obj, event) -> bool:
        """In draw mode, turn a left-drag on the image into a new ellipse ROI."""
        if not self._draw_mode:
            return super().eventFilter(obj, event)

        et = event.type()
        view = self.image_view.getView()
        if (
            et == QEvent.Type.GraphicsSceneMousePress
            and event.button() == Qt.MouseButton.LeftButton
        ):
            p = view.mapSceneToView(event.scenePos())
            self._draw_start = (p.x(), p.y())
            self._draw_item = self.add_ellipse_roi(pos=self._draw_start, size=(1.0, 1.0))
            return True
        if et == QEvent.Type.GraphicsSceneMouseMove and self._draw_start is not None:
            p = view.mapSceneToView(event.scenePos())
            x0, y0 = self._draw_start
            x1, y1 = p.x(), p.y()
            self._draw_item.roi.setPos((min(x0, x1), min(y0, y1)))
            self._draw_item.roi.setSize((abs(x1 - x0), abs(y1 - y0)))
            return True
        if et == QEvent.Type.GraphicsSceneMouseRelease and self._draw_start is not None:
            item = self._draw_item
            self._draw_start = None
            self._draw_item = None
            # Discard accidental clicks that produced a degenerate ellipse.
            if item is not None and (item.roi.size()[0] < 3 or item.roi.size()[1] < 3):
                self.remove_roi(item.id)
            return True
        return super().eventFilter(obj, event)

    def extract_dff(self, roi_ids: list[str]) -> None:
        """Extract ΔF/F for the given ROIs off-thread; emit `dff_extracted` when done."""
        if self._roi_worker is not None and self._roi_worker.isRunning():
            self._emit_status("ΔF/F extraction already running for this video")
            return  # one extraction per panel at a time
        items = [r for r in self._rois if r.id in set(roi_ids)]
        if not items:
            return

        hw = self.frame_hw
        masks_with_labels = [
            (it.name, ellipse_mask(it.roi.pos(), it.roi.size(), it.roi.angle(), hw))
            for it in items
        ]
        worker = TraceExtractWorker(
            self._array, masks_with_labels, self._n_frames, self._fps, parent=self
        )
        worker.progress.connect(self._emit_status)
        worker.traces_ready.connect(self._on_traces_ready)
        self._roi_worker = worker
        self._emit_status(
            f"Extracting ΔF/F for {len(items)} ROI(s) over {self._n_frames} frames…"
        )
        worker.start()

    def _emit_status(self, msg: str) -> None:
        print(f"[{self._name}] {msg}")
        self.roi_status.emit(msg)

    def _on_traces_ready(self, traces) -> None:
        self._roi_worker = None
        self._emit_status(f"ΔF/F extraction done: {len(traces)} trace(s)")
        self.dff_extracted.emit(self._name, traces)

    def stop_roi_worker(self) -> None:
        """Cancel and join any running extraction so the app can exit cleanly."""
        worker = self._roi_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(3000)
        self._roi_worker = None

    # ----- Overlay management -----

    def add_overlay(self, path: str | Path) -> int:
        """Load `path` as an overlay layer and return its index."""
        data, kind, name = load_overlay(path)

        view_box = self.image_view.getView()
        item = pg.ImageItem()
        item.setZValue(10 + len(self._overlays))
        view_box.addItem(item)

        ov = _Overlay(
            name=name,
            source=data,
            kind=kind,
            image_item=item,
            row=QWidget(),  # placeholder, replaced by _build_overlay_widget
            fps=DEFAULT_FPS if kind == "video" else 0.0,
        )

        widget = self._build_overlay_widget(ov)
        ov.row = widget
        self._overlays.append(ov)
        self._overlays_layout.addWidget(widget)
        self._overlays_scroll.setVisible(True)

        # Initial render + LUT.
        self._apply_overlay_colormap(ov)
        item.setOpacity(ov.opacity)
        if kind == "static":
            img = self._normalize_frame_layout(np.asarray(data))
            item.setImage(img, autoLevels=False)
        else:
            self._render_overlay_frame(ov, 0)
        self.auto_overlay_levels(len(self._overlays) - 1)
        self.overlay_added.emit(self)
        return len(self._overlays) - 1

    def remove_overlay(self, index: int) -> None:
        if not (0 <= index < len(self._overlays)):
            return
        ov = self._overlays.pop(index)
        view_box = self.image_view.getView()
        view_box.removeItem(ov.image_item)
        self._overlays_layout.removeWidget(ov.row)
        ov.row.deleteLater()
        if ov.hist_dialog is not None:
            ov.hist_dialog.close()
            ov.hist_dialog.deleteLater()
            ov.hist_dialog = None
        if not self._overlays:
            self._overlays_scroll.setVisible(False)
        self.overlay_removed.emit(self)

    def set_overlay_visible(self, index: int, visible: bool) -> None:
        ov = self._overlays[index]
        ov.visible = bool(visible)
        ov.image_item.setVisible(ov.visible)

    def set_overlay_opacity(self, index: int, opacity: float) -> None:
        ov = self._overlays[index]
        ov.opacity = max(0.0, min(1.0, float(opacity)))
        ov.image_item.setOpacity(ov.opacity)

    def set_overlay_colormap(self, index: int, cmap_name: str) -> None:
        ov = self._overlays[index]
        ov.cmap_name = cmap_name
        self._apply_overlay_colormap(ov)

    def set_overlay_levels(self, index: int, lo: float, hi: float) -> None:
        ov = self._overlays[index]
        ov.levels = (float(lo), float(hi))
        if not _is_rgb_array(ov.source if ov.kind == "static" else None):
            ov.image_item.setLevels(ov.levels)

    def auto_overlay_levels(self, index: int) -> None:
        ov = self._overlays[index]
        img = self._overlay_current_image(ov)
        if img is None or _is_rgb(img):
            return
        lo, hi = float(np.min(img)), float(np.max(img))
        if hi <= lo:
            hi = lo + 1.0
        ov.levels = (lo, hi)
        ov.image_item.setLevels(ov.levels)

    def _overlay_current_image(self, ov: _Overlay) -> np.ndarray | None:
        """Return the raw (pre-layout-normalize) image for this overlay's
        current display state, or None if unavailable. For video overlays
        this reads `ov.current_frame` — independent of base."""
        if ov.kind == "static":
            return np.asarray(ov.source)
        n = int(ov.source.shape[0])
        f = max(0, min(n - 1, int(ov.current_frame)))
        return np.asarray(ov.source[f])

    def _apply_overlay_colormap(self, ov: _Overlay) -> None:
        """RGB(A) overlays ignore LUTs; for single-channel data, set the LUT
        from the chosen tint colormap."""
        img = self._overlay_current_image(ov)
        if img is None or _is_rgb(img):
            ov.image_item.setLookupTable(None)
            return
        lut = _make_colormap(ov.cmap_name).getLookupTable(0.0, 1.0, 256)
        ov.image_item.setLookupTable(lut)

    def _build_overlay_widget(self, ov: _Overlay) -> QWidget:
        """Build the overlay control widget. Static overlays get a single
        row; video overlays get a framed multi-row block with their own
        slider, step buttons, and fps spinbox (independent of base)."""
        if ov.kind == "static":
            return self._build_overlay_static_row(ov)
        return self._build_overlay_video_block(ov)

    def _build_overlay_static_row(self, ov: _Overlay) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(2, 1, 2, 1)
        h.setSpacing(4)

        cb_visible = QCheckBox()
        cb_visible.setChecked(True)
        cb_visible.setToolTip("Show / hide this overlay")
        cb_visible.toggled.connect(lambda on, o=ov: self._on_overlay_visibility_toggled(o, on))
        ov.visible_cb = cb_visible
        h.addWidget(cb_visible)

        label = QLabel(f"{ov.name} [static]")
        label.setToolTip(ov.name)
        label.setMinimumWidth(80)
        h.addWidget(label, stretch=1)

        h.addWidget(self._build_overlay_cmap(ov))
        h.addWidget(QLabel("α"))
        h.addWidget(self._build_overlay_opacity(ov))
        h.addWidget(self._build_overlay_auto_btn(ov))
        h.addWidget(self._build_overlay_hist_btn(ov))
        h.addWidget(self._build_overlay_remove_btn(ov))

        return row

    def _build_overlay_video_block(self, ov: _Overlay) -> QWidget:
        block = QFrame()
        block.setFrameShape(QFrame.Shape.StyledPanel)
        v = QVBoxLayout(block)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(2)

        # Header row: visibility + name + remove (right-aligned)
        header = QHBoxLayout()
        header.setSpacing(4)
        cb_visible = QCheckBox()
        cb_visible.setChecked(True)
        cb_visible.setToolTip("Show / hide this overlay")
        cb_visible.toggled.connect(lambda on, o=ov: self._on_overlay_visibility_toggled(o, on))
        ov.visible_cb = cb_visible
        header.addWidget(cb_visible)
        label = QLabel(f"{ov.name} [video]")
        label.setToolTip(ov.name)
        header.addWidget(label, stretch=1)
        header.addWidget(self._build_overlay_remove_btn(ov))
        v.addLayout(header)

        # Slider row.
        n_frames = int(ov.source.shape[0])
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, max(0, n_frames - 1))
        slider.setValue(0)
        slider.valueChanged.connect(lambda val, o=ov: self._on_overlay_slider_changed(o, val))
        v.addWidget(slider)
        ov.slider = slider

        # Mid row: frame label + cmap + α + Auto.
        mid = QHBoxLayout()
        mid.setSpacing(4)
        frame_label = QLabel(f"Frame 0 / {max(0, n_frames - 1)}  |  t = 0.000 s")
        mid.addWidget(frame_label)
        ov.frame_label = frame_label
        mid.addStretch()
        mid.addWidget(self._build_overlay_cmap(ov))
        mid.addWidget(QLabel("α"))
        mid.addWidget(self._build_overlay_opacity(ov))
        mid.addWidget(self._build_overlay_auto_btn(ov))
        mid.addWidget(self._build_overlay_hist_btn(ov))
        v.addLayout(mid)

        # Bottom row: fps + step buttons.
        bot = QHBoxLayout()
        bot.setSpacing(4)
        bot.addWidget(QLabel("fps:"))
        fps_spin = QDoubleSpinBox()
        fps_spin.setRange(0.1, 1000.0)
        fps_spin.setDecimals(2)
        fps_spin.setSingleStep(1.0)
        fps_spin.setValue(ov.fps)
        fps_spin.setMinimumWidth(96)
        fps_spin.valueChanged.connect(lambda val, o=ov: self._on_overlay_fps_changed(o, val))
        bot.addWidget(fps_spin)
        bot.addStretch()
        for text, delta in (("◀◀", -10), ("◀", -1), ("▶", 1), ("▶▶", 10)):
            btn = QPushButton(text)
            btn.setFixedWidth(36)
            btn.clicked.connect(lambda _, o=ov, d=delta: self._step_overlay(o, d))
            bot.addWidget(btn)
        v.addLayout(bot)

        return block

    # ----- Per-overlay reusable widget factories -----

    def _build_overlay_cmap(self, ov: _Overlay) -> QComboBox:
        cmap = QComboBox()
        cmap.addItems(list(_TINT_STOPS.keys()))
        cmap.setCurrentText(ov.cmap_name)
        cmap.setMinimumWidth(100)
        cmap.currentTextChanged.connect(lambda name, o=ov: self._on_overlay_cmap_changed(o, name))
        return cmap

    def _build_overlay_opacity(self, ov: _Overlay) -> QDoubleSpinBox:
        opacity = QDoubleSpinBox()
        opacity.setRange(0.0, 1.0)
        opacity.setSingleStep(0.05)
        opacity.setDecimals(2)
        opacity.setValue(ov.opacity)
        opacity.setMinimumWidth(88)
        opacity.valueChanged.connect(lambda v, o=ov: self._on_overlay_opacity_changed(o, v))
        return opacity

    def _build_overlay_auto_btn(self, ov: _Overlay) -> QPushButton:
        btn = QPushButton("Auto")
        btn.setFixedWidth(48)
        btn.setToolTip("Recompute contrast levels from this overlay's current frame")
        btn.clicked.connect(lambda _, o=ov: self._on_overlay_auto(o))
        return btn

    def _build_overlay_remove_btn(self, ov: _Overlay) -> QPushButton:
        btn = QPushButton("×")
        btn.setFixedWidth(28)
        btn.setToolTip("Remove this overlay")
        btn.clicked.connect(lambda _, o=ov: self._on_overlay_remove(o))
        return btn

    def _build_overlay_hist_btn(self, ov: _Overlay) -> QPushButton:
        btn = QPushButton("Hist")
        btn.setFixedWidth(48)
        btn.setToolTip("Show histogram dialog for manual contrast adjustment")
        btn.clicked.connect(lambda _, o=ov: self._toggle_overlay_hist(o))
        return btn

    def _toggle_overlay_hist(self, ov: _Overlay) -> None:
        """Lazy-create a floating histogram dialog bound to this overlay's
        ImageItem; toggle its visibility on subsequent clicks."""
        if ov.hist_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Histogram — {ov.name}")
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(4, 4, 4, 4)
            hist = pg.HistogramLUTWidget()
            hist.setImageItem(ov.image_item)
            hist.item.sigLevelsChanged.connect(
                lambda *_, h=hist, o=ov: self._on_overlay_hist_levels_changed(h, o)
            )
            layout.addWidget(hist)
            dialog.resize(240, 420)
            ov.hist_dialog = dialog
        if ov.hist_dialog.isVisible():
            ov.hist_dialog.hide()
        else:
            ov.hist_dialog.show()
            ov.hist_dialog.raise_()
            ov.hist_dialog.activateWindow()

    def _on_overlay_hist_levels_changed(self, hist: Any, ov: _Overlay) -> None:
        """Write histogram-driven levels back into ov.levels so they survive
        frame changes (since _render_overlay_frame re-applies ov.levels)."""
        lo, hi = hist.item.getLevels()
        ov.levels = (float(lo), float(hi))

    # ----- Overlay signal handlers (resolve overlay by identity) -----

    def _overlay_index(self, ov: _Overlay) -> int | None:
        for i, x in enumerate(self._overlays):
            if x is ov:
                return i
        return None

    def _on_overlay_visibility_toggled(self, ov: _Overlay, on: bool) -> None:
        i = self._overlay_index(ov)
        if i is not None:
            self.set_overlay_visible(i, on)

    def _on_overlay_opacity_changed(self, ov: _Overlay, value: float) -> None:
        i = self._overlay_index(ov)
        if i is not None:
            self.set_overlay_opacity(i, value)

    def _on_overlay_cmap_changed(self, ov: _Overlay, name: str) -> None:
        i = self._overlay_index(ov)
        if i is not None:
            self.set_overlay_colormap(i, name)

    def _on_overlay_auto(self, ov: _Overlay) -> None:
        i = self._overlay_index(ov)
        if i is not None:
            self.auto_overlay_levels(i)

    def _on_overlay_remove(self, ov: _Overlay) -> None:
        i = self._overlay_index(ov)
        if i is not None:
            self.remove_overlay(i)

    def _on_overlay_slider_changed(self, ov: _Overlay, value: int) -> None:
        if value == ov.current_frame:
            return
        self._render_overlay_frame(ov, value)

    def _step_overlay(self, ov: _Overlay, delta: int) -> None:
        if ov.slider is None:
            return
        ov.slider.setValue(ov.slider.value() + delta)

    def _on_overlay_fps_changed(self, ov: _Overlay, fps: float) -> None:
        if fps <= 0:
            return
        ov.fps = float(fps)
        # Refresh the frame label so the "t = ..." text reflects the new fps.
        if ov.frame_label is not None and ov.kind == "video":
            n = int(ov.source.shape[0])
            t = ov.current_frame / ov.fps
            ov.frame_label.setText(
                f"Frame {ov.current_frame} / {n - 1}  |  t = {t:.3f} s"
            )

    # ----- Add overlay entry points -----

    def _on_add_overlay_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Add overlay…",
            get_last_dir(),
            OVERLAY_FILE_FILTER,
        )
        if path:
            set_last_dir_from_path(path)
            self._try_add_overlay(path)

    def _try_add_overlay(self, path: str | Path) -> None:
        try:
            self.add_overlay(path)
        except LoadError as exc:
            print(f"Overlay load failed for {path}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"Overlay load failed for {path}: {exc}")

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802
        menu = QMenu(self)
        act_add = QAction("Add Overlay…", self)
        act_add.triggered.connect(self._on_add_overlay_clicked)
        menu.addAction(act_add)
        if self._overlays:
            menu.addSeparator()
            for ov in list(self._overlays):
                act = QAction(f'Remove "{ov.name}"', self)
                act.triggered.connect(lambda _, o=ov: self._on_overlay_remove(o))
                menu.addAction(act)
        menu.exec(event.globalPos())

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        last: str | None = None
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local:
                self._try_add_overlay(local)
                last = local
        if last:
            set_last_dir_from_path(last)
        event.acceptProposedAction()


def _is_rgb(img: np.ndarray) -> bool:
    """True if `img` is an RGB(A) image (channel axis at end with 3 or 4)."""
    return img.ndim == 3 and img.shape[-1] in (3, 4)


def _is_rgb_array(arr: Any) -> bool:
    if arr is None:
        return False
    try:
        a = np.asarray(arr) if not hasattr(arr, "ndim") else arr
    except Exception:  # noqa: BLE001
        return False
    return a.ndim == 3 and a.shape[-1] in (3, 4)
