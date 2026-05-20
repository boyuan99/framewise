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
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QContextMenuEvent, QDragEnterEvent, QDropEvent
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
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .loaders import LoadError, load_overlay
from .master_clock import MasterClock
from .settings import get_last_dir, set_last_dir_from_path

DEFAULT_FPS = 20.0

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


class VideoPanel(QWidget):
    """One video's display + scrubbing controls.

    Frame ↔ time conversion uses self.fps. Connect a MasterClock with
    bind_master_clock() to participate in frame-lock with other panels.
    """

    # Emitted when the user changes the frame by dragging the slider or
    # clicking step buttons. Other listeners (sync controllers) may also use it.
    frame_changed = pyqtSignal(int)

    # Emitted with `self` whenever this panel's overlay list changes (add /
    # remove). Lets the Resource Manager refresh the tree without polling.
    overlay_added = pyqtSignal(object)
    overlay_removed = pyqtSignal(object)

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

        layout.addLayout(controls)

        # Overlays section — hidden until first overlay is added.
        self._overlays_frame = QFrame()
        self._overlays_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._overlays_layout = QVBoxLayout(self._overlays_frame)
        self._overlays_layout.setContentsMargins(2, 2, 2, 2)
        self._overlays_layout.setSpacing(1)
        self._overlays_frame.setVisible(False)
        layout.addWidget(self._overlays_frame)

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
        self._overlays_frame.setVisible(True)

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
            self._overlays_frame.setVisible(False)
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
