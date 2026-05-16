"""Single video panel: pyqtgraph ImageView + time slider + frame controls.

Uses MasterClock for cross-panel frame-lock: when the user drags this panel's
slider, the master clock is updated; when the master clock updates from
elsewhere, this panel jumps to the corresponding frame (computed via fps).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .master_clock import MasterClock

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


class VideoPanel(QWidget):
    """One video's display + scrubbing controls.

    Frame ↔ time conversion uses self.fps. Connect a MasterClock with
    bind_master_clock() to participate in frame-lock with other panels.
    """

    # Emitted when the user changes the frame by dragging the slider or
    # clicking step buttons. Other listeners (sync controllers) may also use it.
    frame_changed = pyqtSignal(int)

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

        self._build_ui()
        self._show_frame(0, first=True)

    @property
    def name(self) -> str:
        return self._name

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
        self.frame_label = QLabel(self._frame_text(0))
        controls.addWidget(self.frame_label)

        controls.addStretch()

        controls.addWidget(QLabel("fps:"))
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 1000.0)
        self.fps_spin.setDecimals(2)
        self.fps_spin.setSingleStep(1.0)
        self.fps_spin.setValue(self._fps)
        self.fps_spin.setFixedWidth(80)
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        controls.addWidget(self.fps_spin)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(list(_TINT_STOPS.keys()))
        self.cmap_combo.setFixedWidth(85)
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

        layout.addLayout(controls)

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
        # changing fps doesn't desync us.
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
