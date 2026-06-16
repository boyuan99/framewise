"""Time-frequency heatmap panel: pyqtgraph ImageItem + master-clock cursor.

Displays a 2D (n_freqs, n_time) power matrix as a colored heatmap. A vertical
InfiniteLine tracks the master clock; the plot's x-range follows it at ±window
seconds, matching SignalPanel's scrubbing UX. The image is drawn once and
pyqtgraph clips to the visible x-range — fast even for full-session
spectrograms.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .master_clock import MasterClock

DEFAULT_WINDOW_SECONDS = 5.0
MAX_WINDOW_SECONDS = 2000.0


class SpectrogramPanel(QWidget):
    """Static 2D heatmap (time x frequency) with a master-clock cursor."""

    # Emitted with new window width when the user changes it, so paired panels
    # (e.g. a SignalPanel showing the same session) can keep their x-axis in sync.
    window_changed = pyqtSignal(float)

    def __init__(
        self,
        name: str,
        image: np.ndarray,           # (n_freqs, n_time)
        freqs: np.ndarray,           # (n_freqs,) Hz, increasing
        fs: float,                   # time-axis sampling rate (Hz)
        units: str = "dB",
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._name = name
        self._image = np.asarray(image, dtype=np.float32)
        self._freqs = np.asarray(freqs, dtype=np.float64)
        self._fs = float(fs)
        self._units = units
        self._window = float(window_seconds)
        self._master_clock: MasterClock | None = None
        self._syncing_window = False

        if self._image.ndim != 2:
            raise ValueError(
                f"image must be 2D (n_freqs, n_time); got shape {self._image.shape}"
            )
        if len(self._freqs) != self._image.shape[0]:
            raise ValueError(
                f"freqs length {len(self._freqs)} does not match image rows {self._image.shape[0]}"
            )

        self._build_ui()
        self._render_image()
        self._update_xrange(0.0)

    @property
    def name(self) -> str:
        return self._name

    def bind_master_clock(self, clock: MasterClock) -> None:
        self._master_clock = clock
        clock.time_changed.connect(self._on_master_time_changed)
        self._on_master_time_changed(clock.time)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.setLabel("left", "frequency (Hz)")
        self.plot.setLabel("bottom", "time (s)")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        # x-axis is master-clock controlled (no free pan/zoom); y stays interactive.
        self.plot.setMouseEnabled(x=False, y=True)
        self.plot.getViewBox().wheelEvent = self._on_wheel_zoom
        layout.addWidget(self.plot, stretch=1)

        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        # Vertical cursor at master clock time
        self.cursor = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color="#222", width=1.5, style=Qt.PenStyle.DashLine),
        )
        self.plot.addItem(self.cursor, ignoreBounds=True)

        # Color bar on the right
        cmap = pg.colormap.get("viridis")
        lo = float(np.percentile(self._image, 5))
        hi = float(np.percentile(self._image, 99))
        self.color_bar = pg.ColorBarItem(
            values=(lo, hi),
            colorMap=cmap,
            label=self._units,
            interactive=False,
        )
        self.color_bar.setImageItem(self.image_item, insert_in=self.plot.getPlotItem())

        # Controls row
        controls = QHBoxLayout()
        controls.addStretch()
        self.time_label = QLabel("t = 0.000 s")
        self.time_label.setStyleSheet("color: #555;")
        controls.addWidget(self.time_label)

        controls.addWidget(QLabel("window (s):"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(0.1, MAX_WINDOW_SECONDS)
        self.window_spin.setDecimals(1)
        self.window_spin.setSingleStep(1.0)
        self.window_spin.setValue(self._window)
        self.window_spin.setMinimumWidth(96)
        self.window_spin.setToolTip(
            "Time window (s). Scroll over the plot to zoom time; "
            "scroll over the y-axis to zoom frequency."
        )
        self.window_spin.valueChanged.connect(self._on_window_changed)
        controls.addWidget(self.window_spin)

        layout.addLayout(controls)

    def _render_image(self) -> None:
        # pyqtgraph ImageItem with default axisOrder='col-major' wants (W, H).
        # Our data is (n_freqs, n_time) — transpose so W=time, H=freq.
        img_t = self._image.T  # (n_time, n_freqs)
        self.image_item.setImage(img_t, autoLevels=False)

        n_freqs, n_time = self._image.shape
        duration = n_time / self._fs
        freq_min = float(self._freqs[0])
        freq_max = float(self._freqs[-1])

        # Map image pixel space -> plot coordinates: x in [0, duration], y in [freq_min, freq_max]
        self.image_item.setRect(QRectF(0.0, freq_min, duration, freq_max - freq_min))
        self.plot.setYRange(freq_min, freq_max, padding=0)
        self._duration = duration

    def _on_wheel_zoom(self, ev, axis=None) -> None:
        """Wheel over plot body / x-axis resizes the time window; wheel over y-axis
        zooms frequency via default ViewBox behavior."""
        if axis == 1:
            pg.ViewBox.wheelEvent(self.plot.getViewBox(), ev, axis=1)
            return
        try:
            delta = ev.delta()
        except AttributeError:
            delta = ev.angleDelta().y()
        if delta:
            factor = 0.8 if delta > 0 else 1.25
            new = min(MAX_WINDOW_SECONDS, max(0.1, self._window * factor))
            self.window_spin.setValue(round(new, 1))
        ev.accept()

    def _on_master_time_changed(self, t: float) -> None:
        self.cursor.setPos(t)
        self.time_label.setText(f"t = {t:.3f} s")
        self._update_xrange(t)

    def _on_window_changed(self, value: float) -> None:
        self._window = float(value)
        t = self._master_clock.time if self._master_clock else 0.0
        self._update_xrange(t)
        if not self._syncing_window:
            self.window_changed.emit(self._window)

    def set_window(self, value: float) -> None:
        """Set the time window (s) from a paired panel, without echoing back."""
        value = float(value)
        if abs(value - self._window) < 1e-6:
            return
        self._syncing_window = True
        try:
            self.window_spin.setValue(round(value, 1))
        finally:
            self._syncing_window = False

    def _update_xrange(self, t: float) -> None:
        self.plot.setXRange(t - self._window, t + self._window, padding=0)
