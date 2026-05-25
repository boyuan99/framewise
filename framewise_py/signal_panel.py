"""Multi-trace signal panel with time cursor and sliding-window view.

Each panel hosts one pyqtgraph PlotWidget that draws one or more 1D signal
traces sharing a common time axis. A vertical InfiniteLine acts as the
master-clock cursor; the plot's x-range follows the cursor at ±window seconds.

Traces are not re-masked on every clock tick — the full trace is drawn once
and pyqtgraph clips to the visible x-range, which is much faster than
re-uploading sliced data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .master_clock import MasterClock

DEFAULT_WINDOW_SECONDS = 5.0
# Largest selectable time window (spinbox max + wheel-zoom clamp).
MAX_WINDOW_SECONDS = 2000.0

# Distinct, color-blind-friendly palette (Tol bright)
_TRACE_COLORS = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
]


@dataclass
class Trace:
    name: str
    data: np.ndarray  # 1D
    sampling_rate: float  # Hz

    @property
    def duration(self) -> float:
        return len(self.data) / self.sampling_rate

    def time_axis(self) -> np.ndarray:
        return np.arange(len(self.data)) / self.sampling_rate


class SignalPanel(QWidget):
    """One signal panel with N traces sharing the master clock."""

    # Emitted whenever the trace set changes (add/remove) so the Resource
    # Manager can refresh this panel's per-trace visibility children.
    traces_changed = pyqtSignal()

    # Emitted with the new window width (s) when the user changes it, so paired
    # panels (e.g. raw + demix) can keep their x-axis time window in sync.
    window_changed = pyqtSignal(float)

    def __init__(
        self,
        name: str,
        traces: list[Trace] | None = None,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._name = name
        self._window = float(window_seconds)
        self._syncing_window = False  # guards window_changed loops between paired panels
        self._master_clock: MasterClock | None = None
        self._trace_items: dict[str, pg.PlotDataItem] = {}
        self._trace_colors: dict[str, str] = {}  # name -> color, stable across re-adds

        self._build_ui()
        for t in traces or []:
            self.add_trace(t)
        self._update_xrange(0.0)

    @property
    def name(self) -> str:
        return self._name

    def bind_master_clock(self, clock: MasterClock) -> None:
        self._master_clock = clock
        clock.time_changed.connect(self._on_master_time_changed)
        self._on_master_time_changed(clock.time)

    def add_trace(self, trace: Trace, color: str | None = None) -> None:
        # Stable color per name so re-adding a trace keeps its color. An explicit
        # `color` (e.g. to match a neuron footprint) overrides the cycling palette
        # and is remembered for this name.
        if color is not None:
            self._trace_colors[trace.name] = color
        color = self._trace_colors.get(trace.name)
        if color is None:
            color = _TRACE_COLORS[len(self._trace_colors) % len(_TRACE_COLORS)]
            self._trace_colors[trace.name] = color
        pen = pg.mkPen(color=color, width=1.5)
        item = self.plot.plot(
            trace.time_axis(),
            trace.data,
            pen=pen,
            name=trace.name,
        )
        self._trace_items[trace.name] = item
        self.traces_changed.emit()

    def remove_trace(self, name: str) -> None:
        item = self._trace_items.pop(name, None)
        if item is not None:
            self.plot.removeItem(item)
        self.traces_changed.emit()

    # ----- per-trace visibility (managed from the Resource Manager) -----

    def trace_names(self) -> list[str]:
        return list(self._trace_items.keys())

    def trace_color(self, name: str) -> str | None:
        return self._trace_colors.get(name)

    def is_trace_visible(self, name: str) -> bool:
        item = self._trace_items.get(name)
        return bool(item.isVisible()) if item is not None else False

    def set_trace_visible(self, name: str, visible: bool) -> None:
        item = self._trace_items.get(name)
        if item is not None:
            item.setVisible(bool(visible))

    def set_legend_visible(self, on: bool) -> None:
        if self.legend is not None:
            self.legend.setVisible(bool(on))

    def trace_arrays(self) -> dict[str, np.ndarray]:
        """{trace name: full y-data} — lets the console read trace values.

        Reads the original `yData`, NOT `getData()`, because the plot
        auto-downsamples/clips for rendering and `getData()` would return the
        reduced display data."""
        out: dict[str, np.ndarray] = {}
        for name, item in self._trace_items.items():
            y = getattr(item, "yData", None)
            if y is None:
                data = item.getData()
                y = data[1] if data is not None else None
            if y is not None:
                out[name] = np.asarray(y)
        return out

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        # Render cost scales with points-on-screen, not trace length: clip to the
        # visible x-window (the cursor shows only ±window seconds of a long trace)
        # and auto-downsample. This keeps many/long traces (e.g. dozens of
        # box-selected neuron C traces, 36k samples each) responsive — the data
        # itself is read from C instantly; the cost is drawing the curves.
        self.plot.setClipToView(True)
        self.plot.setDownsampling(auto=True, mode="peak")
        self.legend = self.plot.addLegend(offset=(10, 10))
        self.plot.setMouseEnabled(x=False, y=True)  # x is master-clock controlled
        # x can't be free-zoomed (it re-centers on the clock each tick), so make
        # the mouse wheel resize the time window instead — centered on the cursor.
        self.plot.getViewBox().wheelEvent = self._on_wheel_zoom
        layout.addWidget(self.plot, stretch=1)

        # Vertical cursor at master clock time
        self.cursor = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color="#222", width=1.5, style=Qt.PenStyle.DashLine),
        )
        self.plot.addItem(self.cursor, ignoreBounds=True)

        controls = QHBoxLayout()

        self.cb_legend = QCheckBox("Legend")
        self.cb_legend.setChecked(True)
        self.cb_legend.setToolTip(
            "Show/hide the plot legend. With many traces, hide it and manage "
            "trace visibility in the Resource Manager instead."
        )
        self.cb_legend.toggled.connect(self.set_legend_visible)
        controls.addWidget(self.cb_legend)

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
        # Wide enough for "2000.0" plus the up/down arrows.
        self.window_spin.setMinimumWidth(96)
        self.window_spin.setToolTip(
            "Time window (s). Scroll over the plot / x-axis to zoom time; "
            "scroll over the y-axis to zoom amplitude."
        )
        self.window_spin.valueChanged.connect(self._on_window_changed)
        controls.addWidget(self.window_spin)

        layout.addLayout(controls)

    def _on_wheel_zoom(self, ev, axis=None) -> None:
        """Wheel over the plot body or x-axis resizes the time window (scroll up
        = smaller window). Wheel over the y-axis zooms amplitude via the default
        ViewBox behavior. `axis` is 1 for the y-axis, 0 for x, None for the body
        (pyqtgraph's AxisItem forwards the index)."""
        if axis == 1:
            pg.ViewBox.wheelEvent(self.plot.getViewBox(), ev, axis=1)
            return
        try:
            delta = ev.delta()
        except AttributeError:  # newer Qt wheel event
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
