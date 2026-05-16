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
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .master_clock import MasterClock

DEFAULT_WINDOW_SECONDS = 5.0

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
        self._master_clock: MasterClock | None = None
        self._trace_items: dict[str, pg.PlotDataItem] = {}

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

    def add_trace(self, trace: Trace) -> None:
        color = _TRACE_COLORS[len(self._trace_items) % len(_TRACE_COLORS)]
        pen = pg.mkPen(color=color, width=1.5)
        item = self.plot.plot(
            trace.time_axis(),
            trace.data,
            pen=pen,
            name=trace.name,
        )
        self._trace_items[trace.name] = item

    def remove_trace(self, name: str) -> None:
        item = self._trace_items.pop(name, None)
        if item is not None:
            self.plot.removeItem(item)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.addLegend(offset=(10, 10))
        self.plot.setMouseEnabled(x=False, y=True)  # x is master-clock controlled
        layout.addWidget(self.plot, stretch=1)

        # Vertical cursor at master clock time
        self.cursor = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color="#222", width=1.5, style=Qt.PenStyle.DashLine),
        )
        self.plot.addItem(self.cursor, ignoreBounds=True)

        controls = QHBoxLayout()
        controls.addStretch()

        self.time_label = QLabel("t = 0.000 s")
        self.time_label.setStyleSheet("color: #555;")
        controls.addWidget(self.time_label)

        controls.addWidget(QLabel("window (s):"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(0.1, 600.0)
        self.window_spin.setDecimals(1)
        self.window_spin.setSingleStep(1.0)
        self.window_spin.setValue(self._window)
        self.window_spin.setFixedWidth(70)
        self.window_spin.valueChanged.connect(self._on_window_changed)
        controls.addWidget(self.window_spin)

        layout.addLayout(controls)

    def _on_master_time_changed(self, t: float) -> None:
        self.cursor.setPos(t)
        self.time_label.setText(f"t = {t:.3f} s")
        self._update_xrange(t)

    def _on_window_changed(self, value: float) -> None:
        self._window = float(value)
        t = self._master_clock.time if self._master_clock else 0.0
        self._update_xrange(t)

    def _update_xrange(self, t: float) -> None:
        self.plot.setXRange(t - self._window, t + self._window, padding=0)
