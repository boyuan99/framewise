"""Single source of truth for the current playback time.

Every panel (video, signal, trial, …) that wants to participate in frame-lock
listens to MasterClock.time_changed. Panels that drive scrubbing (e.g. a
VideoPanel slider) call set_time() to update the master.

The class is GUI-framework independent (just QObject + a pyqtSignal) so it can
be unit-tested without spinning up a window.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal


class MasterClock(QObject):
    time_changed = pyqtSignal(float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._t = 0.0
        self._dispatching = False

    @property
    def time(self) -> float:
        return self._t

    def set_time(self, t: float) -> None:
        """Update master time. No-op if already at this time (within float
        precision) or if we're mid-dispatch (prevents A→B→A loops between
        panels that both drive and listen)."""
        if self._dispatching:
            return
        if abs(t - self._t) < 1e-9:
            return
        self._t = float(t)
        self._dispatching = True
        try:
            self.time_changed.emit(self._t)
        finally:
            self._dispatching = False
