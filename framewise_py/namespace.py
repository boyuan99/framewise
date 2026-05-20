"""Pluggable population of the embedded IPython kernel's user namespace.

Each subsystem that wants to expose objects to the console implements
NamespaceProvider. MainWindow holds an ordered list of providers and pushes
the merged result into the kernel on startup and on every kernel switch.
Phase 2 (ROI / neuron labeling) will append its own provider here without
touching console_panel.py or main_window.py's core wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .console_panel import KernelHost
    from .main_window import MainWindow


@runtime_checkable
class NamespaceProvider(Protocol):
    def collect(self, host: "KernelHost") -> dict[str, Any]:
        """Return {name: object} to inject into the kernel namespace.

        `host.is_same_process` tells the provider whether the kernel shares
        framewise's process (live Qt objects are safe to push) or runs in a
        subprocess (only picklable values survive the wire)."""
        ...


class RemoteWindowStub:
    """Placeholder pushed in out-of-process mode where the live MainWindow
    cannot cross the process boundary."""

    def __init__(self, pid: int) -> None:
        self._pid = pid

    def __repr__(self) -> str:
        return (
            f"<RemoteWindowStub: framewise GUI runs in process {self._pid}. "
            "Switch View → Console kernel → Same-process for live access.>"
        )


class RemotePanelStub:
    """Placeholder for a VideoPanel in out-of-process mode. Carries only the
    picklable metadata; any other attribute access explains how to get live
    access."""

    def __init__(self, name: str, path: str, n_frames: int) -> None:
        self.name = name
        self.path = path
        self.n_frames = n_frames

    def __getattr__(self, item: str):
        raise AttributeError(
            f"{item!r} is unavailable in out-of-process mode. The live panel "
            "lives in the framewise GUI process. Switch View → Console kernel "
            "→ Same-process, or reload the file from `path` in this kernel."
        )

    def __repr__(self) -> str:
        return f"<RemotePanelStub name={self.name!r} n_frames={self.n_frames}>"


class CoreNamespaceProvider:
    """Supplies the always-available framewise objects and helpers."""

    def __init__(self, window: "MainWindow") -> None:
        self._window = window

    def collect(self, host: "KernelHost") -> dict[str, Any]:
        import numpy as np
        import pyqtgraph as pg

        if host.is_same_process:
            return self._live_namespace(np, pg)
        return self._remote_namespace(np, pg)

    # ----- same-process: live objects -----

    def _live_namespace(self, np, pg) -> dict[str, Any]:
        w = self._window

        def panel(name: str):
            for e in w.panel_manager.entries:
                if e.panel.name == name:
                    return e.panel
            raise KeyError(
                f"No panel named {name!r}. Available: "
                f"{[e.panel.name for e in w.panel_manager.entries]}"
            )

        def panels() -> list:
            return [e.panel for e in w.panel_manager.entries]

        def current_time() -> float:
            return w.master_clock.time

        def current_frame(name: str):
            return getattr(panel(name), "current_frame", None)

        return {
            "window": w,
            "mc": w.master_clock,
            "master_clock": w.master_clock,
            "pm": w.panel_manager,
            "panel_manager": w.panel_manager,
            "sync": w.sync_controller,
            "panel": panel,
            "panels": panels,
            "current_time": current_time,
            "current_frame": current_frame,
            "np": np,
            "pg": pg,
        }

    # ----- out-of-process: picklable stubs only -----

    def _remote_namespace(self, np, pg) -> dict[str, Any]:
        import os

        w = self._window
        stubs = {
            e.panel.name: RemotePanelStub(
                name=e.panel.name,
                path=str(e.path),
                n_frames=getattr(e.panel, "n_frames", -1),
            )
            for e in w.panel_manager.entries
        }

        def panel(name: str):
            if name not in stubs:
                raise KeyError(
                    f"No panel named {name!r}. Available: {list(stubs)}"
                )
            return stubs[name]

        return {
            "window": RemoteWindowStub(os.getpid()),
            "panel": panel,
            "panels": lambda: list(stubs.values()),
            "np": np,
            "pg": pg,
        }
