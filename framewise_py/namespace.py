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


class RoiNamespaceProvider:
    """Phase 2: exposes drawn ROIs, their masks, and extracted ΔF/F traces to
    the console / embedded Jupyter Lab.

    ROIs are live Qt objects (and masks are computed from live geometry), so
    these helpers only work with the in-process kernel. In out-of-process mode
    nothing is injected — switch View → Console kernel → Same-process.
    """

    def __init__(self, window: "MainWindow") -> None:
        self._window = window

    def collect(self, host: "KernelHost") -> dict[str, Any]:
        if not host.is_same_process:
            return {}

        from .roi import ellipse_mask

        w = self._window

        def _videos() -> dict:
            return {
                e.panel.name: e.panel
                for e in w.panel_manager.entries
                if e.kind == "video"
            }

        def _require(video: str):
            vids = _videos()
            if video not in vids:
                raise KeyError(f"No video named {video!r}. Available: {list(vids)}")
            return vids[video]

        def rois(video: str | None = None):
            """ROI items. With no arg: {video_name: [RoiItem, ...]}; with a
            video name: that video's list of RoiItem (each has .id .name
            .color .roi …)."""
            if video is None:
                return {name: list(p.rois) for name, p in _videos().items()}
            return list(_require(video).rois)

        def roi_masks(video: str):
            """{roi_name: bool ndarray (H, W)} computed from current geometry."""
            p = _require(video)
            hw = p.frame_hw
            return {
                it.name: ellipse_mask(
                    it.roi.pos(), it.roi.size(), it.roi.angle(), hw
                )
                for it in p.rois
            }

        def roi_dff(video: str | None = None):
            """Most recently extracted ΔF/F. With no arg: {video_name:
            {roi_name: ndarray}}; with a video name: {roi_name: ndarray}.
            Empty until you run Extract ΔF/F for that video."""
            panels = getattr(w, "_roi_trace_panels", {})
            live = w.panel_manager.entries

            def arrays(entry):
                return entry.panel.trace_arrays() if entry in live else {}

            if video is None:
                return {name: arrays(e) for name, e in panels.items()}
            entry = panels.get(video)
            return arrays(entry) if entry is not None else {}

        return {"rois": rois, "roi_masks": roi_masks, "roi_dff": roi_dff}


class SegmentationNamespaceProvider:
    """Exposes loaded segmentation results (footprints A, activity C, selection)
    to the console / embedded Jupyter Lab.

    Like the ROI provider, these are live objects (the sparse `A`, the in-RAM
    `C`, the live panel selection), so they are injected only for the in-process
    kernel. Out-of-process mode injects nothing — switch View → Console kernel →
    Same-process.
    """

    def __init__(self, window: "MainWindow") -> None:
        self._window = window

    def collect(self, host: "KernelHost") -> dict[str, Any]:
        if not host.is_same_process:
            return {}

        w = self._window

        def _panels() -> dict:
            return {
                e.panel.name: e.panel
                for e in w.panel_manager.entries
                if e.kind == "segmentation"
            }

        def _require(name: str):
            ps = _panels()
            if name not in ps:
                raise KeyError(f"No segmentation named {name!r}. Available: {list(ps)}")
            return ps[name]

        def seg(name: str | None = None):
            """SegmentationResult(s). With no arg: {name: SegmentationResult};
            with a name: that one (has .A .C .footprint(n) .trace(n) .neuron_at …)."""
            if name is None:
                return {n: p.seg for n, p in _panels().items()}
            return _require(name).seg

        def selected_neurons(name: str) -> list[int]:
            """0-based indices of neurons currently selected on that panel."""
            return _require(name).selected_neurons

        return {"seg": seg, "selected_neurons": selected_neurons}
