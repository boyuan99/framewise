"""Main application window: hosts video dock widgets + sync panel + playback."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QGuiApplication, QKeySequence
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QTabBar,
    QToolBar,
    QVBoxLayout,
)

from .console_panel import ConsolePanel, JupyterLabLauncher
from .master_clock import MasterClock
from .namespace import (
    CoreNamespaceProvider,
    RoiNamespaceProvider,
    SegmentationNamespaceProvider,
)
from .notebook_panel import WEBENGINE_AVAILABLE, NotebookPanel
from .panels import PanelManager, VideoGrid
from .settings import get_last_dir, set_last_dir_from_path, settings
from .signal_panel import SignalPanel
from .sync import FILE_FILTER, ResourceManagerPanel, SyncController

# QTimer interval for playback ticks. 33ms ≈ 30Hz, smooth enough for any video
# regardless of its native fps (each panel rounds master_time to its own frame).
PLAYBACK_TICK_MS = 33

PLAYBACK_SPEEDS = [
    ("0.25x", 0.25),
    ("0.5x", 0.5),
    ("1x", 1.0),
    ("2x", 2.0),
    ("4x", 4.0),
]

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Framewise")
        self.resize(1400, 900)
        self.setDockNestingEnabled(True)

        self.master_clock = MasterClock(self)

        # Central area is a 2-page stack: the video grid and the embedded
        # notebook. Both stay alive simultaneously; the workspace tabs just
        # switch which page is shown (video playback and the kernel/Lab keep
        # running on the hidden page).
        self.video_grid = VideoGrid()
        self._central_stack = QStackedWidget()
        self._central_stack.addWidget(self.video_grid)  # page 0 — Video

        self.panel_manager = PanelManager(self.video_grid, master_clock=self.master_clock)
        self.sync_controller = SyncController()

        self.sync_panel = ResourceManagerPanel(self.sync_controller, self.panel_manager)
        self.sync_panel.add_video_requested.connect(self._add_videos)
        self.sync_dock = QDockWidget("Resource Manager", self)
        self.sync_dock.setWidget(self.sync_panel)
        # Keep "sync_manager" objectName so QSettings windowState restore still
        # finds this dock across the rename.
        self.sync_dock.setObjectName("sync_manager")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.sync_dock)

        # Console: ordered list of namespace providers (Phase 2 ROI subsystem
        # appends its own). The factory merges them in registration order.
        self._namespace_providers = [
            CoreNamespaceProvider(self),
            RoiNamespaceProvider(self),
            SegmentationNamespaceProvider(self),
        ]
        self.console_panel = ConsolePanel(self._collect_namespace, parent=self)
        self.console_dock = QDockWidget("Console", self)
        self.console_dock.setObjectName("console_dock")
        self.console_dock.setWidget(self.console_panel)
        self.console_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)
        # Hidden by default — show it via View → Console (Ctrl+`) when needed.
        self.console_dock.hide()

        self.jupyter_launcher = JupyterLabLauncher(
            self.console_panel.external_connection_dir
        )

        # Notebook = page 1 of the central stack (embedded Lab if PyQt6-WebEngine
        # is installed; otherwise a hint pointing at the external-browser flow).
        self.notebook_panel = None
        if WEBENGINE_AVAILABLE:
            self.notebook_panel = NotebookPanel(self)
            self._central_stack.addWidget(self.notebook_panel)  # page 1
        else:
            hint = QLabel(
                "Embedded Jupyter Lab needs PyQt6-WebEngine.\n"
                'Install with:  pip install -e ".[notebook]"\n\n'
                "Or use View → Open in external browser to run Lab in your browser."
            )
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hint.setStyleSheet("color: #888;")
            self._central_stack.addWidget(hint)  # page 1

        self.setCentralWidget(self._central_stack)

        self.panel_manager.on_added(self._on_panel_added)
        self.panel_manager.on_removed(self._on_panel_removed)

        # One dedicated ΔF/F signal panel per source video, keyed by video name.
        self._roi_trace_panels: dict = {}
        # One dedicated activity-trace panel per segmentation, keyed by name.
        self._neuron_trace_panels: dict = {}

        # Workspace = which central page is shown (Video grid vs Notebook).
        self._ws_tabbar: QTabBar | None = None

        # Playback timer — advances master_clock by real elapsed wall time so
        # playback stays at true speed even if rendering can't keep up with
        # PLAYBACK_TICK_MS (in which case we skip frames rather than slow down).
        self._playback_speed = 1.0
        self._last_tick: float | None = None
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(PLAYBACK_TICK_MS)
        self._playback_timer.timeout.connect(self._on_playback_tick)

        self._build_menus()
        self._build_toolbar()
        self._build_workspace_bar()
        self._restore_settings()
        # Start in the Video workspace with a freshly built default layout.
        self._set_workspace("Video")

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        act_open = QAction("&Open…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_dialog)
        file_menu.addAction(act_open)

        act_open_folder = QAction("Open &Folder…", self)
        act_open_folder.setShortcut("Ctrl+Shift+O")
        act_open_folder.setToolTip(
            "Open a folder: a segmentation result (SEG/ + rmbg/) or a TDT block"
        )
        act_open_folder.triggered.connect(self._open_folder_dialog)
        file_menu.addAction(act_open_folder)

        file_menu.addSeparator()

        act_save = QAction("&Save Labels", self)
        act_save.setShortcut("Ctrl+S")
        act_save.setToolTip(
            "Save cell-type labels for all open segmentations to SEG/cell_labels.json"
        )
        act_save.triggered.connect(self._save_labels)
        file_menu.addAction(act_save)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = self.menuBar().addMenu("&View")

        toggle_console = self.console_dock.toggleViewAction()
        toggle_console.setText("&Console")
        toggle_console.setShortcut("Ctrl+`")
        view_menu.addAction(toggle_console)
        view_menu.addAction(self.sync_dock.toggleViewAction())

        view_menu.addSeparator()

        kernel_menu = view_menu.addMenu("Console &kernel")
        self._kernel_mode_group = QActionGroup(self)
        self._kernel_mode_group.setExclusive(True)
        for label, mode in [
            ("&Same-process (shared live objects)", "same_process"),
            ("&Out-of-process (heavy compute)", "out_of_process"),
        ]:
            act = QAction(label, self, checkable=True)
            act.setChecked(mode == self.console_panel.mode)
            act.triggered.connect(lambda _checked, m=mode: self._switch_kernel(m))
            self._kernel_mode_group.addAction(act)
            kernel_menu.addAction(act)

        view_menu.addSeparator()

        self.act_open_lab = QAction("&Open in Jupyter Lab", self)
        self.act_open_lab.triggered.connect(self._open_jupyter_lab)
        view_menu.addAction(self.act_open_lab)

        self.act_open_lab_browser = QAction("Open in external &browser…", self)
        self.act_open_lab_browser.triggered.connect(self._open_jupyter_lab_browser)
        view_menu.addAction(self.act_open_lab_browser)

        self.act_stop_lab = QAction("S&top Jupyter Lab", self)
        self.act_stop_lab.setEnabled(False)
        self.act_stop_lab.triggered.connect(self._stop_jupyter_lab)
        view_menu.addAction(self.act_stop_lab)

        act_conn = QAction("Show &connection info…", self)
        act_conn.triggered.connect(self._show_connection_info)
        view_menu.addAction(act_conn)

        view_menu.addSeparator()

        act_tile = QAction("&Tile Videos", self)
        act_tile.triggered.connect(lambda: self.video_grid.tile())
        view_menu.addAction(act_tile)

        act_cascade = QAction("Ca&scade Videos", self)
        act_cascade.triggered.connect(lambda: self.video_grid.cascade())
        view_menu.addAction(act_cascade)

        act_reset = QAction("&Reset Layout", self)
        act_reset.triggered.connect(self._reset_layout)
        view_menu.addAction(act_reset)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Playback", self)
        toolbar.setObjectName("playback_toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.act_play = QAction("▶ Play", self)
        self.act_play.setCheckable(True)
        self.act_play.setShortcut(QKeySequence(Qt.Key.Key_Space))
        self.act_play.setToolTip("Play / Pause (Space)")
        self.act_play.toggled.connect(self._set_playing)
        toolbar.addAction(self.act_play)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Speed: "))
        self.speed_combo = QComboBox()
        for label, _ in PLAYBACK_SPEEDS:
            self.speed_combo.addItem(label)
        self.speed_combo.setCurrentIndex(
            next(i for i, (_, v) in enumerate(PLAYBACK_SPEEDS) if v == 1.0)
        )
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        toolbar.addWidget(self.speed_combo)

        toolbar.addSeparator()

        self.time_label = QLabel("t = 0.000 s")
        self.time_label.setMinimumWidth(110)
        toolbar.addWidget(self.time_label)
        self.master_clock.time_changed.connect(
            lambda t: self.time_label.setText(f"t = {t:.3f} s")
        )

    def add_video(self, path: str | Path) -> bool:
        try:
            self.panel_manager.add(path)
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Open failed", f"{path}\n\n{exc}")
            print(f"Failed to open {path}: {exc}", file=sys.stderr)
            return False

    def _add_videos(self, paths: list[str]) -> None:
        for p in paths:
            self.add_video(p)

    def _open_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open video files",
            get_last_dir(),
            FILE_FILTER,
        )
        if paths:
            set_last_dir_from_path(paths[0])
        for p in paths:
            self.add_video(p)

    def _open_folder_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Open folder (segmentation result or TDT block)",
            get_last_dir(),
        )
        if path:
            set_last_dir_from_path(path)
            self.add_video(path)

    def _save_labels(self) -> None:
        """Write cell-type labels for every open segmentation to its
        SEG/cell_labels.json (created on first save; an entry per neuron)."""
        segs = [e for e in self.panel_manager.entries if e.kind == "segmentation"]
        if not segs:
            QMessageBox.information(
                self, "Save Labels", "No segmentation is open to save labels for."
            )
            return
        saved, failed = [], []
        for e in segs:
            seg = e.panel.seg
            if seg.save_labels():
                saved.append(f"{seg.name} → {seg.label_path}")
            else:
                failed.append(seg.name)
        if failed:
            QMessageBox.warning(
                self,
                "Save Labels",
                "Could not save labels for:\n  " + "\n  ".join(failed)
                + "\n\n(The destination may be read-only.)",
            )
        if saved:
            self.statusBar().showMessage(
                f"Saved labels for {len(saved)} segmentation(s): "
                + "; ".join(saved),
                6000,
            )

    # ----- Playback -----

    def _set_playing(self, playing: bool) -> None:
        if playing:
            self.act_play.setText("⏸ Pause")
            self._last_tick = None
            self._playback_timer.start()
        else:
            self.act_play.setText("▶ Play")
            self._playback_timer.stop()

    def _on_speed_changed(self, index: int) -> None:
        self._playback_speed = PLAYBACK_SPEEDS[index][1]

    def _on_playback_tick(self) -> None:
        now = time.monotonic()
        if self._last_tick is None:
            dt = PLAYBACK_TICK_MS / 1000.0
        else:
            dt = now - self._last_tick
        self._last_tick = now
        self.master_clock.set_time(self.master_clock.time + dt * self._playback_speed)

    # ----- Console -----

    def _collect_namespace(self, host) -> dict:
        ns: dict = {}
        for provider in self._namespace_providers:
            ns.update(provider.collect(host))
        return ns

    def _switch_kernel(self, mode: str) -> None:
        self.console_panel.switch_to(mode)

    def _refresh_console_namespace(self) -> None:
        self.console_panel.refresh_namespace()

    def _open_jupyter_lab(self) -> None:
        """Switch to the Notebook workspace (embedded Lab). Falls back to the
        system browser if PyQt6-WebEngine isn't installed."""
        if self.notebook_panel is None:
            QMessageBox.information(
                self,
                "Jupyter Lab",
                "Embedded Lab needs PyQt6-WebEngine. Install it with:\n"
                '  pip install -e ".[notebook]"\n\n'
                "Opening in your system browser instead.",
            )
            self._open_jupyter_lab_browser()
            return
        # _set_workspace("Notebook") shows the page and starts Lab on first use.
        self._set_workspace("Notebook")

    def _open_jupyter_lab_browser(self) -> None:
        self.jupyter_launcher.start(embedded=False)
        self.act_stop_lab.setEnabled(True)
        kid = self.console_panel.connection_info()["kernel_id"]
        QMessageBox.information(
            self,
            "Jupyter Lab",
            "Jupyter Lab is starting in your browser.\n\n"
            "In a new Notebook/Console, choose the kernel:\n"
            f"  Connect to Existing → {kid}\n\n"
            "Stopping Jupyter Lab will NOT kill this kernel; framewise owns it.",
        )

    def _stop_jupyter_lab(self) -> None:
        self.jupyter_launcher.stop()
        self.act_stop_lab.setEnabled(False)
        if self.notebook_panel is not None:
            self.notebook_panel.clear()
        # Drop back to the video workspace now that the notebook is empty.
        self._set_workspace("Video")

    def _show_connection_info(self) -> None:
        info = self.console_panel.connection_info()
        text = (
            f"Kernel mode: {info['mode']}\n"
            f"Kernel ID:   {info['kernel_id']}\n"
            f"Connection:  {info['connection_file']}\n"
            f"External dir: {info['external_connection_dir']}\n\n"
            "Easiest: View → Open in Jupyter Lab (auto-configures discovery).\n\n"
            "Manual launch (equivalent):\n"
            "  jupyter lab --ServerApp.allow_external_kernels=True \\\n"
            f"    --ServerApp.external_connection_dir=\"{info['external_connection_dir']}\"\n\n"
            "Then New Console/Notebook → Connect to Existing python Kernel →\n"
            f"  {info['kernel_id']}"
        )
        dlg = QDialog(self)
        dlg.setWindowTitle("Console connection info")
        dlg.setModal(False)
        layout = QVBoxLayout(dlg)
        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        layout.addWidget(view)
        copy_btn = QPushButton("Copy connection file path")
        copy_btn.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(info["connection_file"])
        )
        layout.addWidget(copy_btn)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.close)
        layout.addWidget(buttons)
        dlg.resize(560, 260)
        dlg.show()

    # ----- Workspaces & layout -----

    def _build_workspace_bar(self) -> None:
        self._ws_tabbar = QTabBar()
        self._ws_tabbar.addTab("Video")
        self._ws_tabbar.addTab("Notebook")
        # Connect after addTab so the initial population doesn't fire a switch;
        # __init__ calls _set_workspace("Video") explicitly to build it.
        self._ws_tabbar.currentChanged.connect(self._on_ws_tab_changed)
        bar = QToolBar("Workspace", self)
        bar.setObjectName("workspace_toolbar")
        bar.setMovable(False)
        bar.addWidget(self._ws_tabbar)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)

    def _on_ws_tab_changed(self, index: int) -> None:
        self._set_workspace("Video" if index == 0 else "Notebook")

    def _set_workspace(self, name: str) -> None:
        """Switch the central page. Video grid and notebook both stay alive;
        this only changes which is shown."""
        index = 0 if name == "Video" else 1
        if self._ws_tabbar is not None and self._ws_tabbar.currentIndex() != index:
            self._ws_tabbar.blockSignals(True)
            self._ws_tabbar.setCurrentIndex(index)
            self._ws_tabbar.blockSignals(False)
        self._central_stack.setCurrentIndex(index)
        if name == "Notebook":
            self._ensure_notebook_ready()

    def _ensure_notebook_ready(self) -> None:
        """Start the embedded Lab on first entry to the Notebook workspace."""
        if self.notebook_panel is None:
            return
        if not self.jupyter_launcher.is_running():
            url = self.jupyter_launcher.start(embedded=True)
            self.act_stop_lab.setEnabled(True)
            self.notebook_panel.load(url)

    def _reset_layout(self) -> None:
        # Re-show the auxiliary docks and return to the Video workspace.
        self.sync_dock.show()
        self.console_dock.hide()
        self._set_workspace("Video")

    def _on_panel_added(self, entry) -> None:
        self._refresh_console_namespace()
        if entry.kind == "video":
            entry.panel.dff_extracted.connect(self._on_dff_extracted)
            # No timeout: keep the message visible for the whole extraction
            # (a single big block read can exceed any timeout); the next status
            # message replaces it.
            entry.panel.roi_status.connect(
                lambda msg: self.statusBar().showMessage(msg)
            )
        elif entry.kind == "segmentation":
            entry.panel.neuron_toggled.connect(
                lambda n, sel, e=entry: self._on_neuron_toggled(e, n, sel)
            )
            entry.panel.fps_changed.connect(
                lambda _fps, e=entry: self._on_seg_fps_changed(e)
            )
            entry.panel.roi_status.connect(
                lambda msg: self.statusBar().showMessage(msg, 6000)
            )
        # A freshly loaded video should be visible: jump to the Video workspace.
        self._set_workspace("Video")

    def _on_panel_removed(self, entry) -> None:
        self._refresh_console_namespace()
        # Forget any ROI ΔF/F panel that was just closed so it gets rebuilt later.
        for name, e in list(self._roi_trace_panels.items()):
            if e is entry:
                del self._roi_trace_panels[name]
        for name, panels in list(self._neuron_trace_panels.items()):
            if entry in (panels.get("raw"), panels.get("demix")):
                del self._neuron_trace_panels[name]

    def _on_dff_extracted(self, video_name: str, traces) -> None:
        """Route extracted ΔF/F traces into a dedicated per-video signal panel,
        creating it on first use and updating only the extracted ROIs' traces."""
        entry = self._roi_trace_panels.get(video_name)
        if entry is None or entry not in self.panel_manager.entries:
            panel = SignalPanel(name=f"ROI ΔF/F — {video_name}")
            entry = self.panel_manager.register_signal_panel(panel, panel.name)
            self._roi_trace_panels[video_name] = entry
        for tr in traces:
            entry.panel.remove_trace(tr.name)
            entry.panel.add_trace(tr)

    def _on_neuron_toggled(self, seg_entry, n: int, selected: bool) -> None:
        """Add/remove a clicked neuron's trace in the segmentation's companion
        panel(s): raw C in one, demixed C in a second when demix data is loaded.
        Both are colored to match the footprint and share the master clock, so
        raw vs demix line up in time for side-by-side comparison."""
        panels = self._neuron_companions(seg_entry, create=selected)
        if panels is None:
            return
        seg = seg_entry.panel.seg
        color = "#{:02x}{:02x}{:02x}".format(*seg.neuron_color(n))
        for kind, entry in panels.items():
            if entry is None:
                continue
            trace = seg.trace(n) if kind == "raw" else seg.trace_demix(n)
            entry.panel.remove_trace(trace.name)  # idempotent: avoid duplicates
            if selected:
                entry.panel.add_trace(trace, color=color)
                entry.panel.neuron_of_trace[trace.name] = int(n)
            else:
                entry.panel.neuron_of_trace.pop(trace.name, None)

    def _on_seg_fps_changed(self, seg_entry) -> None:
        """The segmentation panel's fps changed → re-derive its companion neuron
        traces at the new sampling rate so their time axis stays frame-locked."""
        panels = self._neuron_trace_panels.get(seg_entry.panel.seg.name)
        if not panels:
            return
        seg = seg_entry.panel.seg
        for n in seg_entry.panel.selected_neurons:
            color = "#{:02x}{:02x}{:02x}".format(*seg.neuron_color(n))
            for kind, entry in panels.items():
                if entry is None or entry not in self.panel_manager.entries:
                    continue
                trace = seg.trace(n) if kind == "raw" else seg.trace_demix(n)
                entry.panel.remove_trace(trace.name)
                entry.panel.add_trace(trace, color=color)

    def _neuron_companions(self, seg_entry, create: bool):
        """Return {"raw": entry, "demix": entry|None} for this segmentation's
        companion trace panels, (re)creating them on first use. None if they
        don't exist and `create` is False."""
        seg = seg_entry.panel.seg
        name = seg.name
        panels = self._neuron_trace_panels.get(name)
        alive = panels is not None and all(
            e in self.panel_manager.entries
            for e in panels.values()
            if e is not None
        )
        if not alive:
            if not create:
                return None
            has_dm = seg.has_demix
            panels = {
                "raw": self._make_companion(
                    seg_entry, "Neurons raw" if has_dm else "Neurons"
                ),
                "demix": self._make_companion(seg_entry, "Neurons demix")
                if has_dm
                else None,
            }
            self._neuron_trace_panels[name] = panels
            # Keep raw + demix time windows in sync (the cursor already shares the
            # master clock; this links the zoom/window so they stay aligned).
            if panels["demix"] is not None:
                raw_p, dmx_p = panels["raw"].panel, panels["demix"].panel
                raw_p.window_changed.connect(dmx_p.set_window)
                dmx_p.window_changed.connect(raw_p.set_window)
        return panels

    def _make_companion(self, seg_entry, label: str):
        """Build + register a companion SignalPanel linked back to the seg panel
        (so the Resource Manager can map its trace rows to neurons)."""
        panel = SignalPanel(name=f"{label} — {seg_entry.panel.seg.name}")
        panel.segmentation_panel = seg_entry.panel
        panel.neuron_of_trace = {}
        return self.panel_manager.register_signal_panel(panel, panel.name)

    # ----- Persistence -----

    def _restore_settings(self) -> None:
        # Only restore window size/position. Dock layout is driven by the
        # workspace system (built fresh each launch), not a single saved state.
        s = settings()
        geom = s.value("geometry")
        if geom is not None:
            self.restoreGeometry(geom)

    def _save_settings(self) -> None:
        s = settings()
        s.setValue("geometry", self.saveGeometry())

    def closeEvent(self, event) -> None:
        self._save_settings()
        # Cancel any in-flight ROI extraction threads so we don't get
        # "QThread destroyed while running" and a slow, blocked shutdown.
        for entry in self.panel_manager.video_entries:
            try:
                entry.panel.stop_roi_worker()
            except Exception:
                pass
        try:
            self.jupyter_launcher.stop()
        except Exception:
            pass
        try:
            self.console_panel.shutdown()
        except Exception:
            pass
        super().closeEvent(event)
