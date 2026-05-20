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
    QToolBar,
    QVBoxLayout,
)

from .console_panel import ConsolePanel, JupyterLabLauncher
from .master_clock import MasterClock
from .namespace import CoreNamespaceProvider
from .panels import PanelManager
from .settings import get_last_dir, set_last_dir_from_path, settings
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

        placeholder = QLabel("Drop videos here or use File → Open")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #888;")
        self.setCentralWidget(placeholder)

        self.master_clock = MasterClock(self)

        self.panel_manager = PanelManager(self, master_clock=self.master_clock)
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
        self._namespace_providers = [CoreNamespaceProvider(self)]
        self.console_panel = ConsolePanel(self._collect_namespace, parent=self)
        self.console_dock = QDockWidget("Console", self)
        self.console_dock.setObjectName("console_dock")
        self.console_dock.setWidget(self.console_panel)
        self.console_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)

        self.jupyter_launcher = JupyterLabLauncher(
            self.console_panel.external_connection_dir
        )

        # Keep the console namespace current as panels come and go (matters for
        # out-of-process mode, whose namespace is a snapshot of picklable stubs;
        # same-process resolves panels live so the re-push is a harmless no-op).
        self.panel_manager.on_added(lambda *_: self._refresh_console_namespace())
        self.panel_manager.on_removed(lambda *_: self._refresh_console_namespace())

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
        self._restore_settings()

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        act_open = QAction("&Open…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_dialog)
        file_menu.addAction(act_open)

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

        self.act_stop_lab = QAction("S&top Jupyter Lab", self)
        self.act_stop_lab.setEnabled(False)
        self.act_stop_lab.triggered.connect(self._stop_jupyter_lab)
        view_menu.addAction(self.act_stop_lab)

        act_conn = QAction("Show &connection info…", self)
        act_conn.triggered.connect(self._show_connection_info)
        view_menu.addAction(act_conn)

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
        started = self.jupyter_launcher.start()
        self.act_stop_lab.setEnabled(True)
        kid = self.console_panel.connection_info()["kernel_id"]
        if started:
            QMessageBox.information(
                self,
                "Jupyter Lab",
                "Jupyter Lab is starting and will open in your browser.\n\n"
                "In a new Notebook/Console, choose the kernel:\n"
                f"  Connect to Existing → {kid}\n\n"
                "Stopping Jupyter Lab (View → Stop Jupyter Lab) will NOT kill "
                "this kernel; framewise keeps owning it.",
            )
        else:
            QMessageBox.information(
                self,
                "Jupyter Lab",
                "Jupyter Lab is already running. Look for the existing browser "
                f"tab, then connect to existing kernel:\n  {kid}",
            )

    def _stop_jupyter_lab(self) -> None:
        self.jupyter_launcher.stop()
        self.act_stop_lab.setEnabled(False)

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

    # ----- Persistence -----

    def _restore_settings(self) -> None:
        s = settings()
        geom = s.value("geometry")
        if geom is not None:
            self.restoreGeometry(geom)
        state = s.value("windowState")
        if state is not None:
            self.restoreState(state)

    def _save_settings(self) -> None:
        s = settings()
        s.setValue("geometry", self.saveGeometry())
        s.setValue("windowState", self.saveState())

    def closeEvent(self, event) -> None:
        self._save_settings()
        try:
            self.jupyter_launcher.stop()
        except Exception:
            pass
        try:
            self.console_panel.shutdown()
        except Exception:
            pass
        super().closeEvent(event)
