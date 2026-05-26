"""Embedded Jupyter Lab view (optional, requires PyQt6-WebEngine).

Hosts a QWebEngineView that loads framewise's locally-launched headless Jupyter
Lab. Kept import-guarded so the whole feature is optional: if PyQt6-WebEngine is
not installed, WEBENGINE_AVAILABLE is False and MainWindow falls back to opening
Lab in the system browser.
"""

from __future__ import annotations

import os

# Match the binding the rest of framewise uses (see framewise_py/__init__.py).
os.environ.setdefault("QT_API", "pyqt6")

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtWidgets import QLabel, QStackedLayout, QWidget

try:
    from PyQt6.QtWebEngineCore import QWebEngineScript
    from PyQt6.QtWebEngineWidgets import QWebEngineView

    WEBENGINE_AVAILABLE = True
except Exception:  # ImportError, or missing native libs
    QWebEngineScript = None  # type: ignore[assignment]
    QWebEngineView = None  # type: ignore[assignment]
    WEBENGINE_AVAILABLE = False


_POLL_INTERVAL_MS = 500
_READY_TIMEOUT_MS = 30_000

# Forces the embedded Lab page to always report itself as visible. Without this,
# QStackedWidget hiding the webview on a workspace switch flips
# `document.hidden` to true, which makes Lab's frontend back off pollers and
# (more importantly) ends up letting the server drop the external kernel from
# `MultiKernelManager._kernels`. Because jupyter_client never clears its
# `kernel_id_to_connection_file` mapping for removed externals, the kernel can
# only be rediscovered after a full Lab restart — see
# jupyter_client/multikernelmanager.py:list_kernel_ids.
_ALWAYS_VISIBLE_JS = """
(() => {
  try {
    Object.defineProperty(document, 'hidden',
      { configurable: true, get: () => false });
    Object.defineProperty(document, 'visibilityState',
      { configurable: true, get: () => 'visible' });
    Object.defineProperty(document, 'webkitHidden',
      { configurable: true, get: () => false });
    Object.defineProperty(document, 'webkitVisibilityState',
      { configurable: true, get: () => 'visible' });
    const blocked = new Set(['visibilitychange', 'webkitvisibilitychange',
                             'pagehide', 'freeze']);
    const origAdd = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function(type, listener, opts) {
      if (blocked.has(type)) return;
      return origAdd.call(this, type, listener, opts);
    };
  } catch (e) { /* best-effort shim */ }
})();
"""


class NotebookPanel(QWidget):
    """Dock body that renders an embedded Jupyter Lab once its server is up.

    `load(url)` does not block: it polls the server's TCP port on a QTimer and
    only calls `setUrl` once the port accepts connections (the server needs a
    few seconds to start; loading too early would show 'connection refused')."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if not WEBENGINE_AVAILABLE:
            raise RuntimeError("PyQt6-WebEngine is not installed")

        self._view = QWebEngineView(self)
        self._install_always_visible_shim()
        self._status = QLabel("", self)
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color: #888;")

        self._layout = QStackedLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._status)  # index 0
        self._layout.addWidget(self._view)  # index 1

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll)
        self._elapsed = 0
        self._pending_url: str | None = None

    def load(self, url: str | None) -> None:
        if not url:
            self._show_status("No URL to load.")
            return
        self._pending_url = url
        self._elapsed = 0
        self._show_status("Starting Jupyter Lab…")
        self._timer.start()

    def clear(self) -> None:
        self._timer.stop()
        self._pending_url = None
        self._view.setUrl(QUrl("about:blank"))
        self._show_status("")

    # ----- internals -----

    def _install_always_visible_shim(self) -> None:
        script = QWebEngineScript()
        script.setName("framewise-always-visible")
        script.setSourceCode(_ALWAYS_VISIBLE_JS)
        script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        script.setRunsOnSubFrames(True)
        script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        self._view.page().scripts().insert(script)

    def _show_status(self, text: str) -> None:
        self._status.setText(text)
        self._layout.setCurrentIndex(0)

    def _poll(self) -> None:
        url = self._pending_url
        if not url:
            self._timer.stop()
            return

        self._elapsed += _POLL_INTERVAL_MS
        if self._server_ready(url):
            self._timer.stop()
            self._view.setUrl(QUrl(url))
            self._layout.setCurrentIndex(1)
            return

        if self._elapsed >= _READY_TIMEOUT_MS:
            self._timer.stop()
            self._show_status(
                "Jupyter Lab did not become reachable in time.\n"
                "Try View → Stop Jupyter Lab, then Open in Jupyter Lab again."
            )

    @staticmethod
    def _server_ready(url: str) -> bool:
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False
