"""Embedded Jupyter console for framewise.

Hosts a `RichJupyterWidget` docked in the main window and manages the kernel it
talks to. Two kernel backends share one `KernelHost` interface:

* `SameProcessKernelHost` — an `IPKernelApp` running on a daemon thread inside
  framewise's own process. It has real ZMQ sockets (so an external Jupyter Lab
  can attach via "Use Existing Kernel") AND shares live Python objects with the
  GUI (`window`, `mc`, `panel("cam0").array`, …). Because the kernel runs on a
  background thread, *reads* of live objects are safe from any client, but
  *mutating* Qt state must go through the injected `gui(fn)` helper, which
  marshals `fn` onto the GUI thread.

* `OutOfProcessKernelHost` — a subprocess kernel via `qtconsole`'s
  `QtKernelManager`. Heavy compute never freezes the GUI, and external Jupyter
  Lab can attach, but framewise's live objects don't cross the process boundary
  (only picklable values are injected).

Both write their connection file into the Jupyter runtime dir so Jupyter Lab's
kernel picker discovers them. `shutdown()` deletes that file.
"""

from __future__ import annotations

import os

# Must precede any qtconsole/qtpy import so the Qt binding resolves to PyQt6.
os.environ.setdefault("QT_API", "pyqt6")

import threading
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget
from qtconsole.rich_jupyter_widget import RichJupyterWidget


def _external_dir() -> str:
    """Dedicated directory for framewise's kernel connection files. A Jupyter
    server launched with `--ServerApp.external_connection_dir=<this>` will list
    only framewise kernels here (no unrelated/stale connection files)."""
    from jupyter_core.paths import jupyter_runtime_dir

    d = os.path.join(jupyter_runtime_dir(), "framewise")
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
#  GUI-thread marshaling (for same-process kernel running on a daemon thread)
# --------------------------------------------------------------------------- #


class _Result:
    __slots__ = ("value", "error", "event")

    def __init__(self) -> None:
        self.value = None
        self.error: Optional[BaseException] = None
        self.event = threading.Event()


class _GuiInvoker(QObject):
    """Runs a callable on the GUI thread and returns its result, even when
    called from the kernel's background thread."""

    _request = pyqtSignal(object, object)

    def __init__(self) -> None:
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            self.moveToThread(app.thread())
        self._request.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def _run(self, fn: Callable[[], Any], result: _Result) -> None:
        try:
            result.value = fn()
        except BaseException as exc:  # surface user errors back to the caller
            result.error = exc
        finally:
            result.event.set()

    def invoke(self, fn: Callable[[], Any]) -> Any:
        if QThread.currentThread() is self.thread():
            return fn()
        result = _Result()
        self._request.emit(fn, result)
        result.event.wait()
        if result.error is not None:
            raise result.error
        return result.value


# --------------------------------------------------------------------------- #
#  Kernel host interface + implementations
# --------------------------------------------------------------------------- #


@runtime_checkable
class KernelHost(Protocol):
    is_same_process: bool
    kernel_manager: Any  # None for same-process (no managed subprocess)
    connection_file: str

    def start(self) -> None: ...
    def make_client(self) -> Any: ...
    def push(self, ns: dict) -> None: ...
    def expose_connection_file(self) -> str: ...
    def unexpose_connection_file(self) -> None: ...
    def shutdown(self) -> None: ...


class SameProcessKernelHost:
    is_same_process = True

    def __init__(self) -> None:
        self.kernel_manager = None
        self.connection_file = os.path.join(
            _external_dir(), f"kernel-framewise-{os.getpid()}.json"
        )
        self._app = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self.gui_invoker = _GuiInvoker()

    def start(self) -> None:
        connection_file = self.connection_file
        ready = self._ready

        def _run() -> None:
            import asyncio

            # The kernel's asyncio loop lives on this daemon thread; the Qt loop
            # keeps owning the main thread untouched.
            asyncio.set_event_loop(asyncio.new_event_loop())

            from ipykernel.kernelapp import IPKernelApp

            class _EmbeddedKernelApp(IPKernelApp):
                # signal.signal() only works on the main thread; this kernel
                # runs on a daemon thread, so installing SIGINT handlers would
                # raise. The host process owns signal handling anyway.
                def init_signal(self) -> None:  # noqa: D401
                    pass

            app = _EmbeddedKernelApp.instance()
            # Leave the OS-level stdout/stderr fds alone so framewise's own
            # terminal output isn't swallowed by the kernel. Cell output is
            # still captured via the kernel's sys.stdout OutStream.
            try:
                app.capture_fd_output = False
            except Exception:
                pass
            app.initialize(
                ["python", f"--IPKernelApp.connection_file={connection_file}"]
            )
            # IPKernelApp writes an empty kernel_name; set it to a real spec so
            # JupyterLab displays "Python 3 (ipykernel)" instead of "Unknown
            # Kernel" when listing this external kernel.
            try:
                import json

                with open(connection_file) as fh:
                    info = json.load(fh)
                info["kernel_name"] = "python3"
                with open(connection_file, "w") as fh:
                    json.dump(info, fh)
            except Exception:
                pass
            self._app = app
            ready.set()
            app.start()  # blocks this thread, running the kernel loop

        self._thread = threading.Thread(
            target=_run, name="framewise-ipykernel", daemon=True
        )
        self._thread.start()
        # Wait for the kernel app to exist before any push()/client connect.
        self._ready.wait(timeout=30)

    def make_client(self):
        from qtconsole.client import QtKernelClient

        client = QtKernelClient(connection_file=self.connection_file)
        client.load_connection_file()
        client.start_channels()
        return client

    def push(self, ns: dict) -> None:
        if self._app is None:
            raise RuntimeError("kernel not started")
        # shell.push mutates the user namespace; safe to call from any thread
        # because it only touches Python dicts (no Qt objects).
        self._app.shell.push(ns)

    def expose_connection_file(self) -> str:
        # Already written into the runtime dir by initialize(); nothing to copy.
        return self.connection_file

    def unexpose_connection_file(self) -> None:
        try:
            os.remove(self.connection_file)
        except OSError:
            pass

    def shutdown(self) -> None:
        self.unexpose_connection_file()
        app = self._app
        if app is not None:
            # Stop the kernel's IO loop on its own thread so app.start() returns
            # and the daemon thread exits cleanly (otherwise the interpreter can
            # die with "could not acquire lock ... daemon threads" at exit).
            io_loop = getattr(app, "io_loop", None)
            if io_loop is not None:
                try:
                    io_loop.add_callback(io_loop.stop)
                except Exception:
                    pass
            try:
                app.kernel.stop()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


class OutOfProcessKernelHost:
    is_same_process = False

    def __init__(self) -> None:
        self.kernel_manager = None
        self._push_client = None
        self.connection_file = os.path.join(
            _external_dir(), f"kernel-framewise-oop-{os.getpid()}.json"
        )

    def start(self) -> None:
        # Plain KernelManager (not QtKernelManager): its lifecycle/heartbeat does
        # not depend on a running Qt loop, so startup is robust. The widget still
        # talks to it through a QtKernelClient built from the connection file.
        from jupyter_client import KernelManager

        km = KernelManager()
        # Land the connection file in the runtime dir under our stable name so
        # external Jupyter Lab can find it.
        km.connection_file = self.connection_file
        km.start_kernel()
        self.kernel_manager = km

        # A dedicated blocking client for namespace injection — independent of
        # the Qt-driven widget client, so push() works without the Qt loop.
        from jupyter_client import BlockingKernelClient

        pc = BlockingKernelClient(connection_file=self.connection_file)
        pc.load_connection_file()
        pc.start_channels()
        # NB: deliberately no wait_for_ready() — its heartbeat probe can falsely
        # report "kernel died" here even though the kernel is alive. ZMQ buffers
        # the push until the shell channel connects, so injection is safe.
        self._push_client = pc

    def make_client(self):
        from qtconsole.client import QtKernelClient

        client = QtKernelClient(connection_file=self.connection_file)
        client.load_connection_file()
        client.start_channels()
        return client

    def push(self, ns: dict) -> None:
        import base64
        import inspect
        import pickle

        client = self._push_client
        imports: list[str] = []
        data: dict[str, Any] = {}
        for key, val in ns.items():
            if inspect.ismodule(val):
                imports.append(f"import {val.__name__} as {key}")
                continue
            try:
                pickle.dumps(val)
            except Exception:
                continue  # not transferable across the process boundary
            data[key] = val

        code = "\n".join(imports)
        if data:
            blob = base64.b64encode(pickle.dumps(data)).decode("ascii")
            code += (
                "\nimport pickle as _p, base64 as _b\n"
                f"globals().update(_p.loads(_b.b64decode('{blob}')))\n"
                "del _p, _b"
            )
        if code.strip():
            client.execute(code, silent=True)

    def expose_connection_file(self) -> str:
        return self.connection_file

    def unexpose_connection_file(self) -> None:
        try:
            os.remove(self.connection_file)
        except OSError:
            pass

    def shutdown(self) -> None:
        self.unexpose_connection_file()
        if self._push_client is not None:
            try:
                self._push_client.stop_channels()
            except Exception:
                pass
            self._push_client = None
        km = self.kernel_manager
        if km is None:
            return
        try:
            km.shutdown_kernel(now=True)
        except Exception:
            pass


_MODES = {
    "same_process": SameProcessKernelHost,
    "out_of_process": OutOfProcessKernelHost,
}


# --------------------------------------------------------------------------- #
#  The dock widget
# --------------------------------------------------------------------------- #


class ConsolePanel(QWidget):
    """Dockable Jupyter console. Owns one KernelHost and a RichJupyterWidget
    client. `namespace_factory(host)` returns the dict to push — MainWindow
    supplies it so the panel stays decoupled from the provider registry."""

    def __init__(
        self,
        namespace_factory: Callable[[KernelHost], dict],
        parent: QWidget | None = None,
        mode: str = "same_process",
    ) -> None:
        super().__init__(parent)
        self._namespace_factory = namespace_factory
        self._host: Optional[KernelHost] = None
        self._client = None

        self._widget = RichJupyterWidget()
        self._widget.set_default_style(colors="linux")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._widget)

        self._start_host(mode)

    # ----- public API -----

    @property
    def mode(self) -> str:
        return "same_process" if self._host.is_same_process else "out_of_process"

    def switch_to(self, mode: str) -> None:
        if self._host is not None and self.mode == mode:
            return
        self._teardown_client()
        if self._host is not None:
            self._host.shutdown()
        self._start_host(mode)

    def refresh_namespace(self) -> None:
        if self._host is None:
            return
        ns = dict(self._namespace_factory(self._host))
        if self._host.is_same_process:
            # Marshal GUI-mutating calls onto the main thread.
            ns.setdefault("gui", self._host.gui_invoker.invoke)
        self._host.push(ns)

    @property
    def external_connection_dir(self) -> str:
        return os.path.dirname(self._host.connection_file)

    def connection_info(self) -> dict:
        kid = os.path.splitext(os.path.basename(self._host.connection_file))[0]
        return {
            "mode": self.mode,
            "connection_file": self._host.connection_file,
            "external_connection_dir": self.external_connection_dir,
            "kernel_id": kid,
        }

    def shutdown(self) -> None:
        self._teardown_client()
        if self._host is not None:
            self._host.shutdown()
            self._host = None

    def external_clients_possible(self) -> bool:
        """Whether an external client could be attached (kernel exposes ZMQ)."""
        return self._host is not None

    # ----- internals -----

    def _start_host(self, mode: str) -> None:
        host = _MODES[mode]()
        host.start()
        host.expose_connection_file()
        self._host = host

        self._client = host.make_client()
        self._widget.kernel_client = self._client
        self._widget.kernel_manager = host.kernel_manager

        self._widget.banner = self._make_banner()
        self.refresh_namespace()

    def _teardown_client(self) -> None:
        if self._client is not None:
            try:
                self._client.stop_channels()
            except Exception:
                pass
            self._client = None

    def _make_banner(self) -> str:
        info = self.connection_info()
        mode_label = (
            "same-process (live objects shared with the GUI)"
            if self._host.is_same_process
            else "out-of-process (separate process; picklable values only)"
        )
        lines = [
            "Framewise IPython console.",
            f"Kernel mode: {mode_label}.",
            "Pre-populated: window, mc (master clock), pm (panel manager), sync.",
            "Helpers: current_time(), current_frame(name), panel(name), panels().",
            "Aliases: np, pg.",
        ]
        if self._host.is_same_process:
            lines.append(
                "Reads are safe from any client; wrap GUI-mutating calls in "
                "gui(...), e.g. gui(lambda: mc.set_time(2.0))."
            )
        else:
            lines.append(
                "Long computations here will NOT freeze the GUI."
            )
        lines.append(
            "Attach Jupyter Lab: use the View → Open in Jupyter Lab menu, then "
            f"New Console/Notebook → Use Existing Kernel → {info['kernel_id']}."
        )
        return "\n".join(lines) + "\n\n"


class JupyterLabLauncher:
    """Starts/stops a `jupyter lab` subprocess configured to discover
    framewise's kernel via the external-connection-dir mechanism.

    Two launch modes:
    * embedded — headless server on a known port+token; the URL is returned so
      a QWebEngineView can load it inside framewise.
    * external — lets jupyter open the system browser (the original behavior).

    Stopping Lab does NOT kill framewise's kernel: the server adopts it with
    owns_kernel=False, so terminating the server only closes the host."""

    def __init__(self, external_connection_dir: str) -> None:
        self._dir = external_connection_dir
        self._proc = None
        self._url: Optional[str] = None

    @property
    def url(self) -> Optional[str]:
        return self._url

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @staticmethod
    def _free_port() -> int:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def start(self, embedded: bool = True) -> Optional[str]:
        """Launch Jupyter Lab. In embedded mode returns the URL (with token) to
        load in a QWebEngineView; in external mode opens the system browser and
        returns None. If already running, returns the current URL unchanged."""
        import secrets
        import subprocess
        import sys

        if self.is_running():
            return self._url

        common = [
            "--ServerApp.allow_external_kernels=True",
            f"--ServerApp.external_connection_dir={self._dir}",
        ]
        if embedded:
            port = self._free_port()
            token = secrets.token_hex(16)
            args = common + [
                "--no-browser",
                f"--ServerApp.port={port}",
                "--ServerApp.port_retries=0",
                f"--IdentityProvider.token={token}",
            ]
            self._url = f"http://127.0.0.1:{port}/lab?token={token}"
        else:
            args = common
            self._url = None

        self._proc = subprocess.Popen(
            [sys.executable, "-m", "jupyterlab", *args]
        )
        return self._url

    def stop(self) -> None:
        import sys

        if not self.is_running():
            self._proc = None
            self._url = None
            return
        proc = self._proc
        try:
            if sys.platform == "win32":
                # Headless server has no window to close; kill the whole tree so
                # nothing is left running invisibly.
                import subprocess

                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                )
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
        except Exception:
            pass
        finally:
            self._proc = None
            self._url = None
