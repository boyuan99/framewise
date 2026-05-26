"""Jupyter server config loaded by framewise's embedded Lab subprocess.

`JupyterLabLauncher` passes `--config=<path-to-this-file>` so this runs in the
Lab subprocess (NOT in the framewise process).

Why: `jupyter_client.MultiKernelManager.list_kernel_ids` has a known quirk —
when an external kernel is removed from `_kernels` (which happens whenever
Lab's UI shuts it down: "Change Kernel", closing a notebook, restart, etc.),
the `kernel_id_to_connection_file` mapping is left behind. The next scan of
`external_connection_dir` then short-circuits at the "if connection_file in
self.kernel_id_to_connection_file.values(): continue" check, so the kernel
never gets re-added. The only recovery is to restart the Lab server.

Symptom in framewise: pick framewise's `python3` kernel for a notebook, then
change that notebook's kernel to something else (or close the notebook), and
framewise's kernel disappears from the "Connect to Existing python Kernel"
dropdown until you stop+restart Jupyter Lab.

Fix: on every `list_kernel_ids` call, prune `kernel_id_to_connection_file`
entries whose kernel_id is no longer in `_kernels`. The original scan then
re-discovers the connection file and re-adds it under a fresh kernel_id.
"""


def _patch_multi_kernel_manager() -> None:
    import threading

    from jupyter_client.multikernelmanager import MultiKernelManager

    if getattr(MultiKernelManager, "_framewise_patched", False):
        return

    _orig = MultiKernelManager.list_kernel_ids
    # The original `list_kernel_ids` re-enters itself indirectly: the factory
    # call at multikernelmanager.py:167 builds a KernelManager whose traitlets
    # __init__ checks `if self.parent:`, which falls back to MultiKernelManager
    # __len__ → list_kernel_ids. At that moment the kernel_id is already in
    # `kernel_id_to_connection_file` but NOT yet in `_kernels`, so a naive
    # pre-clean would treat it as stale and pop it, causing infinite recursion.
    # The thread-local flag skips pre-clean on the re-entrant call.
    _state = threading.local()

    def list_kernel_ids(self):  # type: ignore[no-untyped-def]
        if getattr(_state, "active", False):
            return _orig(self)
        _state.active = True
        try:
            if self.external_connection_dir is not None:
                stale = [
                    kid
                    for kid in list(self.kernel_id_to_connection_file)
                    if kid not in self._kernels
                ]
                for kid in stale:
                    self.kernel_id_to_connection_file.pop(kid, None)
            return _orig(self)
        finally:
            _state.active = False

    MultiKernelManager.list_kernel_ids = list_kernel_ids
    MultiKernelManager._framewise_patched = True


_patch_multi_kernel_manager()
