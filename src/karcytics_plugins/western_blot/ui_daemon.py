"""Western Blot UI Daemon — hosts the module's own window in its own process.

Run by `karcytics_sdk.plugin.PluginUIDaemon` from this plugin's own `.venv`
interpreter (never imported into the Hub's process). Owns its own
`QApplication` and its own copies of numpy/scipy/matplotlib/PySide6, so
switching to or from this module never touches the Hub's `sys.modules` —
the whole class of shadow-copy/purge collisions this exists to avoid (the
Hub's `PluginEnvironmentInjector.enforce_priority()` purging a shared
`cryptography`/`_cffi_backend` C extension mid-process is what crashed the
in-process load with a bus error).

Everything protocol-related (frame transport, the ready handshake, request
dispatch, noticing a native window close) lives in the SDK's
`karcytics_sdk.plugin.run_ui_daemon` and is identical for every isolated
plugin; this file only does what's genuinely plugin-specific: sys.path
setup, env vars that must be set before numpy imports, and building this
plugin's `PluginContext` from the SDK's `runtime_services` singletons —
the same `task_scheduler`/`event_bus` instances this plugin's own widgets
import directly (see `karcytics_sdk.plugin.runtime_services`), so there is
exactly one shared scheduler/event bus per process, not two.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Run directly as `python ui_daemon.py` by PluginUIDaemon rather than imported
# as part of the `karcytics_plugins` package — nothing else puts this
# plugin's own src/ on sys.path for a freestanding subprocess the way
# PluginEnvironmentInjector.inject_path() does for the in-process legacy
# load path, so it has to do that for itself before it can import itself.
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# CRITICAL: must be set before any numpy/scipy import.
import os  # noqa: E402

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# CRITICAL: must happen before run_ui_daemon() (below, via main()) is ever
# called — that's what starts the SDK's background stdin-reader thread.
# Importing numpy while another thread is blocked on a concurrent
# sys.stdin.buffer.read() call deadlocks on Windows: numpy's Windows-specific
# console/codepage setup touches sys.stdin at import time and blocks trying
# to acquire the same lock the reader thread holds mid-read (matches
# numpy/numpy#24290). Importing numpy/pandas here, before that thread
# exists, means the later imports inside the analysis/UI modules are just
# sys.modules cache hits with nothing left to contend over.
import numpy  # noqa: E402, F401
import pandas  # noqa: E402, F401


def _build_plugin_context() -> Any:
    from karcytics_sdk.plugin.context import PluginContext
    from karcytics_sdk.plugin.manifest import PluginManifest
    from karcytics_sdk.plugin.runtime_services import event_bus, task_scheduler

    manifest = PluginManifest(
        name="western_blot",
        entry_point="karcytics_plugins.western_blot:initialize",
        sdk_version="2.0",
        requires=["task_scheduler", "logger", "event_bus"],
    )
    services = {
        "task_scheduler": task_scheduler,
        "logger": __import__("logging").getLogger("plugin.western_blot"),
        "event_bus": event_bus,
    }
    return PluginContext(services=services, manifest=manifest)


def main() -> None:
    from karcytics_sdk.plugin import run_ui_daemon
    from karcytics_sdk.plugin.ui_daemon_runtime import send_event

    def _build_panel() -> Any:
        from karcytics_plugins.western_blot import initialize

        context = _build_plugin_context()
        plugin_module = initialize(context)
        panel_class = plugin_module.get_panel_class()
        panel = panel_class()

        if hasattr(panel, "state_changed"):
            panel.state_changed.connect(lambda: send_event("state_changed", {}))
        if hasattr(panel, "status_message"):
            panel.status_message.connect(lambda msg: send_event("status_message", msg))

        return panel

    # No local "inject_workflow" handler here — ui_daemon_runtime.run() already
    # registers one that's aware of the Ready Gate protocol (stages the
    # payload and only calls begin_async_init() once the panel exists). A
    # local override with the same method name would silently replace it,
    # since RequestDispatcher.register() replaces on name collision.
    run_ui_daemon(
        _build_panel,
        window_title="Western Blot Densitometry",
        window_size=(1400, 900),
        plugin_id="western_blot",
    )


if __name__ == "__main__":
    main()
