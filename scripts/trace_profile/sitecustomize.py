"""Spawn-safe per-process profiler for GuideLLM worker processes.

Python imports this module automatically when its directory is on
``PYTHONPATH``. Each spawned worker is a fresh interpreter, so every PID
dumps its own profile.

Activate with ``GUIDELLM_PROFILE=cprofile|pyinstrument|line``. Dumps go to
``GUIDELLM_PROFILE_DIR`` (default ``./profiles``).

cProfile is stdlib. pyinstrument and line_profiler are optional and are
skipped (with a message) if they are not installed.

## WRITTEN BY AI ##
"""

from __future__ import annotations

import atexit
import builtins
import os
import sys
import traceback

_MODE = os.environ.get("GUIDELLM_PROFILE", "").strip().lower()
_ENABLED = _MODE in {"cprofile", "pyinstrument", "line"}
_INSTALLED = False

# Functions to wrap for line_profiler. Imported lazily after each module loads
# so this file does not need to patch src/.
_LINE_TARGETS: dict[str, tuple[str, ...]] = {
    "guidellm.scheduler.worker": (
        "WorkerProcess._process_requests_loop",
        "WorkerProcess._get_next_ready_node",
        "WorkerProcess._process_next_graph_node",
        "WorkerProcess._prepare_node",
        "WorkerProcess._execute_node",
        "WorkerProcess._schedule_request",
        "WorkerProcess._send_update",
        "WorkerProcess._finalize_node",
    ),
    "guidellm.scheduler.worker_group": (
        "WorkerProcessGroup.requests_generator",
        "WorkerGroupState.update_state",
    ),
    "guidellm.data.deserializers.trace_weka": (
        "WEKATraceFormat.build_conversation_graph",
    ),
    "guidellm.data.deserializers.trace_otel": (
        "OTELTraceFormat.build_conversation_graph",
    ),
    "guidellm.backends.openai.http": ("OpenAIHTTPBackend.resolve",),
}


def _profile_dir() -> str:
    path = os.environ.get("GUIDELLM_PROFILE_DIR", "").strip()
    if not path:
        path = os.path.join(os.getcwd(), "profiles")
    os.makedirs(path, exist_ok=True)
    return path


def _dump_path(suffix: str) -> str:
    return os.path.join(_profile_dir(), f"pid-{os.getpid()}.{suffix}")


def _install_cprofile() -> None:
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    def _dump() -> None:
        profiler.disable()
        stats_path = _dump_path("prof")
        text_path = _dump_path("txt")
        profiler.dump_stats(stats_path)
        with open(text_path, "w", encoding="utf-8") as handle:
            stats = pstats.Stats(profiler, stream=handle)
            stats.sort_stats("cumulative")
            stats.print_stats(80)
            handle.write("\n\n--- sort by tottime (CPU-ish) ---\n\n")
            stats.sort_stats("tottime")
            stats.print_stats(80)
        print(f"[guidellm-profile] cProfile wrote {stats_path} and {text_path}")

    atexit.register(_dump)


def _install_pyinstrument() -> None:
    try:
        from pyinstrument import Profiler
    except ImportError:
        print(
            "[guidellm-profile] pyinstrument is not installed; "
            "pip install pyinstrument or use GUIDELLM_PROFILE=cprofile",
            file=sys.stderr,
        )
        return

    # async_mode captures time in asyncio workers; uvloop is still attributed.
    profiler = Profiler(async_mode="enabled")
    profiler.start()

    def _dump() -> None:
        profiler.stop()
        html_path = _dump_path("html")
        text_path = _dump_path("txt")
        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(profiler.output_html())
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(profiler.output_text(unicode=True, color=False, show_all=True))
        print(f"[guidellm-profile] pyinstrument wrote {html_path} and {text_path}")

    atexit.register(_dump)


def _resolve_attr(module: object, dotted: str) -> tuple[object, str, object] | None:
    """Return ``(owner, attr_name, function)`` for ``Class.method`` on ``module``."""
    owner: object = module
    parts = dotted.split(".")
    try:
        for part in parts[:-1]:
            owner = getattr(owner, part)
        name = parts[-1]
        func = getattr(owner, name)
    except AttributeError:
        return None
    return owner, name, func


def _install_line_profiler() -> None:
    try:
        from line_profiler import LineProfiler
    except ImportError:
        print(
            "[guidellm-profile] line_profiler is not installed; "
            "pip install line_profiler or use GUIDELLM_PROFILE=cprofile",
            file=sys.stderr,
        )
        return

    profiler = LineProfiler()
    wrapped_modules: set[str] = set()
    orig_import = builtins.__import__

    def _wrap_module(name: str) -> None:
        if name in wrapped_modules or name not in _LINE_TARGETS:
            return
        module = sys.modules.get(name)
        if module is None:
            return
        wrapped_modules.add(name)
        for dotted in _LINE_TARGETS[name]:
            resolved = _resolve_attr(module, dotted)
            if resolved is None:
                print(
                    f"[guidellm-profile] skip missing {name}.{dotted}",
                    file=sys.stderr,
                )
                continue
            owner, attr_name, func = resolved
            try:
                profiler.add_function(func)
                setattr(owner, attr_name, profiler.wrap_function(func))
            except Exception:  # noqa: BLE001
                print(
                    f"[guidellm-profile] failed to wrap {name}.{dotted}:\n"
                    f"{traceback.format_exc()}",
                    file=sys.stderr,
                )

    def _hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = orig_import(name, globals, locals, fromlist, level)
        _wrap_module(name)
        if fromlist:
            package = sys.modules.get(name)
            if package is not None and hasattr(package, "__path__"):
                for item in fromlist:
                    if item == "*":
                        continue
                    _wrap_module(f"{name}.{item}")
        return module

    builtins.__import__ = _hooked_import
    for already in list(sys.modules):
        _wrap_module(already)

    def _dump() -> None:
        text_path = _dump_path("lprof.txt")
        with open(text_path, "w", encoding="utf-8") as handle:
            profiler.print_stats(stream=handle, output_unit=1e-6)
        print(f"[guidellm-profile] line_profiler wrote {text_path}")

    atexit.register(_dump)


def _install() -> None:
    global _INSTALLED
    if not _ENABLED or _INSTALLED:
        return
    _INSTALLED = True
    print(
        f"[guidellm-profile] pid={os.getpid()} mode={_MODE} "
        f"dir={_profile_dir()}",
        file=sys.stderr,
    )
    if _MODE == "cprofile":
        _install_cprofile()
    elif _MODE == "pyinstrument":
        _install_pyinstrument()
    elif _MODE == "line":
        _install_line_profiler()


_install()
