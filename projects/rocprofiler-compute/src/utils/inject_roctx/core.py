# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Backend-agnostic core for ROCTX scope tracking.

Maintains the per-thread marker and context stacks and the Python-tier
rangePush/rangePop callbacks shared by all backends.
"""

import importlib.util
import inspect
import os
import sys
import threading
from collections.abc import Iterable
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Callable, Union

from .constants import API_ALIAS, KNOWN_BACKENDS


def _missing_range_push(_label: str) -> None:
    raise RuntimeError(
        "inject_roctx.core: Python tier rangePush is not configured.",
    )


def _missing_range_pop() -> None:
    raise RuntimeError(
        "inject_roctx.core: Python tier rangePop is not configured.",
    )


class _CoreState:
    """Mutable core state: the Python-tier rangePush/rangePop callbacks, the
    registered framework roots, and the ROCm directories probed for the roctx
    module.
    """

    def __init__(self) -> None:
        self.range_push: Callable[[str], None] = _missing_range_push
        self.range_pop: Callable[[], None] = _missing_range_pop
        self.framework_roots: list[str] = []
        self.roctx_candidate_paths: list[str] = []


_STATE = _CoreState()

# Frames under the package directory are skipped during caller-location
# resolution.
_PACKAGE_ROOT: str = str(Path(__file__).resolve().parent) + os.sep


def set_python_tier_io(
    push: Callable[[str], None],
    pop: Callable[[], None],
) -> None:
    _STATE.range_push = push
    _STATE.range_pop = pop


def python_tier_configured() -> bool:
    """True once the Python-tier push/pop callbacks have been wired."""
    return _STATE.range_push is not _missing_range_push


def get_python_tier_io() -> tuple[Callable[[str], None], Callable[[], None]]:
    """Return the currently wired (rangePush, rangePop) callbacks."""
    return _STATE.range_push, _STATE.range_pop


def roctx_candidate_paths() -> list[str]:
    """Return the directories probed for the roctx module."""
    return list(_STATE.roctx_candidate_paths)


def ensure_python_tier() -> bool:
    """Wire the Python-tier rangePush/rangePop callbacks from the ROCm roctx
    module. Idempotent; returns True if the Python tier is configured.
    """
    if python_tier_configured():
        return True
    rocm_root = os.environ.get("ROCM_PATH", "/opt/rocm")
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _STATE.roctx_candidate_paths = [
        f"{rocm_root}/lib/{py}/site-packages",
        f"{rocm_root}/libexec/rocprofiler-sdk/python",
    ]
    spec = PathFinder.find_spec("roctx", _STATE.roctx_candidate_paths)
    if spec is None or spec.loader is None:
        return False
    roctx_mod = importlib.util.module_from_spec(spec)
    # Register before exec so relative submodule imports resolve.
    sys.modules["roctx"] = roctx_mod
    try:
        spec.loader.exec_module(roctx_mod)
    except Exception:
        sys.modules.pop("roctx", None)
        return False
    set_python_tier_io(roctx_mod.rangePush, roctx_mod.rangePop)
    return True


def add_framework_root(path: str) -> None:
    # Store with a trailing separator so prefix matching is exact.
    if not path:
        return
    root = path if path.endswith(os.sep) else path + os.sep
    if root not in _STATE.framework_roots:
        _STATE.framework_roots.append(root)


# Per-thread stacks shared by all backends so nested scopes compose correctly.
_thread_local = threading.local()


def get_marker_stack() -> list[str]:
    if not hasattr(_thread_local, "marker_stack"):
        _thread_local.marker_stack = []
    return _thread_local.marker_stack


def get_context_stack() -> list[str]:
    if not hasattr(_thread_local, "context_stack"):
        _thread_local.context_stack = []
    return _thread_local.context_stack


def resolve_user_caller_location() -> str:
    """'file:line' for the nearest user frame, or 'python.dispatch:0'."""
    frame = inspect.currentframe()
    while frame is not None:
        fn_path = frame.f_code.co_filename
        in_package = fn_path.startswith(_PACKAGE_ROOT)
        in_framework = any(fn_path.startswith(root) for root in _STATE.framework_roots)
        if not in_package and not in_framework:
            return f"{Path(fn_path).name}:{frame.f_lineno}"
        frame = frame.f_back
    return "python.dispatch:0"


# Wire format: "<op_path>:#N@file:line/...[|<backend>]". The optional
# "|<backend>" suffix attributes the scope to its backend.


def compose_marker(marker: str, context: str, backend: str = "") -> str:
    """Return the wire-format string for a scope nested under the current
    marker and context stacks.
    """
    marker_stack = get_marker_stack()
    context_stack = get_context_stack()
    full = "/".join([*marker_stack, marker]) + ":" + "/".join([*context_stack, context])
    if backend:
        full = f"{full}|{backend}"
    return full


def _push_scope(marker: str, context: str, backend: str = "") -> None:
    marker_stack = get_marker_stack()
    context_stack = get_context_stack()

    _STATE.range_push(compose_marker(marker, context, backend))

    marker_stack.append(marker)
    context_stack.append(context)


def _pop_scope() -> None:
    marker_stack = get_marker_stack()
    context_stack = get_context_stack()

    # Unmatched pop: no-op.
    if not marker_stack:
        return

    try:
        _STATE.range_pop()
    finally:
        if marker_stack:
            marker_stack.pop()
        if context_stack:
            context_stack.pop()


def install_global_wraps(backends: Union[str, Iterable[str]] = "") -> None:
    """Install ROCTX instrumentation for each backend in backends.

    "api" expands to every known backend. Empty input is a no-op.
    """
    from .registry import install_many

    if isinstance(backends, str):
        names = [n.strip() for n in backends.split(",") if n.strip()]
    else:
        names = [str(n).strip() for n in backends if str(n).strip()]

    expanded: list[str] = []
    for n in names:
        if n == API_ALIAS:
            expanded.extend(KNOWN_BACKENDS)
        else:
            expanded.append(n)

    if not expanded:
        return
    install_many(expanded)
