# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Backend-agnostic core for ROCTX scope tracking.

Maintains the per-thread marker and context stacks and the Python-tier
range push and pop callbacks. Backends interact only through _push_scope,
_pop_scope, and the helpers defined here.
"""

import importlib
import inspect
import os
import sys
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

# Backends recognized by install_global_wraps and the "api" alias.
KNOWN_BACKENDS: tuple[str, ...] = ("torch", "triton")


def _missing_range_push(_label: str) -> None:
    raise RuntimeError(
        "inject_roctx._core: Python tier rangePush is not configured.",
    )


def _missing_range_pop() -> None:
    raise RuntimeError(
        "inject_roctx._core: Python tier rangePop is not configured.",
    )


_range_push: Callable[[str], None] = _missing_range_push
_range_pop: Callable[[], None] = _missing_range_pop
_framework_roots: list[str] = []

# ROCm site directories probed for the roctx module.
_roctx_candidate_paths: list[str] = []

# Frames under the package directory are skipped during caller-location
# resolution.
_PACKAGE_ROOT: str = str(Path(__file__).resolve().parent) + os.sep


def set_python_tier_io(
    push: Callable[[str], None],
    pop: Callable[[], None],
) -> None:
    global _range_push, _range_pop
    _range_push = push
    _range_pop = pop


def python_tier_configured() -> bool:
    """True once the Python-tier push/pop callbacks have been wired."""
    return _range_push is not _missing_range_push


def get_python_tier_io() -> tuple[Callable[[str], None], Callable[[], None]]:
    """Return the currently wired (rangePush, rangePop) callbacks."""
    return _range_push, _range_pop


def roctx_candidate_paths() -> list[str]:
    """Return the directories probed for the roctx module."""
    return list(_roctx_candidate_paths)


def ensure_python_tier() -> bool:
    """Wire the Python-tier rangePush/rangePop callbacks from the ROCm roctx
    module. Idempotent; returns True if the Python tier is configured.
    """
    if python_tier_configured():
        return True
    global _roctx_candidate_paths
    rocm_root = os.environ.get("ROCM_PATH", "/opt/rocm")
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _roctx_candidate_paths = [
        f"{rocm_root}/lib/{py}/site-packages",
        f"{rocm_root}/libexec/rocprofiler-sdk/python",
    ]
    for candidate in _roctx_candidate_paths:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    try:
        roctx_mod = importlib.import_module("roctx")
    except ImportError:
        return False
    set_python_tier_io(roctx_mod.rangePush, roctx_mod.rangePop)
    return True


def add_framework_root(path: str) -> None:
    # Store with a trailing separator so prefix matching is exact.
    if not path:
        return
    root = path if path.endswith(os.sep) else path + os.sep
    if root not in _framework_roots:
        _framework_roots.append(root)


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
        in_package = fn_path.startswith(_PACKAGE_ROOT) or fn_path == __file__
        in_framework = any(fn_path.startswith(root) for root in _framework_roots)
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

    _range_push(compose_marker(marker, context, backend))

    snapshot_len = len(marker_stack)
    try:
        marker_stack.append(marker)
        context_stack.append(context)
    except Exception:
        del marker_stack[snapshot_len:]
        del context_stack[snapshot_len:]
        try:
            _range_pop()
        except Exception:
            pass
        raise


def _pop_scope() -> None:
    marker_stack = get_marker_stack()
    context_stack = get_context_stack()

    # Unmatched pop: no-op.
    if not marker_stack:
        return

    try:
        _range_pop()
    finally:
        if marker_stack:
            marker_stack.pop()
        if context_stack:
            context_stack.pop()


# Structural wrappers for entry points the ATen dispatcher does not record.


def roctx_wrapper(
    func: Callable[..., Any],
    name: Optional[str] = None,
    backend: str = "",
    push: Callable[..., None] = _push_scope,
    pop: Callable[[], None] = _pop_scope,
) -> Callable[..., Any]:
    """Wrap func so each call emits a ROCTX range. Idempotent.

    When set, backend attributes the scope to that backend. push and pop
    default to the Python tier; a backend may pass its own to route the scope
    through another tier.
    """
    if getattr(func, "_roctx_wrapped", False):
        return func
    func_name = name or func.__name__
    call_counter = {"count": 0}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> object:
        call_counter["count"] += 1
        location = resolve_user_caller_location()
        push(func_name, f"#{call_counter['count']}@{location}", backend=backend)
        try:
            return func(*args, **kwargs)
        finally:
            pop()

    wrapper._roctx_wrapped = True
    return wrapper


def _marker_only_init_wrapper(
    name: str,
    backend: str = "",
    push: Callable[..., None] = _push_scope,
    pop: Callable[[], None] = _pop_scope,
) -> Callable[..., Any]:
    """Build an __init__ that emits a ROCTX range, then calls object.__init__.

    Used for classes whose construction occurs in __new__ (e.g. cuda.Event,
    cuda.Stream). push and pop default to the Python tier.
    """
    call_counter = {"count": 0}

    def marker_only_init(self: object, *args: Any, **kwargs: Any) -> None:
        call_counter["count"] += 1
        location = resolve_user_caller_location()
        push(name, f"#{call_counter['count']}@{location}", backend=backend)
        try:
            return object.__init__(self)
        finally:
            pop()

    marker_only_init._roctx_wrapped = True
    return marker_only_init


def _walk_subclasses(cls: type, fn: Callable[[type], None]) -> None:
    """Apply `fn` to every (transitive) subclass of `cls`."""
    for sub in cls.__subclasses__():
        fn(sub)
        _walk_subclasses(sub, fn)
