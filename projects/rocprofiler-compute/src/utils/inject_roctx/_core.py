# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Backend-agnostic core for ROCTX scope tracking.

Maintains the per-thread marker, context, and tier stacks, the Python-tier
range push and pop callbacks, and an optional native-tier hook for the C++
RecordFunction backend. Backends interact only through _push_scope,
_pop_scope, and the helpers defined here.
"""

import inspect
import os
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Protocol


class NativeTierHook(Protocol):
    def active(self) -> bool: ...
    def push(self, marker: str, context: str, backend: str) -> bool: ...
    def pop(self) -> None: ...


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
_native_tier_hook: Optional[NativeTierHook] = None
_framework_roots: list[str] = []

# Frames under the package directory are skipped during caller-location
# resolution.
_PACKAGE_ROOT: str = str(Path(__file__).resolve().parent) + "/"


def set_python_tier_io(
    push: Callable[[str], None],
    pop: Callable[[], None],
) -> None:
    global _range_push, _range_pop
    _range_push = push
    _range_pop = pop


def set_native_tier_hook(hook: Optional[NativeTierHook]) -> None:
    global _native_tier_hook
    _native_tier_hook = hook


def get_native_tier_hook() -> Optional[NativeTierHook]:
    return _native_tier_hook


def add_framework_root(path: str) -> None:
    # Store with a trailing separator so prefix matching is exact.
    if not path:
        return
    root = path if path.endswith(os.sep) else path + os.sep
    if root not in _framework_roots:
        _framework_roots.append(root)


# Per-thread stacks. Source of truth on Python tier; mirror on C++ tier.
_thread_local = threading.local()


def get_marker_stack() -> list[str]:
    if not hasattr(_thread_local, "marker_stack"):
        _thread_local.marker_stack = []
    return _thread_local.marker_stack


def get_context_stack() -> list[str]:
    if not hasattr(_thread_local, "context_stack"):
        _thread_local.context_stack = []
    return _thread_local.context_stack


def get_tier_stack() -> list[bool]:
    # Each frame records the tier that handled its push: True for the C++
    # RecordFunction tier, False for the Python tier. _pop_scope uses this
    # to route the matching pop.
    if not hasattr(_thread_local, "tier_stack"):
        _thread_local.tier_stack = []
    return _thread_local.tier_stack


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


def _push_scope(marker: str, context: str, backend: str = "") -> None:
    marker_stack = get_marker_stack()
    context_stack = get_context_stack()
    tier_stack = get_tier_stack()

    used_native = False
    hook = _native_tier_hook
    if hook is not None and hook.active():
        try:
            used_native = bool(hook.push(marker, context, backend))
        except Exception:
            used_native = False

    if not used_native:
        # Compose the marker string for this scope.
        full = (
            "/".join([*marker_stack, marker])
            + ":"
            + "/".join([*context_stack, context])
        )
        if backend:
            full = f"{full}|{backend}"
        _range_push(full)

    # On a partial append, restore all three stacks and undo the push.
    snapshot_len = len(tier_stack)
    try:
        tier_stack.append(used_native)
        marker_stack.append(marker)
        context_stack.append(context)
    except Exception:
        del tier_stack[snapshot_len:]
        del marker_stack[snapshot_len:]
        del context_stack[snapshot_len:]
        try:
            if used_native and _native_tier_hook is not None:
                _native_tier_hook.pop()
            else:
                _range_pop()
        except Exception:
            pass
        raise


def _pop_scope() -> None:
    marker_stack = get_marker_stack()
    context_stack = get_context_stack()
    tier_stack = get_tier_stack()

    # Unmatched pop: no-op.
    if not tier_stack:
        return

    used_native = tier_stack.pop()
    try:
        if used_native and _native_tier_hook is not None:
            _native_tier_hook.pop()
        else:
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
) -> Callable[..., Any]:
    """Wrap func so each call emits a ROCTX range. Idempotent.

    When set, backend attributes the scope to that backend.
    """
    if getattr(func, "_roctx_wrapped", False):
        return func
    func_name = name or func.__name__
    call_counter = {"count": 0}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> object:
        call_counter["count"] += 1
        location = resolve_user_caller_location()
        _push_scope(func_name, f"#{call_counter['count']}@{location}", backend=backend)
        try:
            return func(*args, **kwargs)
        finally:
            _pop_scope()

    wrapper._roctx_wrapped = True
    return wrapper


def _marker_only_init_wrapper(name: str, backend: str = "") -> Callable[..., Any]:
    """Build an __init__ that emits a ROCTX range, then calls object.__init__.

    Used for classes whose construction occurs in __new__ (e.g. cuda.Event,
    cuda.Stream).
    """
    call_counter = {"count": 0}

    def marker_only_init(self: object, *args: Any, **kwargs: Any) -> None:
        call_counter["count"] += 1
        location = resolve_user_caller_location()
        _push_scope(name, f"#{call_counter['count']}@{location}", backend=backend)
        try:
            return object.__init__(self)
        finally:
            _pop_scope()

    marker_only_init._roctx_wrapped = True
    return marker_only_init


def _walk_subclasses(cls: type, fn: Callable[[type], None]) -> None:
    """Apply `fn` to every (transitive) subclass of `cls`."""
    for sub in cls.__subclasses__():
        fn(sub)
        _walk_subclasses(sub, fn)
