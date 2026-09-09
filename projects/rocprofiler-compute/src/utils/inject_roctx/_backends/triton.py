# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX instrumentation backend for Triton.

Wraps the Triton kernel-launch entry points (``JITFunction.run`` and
``CompiledKernel.run`` / ``CompiledKernel.__call__``) so that Triton and
Inductor kernel launches appear in ROCTX markers.
"""

import importlib.util
import inspect
import threading
from functools import partial, partialmethod
from pathlib import Path
from typing import Any, Callable

from utils.inject_roctx import core
from utils.inject_roctx.core import (
    _pop_scope,
    _push_scope,
    resolve_user_caller_location,
)
from utils.inject_roctx.registry import register
from utils.logger import console_log, console_warning

_BACKEND_NAME = "triton"


class _TritonState:
    """Resolved Triton launch entry-point handles, populated by _resolve_triton()."""

    def __init__(self) -> None:
        self.compiled_kernel: Any = None
        self.jit_function: Any = None


_STATE = _TritonState()

# Per-thread guard so nested launches emit a single marker.
_thread_local = threading.local()


def _in_launch() -> bool:
    return getattr(_thread_local, "in_launch", False)


def _next_launch_index(marker: str) -> int:
    """Per-thread occurrence count for marker."""
    counters = getattr(_thread_local, "launch_counters", None)
    if counters is None:
        counters = {}
        _thread_local.launch_counters = counters
    counters[marker] = counters.get(marker, 0) + 1
    return counters[marker]


def _resolve_triton() -> bool:
    """Bind the triton handles on _STATE. Returns True if triton is importable."""
    if importlib.util.find_spec("triton") is None:
        return False
    try:
        from triton.compiler import CompiledKernel as _CK

        _STATE.compiled_kernel = _CK
    except Exception:
        _STATE.compiled_kernel = None
    try:
        from triton.runtime.jit import JITFunction as _JIT

        _STATE.jit_function = _JIT
    except Exception:
        _STATE.jit_function = None
    return _STATE.compiled_kernel is not None or _STATE.jit_function is not None


def _register_framework_root() -> None:
    """Register triton's package directory as a framework root for
    caller-location resolution."""
    try:
        import triton

        console_log(
            "ml api trace",
            f"Triton version: {getattr(triton, '__version__', '<unknown>')}",
        )
        triton_file = getattr(triton, "__file__", None)
        if triton_file:
            core.add_framework_root(str(Path(triton_file).parent))
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"Could not register triton framework root: {exc}",
        )


def _extract_kernel_name(obj: object, default: str = "<triton_kernel>") -> str:
    """Resolve the kernel name from ``name``, ``metadata``, or ``fn``,
    returning ``default`` when none is available."""
    name = getattr(obj, "name", None)
    if isinstance(name, str) and name:
        return name

    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, dict):
        meta_name = metadata.get("name")
        if isinstance(meta_name, str) and meta_name:
            return meta_name
    else:
        meta_name = getattr(metadata, "name", None)
        if isinstance(meta_name, str) and meta_name:
            return meta_name

    fn = getattr(obj, "fn", None)
    fn_name = getattr(fn, "__name__", None)
    if isinstance(fn_name, str) and fn_name:
        return fn_name

    return default


def _run_with_marker(
    self_obj: object,
    marker_prefix: str,
    thunk: Callable[[], Any],
) -> object:
    """Run ``thunk`` inside a ROCTX range; nested launches reuse the outer range."""
    if _in_launch():
        return thunk()
    kernel_name = _extract_kernel_name(self_obj)
    marker = f"{marker_prefix}.{kernel_name}"
    location = resolve_user_caller_location()
    index = _next_launch_index(marker)
    _thread_local.in_launch = True
    pushed = False
    try:
        _push_scope(marker, f"#{index}@{location}", backend=_BACKEND_NAME)
        pushed = True
        return thunk()
    finally:
        if pushed:
            _pop_scope()
        _thread_local.in_launch = False


def _roctx_method_call(
    instance: object,
    marker_prefix: str,
    original: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> object:
    """Run a wrapped method ``original`` inside a ROCTX range."""
    return _run_with_marker(
        instance, marker_prefix, partial(original, instance, *args, **kwargs)
    )


def _wrap_method(
    owner: type, method_name: str, marker_prefix: str, original: Callable[..., Any]
) -> bool:
    wrapper = partialmethod(_roctx_method_call, marker_prefix, original)
    wrapper._roctx_wrapped = True
    setattr(owner, method_name, wrapper)
    return True


def _roctx_launch(
    instance: object,
    marker_prefix: str,
    launcher: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> object:
    """Run a property-returned ``launcher`` inside a ROCTX range."""
    return _run_with_marker(instance, marker_prefix, partial(launcher, *args, **kwargs))


def _roctx_property_get(
    marker_prefix: str, original_getter: Callable[..., Any], instance: object
) -> object:
    """Property getter that wraps the launcher it returns with a ROCTX range."""
    launcher = original_getter(instance)
    if launcher is None or getattr(launcher, "_roctx_launcher", False):
        return launcher
    wrapped = partial(_roctx_launch, instance, marker_prefix, launcher)
    wrapped._roctx_launcher = True
    return wrapped


def _wrap_property(
    owner: type, method_name: str, marker_prefix: str, wrapped_property: property
) -> bool:
    original_getter = wrapped_property.fget
    if original_getter is None:
        return False
    getter = partial(_roctx_property_get, marker_prefix, original_getter)
    getter._roctx_wrapped = True
    setattr(
        owner,
        method_name,
        property(getter, wrapped_property.fset, wrapped_property.fdel),
    )
    return True


def _wrap_launch(
    owner: type,
    method_name: str,
    marker_prefix: str,
) -> bool:
    """Wrap ``owner.method_name`` (a method or property) with a ROCTX range.

    Idempotent. Returns True when the wrapper is installed or already present.
    """
    attr = inspect.getattr_static(owner, method_name, None)
    if attr is None:
        return False
    if isinstance(attr, property):
        if attr.fget is not None and getattr(attr.fget, "_roctx_wrapped", False):
            return True
    elif getattr(attr, "_roctx_wrapped", False):
        return True

    try:
        if isinstance(attr, property):
            installed = _wrap_property(owner, method_name, marker_prefix, attr)
        else:
            installed = _wrap_method(owner, method_name, marker_prefix, attr)
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"Could not patch {owner.__name__}.{method_name}: {exc}",
        )
        return False

    if installed:
        console_log(
            "ml api trace",
            f"Wrapped {owner.__name__}.{method_name} with ROCTX markers",
        )
    return installed


def patch_triton_launcher() -> None:
    """Wrap every available Triton launch entry point."""
    wrapped_any = False
    jit_function = _STATE.jit_function
    compiled_kernel = _STATE.compiled_kernel
    if jit_function is not None:
        wrapped_any |= _wrap_launch(jit_function, "run", "triton.JITFunction")
    if compiled_kernel is not None:
        # Prefer run(); fall back to __call__.
        if hasattr(compiled_kernel, "run"):
            wrapped_any |= _wrap_launch(compiled_kernel, "run", "triton.CompiledKernel")
        else:
            wrapped_any |= _wrap_launch(
                compiled_kernel, "__call__", "triton.CompiledKernel"
            )
    if not wrapped_any:
        console_warning(
            "ml api trace",
            "No Triton launch entry points found to instrument; "
            "Triton API tracing may have no effect.",
        )


class TritonBackend:
    name = "triton"

    def install(self) -> None:
        if not _resolve_triton():
            console_warning(
                "ml api trace",
                "Triton is not installed; skipping triton instrumentation.",
            )
            return
        if not core.ensure_python_tier():
            console_warning(
                "ml api trace",
                "ROCTX bindings not found; skipping triton instrumentation.",
            )
            return
        _register_framework_root()
        patch_triton_launcher()


register(TritonBackend())
