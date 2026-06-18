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
from functools import wraps
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

CompiledKernel: Any = None
JITFunction: Any = None

# Per-thread flag set while a launch marker is open, so nested launch calls
# emit a single marker.
_thread_local = threading.local()


def _in_launch() -> bool:
    return getattr(_thread_local, "in_launch", False)


def _resolve_triton() -> bool:
    """Bind the triton handles. Returns True if triton is importable."""
    global CompiledKernel, JITFunction
    if importlib.util.find_spec("triton") is None:
        return False
    try:
        from triton.compiler import CompiledKernel as _CK

        CompiledKernel = _CK
    except Exception:
        CompiledKernel = None
    try:
        from triton.runtime.jit import JITFunction as _JIT

        JITFunction = _JIT
    except Exception:
        JITFunction = None
    return CompiledKernel is not None or JITFunction is not None


def _register_framework_root() -> None:
    """Register triton's package directory so caller-location resolution
    reports the user's call site rather than triton's internals."""
    try:
        import triton

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
    location = resolve_user_caller_location()
    _thread_local.in_launch = True
    pushed = False
    try:
        _push_scope(
            f"{marker_prefix}.{kernel_name}", f"#1@{location}", backend=_BACKEND_NAME
        )
        pushed = True
        return thunk()
    finally:
        if pushed:
            _pop_scope()
        _thread_local.in_launch = False


def _wrap_method(
    owner: type, method_name: str, marker_prefix: str, original: Callable[..., Any]
) -> bool:
    @wraps(original)
    def launch_with_roctx(self: object, *args: Any, **kwargs: Any) -> object:
        return _run_with_marker(
            self, marker_prefix, lambda: original(self, *args, **kwargs)
        )

    launch_with_roctx._roctx_wrapped = True
    setattr(owner, method_name, launch_with_roctx)
    return True


def _wrap_property(
    owner: type, method_name: str, marker_prefix: str, prop: property
) -> bool:
    orig_get = prop.fget
    if orig_get is None:
        return False

    def roctx_get(self: object) -> object:
        launcher = orig_get(self)
        if launcher is None or getattr(launcher, "_roctx_launcher", False):
            return launcher

        @wraps(launcher)
        def launch(*args: Any, **kwargs: Any) -> object:
            return _run_with_marker(
                self, marker_prefix, lambda: launcher(*args, **kwargs)
            )

        launch._roctx_launcher = True
        return launch

    roctx_get._roctx_wrapped = True
    setattr(owner, method_name, property(roctx_get, prop.fset, prop.fdel))
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
    if JITFunction is not None:
        wrapped_any |= _wrap_launch(JITFunction, "run", "triton.JITFunction")
    if CompiledKernel is not None:
        # Prefer run(); fall back to __call__.
        if hasattr(CompiledKernel, "run"):
            wrapped_any |= _wrap_launch(CompiledKernel, "run", "triton.CompiledKernel")
        else:
            wrapped_any |= _wrap_launch(
                CompiledKernel, "__call__", "triton.CompiledKernel"
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
