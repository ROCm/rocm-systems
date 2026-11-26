# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX instrumentation backend for Triton.

Wraps JITFunction.run and CompiledKernel.run or CompiledKernel.__call__
so @triton.jit launches appear as ROCTX markers. Inductor fused kernels
that use PyTorch's static launcher are not wrapped.
"""

import importlib.util
import inspect
from functools import partial, partialmethod
from pathlib import Path
from typing import Any, Callable, List

from ml_api_tracing.inject_roctx import marker_ranges
from ml_api_tracing.inject_roctx.backend_contract import register
from ml_api_tracing.marker_format import (
    _MAX_TENSOR_ARGUMENT_ITEMS,
    format_args_blob,
    render_dtype_shape,
)
from utils.logger import console_log, console_warning

_BACKEND_NAME = "triton"
_JIT_MARKER_PREFIX = "triton.JITFunction"
_COMPILED_MARKER_PREFIX = "triton.CompiledKernel"


def patch_triton_launcher() -> None:
    """Wrap every available Triton launch entry point."""
    wrapped_any = False
    jit_function = _CoreState.jit_function
    compiled_kernel = _CoreState.compiled_kernel
    if jit_function is not None:
        wrapped_any |= _wrap_launch(jit_function, "run", False)
    if compiled_kernel is not None:
        # Prefer run(); fall back to __call__.
        if hasattr(compiled_kernel, "run"):
            wrapped_any |= _wrap_launch(compiled_kernel, "run", True)
        else:
            wrapped_any |= _wrap_launch(compiled_kernel, "__call__", True)
    if not wrapped_any:
        console_warning(
            "ml api trace",
            "No Triton launch entry points found to instrument; "
            "Triton API tracing may have no effect.",
        )


def _resolve_triton() -> bool:
    """Bind the triton handles on _CoreState. Returns True if triton is importable."""
    if importlib.util.find_spec("triton") is None:
        return False
    try:
        from triton.compiler import CompiledKernel as _CK

        _CoreState.compiled_kernel = _CK
    except Exception:
        _CoreState.compiled_kernel = None
    try:
        from triton.runtime.jit import JITFunction as _JIT

        _CoreState.jit_function = _JIT
    except Exception:
        _CoreState.jit_function = None
    return _CoreState.compiled_kernel is not None or _CoreState.jit_function is not None


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
            marker_ranges.add_framework_root(str(Path(triton_file).parent))
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"Could not register triton framework root: {exc}",
        )


def _extract_kernel_name(obj: object, default: str = "<triton_kernel>") -> str:
    """Resolve the kernel name from name, metadata, or fn,
    returning default when none is available."""
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


def _operand_name(param: object) -> str:
    """Return a kernel parameter name, or empty when param has none."""
    if isinstance(param, str) and param:
        return param
    param_name = getattr(param, "name", None)
    if isinstance(param_name, str) and param_name:
        return param_name
    return ""


def _collect_named_operands(source: object) -> List[str]:
    """Return operand names from params, then arg_names, on source."""
    for attr in ("params", "arg_names"):
        names = []
        for param in getattr(source, attr, None) or ():
            name = _operand_name(param)
            if name:
                names.append(name)
        if names:
            return names
    return []


def _triton_operand_names(instance: object, is_compiled: bool) -> List[str]:
    """Return kernel operand names for the intercepted launch kind.

    JITFunction exposes params / arg_names on the instance. CompiledKernel
    does not; names live on src.fn (ASTSource). IRSource has no fn.
    """
    if is_compiled:
        src = getattr(instance, "src", None)
        names = _collect_named_operands(getattr(src, "fn", None))
        if names:
            return names
    return _collect_named_operands(instance)


def _format_triton_args(
    instance: object,
    is_compiled: bool,
    call_args: object,
    call_kwargs: object,
) -> str:
    """Return the marker args blob for a Triton launch, or n/a.

    Drops bound self. CompiledKernel also drops the launcher prefix.
    """
    try:
        names = _triton_operand_names(instance, is_compiled)
        if not names:
            return "n/a"

        positionals = tuple(call_args)
        if positionals and positionals[0] is instance:
            positionals = positionals[1:]
        if is_compiled and len(positionals) > len(names):
            positionals = positionals[-len(names) :]

        parts = []
        for index, name in enumerate(names):
            if len(parts) >= _MAX_TENSOR_ARGUMENT_ITEMS:
                break
            if index < len(positionals):
                value = positionals[index]
            elif name in call_kwargs:
                value = call_kwargs[name]
            else:
                continue
            rendered = render_dtype_shape(value)
            if rendered is None and isinstance(value, (int, float, bool, str)):
                rendered = repr(value)
            if rendered is None:
                continue
            parts.append(f"{name}={rendered}")
        return format_args_blob(parts)
    except Exception:
        return "n/a"


def _run_with_marker(
    instance: object,
    is_compiled: bool,
    call: Callable[..., Any],
    /,
    *call_args: Any,
    **call_kwargs: Any,
) -> object:
    """Run call(*call_args, **call_kwargs) inside a ROCTX range.

    The first three parameters are positional-only so a keyword argument of the
    wrapped call can never collide with them.
    """
    kernel_name = _extract_kernel_name(instance)
    marker_prefix = _COMPILED_MARKER_PREFIX if is_compiled else _JIT_MARKER_PREFIX
    marker = f"{marker_prefix}.{kernel_name}"
    location = marker_ranges.resolve_user_caller_location()
    marker_ranges.push_scope(
        marker,
        location,
        backend=_BACKEND_NAME,
        args=_format_triton_args(instance, is_compiled, call_args, call_kwargs),
    )
    try:
        return call(*call_args, **call_kwargs)
    finally:
        marker_ranges.pop_scope()


def _roctx_method_call(
    instance: object,
    is_compiled: bool,
    original: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> object:
    """Run a wrapped method original inside a ROCTX range."""
    return _run_with_marker(instance, is_compiled, original, instance, *args, **kwargs)


def _wrap_method(
    owner: type,
    method_name: str,
    is_compiled: bool,
    original: Callable[..., Any],
) -> bool:
    wrapper = partialmethod(_roctx_method_call, is_compiled, original)
    wrapper._roctx_wrapped = True
    setattr(owner, method_name, wrapper)
    return True


def _roctx_launch(
    instance: object,
    is_compiled: bool,
    launcher: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> object:
    """Run a property-returned launcher inside a ROCTX range."""
    return _run_with_marker(instance, is_compiled, launcher, *args, **kwargs)


def _roctx_property_get(
    is_compiled: bool, original_getter: Callable[..., Any], instance: object
) -> object:
    """Property getter that wraps the launcher it returns with a ROCTX range."""
    launcher = original_getter(instance)
    if launcher is None or getattr(launcher, "_roctx_launcher", False):
        return launcher
    wrapped = partial(_roctx_launch, instance, is_compiled, launcher)
    wrapped._roctx_launcher = True
    return wrapped


def _wrap_property(
    owner: type,
    method_name: str,
    is_compiled: bool,
    wrapped_property: property,
) -> bool:
    original_getter = wrapped_property.fget
    if original_getter is None:
        return False
    getter = partial(_roctx_property_get, is_compiled, original_getter)
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
    is_compiled: bool,
) -> bool:
    """Wrap owner.method_name (a method or property) with a ROCTX range.

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
            installed = _wrap_property(owner, method_name, is_compiled, attr)
        else:
            installed = _wrap_method(owner, method_name, is_compiled, attr)
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


class _TritonState:
    """Resolved Triton launch entry-point handles, populated by _resolve_triton()."""

    def __init__(self) -> None:
        self.compiled_kernel: Any = None
        self.jit_function: Any = None


_CoreState = _TritonState()


class TritonBackend:
    name = "triton"

    def install(self) -> None:
        if not _resolve_triton():
            console_warning(
                "ml api trace",
                "Triton is not installed; skipping triton instrumentation.",
            )
            return
        if not marker_ranges.ensure_python_tier():
            console_warning(
                "ml api trace",
                "ROCTX bindings not found; skipping triton instrumentation.",
            )
            return
        _register_framework_root()
        patch_triton_launcher()


register(TritonBackend())
