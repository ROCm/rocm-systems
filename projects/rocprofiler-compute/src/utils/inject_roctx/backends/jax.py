# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX instrumentation backend for JAX.

Wraps the JAX transformation entry points (``jax.jit`` and ``jax.pmap``) so
that the kernels launched by each compiled-function invocation appear in ROCTX
markers labelled with the function name.
"""

import importlib.util
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

_BACKEND_NAME = "jax"

# Transformations wrapped on the top-level jax module.
_TRANSFORMS = ("jit", "pmap")

_thread_local = threading.local()


def _in_call() -> bool:
    return getattr(_thread_local, "in_call", False)


def _next_call_index(marker: str) -> int:
    """Per-thread occurrence count for marker."""
    counters = getattr(_thread_local, "call_counters", None)
    if counters is None:
        counters = {}
        _thread_local.call_counters = counters
    counters[marker] = counters.get(marker, 0) + 1
    return counters[marker]


def _function_label(fn: object) -> str:
    """Resolve a display name for a transformed function."""
    name = getattr(fn, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return type(fn).__name__


def _run_with_marker(marker: str, thunk: Callable[[], Any]) -> object:
    """Run ``thunk`` inside a ROCTX range; nested calls reuse the outer range."""
    if _in_call():
        return thunk()
    location = resolve_user_caller_location()
    index = _next_call_index(marker)
    _thread_local.in_call = True
    pushed = False
    try:
        _push_scope(marker, f"#{index}@{location}", backend=_BACKEND_NAME)
        pushed = True
        return thunk()
    finally:
        if pushed:
            _pop_scope()
        _thread_local.in_call = False


def _wrap_transform(
    transform: Callable[..., Any], transform_name: str
) -> Callable[..., Any]:
    """Wrap a JAX transformation so its compiled callable emits ROCTX ranges."""

    @wraps(transform)
    def transform_wrapper(fn: object = None, *args: Any, **kwargs: Any) -> object:
        compiled = transform(fn, *args, **kwargs)
        if compiled is None or not callable(compiled):
            return compiled
        marker = f"jax.{transform_name}.{_function_label(fn)}"

        @wraps(compiled)
        def invocation_wrapper(*call_args: Any, **call_kwargs: Any) -> object:
            return _run_with_marker(marker, lambda: compiled(*call_args, **call_kwargs))

        invocation_wrapper._roctx_wrapped = True
        return invocation_wrapper

    transform_wrapper._roctx_wrapped = True
    return transform_wrapper


def patch_jax_transforms() -> None:
    """Wrap every available JAX transformation entry point."""
    import jax

    wrapped = []
    for transform_name in _TRANSFORMS:
        transform = getattr(jax, transform_name, None)
        if transform is None or not callable(transform):
            continue
        if getattr(transform, "_roctx_wrapped", False):
            continue
        try:
            setattr(jax, transform_name, _wrap_transform(transform, transform_name))
            wrapped.append(transform_name)
        except Exception as exc:
            console_warning(
                "ml api trace",
                f"Could not patch jax.{transform_name}: {exc}",
            )

    if wrapped:
        console_log(
            "ml api trace",
            f"Wrapped jax transforms with ROCTX markers: {', '.join(wrapped)}",
        )
    else:
        console_warning(
            "ml api trace",
            "No JAX transforms found to instrument; "
            "JAX API tracing may have no effect.",
        )


def _register_framework_root() -> None:
    """Register jax's package directory as a framework root for
    caller-location resolution."""
    try:
        import jax

        console_log(
            "ml api trace",
            f"JAX version: {getattr(jax, '__version__', '<unknown>')}",
        )
        jax_file = getattr(jax, "__file__", None)
        if jax_file:
            core.add_framework_root(str(Path(jax_file).parent))
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"Could not register jax framework root: {exc}",
        )


class JaxBackend:
    name = "jax"

    def install(self) -> None:
        if importlib.util.find_spec("jax") is None:
            console_warning(
                "ml api trace",
                "JAX is not installed; skipping jax instrumentation.",
            )
            return
        if not core.ensure_python_tier():
            console_warning(
                "ml api trace",
                "ROCTX bindings not found; skipping jax instrumentation.",
            )
            return
        _register_framework_root()
        patch_jax_transforms()


register(JaxBackend())
