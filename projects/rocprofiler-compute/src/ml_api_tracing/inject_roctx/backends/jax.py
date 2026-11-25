# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX instrumentation backend for JAX.

Resolves jax.jit, jax.pmap, jax.shard_map, and jax.extend.core.Primitive
handles. Wraps those entry points with ROCTX ranges.
"""

import importlib.util
import threading
from pathlib import Path
from typing import Any, Callable

from ml_api_tracing.inject_roctx import marker_ranges
from ml_api_tracing.inject_roctx.backend_contract import register
from utils.logger import console_log, console_warning

_BACKEND_NAME = "jax"
_jax_thread_local = threading.local()


def patch_jax_entry_points() -> None:
    """Wrap JAX compiled-call and primitive-dispatch entry points."""
    _wrap_jit_compiled_call()
    _wrap_primitive_bind_with_trace()


def _resolve_jax() -> bool:
    """Bind the jax handles on _CoreState. Returns True if jax is importable."""
    if importlib.util.find_spec("jax") is None:
        return False
    try:
        import jax
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"Could not import jax: {exc}",
        )
        return False

    _CoreState.jit = getattr(jax, "jit", None)
    _CoreState.pmap = getattr(jax, "pmap", None)
    _CoreState.shard_map = getattr(jax, "shard_map", None)
    try:
        import jax.extend.core as jax_extend_core

        _CoreState.primitive = jax_extend_core.Primitive
    except Exception:
        _CoreState.primitive = None
    return True


def _register_framework_root() -> None:
    """Register the jax package directory as a framework root for
    caller-location resolution."""
    try:
        import jax

        console_log(
            "ml api trace",
            f"JAX version: {getattr(jax, '__version__', '<unknown>')}",
        )
        jax_file = getattr(jax, "__file__", None)
        if jax_file:
            marker_ranges.add_framework_root(str(Path(jax_file).parent))
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"Could not register jax framework root: {exc}",
        )


def _identity(value: object) -> object:
    """Return value unchanged so jax.jit can produce a compiled type at install."""
    return value


def _compiled_function_name(compiled: object) -> str:
    """Name for a jax.jit, jax.pmap, or jax.shard_map marker from the callable."""
    name = getattr(compiled, "__name__", None)
    if isinstance(name, str) and name:
        return name
    wrapped = getattr(compiled, "__wrapped__", None)
    wrapped_name = getattr(wrapped, "__name__", None)
    if isinstance(wrapped_name, str) and wrapped_name:
        return wrapped_name
    return "<unknown>"


def _primitive_name(primitive: object) -> str:
    """Name for a jax.{name} marker from the Primitive instance."""
    name = getattr(primitive, "name", None)
    if isinstance(name, str) and name:
        return name
    return "<unknown>"


def _run_with_jax_range(
    marker: str,
    original: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> object:
    """Run original inside a JAX ROCTX range, or bare if one is already open."""
    depth = getattr(_jax_thread_local, "depth", 0)
    if depth > 0:
        return original(*args, **kwargs)
    location = marker_ranges.resolve_user_caller_location()
    marker_ranges.push_scope(marker, location, backend=_BACKEND_NAME)
    _jax_thread_local.depth = depth + 1
    try:
        return original(*args, **kwargs)
    finally:
        _jax_thread_local.depth = depth
        marker_ranges.pop_scope()


def _roctx_compiled_call(compiled: object, *args: Any, **kwargs: Any) -> object:
    """Run the compiled call under jax.jit.<fn>, skipping a nested JAX range."""
    marker = f"jax.jit.{_compiled_function_name(compiled)}"
    return _run_with_jax_range(
        marker, _CoreState.compiled_call, compiled, *args, **kwargs
    )


_roctx_compiled_call._roctx_wrapped = True


def _wrap_jit_compiled_call() -> None:
    """Replace PjitFunction.__call__ with _roctx_compiled_call."""
    jit = _CoreState.jit
    if jit is None:
        console_warning(
            "ml api trace",
            "jax.jit is not available; skipping jit instrumentation.",
        )
        return
    try:
        compiled_type = type(jit(_identity))
        if getattr(compiled_type.__call__, "_roctx_wrapped", False):
            return
        _CoreState.compiled_call = compiled_type.__call__
        compiled_type.__call__ = _roctx_compiled_call
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"Could not patch jax.jit compiled __call__: {exc}",
        )
        return
    console_log(
        "ml api trace",
        "Wrapped jax.jit compiled calls with ROCTX markers",
    )


def _roctx_bind_with_trace(primitive: object, *args: Any, **kwargs: Any) -> object:
    """Run bind_with_trace under jax.{name}, skipping a nested JAX range."""
    marker = f"jax.{_primitive_name(primitive)}"
    return _run_with_jax_range(
        marker, _CoreState.bind_with_trace, primitive, *args, **kwargs
    )


_roctx_bind_with_trace._roctx_wrapped = True


def _wrap_primitive_bind_with_trace() -> None:
    """Replace Primitive.bind_with_trace with _roctx_bind_with_trace."""
    primitive = _CoreState.primitive
    bind_with_trace = getattr(primitive, "bind_with_trace", None)
    if bind_with_trace is None:
        console_warning(
            "ml api trace",
            "jax Primitive is not available; skipping primitive instrumentation.",
        )
        return
    if getattr(bind_with_trace, "_roctx_wrapped", False):
        return
    try:
        _CoreState.bind_with_trace = bind_with_trace
        primitive.bind_with_trace = _roctx_bind_with_trace
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"Could not patch Primitive.bind_with_trace: {exc}",
        )
        return
    console_log(
        "ml api trace",
        "Wrapped Primitive.bind_with_trace with ROCTX markers",
    )


class _JaxState:
    """Resolved JAX entry-point handles, populated by _resolve_jax()."""

    def __init__(self) -> None:
        self.jit: Any = None
        self.pmap: Any = None
        self.shard_map: Any = None
        self.primitive: Any = None
        self.compiled_call: Any = None
        self.bind_with_trace: Any = None


_CoreState = _JaxState()


class JaxBackend:
    name = "jax"

    def install(self) -> None:
        if not _resolve_jax():
            console_warning(
                "ml api trace",
                "JAX is not installed; skipping jax instrumentation.",
            )
            return
        if not marker_ranges.ensure_python_tier():
            console_warning(
                "ml api trace",
                "ROCTX bindings not found; skipping jax instrumentation.",
            )
            return
        _register_framework_root()
        patch_jax_entry_points()


register(JaxBackend())
