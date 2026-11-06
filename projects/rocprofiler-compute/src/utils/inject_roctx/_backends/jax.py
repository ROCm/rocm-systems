# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX instrumentation backend for JAX.

Resolves jax.jit, jax.pmap, and jax.extend.core.Primitive handles.
Wraps jax.jit compiled __call__ with a jax.jit.<fn> ROCTX range.
"""

import importlib.util
from typing import Any

from utils.inject_roctx import core
from utils.inject_roctx.registry import register
from utils.logger import console_log, console_warning

_BACKEND_NAME = "jax"


class _JaxState:
    """Resolved JAX entry-point handles, populated by _resolve_jax()."""

    def __init__(self) -> None:
        self.jit: Any = None
        self.pmap: Any = None
        self.primitive: Any = None
        self.compiled_call: Any = None


_STATE = _JaxState()


def patch_jax_entry_points() -> None:
    """Wrap jax.jit compiled __call__ with a jax.jit.<fn> ROCTX range."""
    jit = _STATE.jit
    if jit is None:
        console_warning(
            "ml api trace",
            "jax.jit is not available; skipping jit instrumentation.",
        )
        return
    try:
        compiled_type = type(jit(_identity))
        _STATE.compiled_call = compiled_type.__call__
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


def _resolve_jax() -> bool:
    """Bind the jax handles on _STATE. Returns True if jax is importable."""
    if importlib.util.find_spec("jax") is None:
        return False
    try:
        import jax
    except Exception:
        return False

    _STATE.jit = getattr(jax, "jit", None)
    _STATE.pmap = getattr(jax, "pmap", None)
    try:
        import jax.extend.core as jax_extend_core

        _STATE.primitive = jax_extend_core.Primitive
    except Exception:
        _STATE.primitive = None
    return True


def _identity(value: object) -> object:
    """Return value unchanged so jax.jit can produce a compiled type at install."""
    return value


def _compiled_function_name(compiled: object) -> str:
    """Name for jax.jit.<fn> from the compiled object or its wrapped function."""
    name = getattr(compiled, "__name__", None)
    if isinstance(name, str) and name:
        return name
    wrapped = getattr(compiled, "__wrapped__", None)
    wrapped_name = getattr(wrapped, "__name__", None)
    if isinstance(wrapped_name, str) and wrapped_name:
        return wrapped_name
    return "<unknown>"


def _roctx_compiled_call(compiled: object, *args: Any, **kwargs: Any) -> object:
    """Push jax.jit.<fn>, run the original compiled __call__, then pop."""
    marker = f"jax.jit.{_compiled_function_name(compiled)}"
    location = core.resolve_user_caller_location()
    core._push_scope(marker, location, backend=_BACKEND_NAME)
    try:
        return _STATE.compiled_call(compiled, *args, **kwargs)
    finally:
        core._pop_scope()


class JaxBackend:
    name = "jax"

    def install(self) -> None:
        """Resolve jax and ROCTX, then wrap jax.jit compiled calls."""
        if not _resolve_jax():
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
        console_log("ml api trace", "JAX backend is installed")
        patch_jax_entry_points()


register(JaxBackend())
