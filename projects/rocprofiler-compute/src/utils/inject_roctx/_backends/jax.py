# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX instrumentation backend for JAX.

Resolves jax.jit, jax.pmap, jax.shard_map, and jax.extend.core.Primitive
handles. Wraps jax.jit compiled __call__, the jax.pmap and jax.shard_map
factories, and Primitive.bind_with_trace with ROCTX ranges.
"""

import importlib.util
import threading
from pathlib import Path
from typing import Any, Callable, List

from utils.inject_roctx import core
from utils.inject_roctx.registry import register
from utils.logger import console_log, console_warning

_BACKEND_NAME = "jax"


class _JaxState:
    """Resolved JAX entry-point handles, populated by _resolve_jax()."""

    def __init__(self) -> None:
        self.jit: Any = None
        self.pmap: Any = None
        self.pmap_factory: Any = None
        self.shard_map: Any = None
        self.shard_map_factory: Any = None
        self.primitive: Any = None
        self.compiled_call: Any = None
        self.bind_with_trace: Any = None


_STATE = _JaxState()
_jax_thread_local = threading.local()


def patch_jax_entry_points() -> None:
    """Wrap JAX compiled-call and primitive-dispatch entry points."""
    _wrap_jit_compiled_call()
    _wrap_pmap_factory()
    _wrap_shard_map_factory()
    _wrap_primitive_bind_with_trace()


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
    _STATE.shard_map = getattr(jax, "shard_map", None)
    try:
        import jax.extend.core as jax_extend_core

        _STATE.primitive = jax_extend_core.Primitive
    except Exception:
        _STATE.primitive = None
    return True


def _register_framework_root() -> None:
    """Register the jax package directory as a framework root for caller location."""
    try:
        import jax

        jax_file = getattr(jax, "__file__", None)
        if jax_file:
            core.add_framework_root(str(Path(jax_file).parent))
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
    location = core.resolve_user_caller_location()
    core._push_scope(marker, location, backend=_BACKEND_NAME)
    _jax_thread_local.depth = depth + 1
    try:
        return original(*args, **kwargs)
    finally:
        _jax_thread_local.depth = depth
        core._pop_scope()


def _roctx_compiled_call(compiled: object, *args: Any, **kwargs: Any) -> object:
    """Run the compiled call under jax.jit.<fn>, skipping a nested JAX range."""
    marker = f"jax.jit.{_compiled_function_name(compiled)}"
    return _run_with_jax_range(marker, _STATE.compiled_call, compiled, *args, **kwargs)


_roctx_compiled_call._roctx_wrapped = True


def _roctx_pmap_call(compiled: object, *args: Any, **kwargs: Any) -> object:
    """Run the pmap result under jax.pmap.<fn>, skipping a nested JAX range."""
    marker = f"jax.pmap.{_compiled_function_name(compiled)}"
    return _run_with_jax_range(marker, compiled, *args, **kwargs)


def _roctx_pmap_factory(fun: object, *args: Any, **kwargs: Any) -> object:
    """Call original jax.pmap and return a marked callable that forwards attributes."""
    compiled = _STATE.pmap_factory(fun, *args, **kwargs)
    if getattr(compiled, "_roctx_wrapped", False):
        return compiled
    return _RoctxPmapCallable(compiled)


_roctx_pmap_factory._roctx_wrapped = True


def _roctx_shard_map_call(compiled: object, *args: Any, **kwargs: Any) -> object:
    """Run the shard_map result under jax.shard_map.<fn>, skipping a nested range."""
    marker = f"jax.shard_map.{_compiled_function_name(compiled)}"
    return _run_with_jax_range(marker, compiled, *args, **kwargs)


def _roctx_shard_map_factory(fun: object, *args: Any, **kwargs: Any) -> object:
    """Call original jax.shard_map and return a marked callable that forwards attrs."""
    compiled = _STATE.shard_map_factory(fun, *args, **kwargs)
    if getattr(compiled, "_roctx_wrapped", False):
        return compiled
    return _RoctxShardMapCallable(compiled)


_roctx_shard_map_factory._roctx_wrapped = True


def _roctx_bind_with_trace(primitive: object, *args: Any, **kwargs: Any) -> object:
    """Run bind_with_trace under jax.{name}, skipping a nested JAX range."""
    marker = f"jax.{_primitive_name(primitive)}"
    return _run_with_jax_range(
        marker, _STATE.bind_with_trace, primitive, *args, **kwargs
    )


_roctx_bind_with_trace._roctx_wrapped = True


def _wrap_jit_compiled_call() -> None:
    """Replace PjitFunction.__call__ with _roctx_compiled_call."""
    jit = _STATE.jit
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


def _wrap_pmap_factory() -> None:
    """Replace jax.pmap with _roctx_pmap_factory."""
    pmap = _STATE.pmap
    if pmap is None:
        console_warning(
            "ml api trace",
            "jax.pmap is not available; skipping pmap instrumentation.",
        )
        return
    if getattr(pmap, "_roctx_wrapped", False):
        return
    try:
        import jax

        _STATE.pmap_factory = pmap
        jax.pmap = _roctx_pmap_factory
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"Could not patch jax.pmap: {exc}",
        )
        return
    console_log(
        "ml api trace",
        "Wrapped jax.pmap compiled calls with ROCTX markers",
    )


def _wrap_shard_map_factory() -> None:
    """Replace jax.shard_map with _roctx_shard_map_factory."""
    shard_map = _STATE.shard_map
    if shard_map is None:
        console_warning(
            "ml api trace",
            "jax.shard_map is not available; skipping shard_map instrumentation.",
        )
        return
    if getattr(shard_map, "_roctx_wrapped", False):
        return
    try:
        import jax

        _STATE.shard_map_factory = shard_map
        jax.shard_map = _roctx_shard_map_factory
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"Could not patch jax.shard_map: {exc}",
        )
        return
    console_log(
        "ml api trace",
        "Wrapped jax.shard_map compiled calls with ROCTX markers",
    )


def _wrap_primitive_bind_with_trace() -> None:
    """Replace Primitive.bind_with_trace with _roctx_bind_with_trace."""
    primitive = _STATE.primitive
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
        _STATE.bind_with_trace = bind_with_trace
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


class _RoctxPmapCallable:
    """pmap result that opens a ROCTX range, then forwards to the compiled function."""

    __slots__ = ("_compiled", "_roctx_wrapped")

    def __init__(self, compiled: object) -> None:
        object.__setattr__(self, "_compiled", compiled)
        object.__setattr__(self, "_roctx_wrapped", True)

    def __call__(self, *args: Any, **kwargs: Any) -> object:
        return _roctx_pmap_call(self._compiled, *args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self._compiled, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in ("_compiled", "_roctx_wrapped"):
            object.__setattr__(self, name, value)
            return
        setattr(self._compiled, name, value)

    def __dir__(self) -> List[str]:
        names = set(dir(type(self)))
        compiled = object.__getattribute__(self, "_compiled")
        names.update(dir(compiled))
        return sorted(names)


class _RoctxShardMapCallable(_RoctxPmapCallable):
    """shard_map result that opens a ROCTX range, then forwards attributes."""

    __slots__ = ()

    def __call__(self, *args: Any, **kwargs: Any) -> object:
        return _roctx_shard_map_call(self._compiled, *args, **kwargs)


class JaxBackend:
    name = "jax"

    def install(self) -> None:
        """Resolve jax and ROCTX, then wrap compiled-call and primitive entry points."""
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
        _register_framework_root()
        patch_jax_entry_points()


register(JaxBackend())
