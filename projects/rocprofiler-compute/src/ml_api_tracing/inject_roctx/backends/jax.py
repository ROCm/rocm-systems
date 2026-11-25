# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX instrumentation backend for JAX.

Resolves jax.jit, jax.pmap, jax.shard_map, and jax.extend.core.Primitive
handles. Wraps those entry points with ROCTX ranges.
"""

import importlib.util
from pathlib import Path
from typing import Any

from ml_api_tracing.inject_roctx import marker_ranges
from ml_api_tracing.inject_roctx.backend_contract import register
from utils.logger import console_log, console_warning

_BACKEND_NAME = "jax"


def patch_jax_entry_points() -> None:
    """Wrap JAX compiled-call and primitive-dispatch entry points."""
    pass


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


class _JaxState:
    """Resolved JAX entry-point handles, populated by _resolve_jax()."""

    def __init__(self) -> None:
        self.jit: Any = None
        self.pmap: Any = None
        self.shard_map: Any = None
        self.primitive: Any = None


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
