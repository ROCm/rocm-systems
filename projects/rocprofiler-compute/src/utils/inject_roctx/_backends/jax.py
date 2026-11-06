# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX instrumentation backend for JAX.

Resolves jax.jit, jax.pmap, and jax.extend.core.Primitive handles.
This module registers and installs the backend; wraps land in later patches.
"""

import importlib.util
from typing import Any

from utils.inject_roctx import core
from utils.inject_roctx.registry import register
from utils.logger import console_log, console_warning


class _JaxState:
    """Resolved JAX entry-point handles, populated by _resolve_jax()."""

    def __init__(self) -> None:
        self.jit: Any = None
        self.pmap: Any = None
        self.primitive: Any = None


_STATE = _JaxState()


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


def patch_jax_entry_points() -> None:
    """Wrap JAX compiled calls and primitive dispatch. No-op until wraps land."""
    pass


class JaxBackend:
    name = "jax"

    def install(self) -> None:
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
