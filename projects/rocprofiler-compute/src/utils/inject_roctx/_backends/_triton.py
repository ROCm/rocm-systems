# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX instrumentation backend for Triton.

Wraps triton.compiler.CompiledKernel.__call__ so Triton and Inductor kernel
launches appear in ROCTX markers.
"""

import importlib.util
from functools import wraps
from typing import Any

from utils.inject_roctx._core import (
    _pop_scope,
    _push_scope,
    resolve_user_caller_location,
)
from utils.logger import console_log, console_warning

from . import register

_BACKEND_NAME = "triton"

CompiledKernel: Any = None


def _resolve_triton() -> bool:
    """Bind the CompiledKernel handle. Returns True if triton is importable."""
    global CompiledKernel
    if importlib.util.find_spec("triton") is None:
        return False
    try:
        from triton.compiler import CompiledKernel as _CK
    except Exception:
        return False
    CompiledKernel = _CK
    return True


def patch_triton_launcher() -> None:
    """Wrap CompiledKernel.__call__ so Triton/Inductor launches show in markers."""
    if CompiledKernel is None:
        return

    original_call = getattr(CompiledKernel, "__call__", None)
    if original_call is None:
        return
    if getattr(original_call, "_roctx_wrapped", False):
        return

    @wraps(original_call)
    def call_with_roctx(self: object, *args: Any, **kwargs: Any) -> object:
        kernel_name = (
            getattr(self, "name", None)
            or getattr(self, "metadata", None)
            or "<triton_kernel>"
        )
        if isinstance(kernel_name, dict):
            kernel_name = kernel_name.get("name", "<triton_kernel>")
        location = resolve_user_caller_location()
        marker = f"triton.CompiledKernel.{kernel_name}"
        _push_scope(marker, f"#1@{location}", backend=_BACKEND_NAME)
        try:
            return original_call(self, *args, **kwargs)
        finally:
            _pop_scope()

    call_with_roctx._roctx_wrapped = True
    try:
        CompiledKernel.__call__ = call_with_roctx
        console_log(
            "api trace", "Wrapped triton.CompiledKernel.__call__ with ROCTX markers"
        )
    except Exception as exc:
        console_warning(
            "api trace",
            f"Could not patch triton.CompiledKernel.__call__: {exc}",
        )


class TritonBackend:
    name = "triton"

    def install(self) -> None:
        if not _resolve_triton():
            console_warning(
                "api trace",
                "Triton is not installed; skipping triton instrumentation.",
            )
            return
        patch_triton_launcher()


register(TritonBackend())
