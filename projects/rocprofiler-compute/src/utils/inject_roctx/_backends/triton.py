# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX instrumentation backend for Triton.

Wraps triton.compiler.CompiledKernel.__call__ so Triton and Inductor kernel
launches appear in ROCTX markers.
"""

import importlib.util
from functools import wraps
from typing import Any

from utils.inject_roctx.core import (
    _pop_scope,
    _push_scope,
    ensure_python_tier,
    resolve_user_caller_location,
)
from utils.inject_roctx.registry import register
from utils.logger import console_log, console_warning

_BACKEND_NAME = "triton"


class TritonBackend:
    name = "triton"

    def __init__(self) -> None:
        self._compiled_kernel: Any = None

    def _resolve(self) -> bool:
        """Bind the CompiledKernel handle. Returns True if triton is importable."""
        if importlib.util.find_spec("triton") is None:
            return False
        try:
            from triton.compiler import CompiledKernel
        except Exception:
            return False
        self._compiled_kernel = CompiledKernel
        return True

    def patch_launcher(self) -> None:
        """Wrap CompiledKernel.__call__ so Triton/Inductor launches show in markers."""
        compiled_kernel = self._compiled_kernel
        if compiled_kernel is None:
            return

        original_call = getattr(compiled_kernel, "__call__", None)
        if original_call is None:
            return
        if getattr(original_call, "_roctx_wrapped", False):
            return

        call_counts: dict[str, int] = {}

        @wraps(original_call)
        def call_with_roctx(kernel: object, *args: Any, **kwargs: Any) -> object:
            kernel_name = (
                getattr(kernel, "name", None)
                or getattr(kernel, "metadata", None)
                or "<triton_kernel>"
            )
            if isinstance(kernel_name, dict):
                kernel_name = kernel_name.get("name", "<triton_kernel>")
            marker = f"triton.CompiledKernel.{kernel_name}"
            call_counts[marker] = call_counts.get(marker, 0) + 1
            location = resolve_user_caller_location()
            _push_scope(
                marker,
                f"#{call_counts[marker]}@{location}",
                backend=_BACKEND_NAME,
            )
            try:
                return original_call(kernel, *args, **kwargs)
            finally:
                _pop_scope()

        call_with_roctx._roctx_wrapped = True
        try:
            compiled_kernel.__call__ = call_with_roctx
            console_log(
                "ml api trace",
                "Wrapped triton.CompiledKernel.__call__ with ROCTX markers",
            )
        except Exception as exc:
            console_warning(
                "ml api trace",
                f"Could not patch triton.CompiledKernel.__call__: {exc}",
            )

    def install(self) -> None:
        if not self._resolve():
            console_warning(
                "ml api trace",
                "Triton is not installed; skipping triton instrumentation.",
            )
            return
        if not ensure_python_tier():
            console_warning(
                "ml api trace",
                "ROCTX bindings not found; skipping triton instrumentation.",
            )
            return
        self.patch_launcher()


register(TritonBackend())
