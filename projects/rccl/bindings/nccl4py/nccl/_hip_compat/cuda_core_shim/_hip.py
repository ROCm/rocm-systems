# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin wrapper around hip-python.

Imports the ``hip`` module once and exposes a single error-checking
helper that unwraps the ``(status, *values)`` tuples returned by
hip-python's runtime functions and raises :class:`HIPError` on
non-success status.

All shim modules go through :func:`check_hip` so the failure surface is
uniform and the shim itself never silently ignores HIP errors.
"""

from __future__ import annotations

from hip import hip as _hip

hip = _hip


class HIPError(RuntimeError):
    """HIP runtime / driver call returned a non-success status."""

    def __init__(self, status, ctx: str = ""):
        self.status = status
        self.ctx = ctx
        msg = _format_error(status)
        super().__init__(f"HIP error{f' in {ctx}' if ctx else ''}: {msg}")


def _format_error(status) -> str:
    """Best-effort textual description for a HIP error code."""
    try:
        out = _hip.hipGetErrorString(status)
    except Exception:
        return f"status={status}"
    if isinstance(out, tuple):
        out = out[-1] if len(out) > 1 else out[0]
    if isinstance(out, bytes):
        try:
            return out.decode("utf-8", errors="replace")
        except Exception:
            return repr(out)
    return str(out)


def check_hip(result, ctx: str = ""):
    """Validate the (status, *values) tuple returned by a hip-python call.

    hip-python returns either a bare status (for void-returning calls)
    or a tuple ``(status, value, ...)``. On success this helper returns
    ``None`` / ``value`` / ``tuple(values)`` accordingly; on failure it
    raises :class:`HIPError` with a textual diagnostic.
    """
    if isinstance(result, tuple):
        status = result[0]
        values = result[1:]
    else:
        status = result
        values = ()
    if int(status) != 0:
        raise HIPError(status, ctx)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values
