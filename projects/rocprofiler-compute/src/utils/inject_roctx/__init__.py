# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Public API for the inject_roctx package.

Exposes install_global_wraps, the scope push and pop primitives, and the
KNOWN_BACKENDS registry. Backend modules are imported on first use.
"""

from collections.abc import Iterable
from typing import Any, Union

from ._core import (
    _pop_scope,
    _push_scope,
    resolve_user_caller_location,
)

__all__ = [
    "KNOWN_BACKENDS",
    "_pop_scope",
    "_push_scope",
    "install_global_wraps",
    "resolve_user_caller_location",
]

# Supported backend names.
KNOWN_BACKENDS: tuple[str, ...] = ("torch", "triton")

# The "api" alias selects every supported backend.
_API_ALIAS = "api"


_LAZY_TORCH_SYMBOLS: frozenset[str] = frozenset({
    "using_c_tier",
    "dump_recordfn_stats",
    "install_function_apply_wrappers",
})


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazily re-export torch-backend symbols, deferring the PyTorch import."""
    if name in _LAZY_TORCH_SYMBOLS:
        from ._backends._torch import (
            dump_recordfn_stats,
            install_function_apply_wrappers,
            using_c_tier,
        )

        resolved = {
            "using_c_tier": using_c_tier,
            "dump_recordfn_stats": dump_recordfn_stats,
            "install_function_apply_wrappers": install_function_apply_wrappers,
        }
        globals().update(resolved)
        return resolved[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def install_global_wraps(backends: Union[str, Iterable[str]] = "") -> None:
    """Install ROCTX instrumentation for each backend in backends.

    "api" expands to every known backend. Empty input is a no-op.
    """
    from ._backends import install_many

    if isinstance(backends, str):
        names = [n.strip() for n in backends.split(",") if n.strip()]
    else:
        names = [str(n).strip() for n in backends if str(n).strip()]

    expanded: list[str] = []
    for n in names:
        if n == _API_ALIAS:
            expanded.extend(KNOWN_BACKENDS)
        else:
            expanded.append(n)

    if not expanded:
        return
    install_many(expanded)
