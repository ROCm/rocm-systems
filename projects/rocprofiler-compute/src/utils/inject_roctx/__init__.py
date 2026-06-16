# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Public API for the inject_roctx package.

Exposes install_global_wraps, the scope push and pop primitives, and the
KNOWN_BACKENDS registry. Backend modules are imported on first use.
"""

from collections.abc import Iterable
from typing import Union

from ._core import (
    _pop_scope,
    _push_scope,
    resolve_user_caller_location,
)
from .constants import KNOWN_BACKENDS

__all__ = [
    "KNOWN_BACKENDS",
    "_pop_scope",
    "_push_scope",
    "install_global_wraps",
    "resolve_user_caller_location",
]

# The "api" alias selects every supported backend.
_API_ALIAS = "api"


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
