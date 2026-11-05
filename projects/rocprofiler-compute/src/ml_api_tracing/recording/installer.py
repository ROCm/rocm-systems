# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Workload-time discovery and installation of ML API tracing backends.

Backends are discovered by importing backends.<name> on demand. Each
backend module must call register at import time.
"""

import importlib
from typing import Iterable, Set

from utils.logger import console_warning

from .backend_contract import get_backend

_BACKENDS_PKG = f"{__package__}.backends"


def install_global_wraps(backends: Iterable[str] = ()) -> None:
    """Install ROCTX instrumentation for each named backend.

    backends is an iterable of framework names (e.g. ["torch"]).
    Empty input is a no-op.
    """
    names = [str(n).strip() for n in backends if str(n).strip()]
    if not names:
        return
    install_many(names)


def install_many(names: Iterable[str]) -> None:
    """Install each named backend.

    Unknown or failing backends are skipped after a warning.
    """
    seen: Set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            backend = get_backend(name)
            if backend is None:
                importlib.import_module(f".{name}", _BACKENDS_PKG)
                backend = get_backend(name)
            if backend is None:
                raise RuntimeError(
                    f"module {_BACKENDS_PKG}.{name} loaded "
                    f"but did not register {name!r}"
                )
            backend.install()
        except Exception as exc:
            console_warning(
                "inject_roctx",
                f"backend {name!r} install failed; continuing: {exc}",
            )
