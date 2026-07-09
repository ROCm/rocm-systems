# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Backend registry for inject_roctx.

Backends are discovered by importing _backends.<name> on demand. Each
backend module must call register at import time.
"""

import importlib
from collections.abc import Iterable
from typing import Protocol

_BACKENDS_PKG = f"{__package__}._backends"


class Backend(Protocol):
    name: str

    def install(self) -> None: ...


_REGISTRY: dict[str, Backend] = {}


def register(backend: Backend) -> None:
    """Register backend under backend.name."""
    _REGISTRY[backend.name] = backend


def install_many(names: Iterable[str]) -> None:
    """Install each backend in names.

    Unknown or failing entries are warned and skipped.
    """
    from utils.logger import console_warning

    seen: set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            if name not in _REGISTRY:
                importlib.import_module(f".{name}", _BACKENDS_PKG)
            if name not in _REGISTRY:
                raise RuntimeError(
                    f"module {_BACKENDS_PKG}.{name} loaded "
                    f"but did not register {name!r}"
                )
            _REGISTRY[name].install()
        except Exception as exc:
            console_warning(
                "inject_roctx",
                f"backend {name!r} install failed; continuing: {exc}",
            )
