# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Backend contract and the single registry for ML API tracing recording."""

from typing import Dict, Optional, Protocol

# Annotations name Backend as a string because the class is defined last.
_REGISTRY: Dict[str, "Backend"] = {}


def register(backend: "Backend") -> None:
    """Register a backend, replacing any backend with the same name."""
    _REGISTRY[backend.name] = backend


def get_backend(name: str) -> Optional["Backend"]:
    """Return the registered backend, or None when discovery is needed."""
    return _REGISTRY.get(name)


class Backend(Protocol):
    name: str

    def install(self) -> None: ...
