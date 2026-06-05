# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reader registry for rocprofsys_validator.

Phase 1: provides a no-op @reader decorator and the _READER_REGISTRY dict.
Phase 5: discover_validators() loads entry-pointed third-party validators lazily
on the first expect()/expect_all() call, using a module-level _discovered flag
for idempotency and RuntimeWarning for non-blocking failure reporting.
"""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points
from typing import Callable, TypeVar

_T = TypeVar("_T")

_READER_REGISTRY: dict[str, type] = {}
_discovered: bool = False

__all__ = ["reader", "_READER_REGISTRY", "register_validator", "discover_validators", "_discovered"]

def reader(format_name: str) -> Callable[[type[_T]], type[_T]]:
    """Register a FormatReader subclass in the global registry.

    Phase 1: no-op decorator — stores cls in _READER_REGISTRY.
    Phase 5: entry-point discovery is triggered lazily from expect(), not here.

    Args:
        format_name: The unique string key for this format reader
                     (e.g., "perfetto", "rocpd", "timemory").

    Returns:
        A decorator that registers and returns the decorated class unchanged.

    Example::

        @reader("perfetto")
        class PerfettoReader(FormatReader):
            ...
    """

    def decorator(cls: type[_T]) -> type[_T]:
        _READER_REGISTRY[format_name] = cls  # type: ignore[assignment]
        return cls

    return decorator

register_validator = reader

def discover_validators(_stacklevel: int = 2) -> None:
    """Discover and register third-party validators from entry points.

    Idempotent — safe to call multiple times; only runs once per process.
    Failed imports emit RuntimeWarning and continue; never blocks caller.

    Entry-point group: ``rocprofsys_validator.validators``

    Args:
        _stacklevel: Passed to ``warnings.warn`` to attribute warnings correctly.
            Default 2 (direct caller); pass 4 when called through expect()/expect_all()
            so the warning surfaces at the user's call site.

    Third-party pyproject.toml::

        [project.entry-points."rocprofsys_validator.validators"]
        my_format = "mypackage.readers:MyFormatReader"
    """
    global _discovered
    if _discovered:
        return
    _discovered = True          # set BEFORE the loop — interrupt-safe
    from rocprofsys_validator.core import FormatReader  # once, outside loop; avoids circular import
    for ep in entry_points(group="rocprofsys_validator.validators"):
        try:
            cls = ep.load()
            # T-05-01: guard against non-FormatReader-subclass entry points
            if not (isinstance(cls, type) and issubclass(cls, FormatReader)):
                warnings.warn(
                    f"Entry point '{ep.name}' is not a FormatReader subclass — skipping.",
                    RuntimeWarning,
                    stacklevel=_stacklevel,
                )
                continue
            reader(ep.name)(cls)
        except Exception as exc:
            warnings.warn(
                f"Failed to load validator '{ep.name}' "
                f"({type(exc).__name__}): {exc}",
                RuntimeWarning,
                stacklevel=_stacklevel,
            )
