# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Source fingerprint for the torch_trace_collector extension."""

from __future__ import annotations

import hashlib
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
# parents[2] is <repo>/src or <install>/libexec/<project>.
_NATIVE_TOOL_ROOT = _THIS_DIR.parents[2]
_SO_SOURCE_DIR = _NATIVE_TOOL_ROOT / "lib" / "torch_trace_collector"
_SO_BUILDFILE = _SO_SOURCE_DIR / "CMakeLists.txt"
_SHARED_UTILS_HEADERS = (
    _NATIVE_TOOL_ROOT / "lib" / "utils" / "synchronized" / "synchronized.hpp",
    _NATIVE_TOOL_ROOT / "lib" / "utils" / "gsl_assert" / "gsl_assert.h",
)
_COLLECTOR_SOURCE_NAMES = (
    "torch_trace_collector.cpp",
    "torch_trace_collector_module.cpp",
)
_COLLECTOR_HEADER_NAMES = (
    "leaf_context.h",
    "marker_stack.h",
    "process_state.h",
    "record_function_callback.h",
    "record_function_installation.h",
    "scope_guard.h",
    "snapshot_store.h",
    "stack_entry.h",
    "stats.h",
    "user_scope.h",
    "wire_format.h",
)


def required_input_paths() -> tuple[Path, ...]:
    """Named collector sources and headers, CMakeLists.txt, and shared headers."""
    return (
        *(_SO_SOURCE_DIR / name for name in _COLLECTOR_SOURCE_NAMES),
        *(_SO_SOURCE_DIR / name for name in _COLLECTOR_HEADER_NAMES),
        _SO_BUILDFILE,
        *_SHARED_UTILS_HEADERS,
    )


def fingerprint_input_paths() -> tuple[Path, ...]:
    """Collector ``*.cpp`` and ``*.h`` files, CMakeLists.txt, and shared headers."""
    inputs = set(_SO_SOURCE_DIR.glob("*.cpp")) | set(_SO_SOURCE_DIR.glob("*.h"))
    inputs |= set(_SHARED_UTILS_HEADERS)
    inputs.add(_SO_BUILDFILE)
    return tuple(sorted(inputs))


def source_fingerprint() -> str:
    """First 12 hex digits of a SHA-256 of ``<basename>:<file-sha256>``
    lines, or ``"missing"``."""
    catalog_hash = hashlib.sha256()
    n_files = 0
    for path in fingerprint_input_paths():
        try:
            data = path.read_bytes()
        except OSError:
            continue
        digest = hashlib.sha256(data).hexdigest()
        catalog_hash.update(f"{path.name}:{digest}\n".encode("ascii"))
        n_files += 1
    if n_files == 0:
        return "missing"
    return catalog_hash.hexdigest()[:12]
