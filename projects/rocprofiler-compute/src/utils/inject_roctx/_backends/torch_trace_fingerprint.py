# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Source fingerprint for the torch_trace_collector extension."""

from __future__ import annotations

import hashlib
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
# parents[2] resolves to <repo>/src in dev and <install>/libexec/<project>
# in installed layouts; both host the torch_trace_collector sources at lib/.
_NATIVE_TOOL_ROOT = _THIS_DIR.parents[2]
_SO_SOURCE_DIR = _NATIVE_TOOL_ROOT / "lib" / "torch_trace_collector"
_SO_BUILDFILE = _SO_SOURCE_DIR / "CMakeLists.txt"
_SHARED_UTILS_HEADERS = (
    _NATIVE_TOOL_ROOT / "lib" / "utils" / "synchronized" / "synchronized.hpp",
    _NATIVE_TOOL_ROOT / "lib" / "utils" / "gsl_assert" / "gsl_assert.h",
)


def fingerprint_input_paths() -> tuple[Path, ...]:
    """C++ sources and headers, shared utility headers, CMakeLists.txt, and
    cmake/*.py and cmake/*.cmake, sorted by path.
    """
    inputs = set(_SO_SOURCE_DIR.glob("*.cpp")) | set(_SO_SOURCE_DIR.glob("*.h"))
    inputs |= set(_SO_SOURCE_DIR.glob("cmake/*.py"))
    inputs |= set(_SO_SOURCE_DIR.glob("cmake/*.cmake"))
    inputs |= set(_SHARED_UTILS_HEADERS)
    inputs.add(_SO_BUILDFILE)
    return tuple(sorted(inputs))


def source_fingerprint() -> str:
    """First 12 hex chars of a SHA-256 over the source inputs, or ``"missing"``."""
    h = hashlib.sha256()
    seen = 0
    for path in fingerprint_input_paths():
        try:
            data = path.read_bytes()
        except OSError:
            continue
        h.update(f"{path.name}:{len(data)}\n".encode("ascii"))
        h.update(data)
        seen += 1
    if seen == 0:
        return "missing"
    return h.hexdigest()[:12]
