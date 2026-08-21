# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Artifact tag for the torch_trace_collector extension.

The tag names one build of the extension: the Python and torch versions it was
compiled against, and a fingerprint of the C++ inputs it was compiled from. The
loader uses it to find a matching artifact and CMake uses it to name the one it
produces, so both sides call ``artifact_tag`` rather than assembling their own.

Importing this module must not import torch: the loader imports it in
environments where torch is absent. ``artifact_tag`` imports torch when called.
"""

import hashlib
import sys
from pathlib import Path
from typing import Optional, Tuple

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

MISSING_FINGERPRINT = "missing"


def required_input_paths() -> Tuple[Path, ...]:
    """Named collector sources and headers, CMakeLists.txt, and shared headers."""
    return (
        *(_SO_SOURCE_DIR / name for name in _COLLECTOR_SOURCE_NAMES),
        *(_SO_SOURCE_DIR / name for name in _COLLECTOR_HEADER_NAMES),
        _SO_BUILDFILE,
        *_SHARED_UTILS_HEADERS,
    )


def fingerprint_input_paths() -> Tuple[Path, ...]:
    """Collector ``*.cpp`` and ``*.h`` files, CMakeLists.txt, and shared headers."""
    inputs = set(_SO_SOURCE_DIR.glob("*.cpp")) | set(_SO_SOURCE_DIR.glob("*.h"))
    inputs |= set(_SHARED_UTILS_HEADERS)
    inputs.add(_SO_BUILDFILE)
    return tuple(sorted(inputs))


def source_fingerprint() -> str:
    """First 12 hex digits of a SHA-256 over ``<name>:<file-sha256>`` lines.

    Returns ``MISSING_FINGERPRINT`` when no input is readable, so a tree with no
    sources produces a tag that cannot collide with a real build.
    """
    catalog_hash = hashlib.sha256()
    unreadable = []
    n_files = 0
    for path in fingerprint_input_paths():
        try:
            data = path.read_bytes()
        except OSError:
            unreadable.append(path)
            continue
        digest = hashlib.sha256(data).hexdigest()
        catalog_hash.update(f"{path.name}:{digest}\n".encode("ascii"))
        n_files += 1
    # A tag built from a subset of the inputs would name an artifact that does
    # not correspond to the sources it was built from.
    if unreadable or n_files == 0:
        return MISSING_FINGERPRINT
    return catalog_hash.hexdigest()[:12]


def torch_version() -> Optional[str]:
    """The running interpreter's torch version without its local segment.

    CMake reads the same value from ``Torch_VERSION``, which carries no local
    segment, so the ``+rocm``/``+cu`` suffix is dropped here to match.
    """
    try:
        import torch
    except Exception:
        return None
    return torch.__version__.split("+", 1)[0]


def artifact_tag() -> Optional[str]:
    """Return ``py{major}.{minor}_torch{version}_src{fingerprint}``, or ``None``
    when torch is not importable.
    """
    version = torch_version()
    if version is None:
        return None
    return (
        f"py{sys.version_info.major}.{sys.version_info.minor}"
        f"_torch{version}_src{source_fingerprint()}"
    )


if __name__ == "__main__":
    # CMake resolves the tag by running this module; see the collector's
    # CMakeLists.txt. An empty line means torch was not importable.
    #
    # Running a file as a script puts its own directory first on sys.path, and
    # this one holds torch.py, which would shadow the real torch package.
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _THIS_DIR]
    print(artifact_tag() or "")
