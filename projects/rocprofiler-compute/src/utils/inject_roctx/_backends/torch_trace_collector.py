# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Load the generic collector through its revision-2 plain-C interface.

The collector has no PyTorch link dependency. Its symbols are supplied by
the workload's real libtorch_cpu.so after checking the supported minor version.
"""

import ctypes
import os
import threading
from pathlib import Path
from typing import FrozenSet, Optional, Tuple

from utils.logger import console_log, console_warning
from utils.native_tool_finder import find_prebuilt_artifacts

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _THIS_DIR.parents[2]

_ARTIFACT_NAME = "torch_trace_collector.so"
_TORCH_CPU_LIBRARY_NAME = "libtorch_cpu.so"
_EXPECTED_COLLECTOR_ABI_REVISION = 2

# The temporary shim reads private PyTorch layouts recorded for these minors.
_SUPPORTED_TORCH_VERSIONS: FrozenSet[str] = frozenset({"2.13", "2.14"})

# PyTorch retains the callback and debug-info pointers for the process lifetime.
_TORCH_LIBRARY_LOAD_MODE = os.RTLD_GLOBAL | os.RTLD_LAZY | os.RTLD_NODELETE
_COLLECTOR_LOAD_MODE = ctypes.RTLD_LOCAL | os.RTLD_NOW | os.RTLD_NODELETE

_torch_cpu_library: Optional[ctypes.CDLL] = None
_collector_library: Optional[ctypes.CDLL] = None


def install() -> bool:
    """Install the native callback, or return False to select the Python tier."""
    global _collector_library, _torch_cpu_library
    if _collector_library is not None:
        return True

    torch_version = _workload_torch_version()
    if torch_version not in _SUPPORTED_TORCH_VERSIONS:
        console_warning(
            "ml api trace",
            "torch_trace_collector does not support PyTorch "
            f"{torch_version or 'unknown'}. Supported versions: "
            f"{', '.join(sorted(_SUPPORTED_TORCH_VERSIONS))}.",
        )
        return False

    try:
        collector_path = _find_collector()
        if collector_path is None:
            console_warning(
                "ml api trace",
                "torch_trace_collector was not built for this installation.",
            )
            return False
        if _torch_cpu_library is None:
            _torch_cpu_library = _promote_torch_cpu()
        collector = ctypes.CDLL(str(collector_path), mode=_COLLECTOR_LOAD_MODE)
        _bind_collector_interface(collector)
        if collector.torch_trace_collector_install() != 0:
            console_warning(
                "ml api trace",
                f"torch_trace_collector_install failed for {collector_path}.",
            )
            return False
    except Exception as error:
        console_warning(
            "ml api trace",
            f"Failed to load torch_trace_collector ({type(error).__name__}: {error}).",
        )
        return False

    _collector_library = collector
    console_log("ml api trace", f"Loaded {collector_path}")
    return True


def push_launcher_tid() -> bool:
    """Publish this native thread ID; only a successful push needs a pop."""
    if _collector_library is None:
        return False
    try:
        return (
            _collector_library.torch_trace_collector_push_launcher_tid(
                threading.get_native_id()
            )
            == 0
        )
    except Exception:
        # Instrumentation must not replace the workload's result or exception.
        return False


def pop_launcher_tid() -> bool:
    """Remove one successfully published launcher thread ID."""
    if _collector_library is None:
        return False
    try:
        return _collector_library.torch_trace_collector_pop_launcher_tid() == 0
    except Exception:
        return False


def _bind_collector_interface(collector: ctypes.CDLL) -> None:
    """Check the revision and bind all entry points before installing callbacks."""
    collector.torch_trace_collector_abi_revision.restype = ctypes.c_uint32
    collector.torch_trace_collector_abi_revision.argtypes = []
    revision = collector.torch_trace_collector_abi_revision()
    if revision != _EXPECTED_COLLECTOR_ABI_REVISION:
        raise RuntimeError(
            "torch_trace_collector has incompatible interface revision "
            f"{revision}; expected {_EXPECTED_COLLECTOR_ABI_REVISION}"
        )
    collector.torch_trace_collector_install.restype = ctypes.c_int32
    collector.torch_trace_collector_install.argtypes = []
    collector.torch_trace_collector_push_launcher_tid.restype = ctypes.c_int32
    collector.torch_trace_collector_push_launcher_tid.argtypes = [ctypes.c_uint64]
    collector.torch_trace_collector_pop_launcher_tid.restype = ctypes.c_int32
    collector.torch_trace_collector_pop_launcher_tid.argtypes = []


def _workload_torch_version() -> str:
    """Return the workload PyTorch version as major.minor."""
    try:
        import torch
        from torch.torch_version import Version

        release: Tuple[int, ...] = Version(torch.__version__).release
        return f"{release[0]}.{release[1]}"
    except Exception as error:
        console_warning(
            "ml api trace",
            "Could not determine the PyTorch version "
            f"({type(error).__name__}: {error}).",
        )
        return ""


def _find_collector() -> Optional[Path]:
    """Return the generic collector installed with rocprofiler-compute."""
    artifacts = find_prebuilt_artifacts(_PACKAGE_ROOT, _ARTIFACT_NAME)
    return artifacts[0] if artifacts else None


def _promote_torch_cpu() -> ctypes.CDLL:
    """Expose the workload's libtorch_cpu symbols to the native collector."""
    import torch

    torch_library = (
        Path(torch.__file__).resolve().parent / "lib" / _TORCH_CPU_LIBRARY_NAME
    )
    return ctypes.CDLL(str(torch_library), mode=_TORCH_LIBRARY_LOAD_MODE)
