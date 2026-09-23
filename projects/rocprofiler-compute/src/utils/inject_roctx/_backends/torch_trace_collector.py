# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import ctypes
import os
import threading
from pathlib import Path
from typing import FrozenSet, NamedTuple, Optional

from utils.logger import console_log, console_warning
from utils.native_tool_finder import find_prebuilt_artifacts

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _THIS_DIR.parents[2]

_ARTIFACT_NAME = "torch_trace_collector.so"
_TORCH_CPU_LIBRARY_NAME = "libtorch_cpu.so"
_EXPECTED_COLLECTOR_ABI_REVISION = 2


class _TorchIdentity(NamedTuple):
    version: str
    git_version: str
    debug: bool
    uses_cxx11_abi: bool


# This wheel-only gate is intentionally narrower than the native build-ID
# allowlist, which also covers standalone CPU libtorch archives used by direct
# C++ validation harnesses.
_VALIDATED_TORCH_BUILDS: FrozenSet[_TorchIdentity] = frozenset({
    _TorchIdentity(
        version="2.13.0+cpu",
        git_version="cf30153c4c131c8164ee7798e5022d810682e2cb",
        debug=False,
        uses_cxx11_abi=True,
    ),
    _TorchIdentity(
        version="2.14.0+rocm10.2.0a20260917",
        git_version="50011224068dd1fbc41573a6e43c9d9ffa053a63",
        debug=False,
        uses_cxx11_abi=True,
    ),
})
# PyTorch retains callback pointers for the lifetime of the process.
_TORCH_LIBRARY_LOAD_MODE = os.RTLD_GLOBAL | os.RTLD_LAZY | os.RTLD_NODELETE
_COLLECTOR_LOAD_MODE = ctypes.RTLD_LOCAL | os.RTLD_NOW | os.RTLD_NODELETE

_torch_cpu_library: Optional[ctypes.CDLL] = None
_collector_library: Optional[ctypes.CDLL] = None


def install() -> bool:
    """Load and install the native collector, or select the Python fallback."""
    global _collector_library, _torch_cpu_library
    if _collector_library is not None:
        return True

    identity = _workload_torch_identity()
    if not _validate_identity(identity):
        return False

    try:
        collector_path = _find_collector()
        if collector_path is None:
            console_warning(
                "ml api trace",
                "torch_trace_collector was not built for this installation; "
                "using TorchDispatchMode.",
            )
            return False
        if _torch_cpu_library is None:
            _torch_cpu_library = _promote_torch_cpu()
        collector = ctypes.CDLL(str(collector_path), mode=_COLLECTOR_LOAD_MODE)
        if not _validate_collector_interface(collector):
            return False
        if not _install_collector_callback(collector, collector_path):
            return False
    except Exception as error:
        console_warning(
            "ml api trace",
            "C++ RecordFunction tier unavailable "
            f"({type(error).__name__}: {error}); using TorchDispatchMode.",
        )
        return False

    _collector_library = collector
    console_log("ml api trace", f"loaded prebuilt .so: {collector_path}")
    return True


def _validate_identity(identity: _TorchIdentity) -> bool:
    """Return whether identity is allowlisted, warning on an unsupported build."""
    if not identity.version:
        return False
    if identity in _VALIDATED_TORCH_BUILDS:
        return True

    validated_versions = sorted(build.version for build in _VALIDATED_TORCH_BUILDS)
    console_warning(
        "ml api trace",
        "native torch_trace_collector does not support this PyTorch build "
        f"(version={identity.version}, git={identity.git_version or 'unknown'}, "
        f"debug={identity.debug}, cxx11_abi={identity.uses_cxx11_abi}); using "
        "TorchDispatchMode. "
        f"Validated builds: {', '.join(validated_versions)}.",
    )
    return False


def _validate_collector_interface(collector: ctypes.CDLL) -> bool:
    """Validate the collector's plain-C interface revision."""
    collector.torch_trace_collector_abi_revision.restype = ctypes.c_uint32
    collector.torch_trace_collector_abi_revision.argtypes = []
    revision = collector.torch_trace_collector_abi_revision()
    if revision == _EXPECTED_COLLECTOR_ABI_REVISION:
        collector.torch_trace_collector_push_launcher_tid.restype = ctypes.c_int
        collector.torch_trace_collector_push_launcher_tid.argtypes = [ctypes.c_uint64]
        collector.torch_trace_collector_pop_launcher_tid.restype = ctypes.c_int
        collector.torch_trace_collector_pop_launcher_tid.argtypes = []
        return True

    console_warning(
        "ml api trace",
        "torch_trace_collector has an incompatible interface revision "
        f"{revision}; expected {_EXPECTED_COLLECTOR_ABI_REVISION}. "
        "Using TorchDispatchMode.",
    )
    return False


def _install_collector_callback(collector: ctypes.CDLL, path: Path) -> bool:
    """Install the collector callback and report a native rejection."""
    collector.torch_trace_collector_install.restype = ctypes.c_int
    collector.torch_trace_collector_install.argtypes = []
    if collector.torch_trace_collector_install() == 0:
        return True

    console_warning(
        "ml api trace",
        f"torch_trace_collector_install failed for {path}; using TorchDispatchMode.",
    )
    return False


def _workload_torch_identity() -> _TorchIdentity:
    """Return the workload Torch version, revision, debug, and C++ ABI modes."""
    try:
        import torch

        version = str(torch.__version__)
        git_version = str(getattr(torch.version, "git_version", ""))
        debug = bool(torch.version.debug)
        uses_cxx11_abi = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
        return _TorchIdentity(
            version=version,
            git_version=git_version,
            debug=debug,
            uses_cxx11_abi=uses_cxx11_abi,
        )
    except Exception as error:
        console_warning(
            "ml api trace",
            "could not determine the PyTorch build identity; using "
            "TorchDispatchMode "
            f"({type(error).__name__}: {error}).",
        )
        return _TorchIdentity(
            version="",
            git_version="",
            debug=False,
            uses_cxx11_abi=False,
        )


def _find_collector() -> Optional[Path]:
    """Return the generic native collector, when this installation provides it."""
    artifacts = find_prebuilt_artifacts(_PACKAGE_ROOT, _ARTIFACT_NAME)
    return artifacts[0] if artifacts else None


def _promote_torch_cpu() -> ctypes.CDLL:
    """Expose the workload's libtorch_cpu symbols to the native collector."""
    import torch

    torch_library = (
        Path(torch.__file__).resolve().parent / "lib" / _TORCH_CPU_LIBRARY_NAME
    )
    return ctypes.CDLL(str(torch_library), mode=_TORCH_LIBRARY_LOAD_MODE)


def push_launcher_tid() -> bool:
    """Publish the current native thread ID for autograd worker correlation."""
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
        return False


def pop_launcher_tid() -> bool:
    """Remove one successfully published launcher thread ID."""
    if _collector_library is None:
        return False
    try:
        return _collector_library.torch_trace_collector_pop_launcher_tid() == 0
    except Exception:
        return False
