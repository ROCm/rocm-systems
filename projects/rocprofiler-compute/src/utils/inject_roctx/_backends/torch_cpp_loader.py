# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Load the generic torch_trace_collector.so through its plain-C ABI.

The collector intentionally has no PyTorch DT_NEEDED entry. The workload's
libtorch_cpu.so is therefore promoted to the process-wide symbol scope
before the collector is loaded.
"""

import ctypes
import os
from pathlib import Path
from typing import Dict, FrozenSet, NamedTuple, Optional

from utils.logger import console_log, console_warning
from utils.native_tool_finder import find_prebuilt_artifacts

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _THIS_DIR.parents[2]

_ARTIFACT_NAME = "torch_trace_collector.so"
_TORCH_CPU_LIBRARY_NAME = "libtorch_cpu.so"
_EXPECTED_COLLECTOR_ABI_REVISION = 1


class _TorchIdentity(NamedTuple):
    version: str
    git_version: str
    debug: bool
    uses_cxx11_abi: bool


_UNKNOWN_TORCH_IDENTITY = _TorchIdentity(
    version="",
    git_version="",
    debug=False,
    uses_cxx11_abi=False,
)


# This identity gate is a cheap necessary precondition. The native build-ID
# gate is authoritative and intentionally narrower for exact wheel artifacts.
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

_TORCH_LIBRARY_LOAD_MODE = os.RTLD_GLOBAL | os.RTLD_LAZY | os.RTLD_NODELETE
_COLLECTOR_LOAD_MODE = ctypes.RTLD_LOCAL | os.RTLD_NOW | os.RTLD_NODELETE

_torch_cpu_library: Optional[ctypes.CDLL] = None


class _CollectorStats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("installed", ctypes.c_uint32),
        ("pushes", ctypes.c_uint64),
        ("pops", ctypes.c_uint64),
        ("user_scope_pushes", ctypes.c_uint64),
        ("user_scope_pops", ctypes.c_uint64),
        ("user_scope_inherits", ctypes.c_uint64),
        ("snapshots_saved", ctypes.c_uint64),
        ("snapshots_consumed", ctypes.c_uint64),
        ("snapshots_dropped", ctypes.c_uint64),
        ("snapshots_overwritten", ctypes.c_uint64),
        ("callback_errors", ctypes.c_uint64),
        ("snapshots_pending", ctypes.c_uint64),
    ]


class CollectorUnavailableError(RuntimeError):
    """No torch_trace_collector is usable for this workload."""


class CollectorNotBuiltError(CollectorUnavailableError):
    """This installation has no generic torch_trace_collector.so."""

    def __init__(self, workload_torch_version: str) -> None:
        super().__init__(
            "torch_trace_collector was not built for this installation, so "
            f"PyTorch {workload_torch_version} cannot be traced."
        )


class UnsupportedTorchVersionError(CollectorUnavailableError):
    """The workload PyTorch build has not passed native ABI validation."""

    def __init__(self, identity: _TorchIdentity) -> None:
        validated_versions = sorted(build.version for build in _VALIDATED_TORCH_BUILDS)
        super().__init__(
            "torch_trace_collector does not support this PyTorch build "
            f"(version={identity.version or 'unknown'}, "
            f"git={identity.git_version or 'unknown'}, debug={identity.debug}, "
            f"cxx11_abi={identity.uses_cxx11_abi}). Validated builds: "
            f"{', '.join(validated_versions)}."
        )


class CollectorLoadError(CollectorUnavailableError):
    """The generic collector exists but could not be loaded safely."""

    def __init__(self, so_path: Path, cause: Exception) -> None:
        super().__init__(f"torch_trace_collector at {so_path} failed to load: {cause}")


class TorchTraceCollector:
    """Python facade preserving the former extension module's method API."""

    def __init__(self, library: ctypes.CDLL, path: Path) -> None:
        self._library = library
        self._path = path
        self._bind_interface()

    def _bind_interface(self) -> None:
        self._library.torch_trace_collector_abi_revision.argtypes = []
        self._library.torch_trace_collector_abi_revision.restype = ctypes.c_uint32
        revision = self._library.torch_trace_collector_abi_revision()
        if revision != _EXPECTED_COLLECTOR_ABI_REVISION:
            raise RuntimeError(
                "torch_trace_collector has incompatible interface revision "
                f"{revision}; expected {_EXPECTED_COLLECTOR_ABI_REVISION}"
            )

        self._library.torch_trace_collector_install.argtypes = []
        self._library.torch_trace_collector_install.restype = ctypes.c_int32
        self._library.torch_trace_collector_uninstall.argtypes = []
        self._library.torch_trace_collector_uninstall.restype = ctypes.c_int32
        self._library.torch_trace_collector_is_installed.argtypes = []
        self._library.torch_trace_collector_is_installed.restype = ctypes.c_int32
        self._library.torch_trace_collector_push_user_scope.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self._library.torch_trace_collector_push_user_scope.restype = ctypes.c_int32
        self._library.torch_trace_collector_pop_user_scope.argtypes = []
        self._library.torch_trace_collector_pop_user_scope.restype = ctypes.c_int32
        self._library.torch_trace_collector_get_stats.argtypes = [
            ctypes.POINTER(_CollectorStats)
        ]
        self._library.torch_trace_collector_get_stats.restype = ctypes.c_int32

    def install(self) -> None:
        """Install the global RecordFunction callbacks."""
        self._require_success(
            self._library.torch_trace_collector_install(),
            "install",
        )

    def uninstall(self) -> None:
        """Remove the global RecordFunction callbacks."""
        self._require_success(
            self._library.torch_trace_collector_uninstall(),
            "uninstall",
        )

    def is_installed(self) -> bool:
        """Return whether the global RecordFunction callbacks are installed."""
        return self._library.torch_trace_collector_is_installed() == 1

    def push_user_scope(self, marker: str, context: str, backend: str = "") -> None:
        """Push one user scope using UTF-8 strings."""
        result = self._library.torch_trace_collector_push_user_scope(
            marker.encode("utf-8"),
            context.encode("utf-8"),
            backend.encode("utf-8"),
        )
        self._require_success(result, "push_user_scope")

    def pop_user_scope(self) -> None:
        """Pop the most recent user scope on this thread."""
        self._require_success(
            self._library.torch_trace_collector_pop_user_scope(),
            "pop_user_scope",
        )

    def dump_stats(self) -> Dict[str, object]:
        """Return a snapshot of the native collector counters."""
        stats = _CollectorStats()
        stats.struct_size = ctypes.sizeof(_CollectorStats)
        self._require_success(
            self._library.torch_trace_collector_get_stats(ctypes.byref(stats)),
            "get_stats",
        )
        return {
            "installed": bool(stats.installed),
            "pushes": stats.pushes,
            "pops": stats.pops,
            "user_scope_pushes": stats.user_scope_pushes,
            "user_scope_pops": stats.user_scope_pops,
            "user_scope_inherits": stats.user_scope_inherits,
            "snapshots_saved": stats.snapshots_saved,
            "snapshots_consumed": stats.snapshots_consumed,
            "snapshots_dropped": stats.snapshots_dropped,
            "snapshots_overwritten": stats.snapshots_overwritten,
            "callback_errors": stats.callback_errors,
            "snapshots_pending": stats.snapshots_pending,
        }

    def _require_success(self, result: int, operation: str) -> None:
        if result != 0:
            raise RuntimeError(
                f"torch_trace_collector {operation} failed for {self._path}"
            )


def _workload_torch_identity() -> _TorchIdentity:
    """Return the exact workload Torch identity used by the ABI allowlist."""
    try:
        import torch

        return _TorchIdentity(
            version=str(torch.__version__),
            git_version=str(getattr(torch.version, "git_version", "")),
            debug=bool(torch.version.debug),
            uses_cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        )
    except Exception as error:
        console_warning(
            "ml api trace",
            "Could not determine the PyTorch build identity "
            f"({type(error).__name__}: {error}).",
        )
        return _UNKNOWN_TORCH_IDENTITY


def _discover_collector_artifact() -> Optional[Path]:
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


def load() -> TorchTraceCollector:
    """Load the generic collector for an explicitly validated Torch build."""
    global _torch_cpu_library

    identity = _workload_torch_identity()
    if identity not in _VALIDATED_TORCH_BUILDS:
        raise UnsupportedTorchVersionError(identity)

    so_path = _discover_collector_artifact()
    if so_path is None:
        raise CollectorNotBuiltError(identity.version)

    try:
        if _torch_cpu_library is None:
            _torch_cpu_library = _promote_torch_cpu()
        library = ctypes.PyDLL(str(so_path), mode=_COLLECTOR_LOAD_MODE)
        collector = TorchTraceCollector(library, so_path)
    except Exception as error:
        raise CollectorLoadError(so_path, error) from error

    console_log("ml api trace", f"loaded prebuilt .so: {so_path}")
    return collector
