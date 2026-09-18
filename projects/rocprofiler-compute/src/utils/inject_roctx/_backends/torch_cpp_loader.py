# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Load the ``torch_trace_collector`` extension.

Searches ``<prefix>/lib*/rocprofiler-compute/``, ``<project>/build/lib``, and
``src/lib/_build/lib``. The extension links no PyTorch library, so libtorch is
reopened with ``RTLD_GLOBAL`` before it loads.
"""

import ctypes
import importlib.util
import os
import types
from pathlib import Path
from typing import List, Tuple

from utils.logger import console_error, console_log
from utils.native_tool_finder import find_prebuilt_artifacts

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _THIS_DIR.parents[2]

_MODULE_NAME = "torch_trace_collector"
_ARTIFACT_NAME_GLOB = f"{_MODULE_NAME}.*.so"

# In dependency order.
_TORCH_LIBRARIES = ("libc10.so", "libtorch_cpu.so")


class CollectorUnavailableError(RuntimeError):
    """No ``torch_trace_collector`` is usable for this workload."""


class CollectorNotBuiltError(CollectorUnavailableError):
    """This installation has no ``torch_trace_collector`` at all."""

    def __init__(self, workload_torch_version: str) -> None:
        super().__init__(
            "torch_trace_collector was not built for this installation, so "
            f"PyTorch {workload_torch_version} cannot be traced."
        )


class CollectorLoadError(CollectorUnavailableError):
    """The extension could not be loaded."""

    def __init__(self, so_path: Path, cause: Exception) -> None:
        super().__init__(f"torch_trace_collector at {so_path} failed to load: {cause}")


class UnsupportedTorchVersionError(CollectorUnavailableError):
    """The extension does not support the workload PyTorch version."""

    def __init__(
        self,
        workload_torch_version: str,
        supported_torch_versions: Tuple[str, ...],
    ) -> None:
        super().__init__(
            "torch_trace_collector was not built for PyTorch "
            f"{workload_torch_version}. Supported PyTorch versions: "
            f"{', '.join(supported_torch_versions)}."
        )


def torch_version() -> str:
    """Return the workload PyTorch version as ``<major>.<minor>``."""
    try:
        import torch
        from torch.torch_version import Version

        release = Version(torch.__version__).release
        return f"{release[0]}.{release[1]}"
    except Exception as exc:
        console_error(
            "ml api trace",
            f"torch is not importable; cannot profile a torch workload: {exc}",
        )


def list_collector_artifacts() -> List[Path]:
    """Return every collector artifact found on the search path."""
    return find_prebuilt_artifacts(_PACKAGE_ROOT, _ARTIFACT_NAME_GLOB)


def publish_torch_symbols() -> None:
    """Reopen libtorch so its ATen symbols resolve globally."""
    import torch

    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    for library in _TORCH_LIBRARIES:
        ctypes.CDLL(str(torch_lib_dir / library), mode=os.RTLD_GLOBAL | os.RTLD_LAZY)


def load() -> types.ModuleType:
    """Resolve the ``torch_trace_collector`` module."""
    workload_torch_version = torch_version()
    artifacts = list_collector_artifacts()
    if not artifacts:
        raise CollectorNotBuiltError(workload_torch_version)
    so_path = artifacts[0]

    # torch.py falls back to the Python tier on any CollectorUnavailableError.
    try:
        publish_torch_symbols()
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, str(so_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        raise CollectorLoadError(so_path, exc) from exc

    supported = tuple(module.supported_torch_versions)
    if workload_torch_version not in supported:
        raise UnsupportedTorchVersionError(workload_torch_version, supported)

    console_log("ml api trace", f"loaded prebuilt .so: {so_path}")
    return module
