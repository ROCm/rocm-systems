# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Load the ``torch_trace_collector`` extension for the workload PyTorch version.

Searches ``<prefix>/lib*/rocprofiler-compute/``, ``<project>/build/lib``, and
``src/lib/_build/lib``.
"""

import importlib.util
import re
import types
from pathlib import Path
from typing import Dict, Tuple

from utils.logger import console_error, console_log
from utils.native_tool_finder import find_prebuilt_artifacts

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _THIS_DIR.parents[2]

_ARTIFACT_PREFIX = "torch_trace_collector-"
_ARTIFACT_SUFFIX = ".so"
_ARTIFACT_NAME_GLOB = f"{_ARTIFACT_PREFIX}*{_ARTIFACT_SUFFIX}"
# torch_trace_collector-<major>.<minor>.<cpython abi tag>.so
_ARTIFACT_NAME_PATTERN = re.compile(
    r"^"
    + re.escape(_ARTIFACT_PREFIX)
    + r"(\d+\.\d+)\.(.+)"
    + re.escape(_ARTIFACT_SUFFIX)
    + r"$"
)


class CollectorUnavailableError(RuntimeError):
    """No ``torch_trace_collector`` is usable for this workload."""


class CollectorNotBuiltError(CollectorUnavailableError):
    """This installation has no ``torch_trace_collector`` at all."""

    def __init__(self, workload_torch_version: str) -> None:
        super().__init__(
            "torch_trace_collector was not built for this installation, so "
            f"PyTorch {workload_torch_version} cannot be traced."
        )


class UnsupportedTorchVersionError(CollectorUnavailableError):
    """No ``torch_trace_collector`` matches the workload PyTorch version."""

    def __init__(
        self,
        workload_torch_version: str,
        supported_torch_versions: Tuple[str, ...],
    ) -> None:
        super().__init__(
            "torch_trace_collector has no prebuilt extension for PyTorch "
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


def list_collector_artifacts() -> Dict[str, Path]:
    """Return ``{torch version: collector path}`` for each matching ``.so``."""
    artifacts: Dict[str, Path] = {}
    for path in find_prebuilt_artifacts(_PACKAGE_ROOT, _ARTIFACT_NAME_GLOB):
        match = _ARTIFACT_NAME_PATTERN.match(path.name)
        if match is None:
            continue
        version = match.group(1)
        if version not in artifacts:
            artifacts[version] = path
    return artifacts


def load() -> types.ModuleType:
    """Resolve the ``torch_trace_collector`` module."""
    workload_torch_version = torch_version()
    collectors_by_version = list_collector_artifacts()
    if not collectors_by_version:
        raise CollectorNotBuiltError(workload_torch_version)
    so_path = collectors_by_version.get(workload_torch_version)
    if so_path is None:
        raise UnsupportedTorchVersionError(
            workload_torch_version,
            tuple(sorted(collectors_by_version)),
        )

    try:
        spec = importlib.util.spec_from_file_location(
            "torch_trace_collector", str(so_path)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        console_error(
            "ml api trace",
            f"failed to load torch_trace_collector for PyTorch "
            f"{workload_torch_version} from {so_path}: {exc}",
        )

    console_log("ml api trace", f"loaded prebuilt .so: {so_path}")
    return module
