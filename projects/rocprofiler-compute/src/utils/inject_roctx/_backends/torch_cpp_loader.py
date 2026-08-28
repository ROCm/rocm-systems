# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Load the ``torch_trace_collector`` extension for the workload PyTorch version.

Searches ``<prefix>/lib*/rocprofiler-compute/``, ``<project>/build/lib``,
``src/lib/_build/lib``, and ``$CMAKE_BINARY_DIR/lib`` when ``CMAKE_BINARY_DIR``
is set.
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
_ARTIFACT_NAME_PATTERN = re.compile(
    r"^"
    + re.escape(_ARTIFACT_PREFIX)
    + r"(\d+[^/]*)"
    + re.escape(_ARTIFACT_SUFFIX)
    + r"$"
)


class UnsupportedTorchVersionError(RuntimeError):
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
    """Return ``torch.__version__`` without a local ``+...`` suffix."""
    try:
        import torch

        return torch.__version__.split("+", 1)[0]
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
