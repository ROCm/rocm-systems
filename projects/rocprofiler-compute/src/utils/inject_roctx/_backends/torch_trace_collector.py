# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from __future__ import annotations

import ctypes
import re
from pathlib import Path
from typing import Optional

from utils.logger import console_error, console_log, console_warning
from utils.native_tool_finder import find_prebuilt_artifacts

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _THIS_DIR.parents[2]

_ARTIFACT_NAME_GLOB = "torch_trace_collector-*.so"
_ARTIFACT_NAME_PATTERN = re.compile(r"^torch_trace_collector-(\d+\.\d+)\.so$")

_lib: Optional[ctypes.CDLL] = None


def _workload_torch_version() -> str:
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


def _collectors_by_version() -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for path in find_prebuilt_artifacts(_PACKAGE_ROOT, _ARTIFACT_NAME_GLOB):
        match = _ARTIFACT_NAME_PATTERN.match(path.name)
        if match is None:
            continue
        version = match.group(1)
        if version not in artifacts:
            artifacts[version] = path
    return artifacts


def install() -> bool:
    global _lib
    if _lib is not None:
        return True

    workload_torch_version = _workload_torch_version()
    collectors_by_version = _collectors_by_version()
    if not collectors_by_version:
        console_warning(
            "ml api trace",
            "torch_trace_collector was not built for this installation; "
            "using TorchDispatchMode.",
        )
        return False

    so_path = collectors_by_version.get(workload_torch_version)
    if so_path is None:
        console_error(
            "ml api trace",
            "torch_trace_collector has no prebuilt extension for PyTorch "
            f"{workload_torch_version}. Supported PyTorch versions: "
            f"{', '.join(sorted(collectors_by_version))}.",
        )

    try:
        lib = ctypes.CDLL(str(so_path))
        lib.torch_trace_collector_install.restype = ctypes.c_int
        lib.torch_trace_collector_install.argtypes = []
        if lib.torch_trace_collector_install() != 0:
            console_warning(
                "ml api trace",
                f"torch_trace_collector_install failed for {so_path}; "
                "using TorchDispatchMode.",
            )
            return False
    except Exception as exc:
        console_warning(
            "ml api trace",
            "C++ RecordFunction tier unavailable "
            f"({type(exc).__name__}: {exc}); using TorchDispatchMode.",
        )
        return False

    _lib = lib
    console_log("ml api trace", f"loaded prebuilt .so: {so_path}")
    return True
