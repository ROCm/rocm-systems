# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Resolve and load the ``torch_trace_collector`` pybind11 extension.

Selects the installed ``torch_trace_collector-<torch-version>.so`` that matches
the workload PyTorch version.
"""

import importlib.util
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent

_INSTALL_TREE_PROJECT_NAME = "rocprofiler-compute"
_ARTIFACT_PREFIX = "torch_trace_collector-"
_ARTIFACT_SUFFIX = ".so"
_ARTIFACT_NAME_PATTERN = re.compile(
    r"^" + re.escape(_ARTIFACT_PREFIX) + r"(.+)" + re.escape(_ARTIFACT_SUFFIX) + r"$"
)

TIER_PREBUILT = "prebuilt"
C_TIER_NAMES = frozenset((TIER_PREBUILT,))

_Diagnostics = List[Tuple[str, str]]


class UnsupportedTorchVersionError(RuntimeError):
    """Raised when no installed collector matches the workload PyTorch version."""

    def __init__(
        self,
        workload_torch_version: str,
        supported_torch_versions: Tuple[str, ...],
    ) -> None:
        self.workload_torch_version = workload_torch_version
        self.supported_torch_versions = supported_torch_versions
        if supported_torch_versions:
            supported = ", ".join(supported_torch_versions)
        else:
            supported = "none"
        super().__init__(
            "torch_trace_collector has no prebuilt extension for PyTorch "
            f"{workload_torch_version}. Supported PyTorch versions: {supported}."
        )


@dataclass
class LoadResult:
    """Outcome of a ``load()`` attempt."""

    module: Optional[types.ModuleType]
    tier: Optional[str]
    diagnostics: _Diagnostics


def _safe_log(
    level: str,
    msg: str,
    diagnostics: Optional[_Diagnostics] = None,
) -> None:
    """Log via ``utils.logger`` if importable, otherwise stderr."""
    if diagnostics is not None:
        diagnostics.append((level, msg))
    try:
        from utils.logger import console_error, console_log, console_warning

        emit = {"log": console_log, "warning": console_warning, "error": console_error}[
            level
        ]
        emit("ml api trace loader", msg)
    except Exception:
        sys.stderr.write(f"[ml api trace loader] {level.upper()}: {msg}\n")


def format_load_diagnostic_trail(
    diagnostics: _Diagnostics,
    *,
    max_lines: int = 24,
) -> str:
    """Render diagnostics as indented lines, capped at ``max_lines``."""
    if not diagnostics:
        return ""
    rendered = [f"  [{lvl}] {msg}" for lvl, msg in diagnostics[-max_lines:]]
    return "\n".join(rendered)


def torch_version() -> Optional[str]:
    """Return ``torch.__version__`` with any local build suffix removed, or ``None``."""
    try:
        import torch
    except Exception:
        return None
    return torch.__version__.split("+", 1)[0]


def _import_module_from_path(name: str, path: Path) -> types.ModuleType:
    """Import a shared object from a filesystem path."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to build importlib spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_prefix() -> Path:
    """Return the install prefix that hosts ``lib*/rocprofiler-compute/``."""
    return _THIS_DIR.parents[4]


def _installed_collector_artifacts() -> List[Path]:
    """Return installed ``torch_trace_collector-*.so`` paths."""
    pattern = f"lib*/{_INSTALL_TREE_PROJECT_NAME}/{_ARTIFACT_PREFIX}*{_ARTIFACT_SUFFIX}"
    return sorted(path for path in _install_prefix().glob(pattern) if path.is_file())


def _torch_version_from_artifact_name(path: Path) -> Optional[str]:
    """Return the PyTorch version encoded in an artifact filename."""
    match = _ARTIFACT_NAME_PATTERN.match(path.name)
    if match is None:
        return None
    return match.group(1)


def supported_torch_versions() -> Tuple[str, ...]:
    """Return installed collector PyTorch versions, sorted and unique."""
    versions = []
    for path in _installed_collector_artifacts():
        version = _torch_version_from_artifact_name(path)
        if version is not None:
            versions.append(version)
    return tuple(sorted(set(versions)))


def _artifact_paths_for_torch_version(workload_torch_version: str) -> List[Path]:
    """Return installed artifacts for ``workload_torch_version``."""
    so_name = f"{_ARTIFACT_PREFIX}{workload_torch_version}{_ARTIFACT_SUFFIX}"
    return [path for path in _installed_collector_artifacts() if path.name == so_name]


def _load_prebuilt(
    workload_torch_version: str,
    diagnostics: Optional[_Diagnostics] = None,
) -> types.ModuleType:
    """Load the installed collector for ``workload_torch_version``."""
    candidates = _artifact_paths_for_torch_version(workload_torch_version)
    if not candidates:
        raise UnsupportedTorchVersionError(
            workload_torch_version,
            supported_torch_versions(),
        )

    load_errors: List[str] = []
    for so_path in candidates:
        try:
            module = _import_module_from_path("torch_trace_collector", so_path)
        except Exception as exc:
            load_errors.append(f"{so_path}: {exc}")
            _safe_log(
                "warning",
                f"prebuilt .so at {so_path} failed to load: {exc}",
                diagnostics,
            )
            continue
        _safe_log("log", f"loaded prebuilt .so: {so_path}", diagnostics)
        return module

    raise ImportError(
        "failed to load torch_trace_collector for PyTorch "
        f"{workload_torch_version}: {'; '.join(load_errors)}"
    )


def load(force_python_fallback: bool = False) -> LoadResult:
    """Resolve the ``torch_trace_collector`` module."""
    diagnostics: _Diagnostics = []

    if force_python_fallback:
        _safe_log(
            "log", "force_python_fallback=True; declining to load .so", diagnostics
        )
        return LoadResult(None, None, diagnostics)

    workload_torch_version = torch_version()
    if workload_torch_version is None:
        _safe_log(
            "warning", "torch not importable; using Python-only injector", diagnostics
        )
        return LoadResult(None, None, diagnostics)

    module = _load_prebuilt(workload_torch_version, diagnostics=diagnostics)
    return LoadResult(module, TIER_PREBUILT, diagnostics)
