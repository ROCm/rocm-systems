#!/usr/bin/env python3
"""Shared path discovery for the ConSan Tensile validation workload."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess
import sys

TENSILELITE_ROOT_ENV = "CONSAN_VALIDATION_TENSILELITE_ROOT"
ROCM_ROOT_ENV = "CONSAN_VALIDATION_ROCM_ROOT"
TENSILE_CLIENT_ENV = "CONSAN_VALIDATION_TENSILE_CLIENT"
TENSILE_WRAPPER_ENV = "CONSAN_VALIDATION_TENSILE_WRAPPER"
ROCJITSU_EXE_ENV = "CONSAN_VALIDATION_ROCJITSU_EXE"
ROCJITSU_CONFIG_ENV = "CONSAN_VALIDATION_ROCJITSU_CONFIG"
LLVM_READELF_ENV = "CONSAN_VALIDATION_LLVM_READELF"


@dataclass(frozen=True)
class TensileValidationPaths:
    tensilelite: Path
    rocm: Path
    client: Path
    wrapper: Path
    rocjitsu: Path
    rocjitsu_config: Path
    llvm_readelf: Path


def _configured_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return Path(os.path.abspath(Path(value).expanduser()))


def _first_existing(candidates: tuple[Path, ...], *, directory: bool) -> Path:
    predicate = Path.is_dir if directory else Path.is_file
    return next((path for path in candidates if predicate(path)), candidates[0])


@lru_cache(maxsize=None)
def _rocm_sdk_root(workspace: Path) -> Path | None:
    candidates = (
        workspace / "venv" / "bin" / "rocm-sdk",
        workspace / "consan-pytorch-venv" / "bin" / "rocm-sdk",
        Path(sys.executable).parent / "rocm-sdk",
    )
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [str(executable), "path", "--root"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    root = completed.stdout.strip()
    return Path(root) if completed.returncode == 0 and root else None


def resolve_tensile_validation_paths(workspace: Path) -> TensileValidationPaths:
    """Resolves one coherent local Tensile, ROCm, and RocJITsu toolchain."""
    workspace = workspace.resolve()
    tensilelite = _configured_path(TENSILELITE_ROOT_ENV)
    if tensilelite is None:
        relative = Path("projects/hipblaslt/tensilelite")
        tensilelite = _first_existing(
            (
                workspace / "TheRock" / "rocm-libraries" / relative,
                workspace / "upstream-rocm-libraries" / relative,
                workspace.parent / "upstream-rocm-libraries" / relative,
            ),
            directory=True,
        )

    rocm = _configured_path(ROCM_ROOT_ENV)
    if rocm is None:
        packaged = _rocm_sdk_root(workspace)
        candidates = [workspace / "TheRock" / "build" / "dist" / "rocm"]
        if packaged is not None:
            candidates.append(packaged)
        rocm = _first_existing(tuple(candidates), directory=True)

    client = _configured_path(TENSILE_CLIENT_ENV) or (
        tensilelite / "build_tmp" / "tensilelite" / "client" / "tensilelite-client"
    )
    wrapper = _configured_path(TENSILE_WRAPPER_ENV) or (
        Path(__file__).with_name("run_tensile_client_with_rocjitsu.sh")
    )
    rocjitsu = _configured_path(ROCJITSU_EXE_ENV)
    if rocjitsu is None:
        suffix = Path("tools/rocjitsu/rocjitsu")
        rocjitsu = _first_existing(
            (
                workspace / "rocjitsu-build" / suffix,
                workspace / "rocjitsu-main-gpu-build" / suffix,
                workspace / "build" / suffix,
            ),
            directory=False,
        )
    rocjitsu_config = _configured_path(ROCJITSU_CONFIG_ENV) or (
        workspace
        / "rocm-systems"
        / "emulation"
        / "rocjitsu"
        / "configs"
        / "gfx1250.json"
    )
    llvm_readelf = _configured_path(LLVM_READELF_ENV)
    if llvm_readelf is None:
        on_path = shutil.which("llvm-readelf")
        llvm_readelf = _first_existing(
            (
                rocm / "lib" / "llvm" / "bin" / "llvm-readelf",
                rocm / "llvm" / "bin" / "llvm-readelf",
                Path(on_path) if on_path else rocm / "bin" / "llvm-readelf",
            ),
            directory=False,
        )

    return TensileValidationPaths(
        tensilelite=tensilelite,
        rocm=rocm,
        client=client,
        wrapper=wrapper,
        rocjitsu=rocjitsu,
        rocjitsu_config=rocjitsu_config,
        llvm_readelf=llvm_readelf,
    )
