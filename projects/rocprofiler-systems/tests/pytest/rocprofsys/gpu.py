# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from __future__ import annotations
import re
import shutil
import subprocess
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


@dataclass
class GPUInfo:
    """Information about detected GPU(s)

    Attributes:
        available: Whether any GPU is available
        architectures: List of GPU architectures
        device_count: Number of GPUs detected
        categories: Categories the GPU belongs to (instinct, radeon, apu)
    """

    available: bool
    architectures: list[str]
    device_count: int
    categories: set[str]

    @property
    def rocm_events_for_test(self) -> str:
        """Get appropriate ROCm events for testing based on architecture."""
        mi300_or_later = False
        for arch in self.architectures:
            if re.match(r"gfx9[4-9][0-9A-Fa-f]", arch):
                mi300_or_later = True
                break
        if mi300_or_later:
            return "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY:device=0"
        return "SQ_WAVES"

    @property
    def counter_names(self) -> list[str]:
        """Get counter names for validation based on architecture"""
        mi300_or_later = False
        for arch in self.architectures:
            if re.match(r"gfx9[4-9][0-9A-Fa-f]", arch):
                mi300_or_later = True
                break
        if mi300_or_later:
            return ["GRBM_COUNT", "SQ_WAVES", "SQ_INSTS_VALU", "TA_TA_BUSY"]
        return ["SQ_WAVES"]

    @property
    def expected_counter_files(self) -> list[str]:
        """Get expected counter output files based on architecture."""
        return [f"rocprof-device-0-{name}.txt" for name in self.counter_names]


def get_rocminfo(rocm_path: Optional[Path] = None) -> Optional[Path]:
    """Get the path to the rocminfo executable.

    Args:
        rocm_path: Path to the ROCm installation directory

    Returns:
        Path to the rocminfo executable or None if not found
    """
    if rocm_path:
        candidate = rocm_path / "bin" / "rocminfo"
        if candidate.exists():
            return Path(candidate).resolve()
    rocminfo = shutil.which("rocminfo")
    if rocminfo:
        return Path(rocminfo).resolve()
    return None


@lru_cache(maxsize=1)
def detect_gpu(rocm_path: Optional[Path] = None) -> GPUInfo:
    """Detect available AMD GPUs and their capabilities.

    Uses rocminfo to get the list of GPU architectures.
    Regex avoids matching "gfxX-X-generic" which may appear.
    """
    categories: set[str] = set()
    architectures: list[str] = []
    device_count = 0

    # Detect available GPUs
    rocminfo = None
    if rocm_path:
        rocminfo = rocm_path / "bin" / "rocminfo"
    if not rocminfo:
        rocminfo = shutil.which("rocminfo")

    if rocminfo:
        try:
            result = subprocess.run(
                [str(rocminfo)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                # Only match gfx on "Name:"
                name_gfx_pattern = re.compile(
                    r"^\s*Name:\s+(gfx[0-9A-Fa-f][0-9A-Fa-f]+)", re.MULTILINE
                )
                all_matches = name_gfx_pattern.findall(result.stdout)
                # gfx000 is the cpu, remove it
                filtered = [arch for arch in all_matches if arch != "gfx000"]
                device_count = len(filtered)
                # Remove duplicates
                architectures = list(set(filtered))
        except (subprocess.TimeoutExpired, OSError):
            pass

    for arch in architectures:
        categories.update(lookup_gpu_category(arch, rocm_path))

    return GPUInfo(
        available=device_count > 0,
        architectures=sorted(architectures),
        device_count=device_count,
        categories=categories,
    )


def lookup_gpu_category(arch: str, rocm_path: Optional[Path] = None) -> list[str]:
    """Lookup the GPU category for an architecture.

    Args:
        arch: Architecture string (e.g., 'gfx940')

    Returns:
        List of GPU categories the architecture belongs to (instinct, radeon, apu)
    """
    instinct_list = [
        "gfx900",
        "gfx906",  # MI50/MI60
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx950",
    ]

    # Also includes PRO GPUs
    # Ignore Radeon VII (gfx906)
    radeon_list = [
        "gfx1010",
        "gfx1011",
        "gfx1012",
        "gfx1030",
        "gfx1031",
        "gfx1032",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1200",
        "gfx1201",
        "gfx1202",
    ]

    apu_list = [
        "gfx1035",
        "gfx1036",
        "gfx1103",
        "gfx1151",
        "gfx1152",
        "gfx1153",
    ]

    categories: list[str] = []

    if arch in instinct_list:
        categories.append("instinct")
        # Some instinct GPUs may also be an APU (ex: MI300A)
        rocminfo = get_rocminfo(rocm_path)
        if rocminfo:
            try:
                result = subprocess.run(
                    [str(rocminfo)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0 and "APU" in result.stdout:
                    categories.append("apu")
            except (subprocess.TimeoutExpired, OSError):
                pass
    if arch in radeon_list:
        categories.append("radeon")
    if arch in apu_list:
        categories.append("apu")

    if not categories:
        # Unknown architecture, default to instinct
        categories.append("instinct")

    return categories


@lru_cache(maxsize=1)
def get_offload_extractor(rocm_path: Path) -> tuple[Optional[Path], Optional[bool]]:
    """Get offload extractor path

    An offload extractor is one of:
        llvm-objdump (only if version >= 20) - Preferred
        roc-obj-ls (deprecated)              - Fallback

    Args:
        rocm_path: Path to the ROCm installation directory

    Returns:
        Path to the offload extractor
        Bool representing whether found llvm-objdump's version < 20 (None if llvm-objdump not found)
    """

    is_llvm_too_old = None
    offload_extractor = None
    # Check env var - accepts either path to binary or directory containing it
    llvm_objdump_env = os.environ.get("ROCM_LLVM_OBJDUMP")
    if llvm_objdump_env:
        llvm_objdump_path = Path(llvm_objdump_env)
        if llvm_objdump_path.is_file() and llvm_objdump_path.exists():
            offload_extractor = llvm_objdump_path
        elif llvm_objdump_path.is_dir():
            candidate = llvm_objdump_path / "llvm-objdump"
            if candidate.exists():
                offload_extractor = candidate

    # Fallback to ROCm path
    if not offload_extractor and rocm_path:
        llvm_objdump_candidates = [
            rocm_path / "llvm" / "bin" / "llvm-objdump",
            rocm_path / "bin" / "llvm-objdump",
        ]
        for candidate in llvm_objdump_candidates:
            if candidate.exists():
                offload_extractor = candidate
                break

    if offload_extractor:
        # We have found llvm-objdump, check version
        try:
            version_result = subprocess.run(
                [str(offload_extractor), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version_match = re.search(r"version\s+(\d+)", version_result.stdout)
            if version_match:
                major_version = int(version_match.group(1))
                if major_version >= 20:
                    is_llvm_too_old = False
                    return (
                        Path(offload_extractor).resolve(),
                        is_llvm_too_old,
                    )
                else:
                    is_llvm_too_old = True
        except Exception:
            pass

    # Fallback to roc-obj-ls
    offload_extractor = None
    if rocm_path:
        candidate = rocm_path / "bin" / "roc-obj-ls"
        if candidate.exists():
            offload_extractor = Path(candidate).resolve()
            return offload_extractor, is_llvm_too_old
    if not offload_extractor:
        offload_extractor = shutil.which("roc-obj-ls")
    if offload_extractor:
        return offload_extractor, is_llvm_too_old
    return None, is_llvm_too_old


def get_target_gpu_arch(rocm_path: Path, target_path: Path) -> list[str]:
    """Get the list of gpu architectures (gfx) the target was compiled for.

    Args:
        rocm_path: Path to the ROCm installation directory
        target_path: Path to the binary to check

    Returns:
        List of GPU architectures the target was compiled for

    Raises:
        FileNotFoundError: If offload extractor is not found
    """
    import tempfile

    target_archs: set[str] = set()

    result = get_offload_extractor(rocm_path)
    if not result:
        raise FileNotFoundError(
            f"Could not find offload extractor in {rocm_path} "
            "or environment variable ROCM_LLVM_OBJDUMP"
        )
    tool_path, _ = result

    if "llvm-objdump" in tool_path.name:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_symlink = Path(tmpdir) / target_path.name
            try:
                tmp_symlink.symlink_to(target_path)
            except OSError:
                return list(target_archs)

            extracted_files: list[Path] = []
            try:
                result = subprocess.run(
                    [str(tool_path), "--offloading", str(tmp_symlink)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        # Match any gfxXXXX pattern in the line
                        match = re.search(r"(gfx[0-9a-fA-F]+)", line)
                        if match:
                            target_archs.add(match.group(1))

                        # Capture extracted bundle paths for cleanup
                        bundle_match = re.search(
                            r"Extracting offload bundle:\s*(.+)$", line
                        )
                        if bundle_match:
                            extracted_files.append(Path(bundle_match.group(1)))
            except (subprocess.TimeoutExpired, OSError):
                pass

            # Immediately clean up extracted files to free disk space
            for extracted_file in extracted_files:
                try:
                    if extracted_file.exists():
                        extracted_file.unlink()
                except OSError:
                    pass

    elif "roc-obj-ls" in tool_path.name:
        try:
            result = subprocess.run(
                [str(tool_path), str(target_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    # Match any gfxXXXX pattern in the line
                    match = re.search(r"(gfx[0-9a-fA-F]+)", line)
                    if match:
                        target_archs.add(match.group(1))
        except (subprocess.TimeoutExpired, OSError):
            pass

    return list(target_archs)
