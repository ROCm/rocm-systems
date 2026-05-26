# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations
import re
import shutil
import subprocess
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

GFX_XXXX_PATTERN = re.compile(r"(gfx[0-9a-fA-F]+)")


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
    def _is_mi300_or_later(self) -> bool:
        """Check if the GPU is a MI300 or later."""
        for arch in self.architectures:
            if re.match(r"gfx9[4-9][0-9A-Fa-f]", arch):
                return True
        return False

    @property
    def _is_gfx1250(self) -> bool:
        """Check if any detected GPU uses the gfx1250 architecture."""
        return "gfx1250" in self.architectures

    @property
    def rocm_events_for_test(self) -> str:
        """Get appropriate ROCm events for testing based on architecture."""
        if self._is_gfx1250:
            return "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TX_VCA_VCA_BUSY"

        mi300_or_later = self._is_mi300_or_later
        if mi300_or_later:
            return "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY"
        return "SQ_WAVES"

    @property
    def counter_names(self) -> list[str]:
        """Get counter names for validation based on architecture"""
        if self._is_gfx1250:
            return ["GRBM_COUNT", "SQ_WAVES", "SQ_INSTS_VALU", "TX_VCA_VCA_BUSY"]

        mi300_or_later = self._is_mi300_or_later
        if mi300_or_later:
            return ["GRBM_COUNT", "SQ_WAVES", "SQ_INSTS_VALU", "TA_TA_BUSY"]
        return ["SQ_WAVES"]

    @property
    def gpu_perf_counters_for_test(self) -> str:
        """Get appropriate GPU perf counters for testing based on architecture.

        These are the same counters as rocm_events_for_test but used with
        ROCPROFSYS_GPU_PERF_COUNTERS (device counting service) instead of
        ROCPROFSYS_ROCM_EVENTS (kernel dispatch counters).
        """
        return self.rocm_events_for_test

    @property
    def expected_counter_files(self) -> list[str]:
        """Get expected counter output file patterns based on architecture.

        Returns glob patterns that match any device ID (0-9), since the device
        number in the filename depends on device_type_index which varies by
        GPU topology.
        """
        return [f"rocprof-device-[0-9]-{name}.txt" for name in self.counter_names]


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
    rocminfo_stdout: Optional[str] = None

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
            rocminfo_stdout = result.stdout if result.returncode == 0 else None
            if rocminfo_stdout:
                # Only match gfx on "Name:"
                name_gfx_pattern = re.compile(
                    r"^\s*Name:\s+(gfx[0-9A-Fa-f][0-9A-Fa-f]+)", re.MULTILINE
                )
                all_matches = name_gfx_pattern.findall(rocminfo_stdout)
                # gfx000 is the cpu, remove it
                filtered = [arch for arch in all_matches if arch != "gfx000"]
                device_count = len(filtered)
                # Remove duplicates
                architectures = list(set(filtered))
        except (subprocess.TimeoutExpired, OSError):
            pass

    for arch in architectures:
        categories.update(lookup_gpu_category(arch, rocm_path, rocminfo_stdout))

    return GPUInfo(
        available=device_count > 0,
        architectures=sorted(architectures),
        device_count=device_count,
        categories=categories,
    )


def lookup_gpu_category(
    arch: str,
    rocm_path: Optional[Path] = None,
    rocminfo_stdout: Optional[str] = None,
) -> list[str]:
    """Lookup the GPU category for an architecture.

    Args:
        arch: Architecture string (e.g., 'gfx940')
        rocm_path: Optional path to ROCm installation (used only if rocminfo_stdout not provided)
        rocminfo_stdout: Optional pre-captured rocminfo stdout (avoids re-running rocminfo for APU check)

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
        if rocminfo_stdout is None:
            rocminfo = get_rocminfo(rocm_path)
            if rocminfo:
                try:
                    result = subprocess.run(
                        [str(rocminfo)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        rocminfo_stdout = result.stdout
                except (subprocess.TimeoutExpired, OSError):
                    pass
        if rocminfo_stdout and "APU" in rocminfo_stdout:
            categories.append("apu")
    if arch in radeon_list:
        categories.append("radeon")
    if arch in apu_list:
        categories.append("apu")

    if not categories:
        # Unknown architecture, default to instinct
        categories.append("instinct")

    return categories


@lru_cache(maxsize=1)
def get_offload_extractor(
    rocm_path: Optional[Path] = None,
) -> tuple[Optional[Path], Optional[bool]]:
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
            if version_result.returncode == 0:
                version_match = re.search(r"version\s+(\d+)", version_result.stdout or "")
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
        except (subprocess.TimeoutExpired, OSError, ValueError):
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
        return Path(offload_extractor).resolve(), is_llvm_too_old
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
    tool_path, _ = result
    if not tool_path:
        raise FileNotFoundError(
            f"Could not find offload extractor in {rocm_path} "
            "or environment variable ROCM_LLVM_OBJDUMP"
        )

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
                        match = GFX_XXXX_PATTERN.search(line)
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
                    match = GFX_XXXX_PATTERN.search(line)
                    if match:
                        target_archs.add(match.group(1))
        except (subprocess.TimeoutExpired, OSError):
            pass

    return list(target_archs)


@lru_cache(maxsize=1)
def get_xnack_support(rocm_path: Optional[Path] = None) -> bool:
    """Check whether the current GPU is XNACK-capable.

    Run ``rocminfo`` with ``HSA_XNACK=1`` injected into the subprocess
    environment and return True only if the output reports ``xnack+``.

    This keeps the check independent of the caller's shell environment:
    ``xnack-`` remains unsupported for test gating, and GPUs with no
    XNACK qualifier also return False.
    """
    rocminfo = get_rocminfo(rocm_path)
    if not rocminfo:
        return False

    try:
        env = os.environ.copy()
        env["HSA_XNACK"] = "1"
        result = subprocess.run(
            [str(rocminfo)],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if result.returncode == 0:
            return "xnack+" in result.stdout
    except (subprocess.TimeoutExpired, OSError):
        pass

    return False
