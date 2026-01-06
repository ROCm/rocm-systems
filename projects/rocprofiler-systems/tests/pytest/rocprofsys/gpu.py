# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import re
import shutil
import subprocess
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


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


@lru_cache(maxsize=1)
def detect_gpu() -> GPUInfo:
    """Detect available AMD GPUs and their capabilities.

    Uses rocm_agent_enumerator to get the list of GPU architectures.
    """
    categories: set[str] = set()
    architectures: list[str] = []
    device_count = 0

    # Detect available GPUs
    rocm_agent_enumerator = shutil.which("rocm_agent_enumerator")
    if rocm_agent_enumerator:
        try:
            result = subprocess.run(
                [rocm_agent_enumerator],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                all_entries = result.stdout.strip().split("\n")
                # gfx000 is the cpu, remove it
                device_count = sum(
                    1 for entry in all_entries if entry and entry != "gfx000"
                )
                architectures = list(
                    set(entry for entry in all_entries if entry and entry != "gfx000")
                )
        except (subprocess.TimeoutExpired, OSError):
            pass

    for arch in architectures:
        categories.update(lookup_gpu_category(arch))

    return GPUInfo(
        available=device_count > 0,
        architectures=sorted(architectures),
        device_count=device_count,
        categories=categories,
    )


def lookup_gpu_category(arch: str) -> list[str]:
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
    if arch in radeon_list:
        categories.append("radeon")
    if arch in apu_list:
        categories.append("apu")

    if not categories:
        # Unknown architecture, default to instinct
        categories.append("instinct")

    return categories


@lru_cache(maxsize=1)
def get_llvm_objdump(rocm_path: Path) -> Path:
    """Get the path to llvm-objdump.

    Args:
        rocm_path: Path to the ROCm installation directory

    Returns:
        Path to llvm-objdump

    Raises:
        FileNotFoundError: If llvm-objdump is not found
    """
    llvm_objdump_candidates = []
    if rocm_path:
        llvm_objdump_candidates = [
            rocm_path / "llvm" / "bin" / "llvm-objdump",
            rocm_path / "bin" / "llvm-objdump",
        ]
        for candidate in llvm_objdump_candidates:
            if candidate.exists():
                return candidate

    # Check env var - accepts either path to binary or directory containing it
    llvm_objdump_env = os.environ.get("ROCM_LLVM_OBJDUMP")
    if llvm_objdump_env:
        llvm_objdump_path = Path(llvm_objdump_env)
        if llvm_objdump_path.is_file():
            return llvm_objdump_path
        elif llvm_objdump_path.is_dir():
            candidate = llvm_objdump_path / "llvm-objdump"
            if candidate.exists():
                return candidate

    # We explicitly avoid checking PATH here as we require the llvm-objdump
    # from the ROCm installation directory.
    searched_paths = [f"  - {p}" for p in llvm_objdump_candidates]
    searched_paths.append("  - ROCM_LLVM_OBJDUMP environment variable")

    raise FileNotFoundError(
        f"ROCm's llvm-objdump not found. Searched in:\n" + "\n".join(searched_paths)
    )


def get_target_gpu_arch(rocm_path: Path, target_path: Path) -> list[str]:
    """Get the list of gpu architectures (gfx) the target was compiled for.

    Args:
        rocm_path: Path to the ROCm installation directory
        target_path: Path to the binary to check

    Returns:
        List of GPU architectures the target was compiled for

    Raises:
        FileNotFoundError: If llvm-objdump is not found
    """
    import tempfile

    target_archs: set[str] = set()

    # Find llvm-objdump
    llvm_objdump = get_llvm_objdump(rocm_path)

    if llvm_objdump:
        abs_target_path = Path(target_path).resolve()

        # llvm-objdump extracts files in same dir as input path
        # Create symlink in temp directory to avoid permission issues
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_symlink = Path(tmpdir) / abs_target_path.name
            try:
                tmp_symlink.symlink_to(abs_target_path)
            except OSError:
                return list(target_archs)

            extracted_files: list[Path] = []
            try:
                result = subprocess.run(
                    [str(llvm_objdump), "--offloading", str(tmp_symlink)],
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

    return list(target_archs)
