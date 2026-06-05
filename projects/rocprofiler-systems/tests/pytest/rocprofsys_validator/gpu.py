# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU profile detection and configuration for rocprofsys-validator.

Provides:
- GPUProfile: dataclass representing GPU capabilities (arch, counter names, PMC groups)
- detect_gpu(): detect GPU via rocminfo subprocess; cached for process lifetime
- GPUProfile.from_toml(): load GPU profile from a TOML configuration file

Design decisions:
- D-09: GPUProfile fields locked — name, arch, counter_names, pmc_groups
- D-10: detect_gpu() never raises; emits RuntimeWarning and returns unknown profile on failure
- D-12: GPUProfile.has_counter() returns False on unknown arch (all counter-dependent validators skip)
- Pitfall 6: @lru_cache on module-level function, NOT a method (methods are not hashable as cache keys)
- T-03-01: subprocess.run uses a list (not shell=True); shutil.which returns absolute path
- T-03-02: tomllib.load() is a pure data parser; no code execution
- T-03-03: subprocess.run with timeout=30; TimeoutExpired caught; never hangs
"""
from __future__ import annotations

import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-reuse]

# Regex to parse GPU arch from rocminfo output.
# Matches lines like "  Name:          gfx940" or "  Name:                    gfx1201"
_GFX_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*Name:\s+(gfx[0-9A-Fa-f][0-9A-Fa-f]+)", re.MULTILINE
)

_UNKNOWN: str = "unknown"

@dataclass
class GPUProfile:
    """GPU capability profile for adaptive validation.

    Fields (locked per D-09):
    - name: human-readable GPU name (typically the arch string; product name lookup is Phase 2+)
    - arch: GFX architecture string (e.g., "gfx940", "gfx1201")
    - counter_names: set of available hardware counter names (empty for unknown/Phase 1)
    - pmc_groups: list of PMC groups; each group is a list of counter names

    The "unknown" profile is returned when GPU detection fails — all counter-dependent
    validations skip gracefully when arch == "unknown" (D-12).
    """

    name: str
    arch: str
    counter_names: set[str] = field(default_factory=set)
    pmc_groups: list[list[str]] = field(default_factory=list)

    def has_counter(self, name: str) -> bool:
        """Return True if this profile includes the named counter.

        Always returns False on the unknown profile (D-12): counter-dependent validations
        must skip when the GPU arch is not detected, not fail.

        Args:
            name: Hardware counter name to look up (e.g., "SQ_WAVES").

        Returns:
            False if arch == "unknown"; otherwise True iff name is in counter_names.
        """
        if self.arch == _UNKNOWN:
            return False
        return name in self.counter_names

    @classmethod
    def from_toml(cls, path: str | Path) -> GPUProfile:
        """Load a GPU profile from a TOML configuration file.

        Expected TOML schema:
            [gpu]
            name = "MI300X"
            arch = "gfx940"

            [gpu.counters]
            names = ["SQ_WAVES", "TCC_HIT", ...]

            [[gpu.pmc_groups]]
            counters = ["SQ_WAVES", "SQ_ITEMS_PROCESSED"]

            [[gpu.pmc_groups]]
            counters = ["TCC_HIT", "GRBM_COUNT"]

        Security (T-03-02): tomllib.load() is a pure data parser; no code execution.

        Args:
            path: Path to the TOML file (str or Path).

        Returns:
            GPUProfile populated with data from the TOML file.

        Raises:
            tomllib.TOMLDecodeError: If the file is not valid TOML.
            KeyError: If required keys (gpu.name, gpu.arch) are missing.
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)
        gpu = data["gpu"]
        return cls(
            name=gpu["name"],
            arch=gpu["arch"],
            counter_names=set(gpu.get("counters", {}).get("names", [])),
            pmc_groups=[g["counters"] for g in gpu.get("pmc_groups", [])],
        )

@lru_cache(maxsize=1)
def detect_gpu() -> GPUProfile:
    """Detect the GPU model via rocminfo. Cached for process lifetime.

    Resolution:
    1. shutil.which("rocminfo") — portable binary lookup (absolute path or None)
    2. subprocess.run([rocminfo], ..., timeout=30) — with timeout; no shell injection
    3. Regex parse of stdout for "Name: gfxNNN" lines
    4. Filter gfx000 (CPU) and "generic" entries; take first GPU arch found

    On any failure (missing binary, timeout, non-zero exit, no GPU arch found):
    - Emits RuntimeWarning with descriptive message
    - Returns GPUProfile(name="unknown", arch="unknown", ...)
    - Never raises (D-10)

    Security (T-03-01): subprocess list form (not shell=True); path from shutil.which.
    Security (T-03-03): timeout=30 prevents hangs; TimeoutExpired is caught.

    Returns:
        GPUProfile with detected arch and empty counter sets (populated in Phase 2+).
    """
    rocminfo = shutil.which("rocminfo")
    if not rocminfo:
        warnings.warn(
            "rocminfo not found on PATH; GPU detection unavailable. "
            "Using unknown profile — counter-dependent validations will skip.",
            RuntimeWarning,
            stacklevel=2,
        )
        return GPUProfile(name=_UNKNOWN, arch=_UNKNOWN, counter_names=set(), pmc_groups=[])

    try:
        result = subprocess.run(
            [rocminfo],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        warnings.warn(
            f"rocminfo failed ({exc}); using unknown GPU profile.",
            RuntimeWarning,
            stacklevel=2,
        )
        return GPUProfile(name=_UNKNOWN, arch=_UNKNOWN, counter_names=set(), pmc_groups=[])

    if result.returncode != 0:
        warnings.warn(
            f"rocminfo exited with code {result.returncode}; using unknown GPU profile.",
            RuntimeWarning,
            stacklevel=2,
        )
        return GPUProfile(name=_UNKNOWN, arch=_UNKNOWN, counter_names=set(), pmc_groups=[])

    matches = _GFX_PATTERN.findall(result.stdout)
    # Filter CPU (gfx000) and generic entries; take first discrete GPU arch found
    gpu_archs = [a for a in matches if a != "gfx000" and "generic" not in a]
    arch = gpu_archs[0] if gpu_archs else _UNKNOWN

    return GPUProfile(
        name=arch,  # arch string used as name; product name lookup is Phase 2+
        arch=arch,
        counter_names=set(),  # Phase 2: populate from rocprofiler-sdk counter catalog
        pmc_groups=[],
    )
