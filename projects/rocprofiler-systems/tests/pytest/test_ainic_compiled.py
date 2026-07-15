# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests that AI NIC support was compiled into librocprof-sys.so.

This test does NOT require NIC hardware to be present — it checks the
binary artifacts for compile-time indicators. It will fail if the library
was built with ROCPROFSYS_BUILD_AINIC=OFF (e.g. due to ROCM_PATH not
being set during the build, as in ROCM-28005).
"""

from __future__ import annotations

import subprocess
import pytest
from pathlib import Path
from rocprofsys import RocprofsysConfig

pytestmark = [pytest.mark.rocprof_binary]

# Dynamic symbols present in librocprof-sys.so only when
# ROCPROFSYS_BUILD_AINIC=ON (calls to AMD SMI NIC APIs).
_AINIC_SYMBOLS = [
    "amdsmi_get_nic_rdma_port_statistics",
    "amdsmi_get_nic_rdma_dev_info",
    "amdsmi_get_nic_port_info",
]

# String literals embedded only when ROCPROFSYS_BUILD_AINIC=ON.
_AINIC_STRINGS = [
    "ROCPROFSYS_USE_AINIC",
    "ainic",
]

def _find_librocprof_sys(lib_dir: Path) -> Path:
    """Return the versioned or unversioned librocprof-sys.so under lib_dir."""
    for pattern in ("librocprof-sys.so.*", "librocprof-sys.so"):
        candidates = sorted(lib_dir.glob(pattern))
        # Prefer the versioned file (largest, most complete)
        for c in reversed(candidates):
            if c.is_file() and not c.is_symlink():
                return c
        for c in candidates:
            if c.is_file():
                return c
    raise FileNotFoundError(
        f"librocprof-sys.so not found under {lib_dir}"
    )

@pytest.fixture(scope="module")
def librocprof_sys(rocprof_config: RocprofsysConfig) -> Path:
    lib = _find_librocprof_sys(rocprof_config.rocprofsys_lib_dir)
    return lib


class TestAinicCompiledIn:
    """Verify AI NIC support was compiled into librocprof-sys.so."""

    def test_ainic_dynamic_symbols_present(self, librocprof_sys: Path):
        """amdsmi NIC API symbols must appear in the dynamic symbol table.

        These symbols are undefined references (U) in librocprof-sys.so —
        they are resolved at runtime from libamd_smi.so. Their presence
        proves that the NIC collector code was compiled in.
        """
        result = subprocess.run(
            ["nm", "-D", str(librocprof_sys)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"nm failed: {result.stderr}"

        dyn_syms = result.stdout
        missing = [s for s in _AINIC_SYMBOLS if s not in dyn_syms]
        assert not missing, (
            f"AI NIC dynamic symbols not found in {librocprof_sys.name} — "
            f"was it built with ROCPROFSYS_BUILD_AINIC=OFF?\n"
            f"Missing: {missing}"
        )
