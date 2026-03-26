# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for SHMEM.
"""

from __future__ import annotations
import re
import subprocess
import pytest
from pathlib import Path
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.shmem,
    pytest.mark.oshrun_min_version("5.0"),
    pytest.mark.serialize,
]

_SHMEM_NP = 2

# =============================================================================
# SHMEM Fixtures
# =============================================================================


@pytest.fixture
def shmem_env() -> dict[str, str]:
    """Environment variables for SHMEM tests."""
    return {
        "ROCPROFSYS_USE_PID": "OFF",
        "ROCPROFSYS_USE_SHMEM": "ON",
        "OMPI_MCA_memheap_base_max_segments": "64",
    }


@pytest.fixture
def shmem_rules(validation_rules_dir: Path) -> list[Path]:
    rules_dir = validation_rules_dir / "shmem"
    return [
        rules_dir / "validation-rules.json",
    ]


@pytest.fixture(scope="session")
def shmem_validated(rocprof_config) -> tuple[bool, str]:
    """Run ``oshrun -n 2 shmem_hello`` and validate output."""
    oshrun = rocprof_config.capabilities.oshrun_exec
    if oshrun is None:
        return False, "oshrun not found"

    try:
        hello = rocprof_config.get_target_executable("shmem_hello")
    except FileNotFoundError:
        return False, "shmem_hello not found"

    cmd = [str(oshrun), "-n", str(_SHMEM_NP), str(hello)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (subprocess.SubprocessError, OSError):
        return False, f"Failed to run {' '.join(cmd)}"

    if result.returncode != 0:
        return False, f"Failed to run {' '.join(cmd)}: Exit code {result.returncode}"

    output = result.stdout + result.stderr
    if not re.search(rf"Hello from PE [0-9]+ of {_SHMEM_NP}", output):
        return False, f"Hello from PE [0-9]+ of {_SHMEM_NP} not found in output"
    if not re.search(r"PE [0-9]+ received value [0-9]+ from PE [0-9]+", output):
        return (
            False,
            f"PE [0-9]+ received value [0-9]+ from PE [0-9]+ not found in output",
        )
    for pe in range(_SHMEM_NP):
        if f"Hello from PE {pe} of {_SHMEM_NP}" not in output:
            return False, f"Hello from PE {pe} of {_SHMEM_NP} not found in output"

    return True, ""


# =============================================================================
# SHMEM Tests
# =============================================================================


class TestShmem(RocprofsysTest):
    @pytest.mark.parametrize(
        "mode",
        [
            "baseline",
            "sampling",
            pytest.param("sys_run", marks=pytest.mark.rocpd("shmem_env")),
        ],
    )
    def test_pingpong(self, mode, shmem_env, shmem_validated, shmem_rules):
        valid, reason = shmem_validated
        if not valid:
            pytest.skip(f"{reason}")

        result = self.run_test(
            mode,
            "shmem_pingpong",
            env=shmem_env,
            launcher="shmem",
            num_procs=_SHMEM_NP,
        )
        self.assert_regex(result)

        if mode == "sys_run":
            self.assert_perfetto(
                result,
                labels=["shmem_pingpong", "start_pes"],
                counts=[1, 1],
                depths=[0, 1],
            )
            self.assert_rocpd(result, rules_files=shmem_rules)
