# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""R1: cached-Perfetto cross-pid isolation under parallel post-processing.

Runs mpi-example with -np 2 in default cached-Perfetto mode (post-C3 flip:
process_parallel=true). For each per-rank `perfetto-trace-<rank>.proto`,
asserts the file contains exactly one pid in its process table, and that
the pids across files are distinct — proving the parallel parser-thread
pipeline doesn't bleed events across pids.
"""

from __future__ import annotations
import re
import subprocess
import sys

import pytest
from conftest import RocprofsysTest
from rocprofsys import trace_processor_shell_args

pytestmark = [pytest.mark.mpi]

_PID_REPORT_RE = re.compile(r"per-pid-isolation: pid=(\d+) slices=(\d+)")

# Stable lower bound: mpi-example consistently emits >= 50 fixed-count slices
# per rank (host=20 + mpi=31 + a few rocm/pthread). Catches a regression
# where most events get dropped while exactly one survives.
_MIN_SLICES_PER_RANK = 50


@pytest.fixture
def cached_isolation_env() -> dict[str, str]:
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_USE_PID": "ON",
        "ROCPROFSYS_TRACE_LEGACY": "false",
        "ROCPROFSYS_PROFILE": "OFF",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
    }


@pytest.mark.class_name("cached-perfetto-isolation")
class TestCachedPerfettoIsolation(RocprofsysTest):
    @pytest.mark.timeout(180)
    @pytest.mark.parametrize("mode", ["sys_run"])
    def test(self, mode, cached_isolation_env, tests_dir):
        result = self.run_test(
            mode,
            "mpi-example",
            env=cached_isolation_env,
            launcher="mpi",
            num_procs=2,
        )

        proto_files = sorted(result.output_dir.glob("perfetto-trace-*.proto"))
        assert len(proto_files) == 2, (
            f"expected 2 per-rank perfetto-trace-*.proto files in "
            f"{result.output_dir}, found {[p.name for p in proto_files]}"
        )

        validator = tests_dir / "validate-perfetto-proto.py"
        observed_pids: list[int] = []
        for proto in proto_files:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(validator),
                    "--input",
                    str(proto),
                    "--per-pid-isolation",
                ]
                + trace_processor_shell_args(),
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert completed.returncode == 0, (
                f"per-pid isolation failed for {proto}\n"
                f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
            )
            match = _PID_REPORT_RE.search(completed.stdout)
            assert match, (
                f"validator stdout missing pid/slices report for {proto}\n"
                f"stdout: {completed.stdout}"
            )
            pid = int(match.group(1))
            slices = int(match.group(2))
            assert slices >= _MIN_SLICES_PER_RANK, (
                f"{proto} contains only {slices} slices "
                f"(expected >= {_MIN_SLICES_PER_RANK}); possible event drop"
            )
            observed_pids.append(pid)

        assert len(set(observed_pids)) == len(observed_pids), (
            f"per-rank .proto files share pids (cross-rank contamination): "
            f"{observed_pids}"
        )
