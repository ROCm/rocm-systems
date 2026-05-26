# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cached-perfetto output-layout coverage.

Runs mpi-example with -np 2 under both values of
ROCPROFSYS_PERFETTO_OUTPUT_LAYOUT. For 'single_file', asserts each rank
produces exactly one .proto, the file is well-formed, and the recorded
pid set matches the rank's logical pid (proves the seq_id rewrite +
synthetic ProcessTrack pipeline preserves per-pid attribution under
concatenation). For 'per_process', asserts the per-rank file behaves the
same as the cached_perfetto_isolation baseline.
"""

from __future__ import annotations
import re
import subprocess
import sys

import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.mpi]

_SINGLE_FILE_REPORT_RE = re.compile(
    r"single-file-checks: pids=\[([0-9, ]*)\] slices=(\d+)"
)
_PID_REPORT_RE = re.compile(r"per-pid-isolation: pid=(\d+) slices=(\d+)")

# Floor on TrackEvent slice count per rank from mpi-example's instrumented
# loop. The example emits ~70-120 slices per rank on a quiescent
# workstation; 50 keeps headroom for slower runners while still catching
# whole-rank attribution loss (which manifests as 0 or single-digit slices).
# If mpi-example's loop count changes upstream, re-derive this value from a
# fresh local run before lowering.
_MIN_SLICES_PER_RANK = 50


@pytest.fixture
def base_env() -> dict[str, str]:
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_USE_PID": "ON",
        "ROCPROFSYS_TRACE_LEGACY": "false",
        "ROCPROFSYS_PROFILE": "OFF",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
    }


@pytest.mark.class_name("cached-perfetto-layout")
class TestCachedPerfettoLayout(RocprofsysTest):
    @pytest.mark.timeout(180)
    @pytest.mark.parametrize("layout", ["single_file", "per_process"])
    def test(self, layout, base_env, tests_dir):
        env = dict(base_env)
        env["ROCPROFSYS_PERFETTO_OUTPUT_LAYOUT"] = layout

        result = self.run_test(
            "sys_run",
            "mpi-example",
            env=env,
            launcher="mpi",
            num_procs=2,
        )

        proto_files = sorted(result.output_dir.glob("perfetto-trace-*.proto"))
        assert len(proto_files) == 2, (
            f"expected 2 per-rank .proto files (one per rank) in "
            f"{result.output_dir}, found {[p.name for p in proto_files]}"
        )

        validator = tests_dir / "validate-perfetto-proto.py"

        if layout == "single_file":
            for proto in proto_files:
                # Each rank writes ONE concatenated single_file output. With
                # one logical pid per rank (the rank's own pid), the
                # single-file checks reduce to: > 0 slices, exactly the
                # rank's pid in the process table.
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(validator),
                        "--input",
                        str(proto),
                        "--single-file-checks",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                assert completed.returncode == 0, (
                    f"single-file-checks failed for {proto}\n"
                    f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
                )
                match = _SINGLE_FILE_REPORT_RE.search(completed.stdout)
                assert match, (
                    f"validator stdout missing single-file report for {proto}\n"
                    f"stdout: {completed.stdout}"
                )
                pid_list_text = match.group(1).strip()
                slice_count = int(match.group(2))
                pids = (
                    [int(p) for p in pid_list_text.split(",") if p.strip()]
                    if pid_list_text
                    else []
                )
                assert slice_count >= _MIN_SLICES_PER_RANK, (
                    f"{proto} slice count {slice_count} below floor "
                    f"{_MIN_SLICES_PER_RANK}; possible event drop or attribution loss"
                )
                # Synthetic ProcessTrack with the logical pid must produce
                # exactly one pid in the process table, the rank's own pid.
                assert len(pids) == 1, (
                    f"{proto} single_file pid set = {pids}; expected exactly one "
                    f"(synthetic ProcessTrack attribution failed)"
                )
        else:
            for proto in proto_files:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(validator),
                        "--input",
                        str(proto),
                        "--per-pid-isolation",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                assert completed.returncode == 0, (
                    f"per_process layout per-pid check failed for {proto}\n"
                    f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
                )
                match = _PID_REPORT_RE.search(completed.stdout)
                assert match, (
                    f"validator stdout missing pid/slices report for {proto}\n"
                    f"stdout: {completed.stdout}"
                )
                slices = int(match.group(2))
                assert (
                    slices >= _MIN_SLICES_PER_RANK
                ), f"{proto} slice count {slices} below floor {_MIN_SLICES_PER_RANK}"
