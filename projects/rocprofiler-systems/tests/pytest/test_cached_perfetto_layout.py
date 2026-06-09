# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cached-perfetto output-layout coverage.

Runs mpi-example with -np 2 under each value of
ROCPROFSYS_PERFETTO_OUTPUT_LAYOUT:

* ``single_file_only``: a single shared output file produced by every
  rank appending to it under O_APPEND + flock. The base perfetto filename
  is the shared target; per-rank files are NOT written. Each rank's
  rewritten bytes use a disjoint trusted_packet_sequence_id range so the
  concatenation is semantically clean.
* ``per_process_only``: per-rank .proto files only, no cross-rank merge.
  Same per-pid attribution checks as the cached_perfetto_isolation
  baseline.
* ``full``: per-rank .proto files AND a shared ``merged.proto`` produced
  by each rank's tee_sink branch appending under flock. Both outputs are
  present and well-formed.

The assertions are tolerant of both ROCPROFSYS_USE_MPI=ON and
ROCPROFSYS_USE_MPI_HEADERS=ON builds because the file-lock-append
mechanism works without rocprof-sys-internal MPI: under
MPI_HEADERS-only, the workload is still launched under mpiexec/srun and
each independent rocprof-sys process appends its slice via flock.
"""

from __future__ import annotations
import re
import subprocess
import sys

import pytest
from conftest import RocprofsysTest
from rocprofsys import trace_processor_shell_args

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


def _run_single_file_checks(validator, proto):
    """Run --single-file-checks; return (pids, slice_count)."""
    completed = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--input",
            str(proto),
            "--single-file-checks",
        ]
        + trace_processor_shell_args(),
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
        [int(p) for p in pid_list_text.split(",") if p.strip()] if pid_list_text else []
    )
    return pids, slice_count


def _run_per_pid_isolation(validator, proto):
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
        f"per-pid-isolation check failed for {proto}\n"
        f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
    )
    match = _PID_REPORT_RE.search(completed.stdout)
    assert match, (
        f"validator stdout missing pid/slices report for {proto}\n"
        f"stdout: {completed.stdout}"
    )
    return int(match.group(2))


@pytest.mark.class_name("cached-perfetto-layout")
class TestCachedPerfettoLayout(RocprofsysTest):
    @pytest.mark.timeout(180)
    @pytest.mark.parametrize("layout", ["single_file_only", "per_process_only", "full"])
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

        per_rank_files = sorted(result.output_dir.glob("perfetto-trace-*.proto"))
        merged_path = result.output_dir / "merged.proto"
        validator = tests_dir / "validate-perfetto-proto.py"

        if layout == "single_file_only":
            # All ranks append to merged.proto under flock; no per-rank
            # files are written. The merged proto must contain both
            # ranks' pids in its process table (proves the cross-process
            # append-with-flock pipeline actually concatenated bytes from
            # every launcher rank).
            assert not per_rank_files, (
                f"single_file_only: per-rank files must NOT be written; found "
                f"{[p.name for p in per_rank_files]}"
            )
            assert (
                merged_path.exists()
            ), f"single_file_only: expected merged.proto in {result.output_dir}"
            pids, slices = _run_single_file_checks(validator, merged_path)
            assert len(pids) == 2, (
                f"merged.proto pid set = {pids}; expected exactly 2 "
                f"(cross-rank append-with-flock attribution failed)"
            )
            assert slices >= 2 * _MIN_SLICES_PER_RANK, (
                f"merged.proto slice count {slices} below floor "
                f"{2 * _MIN_SLICES_PER_RANK}; a rank's append likely dropped"
            )
        elif layout == "per_process_only":
            assert len(per_rank_files) == 2, (
                f"per_process_only: expected 2 per-rank .proto files in "
                f"{result.output_dir}, found {[p.name for p in per_rank_files]}"
            )
            assert (
                not merged_path.exists()
            ), f"per_process_only: must NOT produce a merged.proto"
            for proto in per_rank_files:
                slices = _run_per_pid_isolation(validator, proto)
                assert slices >= _MIN_SLICES_PER_RANK, (
                    f"{proto} slice count {slices} below floor " f"{_MIN_SLICES_PER_RANK}"
                )
        else:
            # full: per-rank files (per_pid_file_sink branch of tee) AND
            # merged.proto (single_file branch of tee in append+flock mode).
            assert len(per_rank_files) == 2, (
                f"full: expected 2 per-rank .proto files in "
                f"{result.output_dir}, found {[p.name for p in per_rank_files]}"
            )
            assert merged_path.exists(), (
                f"full: expected merged.proto alongside per-rank files in "
                f"{result.output_dir}"
            )
            for proto in per_rank_files:
                slices = _run_per_pid_isolation(validator, proto)
                assert slices >= _MIN_SLICES_PER_RANK, (
                    f"{proto} slice count {slices} below floor " f"{_MIN_SLICES_PER_RANK}"
                )
            pids, slices = _run_single_file_checks(validator, merged_path)
            assert len(pids) == 2, f"merged.proto pid set = {pids}; expected exactly 2"
            assert slices >= 2 * _MIN_SLICES_PER_RANK, (
                f"merged.proto slice count {slices} below floor "
                f"{2 * _MIN_SLICES_PER_RANK}"
            )
