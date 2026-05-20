# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for MPI rank-based console- and file-output filtering.

Covers:
- ROCPROFSYS_RANK_FILTER_OUTPUT / --rank-filter-output (file output)
- ROCPROFSYS_RANK_FILTER_LOGS   / --rank-filter-logs   (console output)
- ROCPROFSYS_RANK_FILTER_ID     / --rank-filter-id     (custom rank source)

Each test runs `mpi-example` under `mpiexec -n 3` and asserts both:
- console output, via banner-line count in stdout
- per-rank file output, by checking presence/absence of perfetto, timemory,
  functions/metadata, and RocPD `.db` files in `test_output_dir`.
"""

from __future__ import annotations
import re
import pytest
from pathlib import Path
from conftest import RocprofsysTest, check_use_rocpd

pytestmark = [pytest.mark.mpi, pytest.mark.rank_filter]

TARGET = "mpi-example"
NUM_PROCS = 3


def banner_count(text: str) -> int:
    # Regex matches the version line in banner
    return len(re.findall(r"rocprof-sys v[0-9]", text))


def assert_per_rank_outputs(
    output_dir: Path,
    ranks_with_output: list[int],
    ranks_without_output: list[int],
) -> None:
    """Assert per-rank file presence/absence for each rank in the given lists.
    Note: `.db` files are PID-named, not rank-named, so they are verified by
    count == len(ranks_with_output) since each producing rank emits one.
    The `.db` check only runs when RocPD is available
    (`check_use_rocpd()` — requires GPU and ROCm >= 7.0)
    """
    per_rank_files = [
        "perfetto-trace-{rank}.proto",
        "wall_clock-{rank}.txt",
        "wall_clock-{rank}.json",
        "functions-{rank}.json",
        "metadata-{rank}.json",
    ]
    for rank in ranks_with_output:
        for name in per_rank_files:
            path = output_dir / name.format(rank=rank)
            assert path.exists(), f"Expected file missing for rank {rank}: {path.name}"
    for rank in ranks_without_output:
        for name in per_rank_files:
            path = output_dir / name.format(rank=rank)
            assert (
                not path.exists()
            ), f"Unexpected file present for rank {rank}: {path.name}"

    if check_use_rocpd():
        db_files = sorted(output_dir.glob("*.db"))
        expected_db = len(ranks_with_output)
        assert len(db_files) == expected_db, (
            f"Expected {expected_db} .db file(s), got {len(db_files)}: "
            f"{[p.name for p in db_files]}"
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rocpd_env() -> dict[str, str]:
    return {}


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.class_name("rank-filter")
@pytest.mark.rocpd("rocpd_env")
@pytest.mark.sys_run
class TestRankFilter(RocprofsysTest):
    """End-to-end tests for the MPI rank-based output filtering feature."""

    @pytest.mark.parametrize(
        "filter_source",
        [
            pytest.param("unset", id=""),
            "via_cli",
            "via_env",
        ],
    )
    def test_no_filter(self, rocpd_env, filter_source):
        """Tests: filter not set + filter options are set (via cli or env) but empty.
        Every rank should produce both file and console output (filtering disabled).
        """
        env = rocpd_env.copy()
        sysrun_args = []
        if filter_source == "unset":
            pass
        elif filter_source == "via_cli":
            sysrun_args = ["--rank-filter-output", "", "--rank-filter-logs", ""]
        elif filter_source == "via_env":
            env["ROCPROFSYS_RANK_FILTER_OUTPUT"] = ""
            env["ROCPROFSYS_RANK_FILTER_LOGS"] = ""

        result = self.run_test(
            "sys_run",
            TARGET,
            env=env,
            sysrun_args=sysrun_args,
            launcher="mpi",
            num_procs=NUM_PROCS,
        )
        self.assert_regex(result)
        assert (
            banner_count(result.test_output) == 3
        ), f"Expected 3 banners, got {banner_count(result.test_output)}"
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[0, 1, 2],
            ranks_without_output=[],
        )

    def test_mixed_env_output_cli_logs(self, rocpd_env):
        """OUTPUT=0 via env, LOGS=2 via CLI
        Rank 0: file output, no banner
        Rank 1: nothing
        Rank 2: banner, no file output
        """
        rocpd_env["ROCPROFSYS_RANK_FILTER_OUTPUT"] = "0"
        result = self.run_test(
            "sys_run",
            TARGET,
            env=rocpd_env,
            sysrun_args=["--rank-filter-logs", "2"],
            launcher="mpi",
            num_procs=NUM_PROCS,
        )
        self.assert_regex(result)
        assert (
            banner_count(result.test_output) == 1
        ), f"Expected 1 banner, got {banner_count(result.test_output)}"
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[0],
            ranks_without_output=[1, 2],
        )

    def test_overlapping_ranges_cli_output_env_logs(self, rocpd_env):
        """OUTPUT=0-1 via CLI, LOGS=1,2 via env
        Rank 0: file output, no banner
        Rank 1: file output AND banner
        Rank 2: no file output, banner
        """
        rocpd_env["ROCPROFSYS_RANK_FILTER_LOGS"] = "1,2"
        result = self.run_test(
            "sys_run",
            TARGET,
            env=rocpd_env,
            sysrun_args=["--rank-filter-output", "0-1"],
            launcher="mpi",
            num_procs=NUM_PROCS,
        )
        self.assert_regex(result)
        assert banner_count(result.test_output) == 2, (
            f"Expected 2 banners, got " f"{banner_count(result.test_output)}"
        )
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[0, 1],
            ranks_without_output=[2],
        )

    def test_custom_id_excludes_all(self, rocpd_env):
        """Custom rank-ID forces every rank to identify as 10; filter is 0-2.
        Since 10 is not in [0,2] for either filter, every rank is silenced
        for both console and file output.
        """
        rocpd_env["MY_CUSTOM_RANK"] = "10"
        result = self.run_test(
            "sys_run",
            TARGET,
            env=rocpd_env,
            sysrun_args=[
                "--rank-filter-id",
                "MY_CUSTOM_RANK",
                "--rank-filter-output",
                "0-2",
                "--rank-filter-logs",
                "0-2",
            ],
            launcher="mpi",
            num_procs=NUM_PROCS,
        )
        self.assert_regex(result)
        assert banner_count(result.test_output) == 0, (
            f"Expected 0 banners, got " f"{banner_count(result.test_output)}"
        )
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[],
            ranks_without_output=[0, 1, 2],
        )

    def test_custom_id_requires_output_or_logs(self, rocpd_env):
        """--rank-filter-id is meaningless without at least one of
        --rank-filter-output / --rank-filter-logs. rocprof-sys-run should
        reject the args, print the required-options error, and refuse to
        launch — no banner, no per-rank output files.
        """
        result = self.run_test(
            "sys_run",
            TARGET,
            env=rocpd_env,
            sysrun_args=["--rank-filter-id", "MY_CUSTOM_RANK"],
            launcher="mpi",
            num_procs=NUM_PROCS,
            fail_on_pass=True,
        )
        self.assert_regex(
            result,
            "sys_run",
            sys_run_pass_regex=[
                r"--rank-filter-id requires one of the options: "
                r"\[--rank-filter-logs, --rank-filter-output\]"
            ],
        )
        assert (
            banner_count(result.test_output) == 0
        ), f"Expected 0 banners, got {banner_count(result.test_output)}"
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[],
            ranks_without_output=[0, 1, 2],
        )
