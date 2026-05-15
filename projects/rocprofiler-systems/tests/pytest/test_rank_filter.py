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
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.mpi,
    pytest.mark.rocprof_config,
]

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
    count == len(ranks_with_output) since each producing rank emits one)
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
    """Shared mutation target for @pytest.mark.rocpd.

    The conftest's autouse `apply_rocpd_marker` looks up this fixture by
    name and adds `ROCPROFSYS_USE_ROCPD=ON` to it (when RocPD is available).
    Each test then layers its filter-specific env vars on top before
    passing the dict to `run_test(env=...)`. Function-scoped, so each
    test gets a fresh dict.
    """
    return {}


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.class_name("rank-filter")
class TestRankFilter(RocprofsysTest):
    """End-to-end tests for the MPI rank-based output filtering feature."""

    @pytest.mark.rocpd("rocpd_env")
    def test_no_filter_present(self, rocpd_env):
        """No filter set anywhere → all 3 ranks produce console + all files."""
        result = self.run_test(
            "sys_run",
            TARGET,
            env=rocpd_env,
            launcher="mpi",
            num_procs=NUM_PROCS,
        )
        assert banner_count(result.test_output) == 3, (
            f"Expected 3 banners (one per rank), got "
            f"{banner_count(result.test_output)}"
        )
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[0, 1, 2],
            ranks_without_output=[],
        )

    @pytest.mark.rocpd("rocpd_env")
    def test_empty_filter_via_cli(self, rocpd_env):
        """Empty filter via CLI flags → all 3 ranks produce console + all files."""
        result = self.run_test(
            "sys_run",
            TARGET,
            env=rocpd_env,
            sysrun_args=[
                "--rank-filter-output",
                "",
                "--rank-filter-logs",
                "",
            ],
            launcher="mpi",
            num_procs=NUM_PROCS,
        )
        assert (
            banner_count(result.test_output) == 3
        ), f"Expected 3 banners, got {banner_count(result.test_output)}"
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[0, 1, 2],
            ranks_without_output=[],
        )

    @pytest.mark.rocpd("rocpd_env")
    def test_empty_filter_via_env(self, rocpd_env):
        """Empty filter via env vars → all 3 ranks produce console + all files."""
        rocpd_env["ROCPROFSYS_RANK_FILTER_OUTPUT"] = ""
        rocpd_env["ROCPROFSYS_RANK_FILTER_LOGS"] = ""
        result = self.run_test(
            "sys_run",
            TARGET,
            env=rocpd_env,
            launcher="mpi",
            num_procs=NUM_PROCS,
        )
        assert (
            banner_count(result.test_output) == 3
        ), f"Expected 3 banners, got {banner_count(result.test_output)}"
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[0, 1, 2],
            ranks_without_output=[],
        )

    @pytest.mark.rocpd("rocpd_env")
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
        assert (
            banner_count(result.test_output) == 1
        ), f"Expected 1 banner, got {banner_count(result.test_output)}"
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[0],
            ranks_without_output=[1, 2],
        )

    @pytest.mark.rocpd("rocpd_env")
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
        assert banner_count(result.test_output) == 2, (
            f"Expected 2 banners, got " f"{banner_count(result.test_output)}"
        )
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[0, 1],
            ranks_without_output=[2],
        )

    @pytest.mark.rocpd("rocpd_env")
    def test_custom_rank_id_excludes_all(self, rocpd_env):
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
        assert banner_count(result.test_output) == 0, (
            f"Expected 0 banners, got " f"{banner_count(result.test_output)}"
        )
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[],
            ranks_without_output=[0, 1, 2],
        )
