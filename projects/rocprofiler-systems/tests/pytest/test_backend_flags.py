# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Backend -I / -E and MPI rank-filter flags on rocprof-sys-sample."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import RocprofsysTest
from test_rank_filter import (
    NUM_PROCS,
    TARGET as MPI_TARGET,
    assert_per_rank_outputs,
    banner_count,
)

pytestmark = [pytest.mark.backend_flags, pytest.mark.sampling]

# Positional args per examples/transpose/README.md: threads, iterations, sync-every-N.
TRANSPOSE_ARGS = ["2", "100", "50"]
SAMPLING_VERBOSE = ["-v", "2"]

_LOCKS_ON = {
    "ROCPROFSYS_TRACE_THREAD_LOCKS": "ON",
    "ROCPROFSYS_TRACE_THREAD_RW_LOCKS": "ON",
    "ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS": "ON",
}

_INCLUDE_ALL_OFF = {
    "ROCPROFSYS_USE_MPIP": "OFF",
    "ROCPROFSYS_USE_OMPT": "OFF",
    "ROCPROFSYS_USE_KOKKOSP": "OFF",
    "ROCPROFSYS_USE_RCCLP": "OFF",
    "ROCPROFSYS_TRACE_THREAD_LOCKS": "OFF",
    "ROCPROFSYS_TRACE_THREAD_RW_LOCKS": "OFF",
    "ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS": "OFF",
}

BACKEND_GPU_ENV_CASES = [
    pytest.param(
        ["-I", "all"],
        [
            "ROCPROFSYS_USE_MPIP=true",
            "ROCPROFSYS_USE_OMPT=true",
            "ROCPROFSYS_USE_KOKKOSP=true",
            "ROCPROFSYS_USE_RCCLP=true",
            "ROCPROFSYS_TRACE_THREAD_LOCKS=true",
            "ROCPROFSYS_TRACE_THREAD_RW_LOCKS=true",
            "ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS=true",
        ],
        _INCLUDE_ALL_OFF,
        marks=pytest.mark.mpi,
        id="include-all",
    ),
    pytest.param(
        ["-I", "mpip"],
        ["ROCPROFSYS_USE_MPIP=true"],
        {"ROCPROFSYS_USE_MPIP": "OFF"},
        marks=pytest.mark.mpi,
        id="mpip",
    ),
    pytest.param(
        ["-I", "ompt"],
        ["ROCPROFSYS_USE_OMPT=true"],
        {"ROCPROFSYS_USE_OMPT": "OFF"},
        id="ompt",
    ),
    pytest.param(
        ["-I", "kokkosp"],
        ["ROCPROFSYS_USE_KOKKOSP=true"],
        {"ROCPROFSYS_USE_KOKKOSP": "OFF"},
        id="kokkosp",
    ),
    pytest.param(
        ["-I", "rcclp"],
        ["ROCPROFSYS_USE_RCCLP=true"],
        {"ROCPROFSYS_USE_RCCLP": "OFF"},
        id="rcclp",
    ),
    pytest.param(
        ["-I", "amd-smi"],
        ["ROCPROFSYS_USE_AMD_SMI=true"],
        {"ROCPROFSYS_USE_AMD_SMI": "OFF"},
        id="include-amd-smi",
    ),
    pytest.param(
        ["-I", "mpip", "-I", "ompt"],
        ["ROCPROFSYS_USE_MPIP=true", "ROCPROFSYS_USE_OMPT=true"],
        {"ROCPROFSYS_USE_MPIP": "OFF", "ROCPROFSYS_USE_OMPT": "OFF"},
        marks=pytest.mark.mpi,
        id="mpip-ompt",
    ),
    pytest.param(
        ["-E", "mutex-locks"],
        ["ROCPROFSYS_TRACE_THREAD_LOCKS=false"],
        _LOCKS_ON,
        id="mutex-locks",
    ),
    pytest.param(
        ["-E", "rw-locks"],
        ["ROCPROFSYS_TRACE_THREAD_RW_LOCKS=false"],
        _LOCKS_ON,
        id="rw-locks",
    ),
    pytest.param(
        ["-E", "spin-locks"],
        ["ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS=false"],
        _LOCKS_ON,
        id="spin-locks",
    ),
    pytest.param(
        ["-E", "mutex-locks", "rw-locks", "spin-locks"],
        [
            "ROCPROFSYS_TRACE_THREAD_LOCKS=false",
            "ROCPROFSYS_TRACE_THREAD_RW_LOCKS=false",
            "ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS=false",
        ],
        _LOCKS_ON,
        id="all-locks",
    ),
    pytest.param(
        ["-E", "amd-smi"],
        ["ROCPROFSYS_USE_AMD_SMI=false"],
        {"ROCPROFSYS_USE_AMD_SMI": "ON"},
        id="exclude-amd-smi",
    ),
]


@pytest.fixture
def backend_flag_env() -> dict[str, str]:
    return {
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_USE_PID": "OFF",
    }


@pytest.fixture
def transpose_baseline_rules(validation_rules_dir: Path) -> list[Path]:
    return [validation_rules_dir / "default-rules.json"]


_ROCM_KERNEL_ARGS = ["--rocm=hip,kernel"]
_PERFETTO_HIP = ["hip_runtime_api"]


def _backend_rocpd_rules(
    *,
    validation_rules_dir: Path,
    transpose_baseline_rules: list[Path],
    pass_regex: list[str],
) -> list[Path]:
    files = [*transpose_baseline_rules]
    if not any("ROCPROFSYS_USE_AMD_SMI=false" in p for p in pass_regex):
        files.append(
            validation_rules_dir / "domain-flags" / "amd-smi-baseline-rules.json"
        )
    return files


def _run_transpose_sample(
    test: RocprofsysTest,
    backend_flag_env: dict[str, str],
    flag_args: list[str],
    *,
    extra_env: dict[str, str] | None = None,
):
    env = {**backend_flag_env, **(extra_env or {})}
    return test.run_test(
        "sampling",
        target="transpose",
        env=env,
        sampling_args=[*flag_args, *_ROCM_KERNEL_ARGS, *SAMPLING_VERBOSE],
        run_args=TRANSPOSE_ARGS,
        check_target_arch=True,
    )


@pytest.mark.timeout(180)
@pytest.mark.gpu
@pytest.mark.hip
@pytest.mark.transpose
@pytest.mark.rocpd("backend_flag_env")
@pytest.mark.class_name("backend-flags")
class TestBackendFlagsOnGpuWorkload(RocprofsysTest):
    @pytest.mark.parametrize("flag_args, pass_regex, seed_env", BACKEND_GPU_ENV_CASES)
    def test_on_transpose(
        self,
        backend_flag_env,
        validation_rules_dir,
        transpose_baseline_rules,
        flag_args,
        pass_regex,
        seed_env,
    ):
        result = _run_transpose_sample(
            self, backend_flag_env, flag_args, extra_env=seed_env
        )
        self.assert_regex(result, pass_regex=pass_regex)
        self.assert_perfetto(
            result,
            subtest_name="Perfetto HIP API",
            categories=_PERFETTO_HIP,
        )
        self.assert_rocpd(
            result,
            subtest_name="RocPD validation",
            rules_files=_backend_rocpd_rules(
                validation_rules_dir=validation_rules_dir,
                transpose_baseline_rules=transpose_baseline_rules,
                pass_regex=pass_regex,
            ),
        )


@pytest.fixture
def rocpd_env() -> dict[str, str]:
    return {}


@pytest.mark.timeout(180)
@pytest.mark.mpi
@pytest.mark.rank_filter
@pytest.mark.sampling
@pytest.mark.rocpd("rocpd_env")
@pytest.mark.class_name("sample-rank-filter")
class TestSampleRankFilter(RocprofsysTest):
    def test_output_range(self, rocpd_env):
        result = self.run_test(
            "sampling",
            MPI_TARGET,
            env=rocpd_env,
            sampling_args=["--rank-filter-output", "0-1"],
            launcher="mpi",
            num_procs=NUM_PROCS,
        )
        self.assert_regex(result)
        assert (
            banner_count(result.test_output) == 3
        ), f"Expected 3 banners, got {banner_count(result.test_output)}"
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[0, 1],
            ranks_without_output=[2],
        )

    def test_logs_single_rank(self, rocpd_env):
        result = self.run_test(
            "sampling",
            MPI_TARGET,
            env=rocpd_env,
            sampling_args=["--rank-filter-logs", "2"],
            launcher="mpi",
            num_procs=NUM_PROCS,
        )
        self.assert_regex(result)
        assert (
            banner_count(result.test_output) == 1
        ), f"Expected 1 banner, got {banner_count(result.test_output)}"
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[0, 1, 2],
            ranks_without_output=[],
        )

    def test_custom_id(self, rocpd_env):
        env = {**rocpd_env, "MY_CUSTOM_RANK": "1"}
        result = self.run_test(
            "sampling",
            MPI_TARGET,
            env=env,
            sampling_args=[
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
        assert (
            banner_count(result.test_output) == 3
        ), f"Expected 3 banners, got {banner_count(result.test_output)}"
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[0, 1, 2],
            ranks_without_output=[],
        )

    def test_output_and_logs_mixed(self, rocpd_env):
        env = {**rocpd_env, "ROCPROFSYS_RANK_FILTER_OUTPUT": "0"}
        result = self.run_test(
            "sampling",
            MPI_TARGET,
            env=env,
            sampling_args=["--rank-filter-logs", "2"],
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

    def test_custom_id_requires_output_or_logs(self, rocpd_env):
        result = self.run_test(
            "sampling",
            MPI_TARGET,
            env=rocpd_env,
            sampling_args=["--rank-filter-id", "MY_CUSTOM_RANK"],
            launcher="mpi",
            num_procs=NUM_PROCS,
            fail_on_pass=True,
        )
        self.assert_regex(
            result,
            "sampling",
            sampling_pass_regex=[
                r"--rank-filter-id requires one of the options: "
                r"\[--rank-filter-logs, --rank-filter-output\]"
            ],
            use_abort_fail_regex=False,
        )
        assert (
            banner_count(result.test_output) == 0
        ), f"Expected 0 banners, got {banner_count(result.test_output)}"
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[],
            ranks_without_output=[0, 1, 2],
        )

    def test_overlapping_output_cli_logs_env(self, rocpd_env):
        env = {**rocpd_env, "ROCPROFSYS_RANK_FILTER_LOGS": "1,2"}
        result = self.run_test(
            "sampling",
            MPI_TARGET,
            env=env,
            sampling_args=["--rank-filter-output", "0-1"],
            launcher="mpi",
            num_procs=NUM_PROCS,
        )
        self.assert_regex(result)
        assert (
            banner_count(result.test_output) == 2
        ), f"Expected 2 banners, got {banner_count(result.test_output)}"
        assert_per_rank_outputs(
            self.test_output_dir,
            ranks_with_output=[0, 1],
            ranks_without_output=[2],
        )

    def test_out_of_range_output_disables_filter(self, rocpd_env):
        result = self.run_test(
            "sampling",
            MPI_TARGET,
            env=rocpd_env,
            sampling_args=["--rank-filter-output", "5,6"],
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

    @pytest.mark.parametrize(
        "filter_source",
        [
            pytest.param("unset", id=""),
            "via_cli",
            "via_env",
        ],
    )
    def test_no_filter(self, rocpd_env, filter_source):
        sampling_args: list[str] = []
        env = rocpd_env
        if filter_source == "via_cli":
            sampling_args = ["--rank-filter-output", "", "--rank-filter-logs", ""]
        elif filter_source == "via_env":
            env = {
                **rocpd_env,
                "ROCPROFSYS_RANK_FILTER_OUTPUT": "",
                "ROCPROFSYS_RANK_FILTER_LOGS": "",
            }

        result = self.run_test(
            "sampling",
            MPI_TARGET,
            env=env,
            sampling_args=sampling_args,
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
