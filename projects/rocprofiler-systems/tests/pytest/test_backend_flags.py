# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Backend -I / -E: env on rocprof-sys-run, GPU artifacts on rocprof-sys-sample."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import RocprofsysTest

pytestmark = [pytest.mark.backend_flags]

RUN_TARGET = "rocprof-sys-run"
VERBOSE_LS = ["-v", "2", "--", "ls"]

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

BACKEND_ENV_CASES = [
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
        marks=pytest.mark.backend_include_ompt,
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
        marks=[pytest.mark.mpi, pytest.mark.gpu],
        id="rcclp",
    ),
    pytest.param(
        ["-I", "mpip", "-I", "ompt"],
        ["ROCPROFSYS_USE_MPIP=true", "ROCPROFSYS_USE_OMPT=true"],
        {"ROCPROFSYS_USE_MPIP": "OFF", "ROCPROFSYS_USE_OMPT": "OFF"},
        marks=[pytest.mark.mpi, pytest.mark.backend_include_ompt],
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
]

BACKEND_GPU_ARTIFACT_CASES = [
    pytest.param(
        ["-I", "amd-smi"],
        ["ROCPROFSYS_USE_AMD_SMI=true"],
        {"ROCPROFSYS_USE_AMD_SMI": "OFF"},
        id="include-amd-smi",
    ),
    pytest.param(
        ["-E", "amd-smi"],
        ["ROCPROFSYS_USE_AMD_SMI=false"],
        {"ROCPROFSYS_USE_AMD_SMI": "ON"},
        id="exclude-amd-smi",
    ),
    pytest.param(
        ["-E", "mutex-locks"],
        ["ROCPROFSYS_TRACE_THREAD_LOCKS=false"],
        _LOCKS_ON,
        id="mutex-locks",
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


@pytest.mark.timeout(60)
@pytest.mark.sys_run
@pytest.mark.class_name("backend-flags-env")
class TestBackendFlagsEnv(RocprofsysTest):
    @pytest.mark.parametrize("flag_args, pass_regex, seed_env", BACKEND_ENV_CASES)
    def test_on_ls(self, backend_flag_env, flag_args, pass_regex, seed_env):
        env = {**backend_flag_env, **(seed_env or {})}
        result = self.run_test(
            "baseline",
            target=RUN_TARGET,
            env=env,
            run_args=[*flag_args, *VERBOSE_LS],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)


@pytest.mark.timeout(180)
@pytest.mark.gpu
@pytest.mark.hip
@pytest.mark.transpose
@pytest.mark.sampling
@pytest.mark.rocpd("backend_flag_env")
@pytest.mark.class_name("backend-flags-artifacts")
class TestBackendFlagsGpuArtifacts(RocprofsysTest):
    @pytest.mark.parametrize(
        "flag_args, pass_regex, seed_env", BACKEND_GPU_ARTIFACT_CASES
    )
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
