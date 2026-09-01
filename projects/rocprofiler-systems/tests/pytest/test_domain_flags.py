# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Composable domain CLI flags on rocprof-sys-sample (transpose GPU workload).

Env echo is in test_presets.py on ls. This module asserts Perfetto/RocPD artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.domain_flags, pytest.mark.sampling]

TRANSPOSE_ARGS = ["2", "100", "50"]
SAMPLING_VERBOSE = ["-v", "2"]

DOMAIN_FILTER_RULES = frozenset(
    {
        "gpu-metrics-selected",
        "gpu-metrics-busy-mem",
        "gpu-metrics-busy-only",
        "rocm-hip-kernel",
        "rocm-hip-api",
        "sampling-target-gpu0",
    }
)

# trace-hpc disables sampling in the preset JSON; re-enable for GPU metric collection.
_PRESET_SAMPLING_ENV = {
    "ROCPROFSYS_USE_SAMPLING": "ON",
    "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
}

_ROCPD_OFF = {"ROCPROFSYS_USE_ROCPD": "OFF"}
_CPU_TARGET_ENV = {
    **_ROCPD_OFF,
    "ROCPROFSYS_USE_SAMPLING": "OFF",
    "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
}


def _run_transpose_sample(
    test: RocprofsysTest,
    env: dict[str, str],
    sampling_args: list[str],
) -> object:
    return test.run_test(
        "sampling",
        target="transpose",
        env=env,
        sampling_args=[*sampling_args, *SAMPLING_VERBOSE],
        run_args=TRANSPOSE_ARGS,
        check_target_arch=True,
    )


def _resolve_rocpd_rules(
    *,
    validation_rules_dir: Path,
    transpose_baseline_rules: list[Path],
    domain_rules,
    with_amd_smi: bool,
    filter_profiles: list[str],
) -> list[Path]:
    if filter_profiles == ["rocm-hip-api"]:
        return [validation_rules_dir / "domain-flags" / "rocm-hip-api-rules.json"]

    files = list(transpose_baseline_rules)
    if with_amd_smi and not filter_profiles:
        files.append(
            validation_rules_dir / "domain-flags" / "amd-smi-baseline-rules.json"
        )
    for name in filter_profiles:
        if name not in DOMAIN_FILTER_RULES:
            pytest.fail(f"Unknown domain-flags filter profile: {name}")
        files.extend(domain_rules(name))
    return files


def _assert_workload_artifacts(
    test: RocprofsysTest,
    result,
    *,
    env: dict[str, str],
    validation_rules_dir: Path,
    transpose_baseline_rules: list[Path],
    domain_rules,
    perfetto_categories: list[str] | None,
    with_amd_smi: bool,
    filter_profiles: list[str],
) -> None:
    if perfetto_categories:
        test.assert_perfetto(
            result,
            subtest_name="Perfetto domain slices",
            categories=perfetto_categories,
        )
    else:
        test.assert_perfetto(result, subtest_name="Perfetto trace file")

    if env.get("ROCPROFSYS_USE_ROCPD", "ON").upper() == "OFF":
        return

    rules_files = _resolve_rocpd_rules(
        validation_rules_dir=validation_rules_dir,
        transpose_baseline_rules=transpose_baseline_rules,
        domain_rules=domain_rules,
        with_amd_smi=with_amd_smi,
        filter_profiles=filter_profiles,
    )
    test.assert_rocpd(
        result,
        subtest_name="RocPD validation",
        rules_files=rules_files,
    )


# flag_args, pass_regex, seed_env, perfetto_categories, with_amd_smi, filter_profiles
DOMAIN_ARTIFACT_CASES = [
    pytest.param(
        ["--gpu"],
        ["ROCPROFSYS_USE_AMD_SMI=true"],
        None,
        None,
        True,
        [],
        id="gpu_bare",
    ),
    pytest.param(
        ["--gpu=temp,power"],
        ["ROCPROFSYS_AMD_SMI_METRICS=temp,power"],
        None,
        None,
        True,
        ["gpu-metrics-selected"],
        id="gpu_temp_power",
    ),
    pytest.param(
        ["--gpu=busy,mem_usage"],
        ["ROCPROFSYS_AMD_SMI_METRICS=busy,mem_usage"],
        None,
        None,
        True,
        ["gpu-metrics-busy-mem"],
        id="gpu_busy_mem_usage",
    ),
    pytest.param(
        ["--gpu=busy"],
        ["ROCPROFSYS_AMD_SMI_METRICS=busy"],
        None,
        None,
        True,
        ["gpu-metrics-busy-only"],
        id="gpu_busy_only",
    ),
    pytest.param(
        ["--rocm"],
        [
            "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,marker_api,"
            "kernel_dispatch,memory_copy,scratch_memory"
        ],
        None,
        ["hip_runtime_api", "kernel_dispatch", "memory_copy"],
        False,
        [],
        id="rocm_bare",
    ),
    pytest.param(
        ["--rocm=hip,kernel,memory"],
        ["ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy"],
        None,
        ["hip_runtime_api", "kernel_dispatch", "memory_copy"],
        False,
        [],
        id="rocm_hip_kernel_memory",
    ),
    pytest.param(
        ["--rocm=hip,kernel"],
        ["ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch"],
        None,
        ["hip_runtime_api"],
        False,
        ["rocm-hip-kernel"],
        id="rocm_hip_kernel",
    ),
    pytest.param(
        ["--rocm=hip"],
        ["ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api"],
        None,
        ["hip_runtime_api"],
        False,
        ["rocm-hip-api"],
        id="rocm_hip_only",
    ),
    pytest.param(
        ["--cpu"],
        ["ROCPROFSYS_USE_SAMPLING=true", "ROCPROFSYS_SAMPLING_FREQ=100"],
        {"ROCPROFSYS_USE_SAMPLING": "OFF"},
        None,
        False,
        [],
        id="cpu_bare",
    ),
    pytest.param(
        ["--cpu=50"],
        ["ROCPROFSYS_SAMPLING_FREQ=50"],
        None,
        None,
        False,
        [],
        id="cpu_freq_50",
    ),
    pytest.param(
        ["--gpu", "--gpus=0"],
        ["ROCPROFSYS_SAMPLING_GPUS=0", "ROCPROFSYS_USE_AMD_SMI=true"],
        None,
        None,
        True,
        ["sampling-target-gpu0"],
        id="gpu_with_gpus_0",
    ),
    pytest.param(
        ["--gpu", "--gpus=0-1"],
        ["ROCPROFSYS_SAMPLING_GPUS=0-1"],
        None,
        None,
        True,
        [],
        id="gpu_with_gpus_range",
    ),
    pytest.param(
        ["--cpu", "--cpus=0-3"],
        ["ROCPROFSYS_SAMPLING_CPUS=0-3"],
        _CPU_TARGET_ENV,
        None,
        False,
        [],
        id="cpu_with_cpus_range",
    ),
    pytest.param(
        ["--cpu", "--cpus=none"],
        ["ROCPROFSYS_SAMPLING_CPUS=none"],
        {**_CPU_TARGET_ENV, "ROCPROFSYS_USE_SAMPLING": "OFF"},
        None,
        False,
        [],
        id="cpu_with_cpus_none",
    ),
    pytest.param(
        ["--gpu", "--cpu", "--rocm"],
        [
            "ROCPROFSYS_USE_AMD_SMI=true",
            "ROCPROFSYS_USE_SAMPLING=true",
            "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,marker_api,"
            "kernel_dispatch,memory_copy,scratch_memory",
        ],
        {"ROCPROFSYS_USE_SAMPLING": "OFF"},
        ["hip_runtime_api", "kernel_dispatch", "memory_copy"],
        True,
        [],
        id="gpu_cpu_rocm_stack",
    ),
    pytest.param(
        ["--gpu", "--rocm=hip,kernel"],
        [
            "ROCPROFSYS_USE_AMD_SMI=true",
            "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch",
        ],
        None,
        ["hip_runtime_api"],
        True,
        ["rocm-hip-kernel"],
        id="gpu_rocm_composed",
    ),
]

PRESET_ARTIFACT_CASES = [
    pytest.param(
        ["--preset=trace-hpc", "--gpu=temp,power", "--gpus=0"],
        [
            r"Preset:\s+trace-hpc",
            "ROCPROFSYS_SAMPLING_GPUS=0",
            "ROCPROFSYS_USE_AMD_SMI=true",
            r"ROCPROFSYS_AMD_SMI_METRICS=.*temp",
            r"ROCPROFSYS_AMD_SMI_METRICS=.*power",
        ],
        _PRESET_SAMPLING_ENV,
        ["hip_runtime_api", "kernel_dispatch"],
        True,
        ["sampling-target-gpu0"],
        id="preset_domain_and_target",
    ),
    pytest.param(
        ["--preset=trace-gpu"],
        [r"Preset:\s+trace-gpu", "ROCPROFSYS_USE_AMD_SMI=true"],
        None,
        ["hip_runtime_api"],
        True,
        [],
        id="trace_gpu_preset",
    ),
    pytest.param(
        ["--preset=trace-gpu", "--gpus=0"],
        [
            r"Preset:\s+trace-gpu",
            "ROCPROFSYS_SAMPLING_GPUS=0",
            "ROCPROFSYS_USE_AMD_SMI=true",
        ],
        None,
        ["hip_runtime_api"],
        True,
        ["sampling-target-gpu0"],
        id="trace_gpu_preset_with_gpus",
    ),
    pytest.param(
        ["--parallel=mpi,openmp", "--rocm=hip,kernel"],
        [
            "ROCPROFSYS_USE_MPIP=true",
            "ROCPROFSYS_USE_OMPT=true",
            "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch",
        ],
        None,
        ["hip_runtime_api"],
        False,
        ["rocm-hip-kernel"],
        id="parallel_with_rocm",
    ),
]


@pytest.fixture
def domain_flag_env() -> dict[str, str]:
    return {
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_USE_PID": "OFF",
    }


@pytest.fixture
def transpose_baseline_rules(validation_rules_dir: Path) -> list[Path]:
    return [validation_rules_dir / "default-rules.json"]


@pytest.fixture
def domain_rules(validation_rules_dir: Path):
    def _rules(*names: str) -> list[Path]:
        return [
            validation_rules_dir / "domain-flags" / f"{name}-rules.json" for name in names
        ]

    return _rules


@pytest.mark.timeout(180)
@pytest.mark.gpu
@pytest.mark.hip
@pytest.mark.transpose
@pytest.mark.rocpd("domain_flag_env")
@pytest.mark.class_name("domain-flags-artifacts")
class TestDomainFlagsArtifactsOnWorkload(RocprofsysTest):
    @pytest.mark.parametrize(
        "flag_args, pass_regex, seed_env, perfetto_categories, with_amd_smi, filter_profiles",
        DOMAIN_ARTIFACT_CASES,
    )
    def test_on_transpose(
        self,
        domain_flag_env,
        validation_rules_dir,
        transpose_baseline_rules,
        domain_rules,
        flag_args,
        pass_regex,
        seed_env,
        perfetto_categories,
        with_amd_smi,
        filter_profiles,
    ):
        env = {**domain_flag_env, **(seed_env or {})}
        result = _run_transpose_sample(self, env, flag_args)
        self.assert_regex(result, pass_regex=pass_regex)
        _assert_workload_artifacts(
            self,
            result,
            env=env,
            validation_rules_dir=validation_rules_dir,
            transpose_baseline_rules=transpose_baseline_rules,
            domain_rules=domain_rules,
            perfetto_categories=perfetto_categories,
            with_amd_smi=with_amd_smi,
            filter_profiles=filter_profiles,
        )

    @pytest.mark.parametrize(
        "flag_args, pass_regex, seed_env, perfetto_categories, with_amd_smi, filter_profiles",
        PRESET_ARTIFACT_CASES,
    )
    def test_preset_on_transpose(
        self,
        domain_flag_env,
        validation_rules_dir,
        transpose_baseline_rules,
        domain_rules,
        flag_args,
        pass_regex,
        seed_env,
        perfetto_categories,
        with_amd_smi,
        filter_profiles,
    ):
        env = {**domain_flag_env, **(seed_env or {})}
        result = _run_transpose_sample(self, env, flag_args)
        self.assert_regex(result, pass_regex=pass_regex)
        _assert_workload_artifacts(
            self,
            result,
            env=env,
            validation_rules_dir=validation_rules_dir,
            transpose_baseline_rules=transpose_baseline_rules,
            domain_rules=domain_rules,
            perfetto_categories=perfetto_categories,
            with_amd_smi=with_amd_smi,
            filter_profiles=filter_profiles,
        )
