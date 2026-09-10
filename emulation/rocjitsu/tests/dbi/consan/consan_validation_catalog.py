"""Canonical ConSan validation profiles, workloads, and target policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

import consan_cdna_hip_moi_registry as cdna_hip_moi_registry


class ValidationError(RuntimeError):
    """The requested validation contract cannot be satisfied."""

SCHEMA_VERSION = 2
EMPIRICAL_CAMPAIGN_SCHEMA_VERSION = 3
PROVENANCE_SCHEMA_VERSION = 3
WORKSPACE_ENV = "CONSAN_VALIDATION_WORKSPACE_DIR"
TARGET_ENV = "CONSAN_VALIDATION_TARGET"
PYTORCH_PYTHON_ENV = "CONSAN_VALIDATION_PYTORCH_PYTHON"
SHARKTANK_PYTHON_ENV = "CONSAN_VALIDATION_SHARKTANK_PYTHON"
TENSILE_PYTHON_ENV = "CONSAN_VALIDATION_TENSILE_PYTHON"
RDNA4_MATMUL_DIR_ENV = "CONSAN_VALIDATION_RDNA4_MATMUL_DIR"
LLAMA_BUILD_DIR_ENV = "CONSAN_VALIDATION_LLAMA_BUILD_DIR"
LLVM_READELF_ENV = "CONSAN_VALIDATION_LLVM_READELF"
HIP_MOI_GPU_BENCHMARK_ITERATIONS_ENV = "HIP_MOI_TEST_GPU_BENCHMARK_ITERATIONS"
HIP_MOI_GPU_BENCHMARK_WARMUP_ITERATIONS_ENV = (
    "HIP_MOI_TEST_GPU_BENCHMARK_WARMUP_ITERATIONS"
)
TIMEOUT_SECONDS = 30
EMPIRICAL_DEFAULT_ROUNDS = 10
EMPIRICAL_DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
EMPIRICAL_DEFAULT_BASELINE_DRIFT_LIMIT = 0.05
EMPIRICAL_MINIMUM_TIMED_MS = 250.0
TENSILE_SHARD_TIMING_CANARY_MS = 1.0
EMPIRICAL_MAX_INNER_REPETITIONS = 1_000_000
PROCESS_OUTPUT_DRAIN_SECONDS = 2
PROCESS_TERMINATION_GRACE_SECONDS = 5
NATIVE_CDNA_TARGETS = frozenset(("gfx942", "gfx950"))
SINGLE_REPETITION_TARGETS = frozenset(("gfx942", "gfx950", "gfx1250"))
QWEN_OVERHEAD_REPETITIONS = {target: 1 for target in SINGLE_REPETITION_TARGETS}
QWEN_BUILD_MANIFEST_SCHEMA_VERSION = 1
QWEN_COMPILE_OPTIONS = (
    "--iree-hal-target-device=hip",
    "--iree-opt-level=O3",
    "--iree-parameter-encoder-output-scope=encoded",
    "--iree-parameter-encoder-mode=overlay",
)
PYTORCH_OVERHEAD_PROCESSES = 10
STREAMK_WORKLOAD_IDS = ("streamk-arrival", "tree-atomic-or")
STREAMK_FAULT_FAMILIES = ("atomic-weaken-order", "atomic-weaken-scope")
CONTROLLED_ENV_PREFIX = "RJ_CONSAN_"
TOOLS = ("iree-run-module", "iree-benchmark-module", "rocminfo")
HSA_TOOL_ENVIRONMENT = {
    "HSA_TOOLS_LIB",
    "HSA_TOOLS_ROCPROFILER_V1_TOOLS",
}
SOFTWARE_MODEL_ENVIRONMENT = {
    "HSA_MODEL_LIB",
    "HSAKMT_SIM_LIB",
    "HSA_MODEL_TOPOLOGY",
    "HSA_MODEL_NUM_THREADS",
    "HSA_ENABLE_SDMA",
    "HSA_ENABLE_SCRATCH_ASYNC_RECLAIM",
    "HSA_ENABLE_INTERRUPT",
}


SETTING_CATEGORIES = {
    "runtime-plumbing": "Locates the target or instrumentation runtime.",
    "instrumentation-selection": (
        "Selects a ConSan flavor or engine, or overrides an event-family default."
    ),
    "acceptance-assertion": "Makes missing or unexpected evidence fail validation.",
    "workload-tuning": "Changes a workload-specific instrumentation operating point.",
    "fault-injection": "Selects and constrains a deliberate mutation.",
    "fault-containment": "Serializes or contains destructive fault execution.",
}

ORDINARY_FORBIDDEN_ENVIRONMENT = (
    "RJ_CONSAN_MAX_PATCHES",
    "RJ_CONSAN_TEST_KERNEL_FILTER",
    "RJ_CONSAN_TMP_VGPR",
    "RJ_CONSAN_SCRATCH_VGPR",
    "RJ_CONSAN_MOI_OWNER_VGPR",
    "RJ_CONSAN_MOI_EPOCH_VGPR",
    "RJ_CONSAN_MOI_EXEC_SAVE_SGPR",
    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL",
)

ORDINARY_MOI_RUNTIME_DEFAULTS = {
    "RJ_CONSAN_MOI_TRACK_BARRIERS": "1",
    "RJ_CONSAN_MOI_TRACK_ATOMICS": "1",
}

SAMPLED_STANDARD_RUNTIME_DEFAULTS = {
    "RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE": "256",
    "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET": "0",
}

RECORD_REPLAY_STANDARD_RUNTIME_DEFAULTS = {
    "RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE": "65536",
    "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET": "0",
}

FAULT_FAMILY_ENVIRONMENTS = {
    "barrier-drop": {
        "RJ_CONSAN_FAULT_DROP_BARRIER": "1",
        "RJ_CONSAN_FAULT_SITE_IDENTITY": "REPLACE_FROM_INVENTORY",
        "RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY": "REPLACE_FROM_INVENTORY",
    },
    "barrier-move": {
        "RJ_CONSAN_FAULT_MOVE_BARRIER": "1",
        "RJ_CONSAN_FAULT_BARRIER_MOVE_DIRECTION": "later",
        "RJ_CONSAN_FAULT_SITE_IDENTITY": "REPLACE_FROM_INVENTORY",
        "RJ_CONSAN_FAULT_BARRIER_DESTINATION_IDENTITY": "REPLACE_FROM_INVENTORY",
    },
    "atomic-weaken-order": {
        "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER": "1",
        "RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE": "release",
        "RJ_CONSAN_FAULT_SITE_IDENTITY": "REPLACE_FROM_INVENTORY",
    },
    "atomic-weaken-scope": {
        "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_SCOPE": "1",
        "RJ_CONSAN_FAULT_SITE_IDENTITY": "REPLACE_FROM_INVENTORY",
    },
    "lds-wrong-address": {
        "RJ_CONSAN_FAULT_LDS_WRONG_ADDRESS": "1",
        "RJ_CONSAN_FAULT_LDS_ADDRESS_VGPR": "0",
        "RJ_CONSAN_FAULT_SITE_IDENTITY": "REPLACE_FROM_INVENTORY",
    },
}

FAULT_FAMILY_SITE_KINDS = {
    "barrier-drop": "barrier",
    "barrier-move": "barrier",
    "atomic-weaken-order": "atomic",
    "atomic-weaken-scope": "atomic",
    "lds-wrong-address": "lds-access",
}

assert (
    FAULT_FAMILY_SITE_KINDS.keys() == FAULT_FAMILY_ENVIRONMENTS.keys()
), "every ConSan fault family must declare both its environment and inventory site kind"


def _fault_family_environment(target: str, family: str) -> dict[str, str]:
    environment = dict(FAULT_FAMILY_ENVIRONMENTS[family])
    if target in NATIVE_CDNA_TARGETS and family == "barrier-drop":
        # CDNA3/4 represent a full workgroup barrier with one s_barrier. Unlike
        # RDNA4's signal/wait pair, it has no two-member logical sequence that
        # must be selected and dropped atomically.
        environment.pop("RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY")
    return environment


@dataclass(frozen=True)
class Profile:
    id: str
    flavor: str
    engine: str
    environment: dict[str, str]


@dataclass(frozen=True)
class Workload:
    id: str
    priority: str
    corpus: str
    kind: str
    relative_path: str
    clean_filter: str | None
    overhead_filter: str | None
    sharktank_workload: str | None
    sharktank_mode: str | None
    tracks_barriers: bool
    tracks_atomics: bool
    overhead_processes: int
    fault_families: tuple[str, ...]
    fault_filter: str | None = None
    command_arguments: tuple[str, ...] = ()
    command_environment: tuple[tuple[str, str], ...] = ()
    targets: tuple[str, ...] | None = None
    moi_record_evidence_expected: bool = True
    record_replay_runtime_sample_stride: int | None = None
    sharktank_skip_warmup: bool = False
    run_timeout_seconds: int = TIMEOUT_SECONDS
    tensile_inner_timeout_seconds: int | None = None
    tensile_expected_numeric_rows: int | None = None
    tensile_exact_problem_size_shards: tuple[
        tuple[tuple[int, ...], ...], ...
    ] = ()
    tensile_expected_source_exact_problem_size_blocks: tuple[
        tuple[tuple[int, ...], ...], ...
    ] = ()
    tensile_expected_numeric_rows_per_shard: tuple[int, ...] = ()
    tensile_expected_client_passes: int | None = None
    tensile_expected_client_passes_per_shard: tuple[int, ...] = ()
    tensile_shard_parallelism: int = 1
    tensile_fault_shard_index: int | None = None
    tensile_streamk_fixed_grid: int | None = None
    tensile_streamk_mode: int | None = None
    tensile_minimum_timed_ms: float = EMPIRICAL_MINIMUM_TIMED_MS
    self_timed_device_minimum_ms: float | None = None
    warm_timing_mode: str | None = None
    empirical_device_timed_minimum_ms: float | None = None
    device_timing_max_iterations: int | None = None
    device_timing_warmup_iterations: int = 5
    device_timing_calibration_iterations: int | None = None
    device_timing_aggregate_headroom: float = 1.0


PROFILES = {
    "supercollider": Profile(
        id="supercollider",
        flavor="supercollider",
        engine="supercollider",
        environment={
            "RJ_CONSAN_MODE": "supercollider",
            "RJ_CONSAN_POLICY": "strict",
        },
    ),
    "record-replay": Profile(
        id="record-replay",
        flavor="moi",
        engine="record_replay",
        environment={
            "RJ_CONSAN_MODE": "record-replay",
            "RJ_CONSAN_POLICY": "strict",
            "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS": "1",
        },
    ),
    "sampled": Profile(
        id="sampled",
        flavor="moi",
        engine="sampled",
        environment={
            "RJ_CONSAN_MODE": "sampled",
            "RJ_CONSAN_POLICY": "strict",
            "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS": "1",
            "RJ_CONSAN_MOI_REQUIRE_RECORDS": "0",
        },
    ),
    "inline-shadow": Profile(
        id="inline-shadow",
        flavor="moi",
        engine="inline_shadow",
        environment={
            "RJ_CONSAN_MODE": "inline-shadow",
            "RJ_CONSAN_POLICY": "strict",
            "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS": "1",
        },
    ),
}

PROFILE_IDS = tuple(PROFILES)
# Mirrors ConSanMoiDiagnosticKind in consan_moi_core_types.h.inc. Unknown
# numeric values remain fail-closed so a device-side enum addition cannot be
# silently accepted by an older validation harness.
MOI_DIAGNOSTIC_KINDS = {
    1: "access-conflict",
    2: "metadata-full",
    3: "barrier-divergence",
}
# Mirrors ConSanMoiShadowAccessKind::Write in consan_moi_abi.h.
MOI_SHADOW_ACCESS_WRITE = 2
WORKLOADS = (
    Workload(
        id="tensile-sk-mxf8gemm-explicit",
        priority="P0",
        corpus="rocjitsu-test-corpus",
        kind="tensile",
        relative_path=(
            "corpus/tensile/configs/Tensile/Tests/common/streamk/gfx1250/"
            "sk_mxf8gemm_explicit.yaml"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
    ),
    Workload(
        id="tensile-sk-mxf4gemm-explicit",
        priority="P0",
        corpus="rocjitsu-test-corpus",
        kind="tensile",
        relative_path=(
            "corpus/tensile/configs/Tensile/Tests/common/streamk/gfx1250/"
            "sk_mxf4gemm_explicit.yaml"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
    ),
    Workload(
        id="tensile-spmm-tdm-f16-transposes",
        priority="P1",
        corpus="rocjitsu-test-corpus",
        kind="tensile",
        relative_path=(
            "corpus/tensile/configs/Tensile/Tests/common/sparse/gfx1250/"
            "spmm_tdm_f16_transposes.yaml"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
    ),
    Workload(
        id="tensile-spmm-tdm-all",
        priority="P1",
        corpus="rocjitsu-test-corpus",
        kind="tensile",
        relative_path=(
            "corpus/tensile/configs/Tensile/Tests/common/sparse/gfx1250/"
            "spmm_tdm_all.yaml"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
    ),
    Workload(
        id="tensile-sk-mxf8f4gemm-tdm",
        priority="P1",
        corpus="rocjitsu-test-corpus",
        kind="tensile",
        relative_path=(
            "corpus/tensile/configs/Tensile/Tests/common/streamk/gfx1250/"
            "sk_mxf8f4gemm_tdm.yaml"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
    ),
    Workload(
        id="tensile-sk-mxf8gemm-tdm",
        priority="P1",
        corpus="rocjitsu-test-corpus",
        kind="tensile",
        relative_path=(
            "corpus/tensile/configs/Tensile/Tests/common/streamk/gfx1250/"
            "sk_mxf8gemm_tdm.yaml"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
    ),
    Workload(
        id="tensile-sk-mxf4gemm-tdm",
        priority="P1",
        corpus="rocjitsu-test-corpus",
        kind="tensile",
        relative_path=(
            "corpus/tensile/configs/Tensile/Tests/common/streamk/gfx1250/"
            "sk_mxf4gemm_tdm.yaml"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
    ),
    Workload(
        id="tensile-sk-sgemm-runtime-smoke",
        priority="P1",
        corpus="rocm-systems",
        kind="tensile",
        relative_path=(
            "emulation/rocjitsu/tests/dbi/consan/fixtures/"
            "gfx1250_tensile_streamk_smoke.yaml"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
        run_timeout_seconds=60,
        tensile_inner_timeout_seconds=55,
        tensile_expected_numeric_rows=1,
        tensile_streamk_fixed_grid=4,
        tensile_streamk_mode=3,
        # This is a one-row functional smoke, not an empirical timing row.
        # Retain a positive device-timing canary without requiring the
        # aggregate duration used by repeated performance measurements.
        tensile_minimum_timed_ms=5.0,
    ),
    Workload(
        id="tensile-gfx950-lds-positive",
        priority="P2",
        corpus="rocm-systems",
        kind="tensile",
        relative_path=(
            "emulation/rocjitsu/tests/dbi/consan/fixtures/"
            "gfx950_tensile_lds_positive.yaml"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("lds-wrong-address",),
        targets=("gfx950",),
        run_timeout_seconds=60,
        tensile_inner_timeout_seconds=55,
        tensile_expected_numeric_rows=1,
    ),
    Workload(
        id="hip-matmul-m128-n128-k128",
        priority="P0",
        corpus="rocjitsu-test-corpus",
        kind="native-executable",
        relative_path=(
            "rocjitsu-test-corpus-build/kernels-gfx950-hip-matmul/cases/"
            "hip-matmul/hip_matmul_matmul"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        command_arguments=("-m", "128", "-n", "128", "-k", "128"),
        command_environment=(("FIXED_ITERATIONS", "1"),),
        targets=("gfx950",),
        run_timeout_seconds=120,
    ),
    Workload(
        id="hipkittens-bf16fp32-16x32",
        priority="P0",
        corpus="rocjitsu-test-corpus",
        kind="native-executable",
        relative_path=(
            "rocjitsu-test-corpus-build/kernels-gfx950-hipkittens/cases/"
            "hipkittens/hipkittens_gemm_bf16fp32_16x32"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        command_arguments=("-m", "256", "-n", "256", "-k", "256"),
        targets=("gfx950",),
        run_timeout_seconds=60,
    ),
    Workload(
        id="hipkittens-fp8fp32-4wave",
        priority="P1",
        corpus="rocjitsu-test-corpus",
        kind="native-executable",
        relative_path=(
            "rocjitsu-test-corpus-build/kernels-gfx950-hipkittens/cases/"
            "hipkittens/hipkittens_gemm_fp8fp32_4wave"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        command_arguments=(
            "-m",
            "256",
            "-n",
            "256",
            "-k",
            "256",
            "--rotating-buffer-count",
            "4",
        ),
        targets=("gfx950",),
        run_timeout_seconds=60,
    ),
    Workload(
        id="hipkittens-mxfp8-4wave",
        priority="P1",
        corpus="rocjitsu-test-corpus",
        kind="native-executable",
        relative_path=(
            "rocjitsu-test-corpus-build/kernels-gfx950-hipkittens/cases/"
            "hipkittens/hipkittens_gemm_mxfp8_4wave"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        command_arguments=("-m", "256", "-n", "256", "-k", "256"),
        targets=("gfx950",),
        run_timeout_seconds=60,
    ),
    Workload(
        id="hipkittens-bf16fp32-cdna5-naive",
        priority="P0",
        corpus="rocjitsu-test-corpus",
        kind="native-executable",
        relative_path=(
            "rocjitsu-test-corpus-build/kernels-gfx1250-hipkittens/cases/"
            "hipkittens/hipkittens_gemm_bf16fp32_gfx1250_naive"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        # One complete 64x64 output tile and one 32-wide reduction step retain
        # every instruction in this fixed kernel while avoiding redundant
        # emulation of identical workgroups and loop iterations.
        command_arguments=("64", "64", "32", "1", "1"),
        targets=("gfx1250",),
        run_timeout_seconds=60,
    ),
    Workload(
        id="hip-streamk-simple-m256-n256-k256",
        priority="P1",
        corpus="rocjitsu-test-corpus",
        kind="native-executable",
        relative_path=(
            "rocjitsu-test-corpus-build/kernels-gfx950-streamk-current/cases/"
            "hip-stream-k/hip_streamk_simple"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        command_arguments=(
            "-m",
            "256",
            "-n",
            "256",
            "-k",
            "256",
            "--grid",
            "4",
            "--num_runs",
            "1",
            "--validate",
        ),
        targets=("gfx950",),
        record_replay_runtime_sample_stride=4,
        run_timeout_seconds=120,
    ),
    Workload(
        id="hip-streamk-two-tile-m256-n256-k256",
        priority="P1",
        corpus="rocjitsu-test-corpus",
        kind="native-executable",
        relative_path=(
            "rocjitsu-test-corpus-build/kernels-gfx950-streamk-current/cases/"
            "hip-stream-k/hip_streamk_two_tile"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        command_arguments=(
            "-m",
            "256",
            "-n",
            "256",
            "-k",
            "256",
            "--num_runs",
            "1",
            "--validate",
        ),
        targets=("gfx950",),
        record_replay_runtime_sample_stride=32,
        run_timeout_seconds=120,
    ),
    Workload(
        id="rocblas-sgemm-square-64",
        priority="P2",
        corpus="rocjitsu-test-corpus",
        kind="native-executable",
        relative_path=(
            "rocjitsu-test-corpus-build/kernels-gfx950-rocblas/cases/"
            "rocblas/rocblas_sgemm"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        command_arguments=("--gtest_filter=RocblasGemmTest.Square_64x64",),
        targets=("gfx950",),
        run_timeout_seconds=120,
    ),
    Workload(
        id="tensile-sk-sgemm-quick",
        priority="P2",
        corpus="rocjitsu-test-corpus",
        kind="tensile",
        relative_path=(
            "corpus/tensile/configs/Tensile/Tests/common/streamk/gfx1250/"
            "sk_sgemm_quick.yaml"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
    ),
    Workload(
        id="tensile-sk-f8gemm-quick",
        priority="P2",
        corpus="rocjitsu-test-corpus",
        kind="tensile",
        relative_path=(
            "corpus/tensile/configs/Tensile/Tests/common/streamk/gfx1250/"
            "sk_f8gemm_quick.yaml"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
    ),
    Workload(
        id="tensile-sk-hgemm-quick",
        priority="P2",
        corpus="rocjitsu-test-corpus",
        kind="tensile",
        relative_path=(
            "corpus/tensile/configs/Tensile/Tests/common/streamk/gfx1250/"
            "sk_hgemm_quick.yaml"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
    ),
    Workload(
        id="tensile-spmm-f16-sb",
        priority="P2",
        corpus="rocjitsu-test-corpus",
        kind="tensile",
        relative_path=(
            "corpus/tensile/configs/Tensile/Tests/common/sparse/gfx1250/"
            "spmm_f16_sb.yaml"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
    ),
    Workload(
        id="tensile-spmm-f8-ml",
        priority="P3",
        corpus="rocjitsu-test-corpus",
        kind="tensile",
        relative_path=(
            "corpus/tensile/configs/Tensile/Tests/common/sparse/gfx1250/"
            "spmm_f8_ml.yaml"
        ),
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
    ),
    Workload(
        id="pytorch-tdm-descriptor-add",
        priority="P0",
        corpus="pytorch",
        kind="pytorch",
        relative_path="consan_pytorch_validation.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
    ),
    Workload(
        id="pytorch-cluster-load-sync",
        priority="P1",
        corpus="pytorch",
        kind="pytorch",
        relative_path="consan_pytorch_validation.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1250",),
    ),
    Workload(
        id="pytorch-torch-mode",
        priority="P0",
        corpus="pytorch",
        kind="pytorch",
        warm_timing_mode="host-json",
        relative_path="consan_pytorch_validation.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx950", "gfx1250", "gfx1201"),
        # This row proves the large object fits the ordinary bound even if the
        # harness-wide default is relaxed later.
        run_timeout_seconds=30,
    ),
    Workload(
        id="pytorch-torch-topk",
        priority="P0",
        corpus="pytorch",
        kind="pytorch",
        relative_path="consan_pytorch_validation.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx950", "gfx1250"),
    ),
    Workload(
        id="pytorch-torch-sort",
        priority="P1",
        corpus="pytorch",
        kind="pytorch",
        relative_path="consan_pytorch_validation.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx950", "gfx1250"),
    ),
    Workload(
        id="pytorch-scatter-reduce",
        priority="P1",
        corpus="pytorch",
        kind="pytorch",
        warm_timing_mode="host-json",
        relative_path="consan_pytorch_validation.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=False,
        tracks_atomics=True,
        overhead_processes=1,
        fault_families=("atomic-weaken-order", "atomic-weaken-scope"),
        targets=("gfx950", "gfx1250", "gfx1201"),
        moi_record_evidence_expected=False,
    ),
    Workload(
        id="pytorch-torch-histc",
        priority="P1",
        corpus="pytorch",
        kind="pytorch",
        warm_timing_mode="host-json",
        relative_path="consan_pytorch_validation.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=True,
        overhead_processes=1,
        fault_families=("barrier-drop", "atomic-weaken-order", "atomic-weaken-scope"),
        # This ordinary upstream operation is selected independently by each
        # installed wheel.  The gfx1201 wheel chooses a native histogram
        # kernel with LDS accesses, split barriers, and LDS atomics; target
        # evidence and qualification remain separate.
        targets=("gfx950", "gfx1250", "gfx1201"),
    ),
    Workload(
        id="pytorch-norm-softmax",
        priority="P2",
        corpus="pytorch",
        kind="pytorch",
        relative_path="consan_pytorch_validation.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx950", "gfx1250"),
    ),
    Workload(
        id="pytorch-rdna4-compiled-softmax",
        priority="P2",
        corpus="pytorch",
        kind="pytorch",
        warm_timing_mode="device-json",
        relative_path="consan_pytorch_validation.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1201",),
        device_timing_calibration_iterations=100,
        device_timing_aggregate_headroom=1.25,
    ),
    Workload(
        id="pytorch-rdna4-split-softmax",
        priority="P2",
        corpus="pytorch",
        kind="pytorch",
        warm_timing_mode="host-json",
        relative_path="consan_pytorch_validation.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1201",),
    ),
    Workload(
        id="pytorch-rdna4-llm-topk",
        priority="P2",
        corpus="pytorch",
        kind="pytorch",
        relative_path="consan_pytorch_validation.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=PYTORCH_OVERHEAD_PROCESSES,
        fault_families=("barrier-drop",),
        targets=("gfx1201",),
        run_timeout_seconds=120,
    ),
    Workload(
        id="rdna4-matmul-fp16-production",
        priority="P0",
        corpus="rdna4-matmul",
        kind="rdna4-matmul",
        warm_timing_mode="device-fixed",
        relative_path="build/rdna4_matmul_production",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop", "lds-wrong-address"),
        targets=("gfx1201",),
        run_timeout_seconds=60,
        self_timed_device_minimum_ms=EMPIRICAL_MINIMUM_TIMED_MS,
    ),
    Workload(
        id="rdna4-matmul-fp8-production",
        priority="P0",
        corpus="rdna4-matmul",
        kind="rdna4-matmul",
        warm_timing_mode="device-fixed",
        relative_path="build/rdna4_matmul_production",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop", "lds-wrong-address"),
        targets=("gfx1201",),
        run_timeout_seconds=60,
        self_timed_device_minimum_ms=EMPIRICAL_MINIMUM_TIMED_MS,
    ),
    Workload(
        id="llama-rdna4-mul-mat-vec-q",
        priority="P2",
        corpus="rocjitsu-test-corpus",
        kind="llama",
        relative_path="llama_cpp_mul_mat_vec_q",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=True,
        overhead_processes=5,
        fault_families=("barrier-drop",),
        targets=("gfx1201",),
        self_timed_device_minimum_ms=EMPIRICAL_MINIMUM_TIMED_MS,
        warm_timing_mode="device-fixed",
    ),
    Workload(
        id="llama-rdna4-rms-norm",
        priority="P3",
        corpus="rocjitsu-test-corpus",
        kind="llama",
        relative_path="llama_cpp_rms_norm",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=5,
        fault_families=("barrier-drop",),
        targets=("gfx1201",),
    ),
    Workload(
        id="qwen-prefill",
        priority="P0",
        corpus="iree-test-suites",
        kind="qwen",
        warm_timing_mode="host-json",
        relative_path="iree-test-suites-build/torch_models/qwen3-600m",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
    ),
    Workload(
        id="tp1-prefill",
        priority="P1",
        corpus="iree-test-suites",
        kind="sharktank",
        warm_timing_mode="host-json",
        relative_path="iree-test-suites/sharktank_models/llama3.1/test_llama.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload="tp1",
        sharktank_mode="prefill",
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
    ),
    Workload(
        id="tp1-decode-combined",
        priority="P1",
        corpus="iree-test-suites",
        kind="sharktank",
        warm_timing_mode="host-json",
        relative_path="iree-test-suites/sharktank_models/llama3.1/test_llama.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload="tp1",
        sharktank_mode="decode-combined",
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
        # The two-mode process performs independent model setup plus decode and
        # combined inference. The gfx1250 simulator's accepted clean row takes
        # roughly 77 seconds, so retain the same bounded 180-second contract
        # for clean, overhead, inventory, and fault automation.
        run_timeout_seconds=180,
    ),
    Workload(
        id="tp2-family",
        priority="P2",
        corpus="iree-test-suites",
        kind="sharktank",
        warm_timing_mode="host-json",
        relative_path="iree-test-suites/sharktank_models/llama3.1/test_llama.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload="tp2",
        sharktank_mode="all",
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
    ),
    Workload(
        id="tp2-decode",
        priority="P2",
        corpus="iree-test-suites",
        kind="sharktank",
        warm_timing_mode="host-json",
        relative_path="iree-test-suites/sharktank_models/llama3.1/test_llama.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload="tp2",
        sharktank_mode="decode",
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        # The reviewed family fault lives in the prefill specialization. This
        # row preserves the independent decode oracle and coverage denominator.
        fault_families=(),
        targets=("gfx950", "gfx1250"),
        run_timeout_seconds=180,
    ),
    Workload(
        id="tp2-combined",
        priority="P2",
        corpus="iree-test-suites",
        kind="sharktank",
        warm_timing_mode="host-json",
        relative_path="iree-test-suites/sharktank_models/llama3.1/test_llama.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload="tp2",
        sharktank_mode="combined",
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        # As above, this is an exact supporting mode of the one TP2 family
        # cell, not a second independently fault-qualified family.
        fault_families=(),
        targets=("gfx950", "gfx1250"),
        run_timeout_seconds=180,
    ),
    Workload(
        id="clip-bf16",
        priority="P3",
        corpus="iree-test-suites",
        kind="sharktank",
        relative_path="iree-test-suites/sharktank_models/clip/test_clip.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload="clip-bf16",
        sharktank_mode="all",
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=10,
        fault_families=("barrier-drop", "barrier-move"),
    ),
    Workload(
        id="d128-block",
        priority="P4",
        corpus="hip-moi",
        kind="gtest",
        relative_path="hip-moi-build/tests/hip_moi_instrumented_rdna4_d128_attention_block_test",
        clean_filter="HipMoiRdna4D128AttentionBlock.*",
        overhead_filter="HipMoiRdna4D128AttentionBlock.ExactContextMatchesHostReference",
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=3,
        fault_families=("barrier-drop",),
        warm_timing_mode="device-gtest",
    ),
    Workload(
        id="d128-pressure",
        priority="P4",
        corpus="hip-moi",
        kind="gtest",
        relative_path="hip-moi-build/tests/hip_moi_instrumented_rdna4_d128_attention_pressure_test",
        clean_filter="HipMoiRdna4D128AttentionPressure.*",
        overhead_filter=(
            "HipMoiRdna4D128AttentionPressure."
            "FullKvDoubleBufferedExactContextMatchesHostReference"
        ),
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=3,
        fault_families=("barrier-drop",),
    ),
    Workload(
        id="wmma-attention",
        priority="P4",
        corpus="hip-moi",
        kind="gtest",
        relative_path="hip-moi-build/tests/hip_moi_instrumented_rdna4_wmma_attention_block_test",
        clean_filter="HipMoiRdna4WmmaAttentionBlock.*",
        overhead_filter="HipMoiRdna4WmmaAttentionBlock.ExactContextMatchesHostReference",
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=3,
        fault_families=("barrier-drop",),
    ),
    Workload(
        id="streamk-arrival",
        priority="P4",
        corpus="hip-moi",
        kind="gtest",
        relative_path=(
            "hip-moi-build/tests/"
            "hip_moi_instrumented_rdna4_wmma_streamk_arrival_counter_test"
        ),
        clean_filter=(
            "HipMoiRdna4WmmaStreamKArrivalCounter." "AcqRelFetchAddOrdersWmmaPartials"
        ),
        overhead_filter=(
            "HipMoiRdna4WmmaStreamKArrivalCounter." "AcqRelFetchAddOrdersWmmaPartials"
        ),
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=True,
        overhead_processes=3,
        fault_families=STREAMK_FAULT_FAMILIES,
        warm_timing_mode="device-gtest",
        empirical_device_timed_minimum_ms=0.5,
        device_timing_max_iterations=1,
        device_timing_warmup_iterations=0,
    ),
    Workload(
        id="tree-atomic-or",
        priority="P4",
        corpus="hip-moi",
        kind="gtest",
        relative_path=(
            "hip-moi-build/tests/"
            "hip_moi_instrumented_rdna4_wmma_streamk_tree_atomic_or_test"
        ),
        clean_filter=(
            "HipMoiRdna4WmmaStreamKTreeAtomicOr." "AcqRelBitmaskOrdersWmmaPartials"
        ),
        overhead_filter=(
            "HipMoiRdna4WmmaStreamKTreeAtomicOr." "AcqRelBitmaskOrdersWmmaPartials"
        ),
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=True,
        overhead_processes=3,
        fault_families=STREAMK_FAULT_FAMILIES,
    ),
    Workload(
        id="jakub-attention",
        priority="P4",
        corpus="hip-moi",
        kind="gtest",
        relative_path="hip-moi-build/tests/hip_moi_reference_rdna4_jakub_matmul",
        clean_filter="SafeFp16Packed/JakubRdna4MatmulReference.MatchesHostReference/*",
        overhead_filter="SafeFp16Packed/JakubRdna4MatmulReference.MatchesHostReference/*",
        sharktank_workload=None,
        sharktank_mode=None,
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=3,
        fault_families=("barrier-drop",),
    ),
)


WORKLOAD_BY_ID = {workload.id: workload for workload in WORKLOADS}


def _validate_tensile_sharding(workload: Workload) -> None:
    shards = workload.tensile_exact_problem_size_shards
    source_blocks = workload.tensile_expected_source_exact_problem_size_blocks
    expected_rows = workload.tensile_expected_numeric_rows_per_shard
    expected_clients = workload.tensile_expected_client_passes_per_shard
    if not shards:
        if (
            source_blocks
            or expected_rows
            or expected_clients
            or workload.tensile_shard_parallelism != 1
            or workload.tensile_fault_shard_index is not None
        ):
            raise RuntimeError(
                f"{workload.id} declares Tensile shard policy without shards"
            )
        return
    if workload.kind != "tensile":
        raise RuntimeError(f"{workload.id} uses Tensile sharding for a non-Tensile row")
    if workload.tensile_expected_numeric_rows is not None:
        raise RuntimeError(
            f"{workload.id} must declare per-shard rather than aggregate numeric rows"
        )
    if expected_rows and (
        len(expected_rows) != len(shards) or any(rows <= 0 for rows in expected_rows)
    ):
        raise RuntimeError(
            f"{workload.id} must declare one positive numeric-row count per shard"
        )
    if expected_clients and (
        len(expected_clients) != len(shards)
        or any(clients <= 0 for clients in expected_clients)
    ):
        raise RuntimeError(
            f"{workload.id} must declare one positive client count per shard"
        )
    if expected_clients and workload.tensile_expected_client_passes is not None:
        raise RuntimeError(
            f"{workload.id} cannot mix aggregate and per-shard client counts"
        )
    if not 1 <= workload.tensile_shard_parallelism <= min(4, len(shards)):
        raise RuntimeError(
            f"{workload.id} Tensile shard parallelism must be between one and four"
        )
    fault_shard_index = workload.tensile_fault_shard_index
    if fault_shard_index is not None and (
        type(fault_shard_index) is not int
        or not 0 <= fault_shard_index < len(shards)
    ):
        raise RuntimeError(
            f"{workload.id} Tensile fault shard index is out of range"
        )
    sizes = []
    for shard in shards:
        if not shard:
            raise RuntimeError(f"{workload.id} has an empty Tensile problem-size shard")
        for size in shard:
            if not size or any(
                type(dimension) is not int or dimension <= 0 for dimension in size
            ):
                raise RuntimeError(
                    f"{workload.id} has a malformed Tensile Exact problem size"
                )
            sizes.append(size)
    if len(set(sizes)) != len(sizes):
        raise RuntimeError(f"{workload.id} repeats a Tensile Exact problem size")
    if source_blocks:
        for block in source_blocks:
            if not block:
                raise RuntimeError(
                    f"{workload.id} has an empty expected source problem-size block"
                )
            for size in block:
                if not size or any(
                    type(dimension) is not int or dimension <= 0
                    for dimension in size
                ):
                    raise RuntimeError(
                        f"{workload.id} has a malformed expected source Exact problem size"
                    )
            if len(set(block)) != len(block):
                raise RuntimeError(
                    f"{workload.id} repeats a size inside an expected source block"
                )
        source_sizes = {size for block in source_blocks for size in block}
        if set(sizes) != source_sizes:
            raise RuntimeError(
                f"{workload.id} shards do not cover its expected source block inventories"
            )


def _validate_workload_manifest() -> None:
    for workload in WORKLOADS:
        _validate_tensile_sharding(workload)
        if any(
            not name
            or not isinstance(value, str)
            or name.startswith(CONTROLLED_ENV_PREFIX)
            or name in HSA_TOOL_ENVIRONMENT
            for name, value in workload.command_environment
        ):
            raise RuntimeError(
                f"{workload.id} has an invalid command environment"
            )
        if any(not argument for argument in workload.command_arguments):
            raise RuntimeError(f"{workload.id} has an empty command argument")
        if workload.kind != "native-executable" and (
            workload.command_arguments or workload.command_environment
        ):
            raise RuntimeError(
                f"{workload.id} declares a native command contract for "
                f"kind {workload.kind}"
            )
        if workload.tensile_expected_client_passes is not None and (
            workload.kind != "tensile"
            or workload.tensile_expected_client_passes <= 0
        ):
            raise RuntimeError(
                f"{workload.id} has an invalid expected Tensile client count"
            )
        if workload.device_timing_calibration_iterations is not None:
            if workload.device_timing_calibration_iterations <= 0:
                raise RuntimeError(
                    f"{workload.id} device timing calibration count must be positive"
                )
            if workload.kind == "pytorch" and (
                workload.device_timing_calibration_iterations < 2
            ):
                raise RuntimeError(
                    f"{workload.id} device timing calibration needs a retained "
                    "sample after discarding the compile transient"
                )
            if workload.warm_timing_mode not in {
                "host-json",
                "device-json",
                "device-fixed",
                "device-gtest",
            }:
                raise RuntimeError(
                    f"{workload.id} device timing calibration requires warm timing"
                )
        if (
            not math.isfinite(workload.device_timing_aggregate_headroom)
            or workload.device_timing_aggregate_headroom < 1.0
        ):
            raise RuntimeError(
                f"{workload.id} device timing aggregate headroom must be at least one"
            )
        if (
            workload.id in STREAMK_WORKLOAD_IDS
            and workload.fault_families != STREAMK_FAULT_FAMILIES
        ):
            raise RuntimeError(
                f"{workload.id} must declare the shared Stream-K fault families"
            )
_validate_workload_manifest()


def _workloads_for_target(target: str) -> tuple[Workload, ...]:
    return tuple(
        workload
        for workload in WORKLOADS
        if workload.targets is None or target in workload.targets
    )


def _workload_for_target(target: str, workload_id: str) -> Workload:
    workload = WORKLOAD_BY_ID[workload_id]
    if workload.targets is not None and target not in workload.targets:
        raise ValidationError(f"{target} manifest excludes workload: {workload_id}")
    return workload


def _target_fault_families(target: str, workload: Workload) -> tuple[str, ...]:
    if target == "gfx1250" and workload.id in (
        "tp1-prefill",
        "tp1-decode-combined",
    ):
        return ("barrier-move",)
    families = workload.fault_families
    if not families:
        # Supporting rows may deliberately carry only an exact oracle and
        # coverage denominator while the family fault remains on a sibling.
        return ()
    if target in NATIVE_CDNA_TARGETS:
        # CDNA compiler atomics encode ordering through surrounding cache and
        # wait operations, but have no RDNA4/CDNA5-style instruction scope field.
        families = tuple(
            family for family in families if family != "atomic-weaken-scope"
        )
        if not families:
            raise ValidationError(
                f"{target} workload has no applicable fault family: {workload.id}"
            )
        return families
    return families


@dataclass(frozen=True)
class _NativeGtestTarget:
    id: str
    build_dir: str
    executable_family: str
    matrix_executable_family: str
    suite_family: str
    matrix_suite_family: str
    matrix_operation: str
    d128_block_oracle: str
    d128_block_fault_uses_oracle: bool = False
    d128_block_run_timeout_seconds: int | None = None
    d128_pressure_run_timeout_seconds: int | None = None
    matrix_run_timeout_seconds: int | None = None


def _cdna_gtest_target(
    target_id: str,
    *,
    suite_family: str,
    matrix_suite_family: str,
    matrix_run_timeout_seconds: int | None = None,
) -> _NativeGtestTarget:
    target = cdna_hip_moi_registry.TARGETS[target_id]
    return _NativeGtestTarget(
        id=target_id,
        build_dir=target.build_dir_name,
        executable_family=target.executable_family,
        matrix_executable_family=f"{target.executable_family}_mfma",
        suite_family=suite_family,
        matrix_suite_family=matrix_suite_family,
        matrix_operation="Mfma",
        d128_block_oracle="SampledFastContextMatchesHostReference",
        matrix_run_timeout_seconds=matrix_run_timeout_seconds,
    )


NATIVE_GTEST_TARGETS = {
    "gfx1201": _NativeGtestTarget(
        id="gfx1201",
        build_dir="hip-moi-build",
        executable_family="rdna4",
        matrix_executable_family="rdna4_wmma",
        suite_family="Rdna4",
        matrix_suite_family="Rdna4Wmma",
        matrix_operation="Wmma",
        d128_block_oracle="ExactContextMatchesHostReference",
    ),
    "gfx942": _cdna_gtest_target(
        "gfx942",
        suite_family="Cdna3",
        matrix_suite_family="Cdna3Mfma",
    ),
    "gfx950": _cdna_gtest_target(
        "gfx950",
        suite_family="Cdna4",
        matrix_suite_family="Cdna4Mfma",
        # Complete Record/Replay coverage scans a large fixed report reservoir
        # after the compact workload finishes. On the physical target that
        # analysis takes about 160 seconds, despite subsecond device execution.
        matrix_run_timeout_seconds=300,
    ),
    "gfx1250": _NativeGtestTarget(
        id="gfx1250",
        build_dir="hip-moi-build-gfx1250-tests",
        executable_family="gfx1250",
        matrix_executable_family="gfx1250_wmma",
        suite_family="Gfx1250",
        matrix_suite_family="Gfx1250Wmma",
        matrix_operation="Wmma",
        d128_block_oracle="SampledFastContextMatchesHostReference",
        d128_block_fault_uses_oracle=True,
        # Current Record/Replay retains thousands of dynamic identity records
        # across both exact host-reference cases. The native gfx1250 simulator
        # completes that stronger contract in roughly 102 seconds.
        d128_block_run_timeout_seconds=150,
        # The pressure suite exercises all four host-reference cases in one
        # process. Inline Shadow currently needs roughly 200 seconds under the
        # native simulator, so retain a 1.5x target-specific margin.
        d128_pressure_run_timeout_seconds=300,
    ),
}


def _native_gtest_path(
    target: _NativeGtestTarget,
    suite_id: str,
    executable: str,
) -> str:
    if target.id in NATIVE_CDNA_TARGETS:
        return str(cdna_hip_moi_registry.relative_executable_path(target.id, suite_id))
    return str(Path(target.build_dir) / "tests" / executable)


def _attention_override(
    relative_path: str,
    suite: str,
    oracle: str,
    *,
    fault_uses_oracle: bool = False,
    run_timeout_seconds: int | None = None,
) -> dict[str, object]:
    oracle_filter = f"{suite}.{oracle}"
    override = {
        "relative_path": relative_path,
        "clean_filter": f"{suite}.*",
        "overhead_filter": oracle_filter,
    }
    if fault_uses_oracle:
        override["fault_filter"] = oracle_filter
    if run_timeout_seconds is not None:
        override["run_timeout_seconds"] = run_timeout_seconds
    return override


def _single_oracle_override(
    relative_path: str,
    oracle_filter: str,
) -> dict[str, str]:
    return {
        "relative_path": relative_path,
        "clean_filter": oracle_filter,
        "overhead_filter": oracle_filter,
    }


def _jakub_override(target: _NativeGtestTarget) -> dict[str, object]:
    relative_path = _native_gtest_path(
        target,
        "jakub-matmul",
        f"hip_moi_reference_{target.executable_family}_jakub_matmul",
    )
    oracle_prefix = (
        f"SafeFp16Packed/Jakub{target.suite_family}MatmulReference."
        "MatchesHostReference"
    )
    if target.id != "gfx1250":
        return _single_oracle_override(relative_path, f"{oracle_prefix}/*")

    # The producer-skew case is the barrier-fault discriminator, not a timing row.
    # Keep clean coverage broad, measure the non-pipelined and double-buffered
    # variants, and mutate only the exact-oracle case selected for inventory.
    return {
        "relative_path": relative_path,
        "clean_filter": f"{oracle_prefix}/*",
        "overhead_filter": (
            f"{oracle_prefix}/NoPipelineProd16x8:"
            f"{oracle_prefix}/DoubleBufferedProd16x8"
        ),
        "fault_filter": f"{oracle_prefix}/ProducerSkewProd16x8",
        # Current full-coverage Record/Replay runs the ordinary exact oracles
        # in roughly 37 seconds, while the intentionally skewed fault oracle
        # completes in about 62 seconds under emulation. Keep a bounded margin
        # without rejecting the expanded 70-access inventory as a timeout.
        "run_timeout_seconds": 90,
    }


STREAMK_WORKLOAD_SHAPES = {
    "streamk-arrival": (
        "streamk_arrival_counter_test",
        "StreamKArrivalCounter",
        "AcqRelFetchAddOrders",
    ),
    "tree-atomic-or": (
        "streamk_tree_atomic_or_test",
        "StreamKTreeAtomicOr",
        "AcqRelBitmaskOrders",
    ),
}


def _validate_exact_keys(
    label: str,
    actual: Iterable[str],
    expected: Iterable[str],
) -> None:
    actual_keys = set(actual)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise RuntimeError(f"{label} mismatch: missing={missing} extra={extra}")


_validate_exact_keys(
    "native Stream-K workload shapes",
    STREAMK_WORKLOAD_SHAPES,
    STREAMK_WORKLOAD_IDS,
)


def _streamk_overrides(
    target: _NativeGtestTarget,
) -> dict[str, dict[str, str]]:
    overrides = {}
    for workload_id in STREAMK_WORKLOAD_IDS:
        executable_stem, suite_stem, oracle_stem = STREAMK_WORKLOAD_SHAPES[workload_id]
        overrides[workload_id] = _single_oracle_override(
            _native_gtest_path(
                target,
                workload_id,
                "hip_moi_instrumented_"
                f"{target.matrix_executable_family}_{executable_stem}",
            ),
            f"HipMoi{target.matrix_suite_family}{suite_stem}."
            f"{oracle_stem}{target.matrix_operation}Partials",
        )
    return overrides


def _native_gtest_overrides(
    target: _NativeGtestTarget,
) -> dict[str, dict[str, object]]:
    base = target.executable_family
    matrix = target.matrix_executable_family
    suite = target.suite_family
    matrix_suite = target.matrix_suite_family
    streamk = _streamk_overrides(target)
    return {
        "d128-block": _attention_override(
            _native_gtest_path(
                target,
                "d128-block",
                f"hip_moi_instrumented_{base}_d128_attention_block_test",
            ),
            f"HipMoi{suite}D128AttentionBlock",
            target.d128_block_oracle,
            fault_uses_oracle=target.d128_block_fault_uses_oracle,
            run_timeout_seconds=target.d128_block_run_timeout_seconds,
        ),
        "d128-pressure": _attention_override(
            _native_gtest_path(
                target,
                "d128-pressure",
                f"hip_moi_instrumented_{base}_d128_attention_pressure_test",
            ),
            f"HipMoi{suite}D128AttentionPressure",
            "FullKvDoubleBufferedExactContextMatchesHostReference",
            run_timeout_seconds=target.d128_pressure_run_timeout_seconds,
        ),
        "wmma-attention": _attention_override(
            _native_gtest_path(
                target,
                "mfma-attention",
                f"hip_moi_instrumented_{matrix}_attention_block_test",
            ),
            f"HipMoi{matrix_suite}AttentionBlock",
            "ExactContextMatchesHostReference",
            run_timeout_seconds=target.matrix_run_timeout_seconds,
        ),
        **streamk,
        "jakub-attention": _jakub_override(target),
    }


NATIVE_GTEST_WORKLOAD_OVERRIDES = {
    target_id: _native_gtest_overrides(target)
    for target_id, target in NATIVE_GTEST_TARGETS.items()
}
NATIVE_GTEST_WORKLOAD_IDS = tuple(
    workload.id for workload in WORKLOADS if workload.kind == "gtest"
)
for target_id, overrides in NATIVE_GTEST_WORKLOAD_OVERRIDES.items():
    _validate_exact_keys(
        f"{target_id} native gtest override matrix",
        overrides,
        NATIVE_GTEST_WORKLOAD_IDS,
    )


TARGET_WORKLOAD_OVERRIDES: dict[str, dict[str, dict[str, object]]] = {
    "gfx1201": {
        # All four exact torch.mode profiles complete on the physical RDNA4
        # target, but owner-local planning of its 50-MiB multi-kernel code
        # object takes 24--40 seconds. Keep a bounded 120-second process
        # envelope instead of misclassifying ordinary patching as a timeout.
        "pytorch-torch-mode": {
            "run_timeout_seconds": 120,
        },
    },
    "gfx950": {
        # These compact schedules select no workgroup at the production
        # stride.  A target-resolved validation cadence retains evidence from
        # the same unmodified workloads.  Native execution itself is fast, but
        # strict replay analysis scans the target's conservative 2M-slot table;
        # retain a bounded margin for that host-side validation step.
        "tp1-prefill": {
            "record_replay_runtime_sample_stride": 256,
            "run_timeout_seconds": 300,
        },
        "tp1-decode-combined": {
            "record_replay_runtime_sample_stride": 256,
            "run_timeout_seconds": 300,
        },
        # Complete SuperCollider coverage of all three TP2 oracles takes
        # roughly 253 seconds through RocJitsu. The former generic 30-second
        # bound expired during normal execution before the workload or hook
        # could publish a verdict. Instrumented modes retain large report
        # buffers across the three model lifecycles, so give each unchanged
        # exact oracle an independent bounded process, as on gfx1250. The
        # measured-only Inline Shadow prefill takes roughly 1,091 seconds in
        # emulation, so retain a bounded 1,800-second margin for that row. The
        # independently resolved decode and combined rows retain their current
        # bounds until their own measurements establish otherwise.
        "tp2-family": {
            "sharktank_mode": "prefill",
            "record_replay_runtime_sample_stride": 256,
            "sharktank_skip_warmup": True,
            "run_timeout_seconds": 1800,
        },
        "tp2-decode": {
            "record_replay_runtime_sample_stride": 1,
            "sharktank_skip_warmup": True,
            "run_timeout_seconds": 600,
        },
        "tp2-combined": {
            "record_replay_runtime_sample_stride": 256,
            "sharktank_skip_warmup": True,
            "run_timeout_seconds": 600,
        },
        "clip-bf16": {
            "record_replay_runtime_sample_stride": 256,
            "run_timeout_seconds": 300,
        },
        # The four Jakub oracle dispatches also miss the denser 256 cadence,
        # so this bounded row selects every workgroup deterministically.
        "jakub-attention": {
            "record_replay_runtime_sample_stride": 1,
            "run_timeout_seconds": 300,
        },
        # The physical gfx950 histogram is one two-workgroup dispatch. Its
        # current dispatch token misses the production stride-65,536 residue,
        # which makes otherwise complete 179-access/84-barrier instrumentation
        # terminate without runtime evidence. Select the complete bounded
        # dispatch for validation and retain enough time for replay's
        # conservative fixed-capacity report scan.
        "pytorch-torch-histc": {
            "record_replay_runtime_sample_stride": 1,
            "run_timeout_seconds": 300,
        },
        # The segmented sort is likewise a compact four-row dispatch. Its
        # current token misses the production stride entirely even though all
        # 56,884 accesses and 6,032 barriers are instrumented. Select each
        # workgroup for validation. With all-supported Inline Shadow placement,
        # the clean process completes in about 198 seconds through gfx950
        # emulation, so a 300-second target envelope bounds every profile
        # without weakening the production Record/Replay cadence.
        "pytorch-torch-sort": {
            "record_replay_runtime_sample_stride": 1,
            "run_timeout_seconds": 300,
        },
        # Inline Shadow completes the exact segmented-mode oracle in roughly
        # 43 seconds through gfx950 emulation. The shared 30-second bound is
        # appropriate for native execution but expires during ordinary
        # emulator work, so retain the same bounded margin as the neighboring
        # segmented-sort row.
        "pytorch-torch-mode": {
            "run_timeout_seconds": 120,
        },
        # The exact HIP matmul row launches only six compact workgroups.  None
        # of their current dispatch identities selects a workgroup at the
        # production stride, so the otherwise complete 739-access/109-barrier
        # transform terminates without dynamic evidence.  Select the complete
        # bounded row deterministically for validation; this is the same
        # stride-one operating point used by its reviewed E2E qualification.
        "hip-matmul-m128-n128-k128": {
            "record_replay_runtime_sample_stride": 1,
        },
        # The exact norm/softmax device interval completes in about 27.5
        # seconds, while the same official PyTorch wheel spends about 4.7
        # seconds in fixed process startup and teardown even without ConSan.
        # Give this physical validation row a bounded whole-process margin;
        # this does not change the production Record/Replay cadence.
        "pytorch-norm-softmax": {
            "run_timeout_seconds": 60,
        },
        # Each compact HipKittens validation dispatch misses the production
        # stride. Select every workgroup for this bounded validation row; the
        # source workloads and their exact numerical oracles are unchanged.
        "hipkittens-bf16fp32-16x32": {
            "record_replay_runtime_sample_stride": 1,
        },
        "hipkittens-fp8fp32-4wave": {
            "record_replay_runtime_sample_stride": 1,
        },
        "hipkittens-mxfp8-4wave": {
            "record_replay_runtime_sample_stride": 1,
        },
    },
    "gfx1250": {
        # Complete Record/Replay instrumentation of the two exact MXF8
        # Stream-K problems exceeds the generic 55-second Tensile subprocess
        # budget while still making forward progress. A 300-second trial
        # completed the first exact oracle and entered the second. Keep the
        # unchanged serialized pair under an explicit bounded margin.
        "tensile-sk-mxf8gemm-explicit": {
            "tensile_inner_timeout_seconds": 900,
            "run_timeout_seconds": 960,
        },
        "tensile-sk-mxf4gemm-explicit": {
            "tensile_inner_timeout_seconds": 900,
            "run_timeout_seconds": 960,
        },
        # The unchanged full 151,936-logit Qwen baseline takes about 65
        # seconds through RocJitsu. Complete Sampled instrumentation has been
        # observed to finish in about 129 seconds but can exceed 180 seconds
        # under emulator load, so retain a bounded twofold-variance margin in
        # the executable manifest instead of requiring an ad hoc CLI override.
        "qwen-prefill": {
            "run_timeout_seconds": 360,
        },
        # The target-specific naive HipKittens runner launches one complete
        # exact-validation workgroup. Select it deterministically instead of
        # relying on its dispatch token to hit the production replay stride.
        "hipkittens-bf16fp32-cdna5-naive": {
            "record_replay_runtime_sample_stride": 1,
        },
        # The strict Inline Shadow row completes in roughly 31 seconds after
        # transformation under RocJitsu. Keep a bounded twofold margin without
        # weakening the ordinary physical-target timeout.  This compact
        # emulated schedule needs a denser Record/Replay cadence than the
        # production operating point to produce validation evidence.
        "tp1-prefill": {
            "record_replay_runtime_sample_stride": 256,
            "run_timeout_seconds": 60,
        },
        # Decode and combined each retain an exact measured oracle. Their
        # instrumented warmups merely repeat the same complete execution and
        # make the two-mode process exceed its bound under Record/Replay.
        # Measured-only production-cadence runs complete in about 149 and 131
        # seconds respectively, so preserve both modes with a bounded margin.
        "tp1-decode-combined": {
            "sharktank_skip_warmup": True,
            "run_timeout_seconds": 360,
        },
        # Running all three TP2 modes in one instrumented process retains two
        # 252-MB rank report buffers across independent model lifecycles and
        # exceeds the emulator bound. Keep the reviewed family fault on
        # prefill, while the adjacent target-only rows retain decode and
        # combined as separate exact, fully covered processes. Once the
        # Record/Replay workgroup predicate is cached correctly, this compact
        # 64-packet prefill schedule selects no workgroup at the production
        # stride. The established bounded-validation cadence produces exact
        # evidence without changing the model, oracle, or static denominator.
        "tp2-family": {
            "sharktank_mode": "prefill",
            "record_replay_runtime_sample_stride": 256,
            "run_timeout_seconds": 180,
        },
        "tp2-decode": {
            # Decode dispatches only one workgroup per kernel. Sparse
            # workgroup cadences can miss every instrumented site, while two
            # dense invocations exceed the emulator bound. Keep the exact
            # timed oracle and omit only the redundant untimed warmup. A
            # complete dense Record/Replay run takes roughly 344 seconds on
            # the current host, so retain a 600-second bound for ordinary host
            # and contention variance.
            "record_replay_runtime_sample_stride": 1,
            "sharktank_skip_warmup": True,
            "run_timeout_seconds": 600,
        },
        "tp2-combined": {
            "record_replay_runtime_sample_stride": 256,
            # Preserve the exact measured two-rank oracle without repeating
            # it as an instrumented warmup. The measured-only row completes
            # in about 210 seconds on the current host.
            "sharktank_skip_warmup": True,
            "run_timeout_seconds": 360,
        },
        # Complete Inline Shadow instrumentation of the unchanged segmented
        # sort oracle takes about 214 seconds through RocJITsu. Preserve the
        # full exact workload under a target-specific bound with enough room
        # for ordinary host-load variance.
        "pytorch-torch-sort": {
            "run_timeout_seconds": 360,
        },
        # Record/Replay of the cross-target LDS-qualifying softmax shape takes
        # roughly 32 seconds in gfx1250 emulation. Retain a bounded whole-row
        # margin rather than relying on the generic 30-second default.
        "pytorch-norm-softmax": {
            "run_timeout_seconds": 60,
        },
        # This sparse matrix has nine benchmark clients with four distinct
        # Exact shapes. Retain every client that owns a selected shape, prune
        # only structurally empty clients, and give each shape an independent
        # oracle and teardown verdict.
        "tensile-spmm-tdm-all": {
            "tensile_inner_timeout_seconds": 300,
            "run_timeout_seconds": 360,
            "tensile_exact_problem_size_shards": (
                ((32, 32, 1, 64),),
                ((64, 64, 1, 128),),
                ((64, 64, 1, 256),),
                ((128, 128, 1, 256),),
            ),
            "tensile_expected_source_exact_problem_size_blocks": (
                (
                    (32, 32, 1, 64),
                    (64, 64, 1, 128),
                    (128, 128, 1, 256),
                ),
                ((32, 32, 1, 64), (64, 64, 1, 128)),
                ((32, 32, 1, 64), (64, 64, 1, 256)),
                ((32, 32, 1, 64), (128, 128, 1, 256)),
                ((32, 32, 1, 64), (64, 64, 1, 128)),
                ((32, 32, 1, 64), (64, 64, 1, 256)),
                ((32, 32, 1, 64), (64, 64, 1, 256)),
                ((32, 32, 1, 64), (64, 64, 1, 256)),
                ((32, 32, 1, 64), (64, 64, 1, 256)),
            ),
            "tensile_expected_client_passes_per_shard": (9, 3, 5, 2),
            "tensile_shard_parallelism": 4,
            "tensile_fault_shard_index": 0,
        },
        # The two SGEMM benchmark blocks repeat the same six Exact sizes but
        # have very different solution spaces. Preserve both blocks in every
        # shard and give each size an independent all-client oracle, teardown
        # verdict, and coverage gate instead of accepting the old partial
        # first-block transcript from one execution-bound monolithic process.
        "tensile-sk-sgemm-quick": {
            "tensile_inner_timeout_seconds": 300,
            "run_timeout_seconds": 360,
            "tensile_exact_problem_size_shards": (
                ((127, 127, 1, 127),),
                ((128, 128, 1, 128),),
                ((129, 129, 1, 129),),
                ((511, 511, 1, 511),),
                ((512, 512, 1, 512),),
                ((513, 513, 1, 513),),
            ),
            "tensile_expected_client_passes": 2,
            "tensile_shard_parallelism": 3,
            "tensile_fault_shard_index": 0,
        },
        # The single F8 client generates 12 solutions for each of its nine
        # exact problems. Give every size an independent numerical oracle,
        # coverage gate, and teardown verdict instead of allowing one slow
        # problem to hide completed work behind the monolithic row's bound.
        # The 511-square shard completes in about 211 seconds alone but can
        # take just over 300 seconds under the retained four-way campaign
        # parallelism, so keep a bounded twofold-variance margin.
        "tensile-sk-f8gemm-quick": {
            "tensile_inner_timeout_seconds": 420,
            "run_timeout_seconds": 480,
            "tensile_exact_problem_size_shards": (
                ((127, 127, 1, 1024),),
                ((128, 128, 1, 1024),),
                ((129, 129, 1, 1024),),
                ((511, 511, 1, 1024),),
                ((512, 512, 1, 1024),),
                ((513, 513, 1, 1024),),
                ((127, 128, 1, 640),),
                ((128, 128, 1, 640),),
                ((129, 128, 1, 640),),
            ),
            "tensile_expected_numeric_rows_per_shard": (12,) * 9,
            "tensile_shard_parallelism": 4,
            "tensile_fault_shard_index": 0,
        },
        # HGEMM has two benchmark clients with intentionally asymmetric Exact
        # inventories: both own the three 127--129 shapes, while only the first
        # owns the three 511--513 shapes. Preserve that source structure as a
        # fail-closed contract, retaining both clients for shared sizes and
        # dropping the empty second client for the larger-size shards.
        "tensile-sk-hgemm-quick": {
            "tensile_inner_timeout_seconds": 300,
            "run_timeout_seconds": 360,
            "tensile_exact_problem_size_shards": (
                ((127, 127, 1, 127),),
                ((128, 128, 1, 128),),
                ((129, 129, 1, 129),),
                ((511, 511, 1, 511),),
                ((512, 512, 1, 512),),
                ((513, 513, 1, 513),),
            ),
            "tensile_expected_source_exact_problem_size_blocks": (
                (
                    (127, 127, 1, 127),
                    (128, 128, 1, 128),
                    (129, 129, 1, 129),
                    (511, 511, 1, 511),
                    (512, 512, 1, 512),
                    (513, 513, 1, 513),
                ),
                (
                    (127, 127, 1, 127),
                    (128, 128, 1, 128),
                    (129, 129, 1, 129),
                ),
            ),
            "tensile_expected_client_passes_per_shard": (2, 2, 2, 1, 1, 1),
            "tensile_shard_parallelism": 3,
            "tensile_fault_shard_index": 0,
        },
        # The six exact MXF8 problems share one generated six-solution object.
        # Give each size an independent numerical oracle, coverage gate, and
        # teardown verdict instead of allowing the monolithic client to consume
        # the complete legacy timeout before its final oracle is visible.
        "tensile-sk-mxf8gemm-tdm": {
            "tensile_inner_timeout_seconds": 120,
            "run_timeout_seconds": 180,
            "tensile_exact_problem_size_shards": (
                ((127, 127, 1, 1024),),
                ((128, 128, 1, 1024),),
                ((129, 129, 1, 1024),),
                ((127, 127, 1, 1056),),
                ((128, 128, 1, 1056),),
                ((129, 129, 1, 1056),),
            ),
            "tensile_expected_numeric_rows_per_shard": (6, 6, 6, 6, 6, 6),
            "tensile_shard_parallelism": 4,
            "tensile_fault_shard_index": 0,
        },
        # The mixed MXF8/F4 object has three exact problems and six generated
        # solutions per problem. Give each size its own numerical oracle,
        # coverage gate, and teardown verdict so a compiler/DBT failure in one
        # slice cannot be obscured by a partial monolithic transcript. The
        # 512-square slice needs roughly 30 seconds per generated solution in
        # Sampled emulation, so retain a bounded five-minute inner allowance.
        "tensile-sk-mxf8f4gemm-tdm": {
            "tensile_inner_timeout_seconds": 300,
            "run_timeout_seconds": 360,
            "tensile_exact_problem_size_shards": (
                ((128, 128, 1, 2048),),
                ((128, 128, 1, 1056),),
                ((512, 512, 1, 2048),),
            ),
            "tensile_expected_numeric_rows_per_shard": (6, 6, 6),
            "tensile_shard_parallelism": 3,
            "tensile_fault_shard_index": 0,
        },
        # Preserve all six exact problems and all 16 generated solutions per
        # problem, but give each exact size an independent numeric oracle,
        # teardown verdict, and coverage gate. Four RocJITsu-backed shards fit
        # the executable manifest's concurrency contract and keep the complete
        # 96-row Record/Replay corpus bounded without accepting a partial
        # monolithic timeout transcript.
        "tensile-sk-mxf4gemm-tdm": {
            "tensile_inner_timeout_seconds": 1200,
            "run_timeout_seconds": 1260,
            "tensile_exact_problem_size_shards": (
                ((120, 120, 1, 1024),),
                ((128, 128, 1, 1024),),
                ((136, 136, 1, 1024),),
                ((120, 120, 1, 1056),),
                ((128, 128, 1, 1056),),
                ((136, 136, 1, 1056),),
            ),
            "tensile_expected_numeric_rows_per_shard": (16, 16, 16, 16, 16, 16),
            "tensile_shard_parallelism": 4,
            # Inventory and contained mutation need one exact numerical oracle,
            # not a second unbounded traversal of the complete 96-row corpus.
            # The clean and overhead phases still qualify every shard.
            "tensile_fault_shard_index": 0,
        },
        # The eight sparse benchmark blocks repeat the same three Exact
        # problem sizes. Give each size an independent all-block numeric
        # oracle and run the three complete size slices concurrently. This
        # retains every generated client and solution while replacing the
        # roughly 20-minute serial Record/Replay traversal with a bounded
        # maximum-shard latency. Fault qualification uses the smallest slice.
        "tensile-spmm-f8-ml": {
            "tensile_inner_timeout_seconds": 900,
            "run_timeout_seconds": 960,
            "tensile_exact_problem_size_shards": (
                ((16, 16, 1, 64),),
                ((16, 16, 1, 256),),
                ((128, 128, 1, 128),),
            ),
            "tensile_expected_client_passes": 8,
            "tensile_shard_parallelism": 3,
            "tensile_fault_shard_index": 0,
        },
    },
}


def _resolved_workload(target: str, workload: Workload) -> Workload:
    """Materialize one target's command overrides from a canonical registry row."""
    override = dict(TARGET_WORKLOAD_OVERRIDES.get(target, {}).get(workload.id, {}))
    if workload.kind == "gtest":
        native_overrides = NATIVE_GTEST_WORKLOAD_OVERRIDES.get(target)
        if native_overrides is not None:
            native_override = native_overrides.get(workload.id)
            if native_override is None:
                raise ValidationError(
                    f"{target} gtest workload has no target-specific registry entry: {workload.id}"
                )
            override.update(native_override)
    resolved = replace(workload, **override) if override else workload
    _validate_tensile_sharding(resolved)
    return resolved


def resolved_workload_relative_path(target: str, workload_id: str) -> str:
    """Return the target-resolved path owned by the workload registry."""
    workload = _workload_for_target(target, workload_id)
    return _resolved_workload(target, workload).relative_path


def _fault_families(target: str, workload: Workload) -> tuple[str, ...]:
    return _target_fault_families(target, _resolved_workload(target, workload))
