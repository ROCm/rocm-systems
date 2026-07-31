#!/usr/bin/env python3
"""Runs ConSan's portable real-workload validation matrix.

The required CONSAN_VALIDATION_WORKSPACE_DIR contains external repositories,
their build outputs, and a rocJITsu build. IREE command-line tools and rocminfo
are resolved from PATH. Run `consan_validation.py doctor` before GPU work and
`consan_validation.py explain` to audit commands, settings, and fault policy.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time

import consan_cdna_hip_moi_registry as cdna_hip_moi_registry
from consan_coverage_gate import CoverageParseError, parse_coverage_evidence
from consan_tensile_support import resolve_tensile_validation_paths
from consan_validation_support import (
    SITE_KINDS,
    atomic_write_json,
    git_identity,
    sha256_file,
)

SCHEMA_VERSION = 2
WORKSPACE_ENV = "CONSAN_VALIDATION_WORKSPACE_DIR"
TARGET_ENV = "CONSAN_VALIDATION_TARGET"
PYTORCH_PYTHON_ENV = "CONSAN_VALIDATION_PYTORCH_PYTHON"
SHARKTANK_PYTHON_ENV = "CONSAN_VALIDATION_SHARKTANK_PYTHON"
TENSILE_PYTHON_ENV = "CONSAN_VALIDATION_TENSILE_PYTHON"
TIMEOUT_SECONDS = 30
PROCESS_OUTPUT_DRAIN_SECONDS = 2
PROCESS_TERMINATION_GRACE_SECONDS = 5
NATIVE_CDNA_TARGETS = frozenset(("gfx942", "gfx950"))
SINGLE_REPETITION_TARGETS = frozenset(("gfx942", "gfx950", "gfx1250"))
QWEN_OVERHEAD_REPETITIONS = {target: 1 for target in SINGLE_REPETITION_TARGETS}
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
    "RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE": "16384",
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
}


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
class CoverageOutputContract:
    # No current workload declares this exception. Keep the typed contract and
    # validation/reporting path available for a future workload only when it
    # explicitly names a bounded, tracked diagnostic contract. Retired
    # producer-shaped examples live in consan_validation_test_support.py.
    profile: str
    diagnostics: tuple[str, ...]
    max_diagnostics: int
    instruction_groups: tuple[tuple[int, ...], ...]
    code_object_fingerprint: str
    tracking_issue: str
    withhold_fault_qualification: bool
    fault_qualification_withheld_reason: str


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
    targets: tuple[str, ...] | None = None
    moi_record_evidence_expected: bool = True
    run_timeout_seconds: int = TIMEOUT_SECONDS
    coverage_output_contract: CoverageOutputContract | None = None
    tensile_inner_timeout_seconds: int | None = None
    tensile_expected_numeric_rows: int | None = None
    tensile_streamk_fixed_grid: int | None = None
    tensile_streamk_mode: int | None = None


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
# The structural parser below currently models Record/Replay output. Do not add
# another profile until its complete runtime diagnostic surface is parsed and
# pinned by producer-shaped fixtures.
COVERAGE_OUTPUT_PROFILE_IDS = ("record-replay",)
COVERAGE_OUTPUT_DIAGNOSTICS = ("exact-lds-write-write",)
# Coverage-output exceptions remain deliberately small even though the runtime
# now sizes replay diagnostic storage from each report's visible record count.
MAX_COVERAGE_OUTPUT_DIAGNOSTICS = 4
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
        id="pytorch-rdna4-split-softmax",
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
        overhead_processes=1,
        fault_families=("barrier-drop",),
        targets=("gfx1201",),
        run_timeout_seconds=120,
    ),
    Workload(
        id="pytorch-rdna4-sdpa",
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
        targets=("gfx1201",),
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
        relative_path="iree-test-suites/sharktank_models/llama3.1/test_llama.py",
        clean_filter=None,
        overhead_filter=None,
        sharktank_workload="tp1",
        sharktank_mode="decode-combined",
        tracks_barriers=True,
        tracks_atomics=False,
        overhead_processes=1,
        fault_families=("barrier-drop",),
    ),
    Workload(
        id="tp2-family",
        priority="P2",
        corpus="iree-test-suites",
        kind="sharktank",
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


def _validate_coverage_output_contract(workload: Workload) -> None:
    contract = workload.coverage_output_contract
    if contract is None:
        return
    if contract.profile not in COVERAGE_OUTPUT_PROFILE_IDS:
        raise RuntimeError(
            f"invalid coverage-output profile for {workload.id}: {contract.profile}"
        )
    invalid_diagnostics = sorted(
        set(contract.diagnostics) - set(COVERAGE_OUTPUT_DIAGNOSTICS)
    )
    if not contract.diagnostics or invalid_diagnostics:
        raise RuntimeError(
            f"invalid coverage-output diagnostics for {workload.id}: "
            f"{', '.join(invalid_diagnostics) or 'none'}"
        )
    if contract.max_diagnostics < 1:
        raise RuntimeError(
            f"{workload.id} coverage-output max_diagnostics must be positive"
        )
    if contract.max_diagnostics > MAX_COVERAGE_OUTPUT_DIAGNOSTICS:
        raise RuntimeError(
            f"{workload.id} coverage-output max_diagnostics exceeds policy limit"
        )
    instruction_sites = [
        instruction for group in contract.instruction_groups for instruction in group
    ]
    if (
        not contract.instruction_groups
        or any(
            not group
            or tuple(sorted(set(group))) != group
            or any(instruction < 0 for instruction in group)
            for group in contract.instruction_groups
        )
        or len(instruction_sites) != len(set(instruction_sites))
    ):
        raise RuntimeError(
            f"{workload.id} coverage-output instruction_groups are invalid"
        )
    if re.fullmatch(r"fnv1a64:[0-9a-f]{16}", contract.code_object_fingerprint) is None:
        raise RuntimeError(
            f"{workload.id} coverage-output code_object_fingerprint is invalid"
        )
    if not contract.tracking_issue.startswith("bd-"):
        raise RuntimeError(
            f"{workload.id} coverage-output contract needs a tracking bead"
        )
    if contract.withhold_fault_qualification and not workload.fault_families:
        raise RuntimeError(
            f"{workload.id} coverage-output contract must not suppress unrelated "
            "profile fault qualification"
        )
    if (
        contract.withhold_fault_qualification
        and not contract.fault_qualification_withheld_reason.strip()
    ):
        raise RuntimeError(
            f"{workload.id} coverage-output contract needs a fault-withholding reason"
        )
    if (
        not contract.withhold_fault_qualification
        and contract.fault_qualification_withheld_reason
    ):
        raise RuntimeError(
            f"{workload.id} coverage-output contract has an unused "
            "fault-withholding reason"
        )


def _validate_workload_manifest() -> None:
    for workload in WORKLOADS:
        _validate_coverage_output_contract(workload)
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
    if target in NATIVE_CDNA_TARGETS:
        # CDNA compiler atomics encode ordering through surrounding cache and
        # wait operations, but have no gfx12-style instruction scope field.
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


def _cdna_gtest_target(
    target_id: str,
    *,
    suite_family: str,
    matrix_suite_family: str,
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
) -> dict[str, str]:
    oracle_filter = f"{suite}.{oracle}"
    override = {
        "relative_path": relative_path,
        "clean_filter": f"{suite}.*",
        "overhead_filter": oracle_filter,
    }
    if fault_uses_oracle:
        override["fault_filter"] = oracle_filter
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
) -> dict[str, dict[str, str]]:
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
        ),
        "d128-pressure": _attention_override(
            _native_gtest_path(
                target,
                "d128-pressure",
                f"hip_moi_instrumented_{base}_d128_attention_pressure_test",
            ),
            f"HipMoi{suite}D128AttentionPressure",
            "FullKvDoubleBufferedExactContextMatchesHostReference",
        ),
        "wmma-attention": _attention_override(
            _native_gtest_path(
                target,
                "mfma-attention",
                f"hip_moi_instrumented_{matrix}_attention_block_test",
            ),
            f"HipMoi{matrix_suite}AttentionBlock",
            "ExactContextMatchesHostReference",
        ),
        **streamk,
        "jakub-attention": _single_oracle_override(
            _native_gtest_path(
                target,
                "jakub-matmul",
                f"hip_moi_reference_{base}_jakub_matmul",
            ),
            f"SafeFp16Packed/Jakub{suite}MatmulReference." "MatchesHostReference/*",
        ),
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


def _resolved_workload(target: str, workload: Workload) -> Workload:
    """Materialize one target's command overrides from a canonical registry row."""
    if workload.kind != "gtest":
        return workload
    overrides = NATIVE_GTEST_WORKLOAD_OVERRIDES.get(target)
    if overrides is None:
        return workload
    override = overrides.get(workload.id)
    if override is None:
        raise ValidationError(
            f"{target} gtest workload has no target-specific registry entry: {workload.id}"
        )
    return replace(workload, **override)


def resolved_workload_relative_path(target: str, workload_id: str) -> str:
    """Return the target-resolved path owned by the workload registry."""
    workload = _workload_for_target(target, workload_id)
    return _resolved_workload(target, workload).relative_path


def _fault_families(target: str, workload: Workload) -> tuple[str, ...]:
    return _target_fault_families(target, _resolved_workload(target, workload))


class ValidationError(RuntimeError):
    pass


def _workspace_from_environment() -> Path:
    value = os.environ.get(WORKSPACE_ENV)
    if not value:
        raise ValidationError(f"{WORKSPACE_ENV} is required")
    workspace = Path(value).expanduser().resolve()
    if not workspace.is_dir():
        raise ValidationError(f"{WORKSPACE_ENV} is not a directory: {workspace}")
    return workspace


def _target(args: argparse.Namespace) -> str:
    value = args.target or os.environ.get(TARGET_ENV)
    if not value or re.fullmatch(r"gfx[0-9a-z]+", value) is None:
        raise ValidationError(
            f"set --target or {TARGET_ENV} to a gfx architecture name"
        )
    return value


@dataclass(frozen=True)
class WorkloadSelection:
    target: str
    workload_id: str
    workload: Workload | None

    @property
    def is_all(self) -> bool:
        return self.workload_id == "all"

    def require_workload(self) -> Workload:
        if self.workload is None:
            raise ValidationError("command requires one concrete workload")
        return self.workload

    def selected_ids(self) -> tuple[str, ...] | None:
        return None if self.is_all else (self.workload_id,)


def _resolve_workload_selection(
    args: argparse.Namespace,
    *,
    allow_all: bool,
) -> WorkloadSelection:
    target = _target(args)
    workload_id = getattr(args, "workload", "all")
    if workload_id == "all":
        if not allow_all:
            command = getattr(args, "command", "this command")
            raise ValidationError(f"{command} requires one concrete workload")
        return WorkloadSelection(target=target, workload_id="all", workload=None)
    return WorkloadSelection(
        target=target,
        workload_id=workload_id,
        workload=_workload_for_target(target, workload_id),
    )


def _command_json(value: str) -> list[str]:
    try:
        command = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("command must be valid JSON") from error
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise argparse.ArgumentTypeError(
            "command must be a non-empty JSON array of non-empty strings"
        )
    return command


def _hook_path(workspace: Path) -> Path:
    suffix = Path("lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so")
    candidates = (
        workspace / "rocjitsu-build" / suffix,
        workspace / "rocjitsu-main-gpu-build" / suffix,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _required_paths(
    workspace: Path, workloads: tuple[Workload, ...]
) -> dict[str, Path]:
    hook = _hook_path(workspace)
    paths = {
        "rocjitsu-build": hook.parents[5],
        "hook": hook,
    }
    if any(workload.corpus == "iree-test-suites" for workload in workloads):
        paths["iree-test-suites"] = workspace / "iree-test-suites"
    if any(workload.kind == "qwen" for workload in workloads):
        paths["iree-test-suites-build"] = workspace / "iree-test-suites-build"
    if any(workload.corpus == "hip-moi" for workload in workloads):
        paths["hip-moi"] = workspace / "hip-moi"
    if any(workload.corpus == "rocjitsu-test-corpus" for workload in workloads):
        paths["rocjitsu-test-corpus"] = workspace / "rocjitsu-test-corpus"
    if any(workload.kind == "tensile" for workload in workloads):
        tensile_paths = resolve_tensile_validation_paths(workspace)
        paths["tensilelite"] = tensile_paths.tensilelite
        paths["rocm"] = tensile_paths.rocm
    return paths


def _pytorch_python(workspace: Path | None = None) -> Path:
    # Preserve a virtual environment's interpreter path. Resolving its python
    # symlink would silently bypass that environment and lose torch/triton.
    configured = os.environ.get(PYTORCH_PYTHON_ENV)
    if configured:
        return Path(os.path.abspath(Path(configured).expanduser()))
    if workspace is not None:
        workspace_interpreter = workspace / "consan-pytorch-venv" / "bin" / "python"
        if workspace_interpreter.is_file():
            return workspace_interpreter
    return Path(os.path.abspath(Path(sys.executable).expanduser()))


def _sharktank_python() -> Path:
    return Path(
        os.path.abspath(
            Path(os.environ.get(SHARKTANK_PYTHON_ENV, sys.executable)).expanduser()
        )
    )


def _pytorch_runtime_probe(
    python: Path, hook: Path, target: str, workload: Workload
) -> dict:
    """Proves that PyTorch can dispatch and that its HSA runtime loads ConSan."""
    probe_source = """
import json
import os
import pathlib
import sys

import torch
import triton

value = torch.ones(1, device="cuda")
torch.cuda.synchronize()
properties = torch.cuda.get_device_properties(0)
maps = pathlib.Path("/proc/self/maps").read_text(encoding="utf-8")
print(json.dumps({
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "triton": triton.__version__,
    "device": torch.cuda.get_device_name(0),
    "arch": getattr(properties, "gcnArchName", None),
    "numeric_oracle": value.item() == 1.0,
    "hook_loaded": sys.argv[1] in maps,
}), flush=True)
# Large precompiled operator libraries can spend longer tearing down a
# software target than executing this linkage canary.  The workload clients
# finalize ConSan explicitly; this doctor probe only needs the flushed result
# and process mappings, so skip unrelated runtime shutdown.
os._exit(0)
"""
    environment = _clean_environment("supercollider", workload, hook, target)
    # This is a runtime-linkage canary, not a coverage row.  Avoid spending
    # preflight time on PyTorch's large bundled kernel object; the real rows
    # run without this filter and enforce complete coverage independently.
    environment.update(
        {
            "RJ_CONSAN_LOG": "0",
            "RJ_CONSAN_REQUIRE_PATCH": "0",
            "RJ_CONSAN_TEST_KERNEL_FILTER": (
                "__consan_pytorch_runtime_probe_never_matches__"
            ),
        }
    )
    try:
        probe = subprocess.run(
            [str(python), "-c", probe_source, str(hook.resolve())],
            check=False,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "ok": False,
            "python": str(python),
            "detail": str(error),
        }
    try:
        payload = json.loads(probe.stdout.strip())
    except json.JSONDecodeError:
        payload = None
    reasons = []
    if probe.returncode != 0:
        reasons.append(f"probe exited with status {probe.returncode}")
    if not isinstance(payload, dict):
        reasons.append("probe did not emit its JSON result")
    else:
        arch = payload.get("arch")
        if not isinstance(arch, str) or not arch.startswith(target):
            reasons.append(f"device architecture is {arch!r}, expected {target!r}")
        if not payload.get("numeric_oracle"):
            reasons.append("GPU numeric oracle failed")
        if not payload.get("hook_loaded"):
            reasons.append("PyTorch HSA runtime did not load the ConSan hook")
    if reasons and probe.stderr.strip():
        reasons.append(probe.stderr.strip())
    return {
        "ok": not reasons,
        "python": str(python),
        "detail": payload if payload is not None else probe.stderr.strip(),
        "reasons": reasons,
    }


def _tensile_python() -> Path:
    return Path(
        os.path.abspath(
            Path(os.environ.get(TENSILE_PYTHON_ENV, sys.executable)).expanduser()
        )
    )


def _llama_executable(workspace: Path, target: str, name: str) -> Path:
    candidates = (
        workspace
        / "rocjitsu-test-corpus-build"
        / "kernels"
        / target
        / "cases"
        / "llama.cpp"
        / name,
        workspace
        / "rocjitsu-test-corpus"
        / ".pytest-artifacts-rdna4-llama-baseline"
        / "_suite_shards"
        / "kernels_shard_0"
        / "kernels"
        / target
        / "build"
        / "cases"
        / "llama.cpp"
        / name,
    )
    return next(
        (candidate for candidate in candidates if candidate.is_file()), candidates[0]
    )


def _input_files(workspace: Path, target: str, workload: Workload) -> dict[str, Path]:
    workload = _resolved_workload(target, workload)
    if workload.kind == "pytorch":
        return {
            "python": _pytorch_python(workspace),
            "workload-source": Path(__file__).with_name(workload.relative_path),
        }
    if workload.kind == "tensile":
        paths = resolve_tensile_validation_paths(workspace)
        return {
            "python": _tensile_python(),
            "workload-source": Path(__file__).with_name("consan_tensile_validation.py"),
            "support-source": Path(__file__).with_name("consan_tensile_support.py"),
            "config": workspace / workload.corpus / workload.relative_path,
            "client": paths.client,
            "wrapper": paths.wrapper,
            "rocjitsu": paths.rocjitsu,
            "rocjitsu-config": paths.rocjitsu_config,
            "llvm-readelf": paths.llvm_readelf,
            "amdclang++": paths.rocm / "bin" / "amdclang++",
        }
    if workload.kind == "llama":
        case = (
            "mul_mat_vec_q"
            if workload.id == "llama-rdna4-mul-mat-vec-q"
            else "rms_norm"
        )
        return {
            "python": Path(os.path.abspath(Path(sys.executable).expanduser())),
            "workload-source": Path(__file__).with_name("consan_llama_validation.py"),
            "case": workspace
            / "rocjitsu-test-corpus"
            / "corpus"
            / "kernels"
            / "cases"
            / "llama.cpp"
            / case
            / "case.json",
            "executable": _llama_executable(workspace, target, workload.relative_path),
        }
    if workload.kind == "qwen":
        root = workspace / workload.relative_path
        data = root / "hf" / "qwen3-600m"
        return {
            "vmfb": root / target / "qwen3-600m.vmfb",
            "parameters": data / "real_weights.irpa",
            "input": data / "inference_input.0.bin",
            "expected": data / "inference_output.0.bin",
        }
    if workload.kind == "gtest":
        return {"executable": workspace / workload.relative_path}
    source = workspace / workload.relative_path
    if workload.sharktank_workload in {"tp1", "tp2"}:
        assets = source.parent / "assets"
        names = (
            ("toy_llama.mlir", "toy_llama.irpa")
            if workload.sharktank_workload == "tp1"
            else (
                "toy_llama_tp2.mlir",
                "toy_llama_tp2.irpa",
                "toy_llama_tp2.rank0.irpa",
                "toy_llama_tp2.rank1.irpa",
            )
        )
        return {"workload-source": source, **{name: assets / name for name in names}}
    assets = source.parent / "assets" / "text_model" / "toy"
    return {
        "workload-source": source,
        "bf16.mlir": assets / "bf16.mlir",
        "bf16_parameters.irpa": assets / "bf16_parameters.irpa",
        "input": assets / "forward_bs4_arg0_input_ids.irpa",
        "expected": assets / "forward_bs4_expected_result0_last_hidden_state_f32.irpa",
    }


def _doctor(
    workspace: Path, target: str, workload_ids: tuple[str, ...] | None = None
) -> dict:
    selected_ids = (
        tuple(workload.id for workload in _workloads_for_target(target))
        if workload_ids is None
        else workload_ids
    )
    workloads = tuple(
        _workload_for_target(target, workload_id) for workload_id in selected_ids
    )
    paths = _required_paths(workspace, workloads)
    path_checks = {
        label: {
            "path": str(path),
            "present": path.is_file() if label == "hook" else path.is_dir(),
        }
        for label, path in paths.items()
    }
    # PyTorch's runtime probe performs a numeric dispatch and reports the
    # device architecture from the same process that loads the ConSan hook.
    # Requiring a separately installed rocminfo for a PyTorch-only row adds no
    # target assurance and can reject an otherwise complete wheel-based setup.
    required_tools = (
        ("rocminfo",)
        if any(workload.kind != "pytorch" for workload in workloads)
        else ()
    )
    if any(workload.kind == "qwen" for workload in workloads):
        required_tools = TOOLS
    tools = {tool: shutil.which(tool) for tool in required_tools}
    for workload in workloads:
        for label, path in _input_files(workspace, target, workload).items():
            executable = workload.kind == "tensile" and label in {
                "client",
                "wrapper",
                "rocjitsu",
                "llvm-readelf",
                "amdclang++",
            }
            path_checks[f"workload:{workload.id}:{label}"] = {
                "path": str(path),
                "present": path.is_file()
                and (not executable or os.access(path, os.X_OK)),
            }
    runtimes = {}
    pytorch_workloads = tuple(
        workload for workload in workloads if workload.kind == "pytorch"
    )
    if pytorch_workloads:
        python = _pytorch_python(workspace)
        if python.is_file():
            runtimes["pytorch"] = _pytorch_runtime_probe(
                python,
                _hook_path(workspace),
                target,
                pytorch_workloads[0],
            )
        else:
            runtimes["pytorch"] = {
                "ok": False,
                "python": str(python),
                "detail": "interpreter is missing",
            }
    ok = (
        all(item["present"] for item in path_checks.values())
        and all(tools.values())
        and all(item["ok"] for item in runtimes.values())
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "workspace": str(workspace),
        "target": target,
        "workloads": list(selected_ids),
        "paths": path_checks,
        "runtimes": runtimes,
        "tools": tools,
    }


def _manifest(target: str) -> dict:
    def manifest_workload(workload: Workload) -> dict:
        # The manifest is the executable target contract, not the union declared
        # by the target-independent Workload row.
        return asdict(_effective_workload(target, workload))

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": "consan-real-workload-validation-v1",
        "workspace_environment": WORKSPACE_ENV,
        "target": target,
        "tools_from_path": list(TOOLS),
        "profiles": [asdict(PROFILES[profile]) for profile in PROFILE_IDS],
        "workloads": [
            manifest_workload(workload) for workload in _workloads_for_target(target)
        ],
        "ordinary_forbidden_environment": list(ORDINARY_FORBIDDEN_ENVIRONMENT),
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_gpu_parallelism": 4,
    }


def _clean_environment(
    profile: str | None,
    workload: Workload,
    hook: Path,
    target: str | None = None,
) -> dict[str, str]:
    if target is not None:
        workload = _resolved_workload(target, workload)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(CONTROLLED_ENV_PREFIX)
        and key not in HSA_TOOL_ENVIRONMENT | {"HIP_TARGET"}
        and not (target in NATIVE_CDNA_TARGETS and key in SOFTWARE_MODEL_ENVIRONMENT)
    }
    if target is not None:
        environment["HIP_TARGET"] = target
    if profile is None:
        return environment
    config = PROFILES[profile]
    environment.update(config.environment)
    environment.update(
        {
            "HSA_TOOLS_LIB": str(hook),
            "RJ_CONSAN_LOG": "1",
        }
    )
    if workload.kind in {"pytorch", "llama"}:
        # These clients use a modern HSA runtime which returns after successful
        # rocprofiler registration unless legacy environment tools are
        # explicitly requested. ConSan is currently such a tool.
        environment["HSA_TOOLS_ROCPROFILER_V1_TOOLS"] = "1"
    if not workload.moi_record_evidence_expected:
        environment["RJ_CONSAN_MOI_REQUIRE_RECORDS"] = "0"
    if workload.id == "qwen-prefill" and profile == "sampled":
        environment["RJ_CONSAN_MOI_REQUIRE_RECORDS"] = "1"
    return environment


def _run_environment(
    profile: str | None,
    workload: Workload,
    hook: Path,
    target: str,
    phase: str,
) -> dict[str, str]:
    if phase not in {"clean", "overhead"}:
        raise ValidationError(f"unsupported validation phase: {phase}")
    environment = _clean_environment(profile, workload, hook, target)
    if _coverage_contract_for_profile(workload, profile):
        # The contract is a property of this exact workload/profile execution,
        # so both correctness and paired-overhead rows use the same bounded,
        # structurally validated diagnostic inventory. Fault qualification is
        # separately disabled while the exception remains open.
        environment.pop("RJ_CONSAN_MOI_FORBID_DIAGNOSTICS", None)
    return environment


def _controlled_environment(environment: dict[str, str]) -> dict[str, str]:
    runtime_names = {
        "HSA_TOOLS_LIB",
        "HSA_TOOLS_ROCPROFILER_V1_TOOLS",
        "CTEST_PARALLEL_LEVEL",
        "HIP_PATH",
        "HIP_TARGET",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
        "ROCM_PATH",
    }
    names = {
        key
        for key in environment
        if key.startswith(CONTROLLED_ENV_PREFIX) or key in runtime_names
    }
    return {key: environment[key] for key in sorted(names)}


def _setting_metadata(name: str) -> dict:
    if name in HSA_TOOL_ENVIRONMENT | {"HIP_TARGET"}:
        category = "runtime-plumbing"
    elif name == "CTEST_PARALLEL_LEVEL":
        category = "fault-containment"
    elif name.startswith("RJ_CONSAN_FAULT_"):
        category = "fault-injection"
    elif name in {
        "RJ_CONSAN_MODE",
        "RJ_CONSAN_POLICY",
        "RJ_CONSAN_FLAVOR",
        "RJ_CONSAN_MOI_ENGINE",
        "RJ_CONSAN_MOI_TRACK_BARRIERS",
        "RJ_CONSAN_MOI_TRACK_ATOMICS",
    }:
        category = "instrumentation-selection"
    elif name in {
        "RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE",
        "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET",
        "RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES",
        "RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT",
        "RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
        "RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES",
        "RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES",
    }:
        category = "workload-tuning"
    elif name == "RJ_CONSAN_LOG" or "_REQUIRE_" in name or "_FORBID_" in name:
        category = "acceptance-assertion"
    else:
        raise ValidationError(f"unclassified validation setting: {name}")
    result = {
        "category": category,
        "category_description": SETTING_CATEGORIES[category],
        "usability_exception": category == "workload-tuning",
    }
    if name in {
        "RJ_CONSAN_MOI_TRACK_BARRIERS",
        "RJ_CONSAN_MOI_TRACK_ATOMICS",
    }:
        result["usability_note"] = (
            "Ordinary MOI enables this event family by default; an explicit "
            "value is an expert compatibility override."
        )
    elif category == "workload-tuning":
        result["usability_note"] = (
            "This is a workload-specific non-default operating point."
        )
    return result


def _audited_settings(environment: dict[str, str]) -> list[dict]:
    names = sorted(
        name
        for name in environment
        if name.startswith(CONTROLLED_ENV_PREFIX)
        or name in HSA_TOOL_ENVIRONMENT | {"HIP_TARGET", "CTEST_PARALLEL_LEVEL"}
    )
    return [
        {"name": name, "value": environment[name], **_setting_metadata(name)}
        for name in names
    ]


def _audited_unsets(names: list[str]) -> list[dict]:
    return [
        {
            "name": name,
            "operation": "unset",
            **_setting_metadata(name),
            "usability_note": (
                "Fault-only policy relaxes this clean-run acceptance assertion."
            ),
        }
        for name in sorted(names)
    ]


def _profile_runtime_defaults(
    profile: str, explicit_environment: dict[str, str] | None = None
) -> list[dict]:
    if PROFILES[profile].flavor != "moi":
        return []
    defaults = dict(ORDINARY_MOI_RUNTIME_DEFAULTS)
    if profile == "sampled":
        defaults.update(SAMPLED_STANDARD_RUNTIME_DEFAULTS)
    explicit_names = set(explicit_environment or {})
    settings = _audited_settings(
        {name: value for name, value in defaults.items() if name not in explicit_names}
    )
    return [
        {
            **setting,
            "source": "standard-profile-runtime-default",
            "usability_exception": False,
            "usability_note": "The standard profile selects this automatically.",
        }
        for setting in settings
    ]


def _qwen_command(
    workspace: Path, target: str, overhead: bool, output: Path
) -> list[str]:
    root = workspace / "iree-test-suites-build" / "torch_models" / "qwen3-600m"
    data = root / "hf" / "qwen3-600m"
    command = [
        "iree-benchmark-module" if overhead else "iree-run-module",
        "--device=hip",
        f"--module={root / target / 'qwen3-600m.vmfb'}",
        f"--parameters=model={data / 'real_weights.irpa'}",
        "--function=main",
        f"--input=1x5xi64=@{data / 'inference_input.0.bin'}",
    ]
    if overhead:
        repetitions = QWEN_OVERHEAD_REPETITIONS.get(target, 10)
        command.extend(
            [
                f"--benchmark_repetitions={repetitions}",
                "--benchmark_min_time=0s",
                f"--benchmark_out={output}",
                "--benchmark_out_format=json",
            ]
        )
    else:
        command.extend(
            [
                f"--expected_output=1x5x151936xf32=@{data / 'inference_output.0.bin'}",
                "--expected_f32_threshold=0.05",
            ]
        )
    return command


def _health_smoke_command(
    workspace: Path, target: str, workload: Workload, output: Path
) -> list[str]:
    qwen = WORKLOAD_BY_ID["qwen-prefill"]
    if all(path.is_file() for path in _input_files(workspace, target, qwen).values()):
        return _qwen_command(workspace, target, False, output)
    # A workload-scoped doctor permits an independently ready row to proceed
    # when unrelated Qwen artifacts are absent. Its destructive health gate
    # must honor the same contract instead of manufacturing an unhealthy GPU
    # result from a missing universal smoke file.
    return _workload_command(workspace, target, workload, "clean", output)


def _workload_command(
    workspace: Path,
    target: str,
    workload: Workload,
    phase: str,
    output: Path,
) -> list[str]:
    workload = _resolved_workload(target, workload)
    overhead = phase == "overhead"
    if workload.kind == "qwen":
        return _qwen_command(workspace, target, overhead, output)
    if workload.kind == "sharktank":
        # The active architecture campaigns use one end-to-end repetition.
        # Keep both the outer process count and this inner suite count at one.
        repetitions = (
            1
            if target in SINGLE_REPETITION_TARGETS or workload.overhead_processes > 1
            else (10 if overhead else 1)
        )
        return [
            str(_sharktank_python()),
            str(Path(__file__).with_name("consan_sharktank_validation.py")),
            "--suite-root",
            str(workspace / "iree-test-suites"),
            "--workload",
            str(workload.sharktank_workload),
            "--mode",
            str(workload.sharktank_mode),
            "--repetitions",
            str(repetitions),
            "--label",
            f"{workload.id}-{phase}",
        ]
    if workload.kind == "pytorch":
        return [
            str(_pytorch_python(workspace)),
            str(Path(__file__).with_name(workload.relative_path)),
            "--workload",
            workload.id.removeprefix("pytorch-"),
            "--repetitions",
            (
                "1"
                if target in SINGLE_REPETITION_TARGETS
                else ("10" if overhead else "1")
            ),
            "--label",
            f"{workload.id}-{phase}",
        ]
    if workload.kind == "tensile":
        command = [
            str(_tensile_python()),
            str(Path(__file__).with_name("consan_tensile_validation.py")),
            "--workspace",
            str(workspace),
            "--config",
            str(workspace / workload.corpus / workload.relative_path),
            "--output-dir",
            str(output.parent / "tensile-work"),
            "--repetitions",
            "1",
            "--label",
            f"{workload.id}-{phase}",
        ]
        if workload.tensile_inner_timeout_seconds is not None:
            command.extend(
                (
                    "--timeout-seconds",
                    str(workload.tensile_inner_timeout_seconds),
                )
            )
        if workload.tensile_expected_numeric_rows is not None:
            command.extend(
                (
                    "--expect-numeric-rows",
                    str(workload.tensile_expected_numeric_rows),
                )
            )
        if workload.tensile_streamk_fixed_grid is not None:
            command.extend(
                (
                    "--streamk-fixed-grid",
                    str(workload.tensile_streamk_fixed_grid),
                )
            )
        if workload.tensile_streamk_mode is not None:
            command.extend(
                (
                    "--require-streamk-mode",
                    str(workload.tensile_streamk_mode),
                )
            )
        return command
    if workload.kind == "llama":
        llama_workload = (
            "mul-mat-vec-q"
            if workload.id == "llama-rdna4-mul-mat-vec-q"
            else "rms-norm"
        )
        return [
            sys.executable,
            str(Path(__file__).with_name("consan_llama_validation.py")),
            "--executable",
            str(_llama_executable(workspace, target, workload.relative_path)),
            "--workload",
            llama_workload,
            "--output-dir",
            str(output.parent / f"{output.stem}-llama-work"),
        ]
    executable = workspace / workload.relative_path
    selected_filter = (
        workload.fault_filter or workload.clean_filter
        if phase == "fault"
        else workload.overhead_filter if overhead else workload.clean_filter
    )
    return [str(executable), f"--gtest_filter={selected_filter}"]


def _write_provenance(
    workspace: Path,
    target: str,
    workload: Workload,
    workload_root: Path,
) -> Path:
    workload_root.mkdir(parents=True, exist_ok=True)
    path = workload_root / "provenance.json"
    hook = _hook_path(workspace)
    files = {"hook": hook, **_input_files(workspace, target, workload)}
    document = {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "workload": workload.id,
        "files": {
            label: {
                "path": str(file),
                "size": file.stat().st_size,
                "sha256": sha256_file(file),
            }
            for label, file in files.items()
        },
        "sources": _source_identities(workspace, workload),
        "manifest": _manifest(target),
    }
    normalized_document = json.loads(json.dumps(document))
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValidationError(
                f"cannot read existing provenance {path}: {error}"
            ) from error
        if existing != normalized_document:
            raise ValidationError(
                f"provenance conflicts with existing artifact: {path}"
            )
        return path
    atomic_write_json(path, document)
    return path


def _source_identities(workspace: Path, workload: Workload) -> list[dict | None]:
    roots = [
        workspace / "iree-test-suites",
        workspace / "hip-moi",
        Path(__file__).resolve().parents[5],
    ]
    if workload.kind == "tensile":
        paths = resolve_tensile_validation_paths(workspace)
        if workload.corpus == "rocjitsu-test-corpus":
            roots.append(workspace / "rocjitsu-test-corpus")
        roots.append(paths.tensilelite)
    if workload.kind == "llama":
        roots.append(workspace / "rocjitsu-test-corpus")
    if workload.kind == "pytorch":
        roots.append(workspace / "pytorch")
    return [git_identity(root) for root in roots]


ReplayIdentity = tuple[int, int]


@dataclass(frozen=True)
class _ReplayDiagnosticRecord:
    signature: str
    reader: int | None
    report_generation: int | None
    generation: int | None
    code_object_fingerprint: str | None
    index: int | None
    kind: int | None
    first_owner: int | None
    second_owner: int | None
    first_instruction: int | None
    second_instruction: int | None
    first_instruction_raw: str | None
    second_instruction_raw: str | None
    first_lds: str | None
    second_lds: str | None
    first_access_kind: int | None
    second_access_kind: int | None

    @property
    def report_identity(self) -> ReplayIdentity | None:
        if self.reader is None or self.report_generation is None:
            return None
        return self.reader, self.report_generation


@dataclass(frozen=True)
class DiagnosticRecord:
    signature: str
    source_identity: str | None
    code_object_fingerprint: str | None
    first_instruction: int | None
    second_instruction: int | None
    first_instruction_raw: str | None
    second_instruction_raw: str | None
    artifact_fields: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class DiagnosticSourceSummary:
    identity: str
    diagnostic_count: int
    record_count: int
    code_object_fingerprint: str
    artifact_fields: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class ParsedDiagnosticOutput:
    profile: str
    sources: tuple[DiagnosticSourceSummary, ...]
    records: tuple[DiagnosticRecord, ...]
    diagnostic_count: int
    pre_output_count: int
    structural_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosticPolicy:
    diagnostics: tuple[str, ...]
    max_diagnostics: int
    kind: str = "clean"
    instruction_groups: tuple[tuple[int, ...], ...] = ()
    code_object_fingerprint: str | None = None
    contract: CoverageOutputContract | None = None

    @classmethod
    def clean(cls) -> DiagnosticPolicy:
        return cls(diagnostics=(), max_diagnostics=0)

    @classmethod
    def from_contract(cls, contract: CoverageOutputContract) -> DiagnosticPolicy:
        return cls(
            kind="coverage-output",
            diagnostics=contract.diagnostics,
            max_diagnostics=contract.max_diagnostics,
            instruction_groups=contract.instruction_groups,
            code_object_fingerprint=contract.code_object_fingerprint,
            contract=contract,
        )


@dataclass(frozen=True)
class _ReplayReport:
    identity: ReplayIdentity
    code_object_fingerprint: str
    reported: int
    sampled_conflicts: int
    replay_required: bool
    visible_records: int
    diagnostic_capacity: int


@dataclass(frozen=True)
class _ReplaySummary:
    identity: ReplayIdentity
    code_object_fingerprint: str
    reported: int
    diagnostic_capacity_exhausted: bool | None
    diagnostic_capacity: int | None
    conflict: bool | None
    metadata_full: bool | None
    provenance_repaired: int
    provenance_unresolved: int | None


@dataclass(frozen=True)
class _ReplaySkipped:
    identity: ReplayIdentity
    code_object_fingerprint: str
    required_shadow_entries: int
    limit: int


class _DiagnosticFieldsError(ValueError):
    pass


def _log_fields(payload: str, context: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in payload.split():
        if "=" not in token:
            raise _DiagnosticFieldsError(
                f"malformed {context}: malformed field {token!r}"
            )
        key, value = token.split("=", 1)
        if re.fullmatch(r"[a-z_]+", key) is None or not value:
            raise _DiagnosticFieldsError(
                f"malformed {context}: malformed field {token!r}"
            )
        if key in fields:
            raise _DiagnosticFieldsError(
                f"malformed {context}: duplicate field {key!r}"
            )
        fields[key] = value
    return fields


def _parse_log_fields(
    line: str, marker: str, context: str, reasons: list[str]
) -> dict[str, str] | None:
    try:
        return _log_fields(line.split(marker, 1)[1], context)
    except (IndexError, _DiagnosticFieldsError) as error:
        reasons.append(str(error) or f"malformed {context}")
        return None


def _unsigned(fields: dict[str, str], name: str) -> int | None:
    try:
        value = int(fields[name], 0)
    except (KeyError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _boolean(fields: dict[str, str], name: str) -> bool | None:
    value = fields.get(name)
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _code_object_fingerprint(fields: dict[str, str]) -> str | None:
    value = fields.get("code_object")
    if value is None or re.fullmatch(r"fnv1a64:[0-9a-f]{16}", value) is None:
        return None
    return value


def _lds_range(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"\[(\d+),(\d+)\)", value)
    if match is None:
        return None
    begin, end = int(match.group(1)), int(match.group(2))
    return (begin, end) if begin < end else None


def _replay_identity(
    fields: dict[str, str], generation_field: str = "generation"
) -> ReplayIdentity | None:
    reader = _unsigned(fields, "reader")
    generation = _unsigned(fields, generation_field)
    if reader is None or generation is None:
        return None
    return reader, generation


def _identity_label(identity: ReplayIdentity) -> str:
    return f"reader={identity[0]},generation={identity[1]}"


def _replay_diagnostic_record(fields: dict[str, str]) -> _ReplayDiagnosticRecord:
    kind = _unsigned(fields, "kind")
    first_lds = _lds_range(fields.get("first_lds"))
    second_lds = _lds_range(fields.get("second_lds"))
    first_owner = _unsigned(fields, "first_owner")
    second_owner = _unsigned(fields, "second_owner")
    generation = _unsigned(fields, "generation")
    first_access_kind = _unsigned(fields, "first_kind")
    second_access_kind = _unsigned(fields, "second_kind")
    signature = (
        "exact-lds-write-write"
        if (
            kind == 1
            and generation is not None
            and fields.get("first_lds_known") == "true"
            and first_lds is not None
            and first_lds == second_lds
            and first_owner is not None
            and second_owner is not None
            and first_owner != second_owner
            and first_access_kind == MOI_SHADOW_ACCESS_WRITE
            and second_access_kind == MOI_SHADOW_ACCESS_WRITE
        )
        else (
            "malformed"
            if kind is None
            else MOI_DIAGNOSTIC_KINDS.get(kind, f"unknown-{kind}")
        )
    )
    return _ReplayDiagnosticRecord(
        signature=signature,
        reader=_unsigned(fields, "reader"),
        report_generation=_unsigned(fields, "report_generation"),
        generation=generation,
        code_object_fingerprint=_code_object_fingerprint(fields),
        index=_unsigned(fields, "index"),
        kind=kind,
        first_owner=first_owner,
        second_owner=second_owner,
        first_instruction=_unsigned(fields, "first_inst"),
        second_instruction=_unsigned(fields, "second_inst"),
        first_instruction_raw=fields.get("first_inst"),
        second_instruction_raw=fields.get("second_inst"),
        first_lds=fields.get("first_lds"),
        second_lds=fields.get("second_lds"),
        first_access_kind=first_access_kind,
        second_access_kind=second_access_kind,
    )


def _bool_label(value: bool | None) -> str:
    if value is None:
        return "missing"
    return "true" if value else "false"


def _instruction_label(value: int | None) -> str:
    return f"0x{value:x}" if value is not None else "missing"


def _parse_record_replay_diagnostic_output(log_text: str) -> ParsedDiagnosticOutput:
    reports: dict[ReplayIdentity, _ReplayReport] = {}
    replays: dict[ReplayIdentity, _ReplaySummary] = {}
    skipped_replays: dict[ReplayIdentity, _ReplaySkipped] = {}
    replay_records: list[_ReplayDiagnosticRecord] = []
    reasons: list[str] = []

    for line in log_text.splitlines():
        if "ConSan MOI auto report reader=" in line:
            reader_match = re.search(r"\breader=(\d+)", line)
            reader_label = (
                f"reader={reader_match.group(1)}"
                if reader_match is not None
                else "reader=missing"
            )
            if " needs hsa_memory_copy for coarse-grained summary" in line:
                reasons.append(
                    "pre-replay report requires hsa_memory_copy: " + reader_label
                )
                continue
            if " hsa_memory_copy failed status=" in line:
                status_match = re.search(r"\bstatus=([^ ]+)", line)
                status = (
                    status_match.group(1) if status_match is not None else "missing"
                )
                reasons.append(
                    "pre-replay hsa_memory_copy failed: "
                    f"{reader_label}, status={status}"
                )
                continue
            if " has invalid header magic=" in line:
                reasons.append("pre-replay report header invalid: " + reader_label)
                continue
            if " has inconsistent ABI-v" in line:
                reasons.append("pre-replay report layout inconsistent: " + reader_label)
                continue
            if " addr=" not in line or " bytes=" not in line:
                reasons.append("malformed pre-replay diagnostic summary")
                continue
            fields = _parse_log_fields(
                line,
                "ConSan MOI auto report ",
                "pre-replay diagnostic summary",
                reasons,
            )
            if fields is None:
                continue
            identity = _replay_identity(fields)
            fingerprint = _code_object_fingerprint(fields)
            diagnostics = _unsigned(fields, "diagnostics")
            diagnostic_capacity = _unsigned(fields, "diagnostic_capacity")
            address = _unsigned(fields, "addr")
            byte_count = _unsigned(fields, "bytes")
            sampled_counts = tuple(
                _unsigned(fields, name)
                for name in ("sampled_conflicts", "sampled_immediate_conflicts")
            )
            visible_counts = tuple(
                _unsigned(fields, name)
                for name in (
                    "visible_records",
                    "visible_barriers",
                    "visible_atomics",
                    "visible_fences",
                )
            )
            if (
                identity is None
                or fingerprint is None
                or diagnostics is None
                or diagnostic_capacity is None
                or address is None
                or byte_count is None
                or any(count is None for count in sampled_counts)
                or any(count is None for count in visible_counts)
            ):
                reasons.append("malformed pre-replay diagnostic summary")
            elif identity in reports:
                reasons.append(
                    f"duplicate pre-replay summary {_identity_label(identity)}"
                )
            else:
                reports[identity] = _ReplayReport(
                    identity=identity,
                    code_object_fingerprint=fingerprint,
                    reported=diagnostics,
                    sampled_conflicts=sum(
                        count for count in sampled_counts if count is not None
                    ),
                    replay_required=any(
                        count for count in visible_counts if count is not None
                    ),
                    visible_records=visible_counts[0],
                    diagnostic_capacity=diagnostic_capacity,
                )
        elif "ConSan MOI auto replay diagnostic reader=" in line:
            fields = _parse_log_fields(
                line,
                "ConSan MOI auto replay diagnostic ",
                "replay diagnostic detail",
                reasons,
            )
            if fields is not None:
                replay_records.append(_replay_diagnostic_record(fields))
        elif "ConSan MOI auto replay reader=" in line:
            payload_line = line
            if " skipped " in f" {line} ":
                payload_line = line.replace(" skipped ", " ", 1)
            fields = _parse_log_fields(
                payload_line,
                "ConSan MOI auto replay ",
                (
                    "replay-skipped summary"
                    if payload_line != line
                    else "replay diagnostic summary"
                ),
                reasons,
            )
            if fields is None:
                continue
            identity = _replay_identity(fields)
            fingerprint = _code_object_fingerprint(fields)
            if payload_line != line:
                required = _unsigned(fields, "required_shadow_entries")
                limit = _unsigned(fields, "limit")
                if (
                    identity is None
                    or fingerprint is None
                    or required is None
                    or limit is None
                ):
                    reasons.append("malformed replay-skipped summary")
                elif identity in skipped_replays:
                    reasons.append(
                        f"duplicate replay-skipped summary {_identity_label(identity)}"
                    )
                else:
                    skipped_replays[identity] = _ReplaySkipped(
                        identity=identity,
                        code_object_fingerprint=fingerprint,
                        required_shadow_entries=required,
                        limit=limit,
                    )
                continue
            diagnostics = _unsigned(fields, "diagnostics")
            provenance_repaired = _unsigned(fields, "provenance_repaired")
            if (
                identity is None
                or fingerprint is None
                or diagnostics is None
                or provenance_repaired is None
            ):
                reasons.append("malformed replay diagnostic summary")
            elif identity in replays:
                reasons.append(f"duplicate replay summary {_identity_label(identity)}")
            else:
                replays[identity] = _ReplaySummary(
                    identity=identity,
                    code_object_fingerprint=fingerprint,
                    reported=diagnostics,
                    diagnostic_capacity_exhausted=_boolean(
                        fields, "diagnostic_capacity_exhausted"
                    ),
                    diagnostic_capacity=_unsigned(fields, "diagnostic_capacity"),
                    conflict=_boolean(fields, "conflict"),
                    metadata_full=_boolean(fields, "metadata_full"),
                    provenance_repaired=provenance_repaired,
                    provenance_unresolved=_unsigned(fields, "provenance_unresolved"),
                )

    details_by_identity: dict[ReplayIdentity | None, list[_ReplayDiagnosticRecord]] = {}
    for record in replay_records:
        if record.generation is None:
            reasons.append("replay diagnostic detail has malformed generation")
        details_by_identity.setdefault(record.report_identity, []).append(record)

    if not reports:
        reasons.append("missing pre-replay report summary")
    # This is the exact producer guard around automatic replay: a report needs
    # replay only when it contains a visible access or synchronization record.
    required_replay = {
        identity for identity, summary in reports.items() if summary.replay_required
    }
    skipped_identities = set(skipped_replays)
    missing_replay = sorted(required_replay - set(replays) - skipped_identities)
    unexpected_replay = sorted(set(replays) - required_replay)
    unexpected_skipped = sorted(skipped_identities - required_replay)
    for identity in sorted(skipped_identities & required_replay):
        skipped = skipped_replays[identity]
        reasons.append(
            f"replay skipped: {_identity_label(identity)}, "
            f"required_shadow_entries={skipped.required_shadow_entries}, "
            f"limit={skipped.limit}"
        )
    if missing_replay:
        reasons.append(
            "missing replay summaries: "
            + ", ".join(_identity_label(identity) for identity in missing_replay)
        )
    if unexpected_replay:
        reasons.append(
            "unexpected replay summaries: "
            + ", ".join(_identity_label(identity) for identity in unexpected_replay)
        )
    if unexpected_skipped:
        reasons.append(
            "unexpected replay-skipped summaries: "
            + ", ".join(_identity_label(identity) for identity in unexpected_skipped)
        )
    conflicting_replay = sorted(set(replays) & skipped_identities)
    if conflicting_replay:
        reasons.append(
            "replay both skipped and summarized: "
            + ", ".join(_identity_label(identity) for identity in conflicting_replay)
        )

    source_summaries = []
    for identity, summary in sorted(replays.items()):
        details = details_by_identity.get(identity, ())
        detailed = len(details)
        report = reports.get(identity)
        report_fingerprint = (
            report.code_object_fingerprint if report is not None else None
        )
        if report_fingerprint != summary.code_object_fingerprint:
            reasons.append(
                "replay code-object fingerprint mismatch: "
                f"{_identity_label(identity)}, "
                f"pre_report={report_fingerprint or 'missing'}, "
                f"replay={summary.code_object_fingerprint}"
            )
        detail_fingerprints = {record.code_object_fingerprint for record in details}
        if details and detail_fingerprints != {summary.code_object_fingerprint}:
            reasons.append(
                "replay diagnostic code-object fingerprint mismatch: "
                f"{_identity_label(identity)}"
            )
        if summary.reported != detailed:
            reasons.append(
                "replay diagnostic inventory incomplete: "
                f"{_identity_label(identity)}, "
                f"reported={summary.reported}, detailed={detailed}"
            )
        indices = [record.index for record in details]
        concrete_indices = [index for index in indices if index is not None]
        indices_are_contiguous = (
            not indices
            if summary.reported == 0
            else (
                len(indices) == summary.reported
                and len(set(concrete_indices)) == summary.reported
                and min(concrete_indices, default=-1) == 0
                and max(concrete_indices, default=-1) == summary.reported - 1
            )
        )
        if any(index is None for index in indices) or not indices_are_contiguous:
            actual_indices = ",".join(
                "malformed" if index is None else str(index) for index in indices
            )
            expected_indices = (
                f"0..{summary.reported - 1}" if summary.reported else "none"
            )
            reasons.append(
                "replay diagnostic indices invalid: "
                f"{_identity_label(identity)}, "
                f"expected={expected_indices}, "
                f"actual={actual_indices or 'none'}"
            )
        expected_diagnostic_capacity = (
            min(report.diagnostic_capacity, report.visible_records)
            if report is not None
            else None
        )
        if summary.diagnostic_capacity != expected_diagnostic_capacity:
            reasons.append(
                "replay diagnostic capacity mismatch: "
                f"{_identity_label(identity)}, "
                f"value={summary.diagnostic_capacity}, "
                f"expected={expected_diagnostic_capacity}"
            )
        expected_conflict = summary.reported != 0
        if summary.conflict is not expected_conflict:
            reasons.append(
                f"replay conflict status invalid: {_identity_label(identity)}, "
                f"value={_bool_label(summary.conflict)}, "
                f"expected={_bool_label(expected_conflict)}"
            )
        if summary.diagnostic_capacity_exhausted is not False:
            reasons.append(
                "replay diagnostic capacity status invalid: "
                f"{_identity_label(identity)}, "
                f"value={_bool_label(summary.diagnostic_capacity_exhausted)}"
            )
        if summary.metadata_full is not False:
            reasons.append(
                f"replay metadata status invalid: {_identity_label(identity)}, "
                f"value={_bool_label(summary.metadata_full)}"
            )
        if summary.provenance_unresolved != 0:
            reasons.append(
                f"replay provenance unresolved: {_identity_label(identity)}, "
                f"count={summary.provenance_unresolved}"
            )
        if summary.provenance_repaired > summary.reported:
            reasons.append(
                "replay provenance repaired exceeds diagnostics: "
                f"{_identity_label(identity)}, "
                f"repaired={summary.provenance_repaired}, "
                f"diagnostics={summary.reported}"
            )
        source_summaries.append(
            DiagnosticSourceSummary(
                identity=_identity_label(identity),
                diagnostic_count=summary.reported,
                record_count=detailed,
                code_object_fingerprint=summary.code_object_fingerprint,
                artifact_fields=(
                    (
                        "diagnostic_capacity_exhausted",
                        summary.diagnostic_capacity_exhausted,
                    ),
                    ("diagnostic_capacity", summary.diagnostic_capacity),
                    ("conflict", summary.conflict),
                    ("metadata_full", summary.metadata_full),
                    ("provenance_repaired", summary.provenance_repaired),
                    ("provenance_unresolved", summary.provenance_unresolved),
                ),
            )
        )

    for identity in details_by_identity:
        if identity is None:
            reasons.append("replay diagnostic detail has malformed identity")
        elif identity not in replays:
            reasons.append(
                "replay diagnostic detail has no summary: "
                f"{_identity_label(identity)}"
            )

    pre_output_count = sum(summary.reported for summary in reports.values())
    sampled_conflict_count = sum(
        summary.sampled_conflicts for summary in reports.values()
    )
    if pre_output_count:
        reasons.append(f"pre-replay diagnostics={pre_output_count}")
    if sampled_conflict_count:
        reasons.append(f"pre-replay sampled conflicts={sampled_conflict_count}")
    records = tuple(
        DiagnosticRecord(
            signature=record.signature,
            source_identity=(
                _identity_label(record.report_identity)
                if record.report_identity is not None
                else None
            ),
            code_object_fingerprint=record.code_object_fingerprint,
            first_instruction=record.first_instruction,
            second_instruction=record.second_instruction,
            first_instruction_raw=record.first_instruction_raw,
            second_instruction_raw=record.second_instruction_raw,
            artifact_fields=(
                ("reader", record.reader),
                ("report_generation", record.report_generation),
                ("generation", record.generation),
                ("index", record.index),
                ("kind", record.kind),
                ("first_owner", record.first_owner),
                ("second_owner", record.second_owner),
                ("first_lds", record.first_lds),
                ("second_lds", record.second_lds),
                ("first_access_kind", record.first_access_kind),
                ("second_access_kind", record.second_access_kind),
            ),
        )
        for record in replay_records
    )
    return ParsedDiagnosticOutput(
        profile="record-replay",
        sources=tuple(source_summaries),
        records=records,
        diagnostic_count=sum(source.diagnostic_count for source in source_summaries),
        pre_output_count=pre_output_count,
        structural_reasons=tuple(reasons),
    )


DIAGNOSTIC_OUTPUT_PARSERS = {
    "record-replay": _parse_record_replay_diagnostic_output,
}


def _diagnostic_record_result(record: DiagnosticRecord) -> dict:
    result = {
        "signature": record.signature,
        "source_identity": record.source_identity,
        "code_object_fingerprint": record.code_object_fingerprint,
        "first_instruction": (
            _instruction_label(record.first_instruction)
            if record.first_instruction is not None
            else None
        ),
        "second_instruction": (
            _instruction_label(record.second_instruction)
            if record.second_instruction is not None
            else None
        ),
        "first_instruction_raw": record.first_instruction_raw,
        "second_instruction_raw": record.second_instruction_raw,
    }
    result.update(record.artifact_fields)
    return result


def _diagnostic_source_result(summary: DiagnosticSourceSummary) -> dict:
    result = {
        "reported": summary.diagnostic_count,
        "detailed": summary.record_count,
        "code_object_fingerprint": summary.code_object_fingerprint,
    }
    result.update(summary.artifact_fields)
    return result


def _evaluate_diagnostic_output(
    output: ParsedDiagnosticOutput,
    policy: DiagnosticPolicy,
) -> dict:
    reasons = list(output.structural_reasons)
    if policy.diagnostics and not policy.instruction_groups:
        reasons.append(
            "policy declares diagnostics without qualified instruction groups"
        )
    observed_signatures = {record.signature for record in output.records}
    unexpected = sorted(observed_signatures - set(policy.diagnostics))
    if output.diagnostic_count > policy.max_diagnostics:
        reasons.append(
            "replay diagnostics exceed declared maximum: "
            f"observed={output.diagnostic_count}, maximum={policy.max_diagnostics}"
        )
    if unexpected:
        reasons.append(f"unexpected diagnostics={','.join(unexpected)}")

    if policy.code_object_fingerprint is not None:
        for source in output.sources:
            if (
                source.diagnostic_count != 0 or source.record_count != 0
            ) and source.code_object_fingerprint != policy.code_object_fingerprint:
                reasons.append(
                    "diagnostic code-object fingerprint does not match contract: "
                    f"{source.identity}, "
                    f"observed={source.code_object_fingerprint}, "
                    f"contract={policy.code_object_fingerprint}"
                )
        if not any(
            source.code_object_fingerprint == policy.code_object_fingerprint
            for source in output.sources
        ):
            reasons.append(
                "contract code-object fingerprint missing from diagnostic summaries: "
                f"expected={policy.code_object_fingerprint}"
            )

    unexpected_sites = []
    if policy.instruction_groups:
        for record in output.records:
            same_qualified_group = (
                record.first_instruction is not None
                and record.second_instruction is not None
                and any(
                    record.first_instruction in group
                    and record.second_instruction in group
                    for group in policy.instruction_groups
                )
            )
            if not same_qualified_group:
                unexpected_sites.append(
                    f"{record.source_identity or 'missing'}:"
                    f"{_instruction_label(record.first_instruction)}->"
                    f"{_instruction_label(record.second_instruction)}"
                )
    if unexpected_sites:
        reasons.append(f"unexpected diagnostic sites={','.join(unexpected_sites)}")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "profile": output.profile,
        "policy": {
            "kind": policy.kind,
            "diagnostics": list(policy.diagnostics),
            "max_diagnostics": policy.max_diagnostics,
            "code_object_fingerprint": policy.code_object_fingerprint,
            "instruction_groups": [list(group) for group in policy.instruction_groups],
        },
        "contract": asdict(policy.contract) if policy.contract is not None else None,
        "observed_signatures": sorted(observed_signatures),
        "observed_code_object_fingerprints": sorted(
            {source.code_object_fingerprint for source in output.sources}
        ),
        "diagnostic_count": output.diagnostic_count,
        "replay_count": output.diagnostic_count,
        "pre_replay_count": output.pre_output_count,
        "readers": {
            source.identity: _diagnostic_source_result(source)
            for source in output.sources
        },
        "records": [_diagnostic_record_result(record) for record in output.records],
    }


def _diagnostic_output_summary(
    log_text: str,
    profile: str,
    contract: CoverageOutputContract | None = None,
) -> dict:
    parser = DIAGNOSTIC_OUTPUT_PARSERS.get(profile)
    if parser is None:
        raise ValidationError(
            f"no complete diagnostic-output parser for profile: {profile}"
        )
    if contract is not None and contract.profile != profile:
        raise ValidationError(
            "diagnostic-output contract profile mismatch: "
            f"run={profile}, contract={contract.profile}"
        )
    output = parser(log_text)
    policy = (
        DiagnosticPolicy.clean()
        if contract is None
        else DiagnosticPolicy.from_contract(contract)
    )
    return _evaluate_diagnostic_output(output, policy)


def _coverage_output_diagnostic_summary(
    log_text: str, contract: CoverageOutputContract
) -> dict:
    return _diagnostic_output_summary(log_text, contract.profile, contract)


def _coverage_summary(
    log_text: str,
    profile: str | None = None,
    coverage_output_contract: CoverageOutputContract | None = None,
) -> dict:
    try:
        evidence = parse_coverage_evidence(log_text)
    except CoverageParseError as error:
        rejection_prefix = "[rocjitsu-dbi-hooks] ConSan load rejection "
        rejection_lines = [
            line[len(rejection_prefix) :]
            for line in log_text.splitlines()
            if line.startswith(rejection_prefix)
        ]
        if rejection_lines:
            fields = dict(re.findall(r"([a-z_]+)=([^ ]+)", rejection_lines[-1]))
            return {
                "accepted": False,
                "error": "ConSan rejected a code object before execution",
                "load_rejection": fields,
            }
        return {"accepted": False, "error": str(error)}
    verdict = evidence.verdict
    reasons = []
    if not verdict.applicable:
        reasons.append("no applicable code object")
    if not verdict.analysis_complete:
        reasons.append("analysis incomplete")
    elif not verdict.static_complete:
        reasons.append("static coverage incomplete")
    if not verdict.dynamic_complete:
        reasons.append("dynamic coverage incomplete")
    if verdict.counts["dynamic_incomplete"] != 0:
        reasons.append(f"dynamic_incomplete={verdict.counts['dynamic_incomplete']}")
    for kind in SITE_KINDS:
        patched, supported = verdict.patched_supported[kind]
        if patched != supported:
            reasons.append(f"{kind}={patched}/{supported}")
    if any(record.expert_limit for record in evidence.coverage):
        reasons.append("expert patch limit enabled")
    summary = {
        "accepted": not reasons,
        "reasons": reasons,
        "analysis_complete": verdict.analysis_complete,
        "static_complete": verdict.static_complete,
        "dynamic_complete": verdict.dynamic_complete,
        "patched_supported": {
            kind: list(verdict.patched_supported[kind]) for kind in SITE_KINDS
        },
        "dynamic_incomplete": verdict.counts["dynamic_incomplete"],
    }
    if coverage_output_contract is not None or profile in DIAGNOSTIC_OUTPUT_PARSERS:
        if profile is None:
            raise ValidationError(
                "diagnostic-output contract requires an explicit run profile"
            )
        diagnostics = _diagnostic_output_summary(
            log_text, profile, coverage_output_contract
        )
        summary["diagnostics"] = diagnostics
        summary["reasons"].extend(diagnostics["reasons"])
        summary["accepted"] = not summary["reasons"]
    return summary


def _benchmark_median(path: Path) -> float:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("benchmarks", [])
    medians = [
        row
        for row in rows
        if row.get("aggregate_name") == "median"
        and str(row.get("name", "")).startswith("BM_main/")
    ]
    if not medians:
        medians = [
            row
            for row in rows
            if row.get("run_type") == "iteration"
            and row.get("repetitions") == 1
            and str(row.get("name", "")).startswith("BM_main/")
        ]
    if len(medians) != 1:
        raise ValidationError(f"expected one Qwen benchmark median in {path}")
    scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
    row = medians[0]
    return float(row["real_time"]) * scale[row["time_unit"]]


def _json_medians(log_text: str, workload_kind: str) -> dict[str, float]:
    documents = []
    for line in log_text.splitlines():
        if line.startswith("{"):
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if len(documents) != 1:
        raise ValidationError(f"expected one {workload_kind} JSON result")
    document = documents[0]
    return {
        key: float(value["median_ms"])
        for key, value in document.items()
        if isinstance(value, dict) and "median_ms" in value
    }


def _sharktank_medians(log_text: str) -> dict[str, float]:
    return _json_medians(log_text, "Sharktank")


def _gtest_median(log_texts: list[str]) -> dict[str, float]:
    pattern = re.compile(r"\[==========\].*\(([0-9]+) ms total\)")
    values = []
    for log_text in log_texts:
        matches = pattern.findall(log_text)
        if not matches:
            raise ValidationError("missing GTest total latency")
        values.append(float(matches[-1]))
    return {"process": statistics.median(values)}


def _gtest_test_count(log_text: str) -> int | None:
    matches = re.findall(
        r"\[==========\]\s+Running\s+([0-9]+)\s+tests?\s+from",
        log_text,
    )
    return int(matches[-1]) if matches else None


def _run_process(
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    timeout: int,
) -> tuple[int, float, str]:
    start = time.monotonic()
    process = subprocess.Popen(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        returncode = 124
        _stop_process_group(process, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired as error:
            _stop_process_group(process, signal.SIGKILL)
            output = _bounded_process_output(process, error.output)
        output = (output or "") + f"\nvalidation timeout after {timeout}s\n"
    except BaseException:
        _stop_process_group(process, signal.SIGTERM)
        try:
            process.communicate(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired as error:
            _stop_process_group(process, signal.SIGKILL)
            _bounded_process_output(process, error.output)
        raise
    elapsed = time.monotonic() - start
    log_path.write_text(output, encoding="utf-8")
    return returncode, elapsed, output


def _launcher_from_json(value: str | None) -> list[str]:
    if value is None:
        return []
    try:
        launcher = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValidationError(f"invalid --launcher-json: {error}") from error
    if (
        not isinstance(launcher, list)
        or not launcher
        or any(not isinstance(item, str) or not item for item in launcher)
    ):
        raise ValidationError(
            "--launcher-json must be a nonempty JSON array of nonempty strings"
        )
    return launcher


def _target_outer_repetitions(target: str, phase: str, workload: Workload) -> int:
    if target in SINGLE_REPETITION_TARGETS or phase != "overhead":
        return 1
    return workload.overhead_processes


def _outer_repetitions(target: str, phase: str, workload: Workload) -> int:
    return _target_outer_repetitions(
        target, phase, _resolved_workload(target, workload)
    )


def _effective_workload(target: str, workload: Workload) -> Workload:
    resolved = _resolved_workload(target, workload)
    return replace(
        resolved,
        fault_families=_target_fault_families(target, resolved),
        overhead_processes=_target_outer_repetitions(target, "overhead", resolved),
    )


def _coverage_contract_for_profile(
    workload: Workload, profile: str | None
) -> CoverageOutputContract | None:
    contract = workload.coverage_output_contract
    return contract if contract is not None and contract.profile == profile else None


def _fault_qualification_contract_for_profile(
    workload: Workload, profile: str
) -> CoverageOutputContract | None:
    """Return the profile-specific contract that withholds fault qualification."""
    contract = _coverage_contract_for_profile(workload, profile)
    return (
        contract
        if contract is not None and contract.withhold_fault_qualification
        else None
    )


def _result_phase(phase: str, profile: str | None, workload: Workload) -> str:
    if phase == "clean" and _coverage_contract_for_profile(workload, profile):
        return "coverage-output"
    return phase


def _workload_provenance_path(artifact_root: Path, workload: Workload) -> Path:
    return artifact_root / workload.id / "provenance.json"


def _run_profile(
    workspace: Path,
    target: str,
    workload: Workload,
    profile: str | None,
    phase: str,
    artifact_root: Path,
    timeout: int,
    row_label: str | None = None,
    launcher: list[str] | None = None,
) -> dict:
    profile_id = profile or "baseline"
    result_phase = _result_phase(phase, profile, workload)
    row_dir = artifact_root / workload.id / result_phase / (row_label or profile_id)
    row_dir.mkdir(parents=True, exist_ok=False)
    hook = _hook_path(workspace)
    repetitions = _outer_repetitions(target, phase, workload)
    logs = []
    commands = []
    returncodes = []
    elapsed_seconds = []
    qwen_json_paths = []
    for index in range(repetitions):
        benchmark_path = row_dir / f"benchmark-{index}.json"
        command = _workload_command(workspace, target, workload, phase, benchmark_path)
        if launcher:
            command = [*launcher, *command]
        log_path = row_dir / f"run-{index}.log"
        environment = _run_environment(profile, workload, hook, target, phase)
        returncode, elapsed, output = _run_process(
            command, environment, log_path, timeout
        )
        commands.append(command)
        returncodes.append(returncode)
        elapsed_seconds.append(elapsed)
        logs.append(output)
        if workload.kind == "qwen" and phase == "overhead":
            qwen_json_paths.append(benchmark_path)

    timing = None
    if phase == "overhead" and all(code == 0 for code in returncodes):
        if workload.kind == "qwen":
            timing = {
                "dispatch": statistics.median(
                    _benchmark_median(path) for path in qwen_json_paths
                )
            }
        elif workload.kind in {"sharktank", "pytorch", "tensile", "llama"}:
            per_run = [_json_medians(log, workload.kind.capitalize()) for log in logs]
            keys = set.intersection(*(set(item) for item in per_run))
            timing = {
                key: statistics.median(item[key] for item in per_run)
                for key in sorted(keys)
            }
        else:
            timing = _gtest_median(logs)

    coverage = None
    coverage_runs = None
    if profile is not None and logs:
        coverage_contract = _coverage_contract_for_profile(workload, profile)
        coverage_runs = [
            _coverage_summary(
                log,
                profile=profile,
                coverage_output_contract=coverage_contract,
            )
            for log in logs
        ]
        coverage = coverage_runs[-1]
    gtest_test_counts = (
        [_gtest_test_count(log) for log in logs] if workload.kind == "gtest" else None
    )
    gtest_executed = gtest_test_counts is None or all(
        count is not None and count > 0 for count in gtest_test_counts
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "workload": workload.id,
        "profile": profile_id,
        "phase": result_phase,
        "target": target,
        "commands": commands,
        "environment": _controlled_environment(
            _run_environment(profile, workload, hook, target, phase)
        ),
        "returncodes": returncodes,
        "elapsed_seconds": elapsed_seconds,
        "timing_median_ms": timing,
        "coverage": coverage,
        "coverage_runs": coverage_runs,
        "gtest_test_counts": gtest_test_counts,
        "accepted": (
            all(code == 0 for code in returncodes)
            and gtest_executed
            and (
                profile is None
                or bool(coverage_runs)
                and all(item["accepted"] for item in coverage_runs)
            )
        ),
        "files": {
            "hook": {
                "path": str(hook),
                "sha256": sha256_file(hook),
            }
        },
        "sources": _source_identities(workspace, workload),
        "provenance": str(_workload_provenance_path(artifact_root, workload)),
    }
    result_path = row_dir / "result.json"
    atomic_write_json(result_path, result)
    return result


def _overhead_summary(results: list[dict]) -> dict:
    baselines = [
        result["timing_median_ms"]
        for result in results
        if result["profile"] == "baseline"
    ]
    if len(baselines) != 2 or any(value is None for value in baselines):
        raise ValidationError("overhead requires baseline-before and baseline-after")
    modes = set(baselines[0]) & set(baselines[1])
    paired = {
        mode: statistics.mean([baselines[0][mode], baselines[1][mode]])
        for mode in sorted(modes)
    }
    profiles = {}
    for result in results:
        if result["profile"] == "baseline":
            continue
        timing = result["timing_median_ms"] or {}
        ratios = {
            mode: timing[mode] / paired[mode]
            for mode in sorted(set(timing) & set(paired))
        }
        profiles[result["profile"]] = {
            "timing_median_ms": timing,
            "slowdown_by_mode": ratios,
            "cell_slowdown": max(ratios.values()) if ratios else None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_policy": "mean-of-before-and-after-medians",
        "paired_baseline_median_ms": paired,
        "profiles": profiles,
    }


def _inventory_records(
    log_text: str, family: str | None = None
) -> dict[str, list[str]]:
    event_kind = None
    if family:
        event_kind = "barrier" if family.startswith("barrier-") else "atomic"
    prefixes = {
        "sites": "ConSan fault site ",
        "sequences": "ConSan sync sequence ",
        "destinations": "ConSan barrier destination ",
    }
    records = {key: [] for key in prefixes}
    for line in log_text.splitlines():
        for key, prefix in prefixes.items():
            if prefix not in line:
                continue
            match = re.search(r"\bidentity=(\S+)", line)
            if match:
                identity = match.group(1)
                if family and key == "sites" and f"|kind={event_kind}|" not in identity:
                    continue
                if (
                    family
                    and key == "sequences"
                    and f"|event={event_kind}|" not in identity
                ):
                    continue
                if family and key == "destinations" and family != "barrier-move":
                    continue
                records[key].append(identity)
        if "ConSan fault site " in line:
            match = re.search(r"\bsync_sequence=(\S+)", line)
            if match:
                identity = match.group(1)
                if identity != "-" and (
                    not family or f"|event={event_kind}|" in identity
                ):
                    records["sequences"].append(identity)
    return {key: sorted(set(values)) for key, values in records.items()}


def _inventory_line_completes(
    line: str, family: str, relevant_readers: set[str]
) -> bool:
    """Tracks a relevant site through the matching code-object coverage record."""
    site_kind = "barrier" if family.startswith("barrier-") else "atomic"
    if "ConSan fault site " in line and f" kind={site_kind} " in line:
        match = re.search(r"\breader=(\S+)", line)
        if match:
            relevant_readers.add(match.group(1))
    if "ConSan coverage " not in line:
        return False
    match = re.search(r"\breader=(\S+)", line)
    return bool(match and match.group(1) in relevant_readers)


def _inventory_collection_complete(log_text: str, family: str) -> bool:
    relevant_readers: set[str] = set()
    return any(
        _inventory_line_completes(line, family, relevant_readers)
        for line in log_text.splitlines()
    )


def _stop_process_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _bounded_process_output(
    process: subprocess.Popen[str], partial_output: str | bytes | None
) -> str:
    try:
        output, _ = process.communicate(timeout=PROCESS_OUTPUT_DRAIN_SECONDS)
        return output
    except subprocess.TimeoutExpired as error:
        if process.stdout is not None:
            process.stdout.close()
        try:
            process.wait(timeout=PROCESS_OUTPUT_DRAIN_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        output = error.output or partial_output or ""
        return output.decode(errors="replace") if isinstance(output, bytes) else output


def _run_inventory_process(
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    timeout: int,
    family: str,
) -> tuple[int, float, str, bool, str]:
    """Collects static identities without waiting for workload execution."""
    start = time.monotonic()
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    lines: list[str] = []
    pending = ""
    relevant_readers: set[str] = set()
    collection_complete = False
    timed_out = False
    deadline = start + timeout
    while process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        events = selector.select(min(remaining, 0.25))
        if not events:
            continue
        chunk = os.read(process.stdout.fileno(), 65536)
        if not chunk:
            continue
        pending += chunk.decode(errors="replace")
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            line += "\n"
            lines.append(line)
            if _inventory_line_completes(line, family, relevant_readers):
                collection_complete = True
                break
        if collection_complete:
            break

    outcome = "natural-exit"
    if collection_complete:
        outcome = "static-inventory-complete"
        _stop_process_group(process, signal.SIGTERM)
    elif timed_out:
        outcome = "timeout"
        _stop_process_group(process, signal.SIGTERM)
    try:
        remainder, _ = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        _stop_process_group(process, signal.SIGKILL)
        remainder, _ = process.communicate()
    selector.close()
    if remainder:
        pending += remainder.decode(errors="replace")
    if pending:
        lines.append(pending)
    if timed_out:
        lines.append(f"\nvalidation timeout after {timeout}s\n")
    output = "".join(lines)
    collection_complete = not timed_out and _inventory_collection_complete(
        output, family
    )
    elapsed = time.monotonic() - start
    log_path.write_text(output, encoding="utf-8")
    returncode = 124 if timed_out else process.returncode
    return returncode, elapsed, output, collection_complete, outcome


def _fault_template(target: str, workload: Workload) -> dict:
    profile_policies = {}
    for profile in PROFILE_IDS:
        contract = _fault_qualification_contract_for_profile(workload, profile)
        profile_policies[profile] = (
            {
                "disposition": "not-applicable",
                "reason": contract.fault_qualification_withheld_reason,
                "tracking_issue": contract.tracking_issue,
            }
            if contract is not None
            else {"detector": "REVIEW_REQUIRED", "oracle": "any"}
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "workload": workload.id,
        "review_required": True,
        "faults": [
            {
                "id": family,
                "family": family,
                "environment": _fault_family_environment(target, family),
                "profiles": profile_policies,
            }
            for family in _fault_families(target, workload)
        ],
    }


def _fault_inventory_environment(family: str) -> dict[str, str]:
    """Enables family-specific analysis without selecting or applying a site."""
    return {
        name: value
        for name, value in FAULT_FAMILY_ENVIRONMENTS[family].items()
        if not name.endswith("_IDENTITY")
    }


def _inventory(args: argparse.Namespace) -> int:
    selection = _resolve_workload_selection(args, allow_all=False)
    target = selection.target
    workload = selection.require_workload()
    workspace = _workspace_from_environment()
    if not _doctor(workspace, target, (workload.id,))["ok"]:
        raise ValidationError("workspace doctor failed; run the doctor subcommand")
    root = args.artifact_root.resolve() / workload.id / "inventory"
    root.mkdir(parents=True, exist_ok=False)
    provenance = _write_provenance(workspace, target, workload, root)
    hook = _hook_path(workspace)
    command = _workload_command(
        workspace, target, workload, "fault", root / "unused.json"
    )
    family_runs = []
    aggregate_records = {"sites": set(), "sequences": set(), "destinations": set()}
    for family in _fault_families(target, workload):
        environment = _clean_environment("supercollider", workload, hook, target)
        # Clean qualification uses compact level-1 summaries. Fault inventory
        # explicitly requests level 2 because it consumes per-site identities.
        environment["RJ_CONSAN_LOG"] = "2"
        environment["RJ_CONSAN_FAULT_DRY_RUN"] = "1"
        environment.update(_fault_inventory_environment(family))
        log_path = root / f"command-{family}.log"
        returncode, elapsed, output, collection_complete, outcome = (
            _run_inventory_process(command, environment, log_path, args.timeout, family)
        )
        records = _inventory_records(output, family)
        for kind, values in records.items():
            aggregate_records[kind].update(values)
        family_runs.append(
            {
                "family": family,
                "environment": _controlled_environment(environment),
                "returncode": returncode,
                "outcome": outcome,
                "collection_complete": collection_complete,
                "elapsed_seconds": elapsed,
                "records": records,
                "log": str(log_path),
            }
        )
    records = {kind: sorted(values) for kind, values in aggregate_records.items()}
    document = {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "workload": workload.id,
        "command": command,
        "family_runs": family_runs,
        "returncode": 0,
        "elapsed_seconds": sum(run["elapsed_seconds"] for run in family_runs),
        "records": records,
        "hook": {"path": str(hook), "sha256": sha256_file(hook)},
        "provenance": str(provenance),
    }
    document["accepted"] = all(
        run["collection_complete"] and bool(run["records"]["sites"])
        for run in family_runs
    )
    if not document["accepted"]:
        document["returncode"] = next(
            (
                run["returncode"]
                for run in family_runs
                if not run["collection_complete"] and run["returncode"] != 0
            ),
            1,
        )
    atomic_write_json(root / "inventory.json", document)
    atomic_write_json(
        root / "fault-spec.template.json", _fault_template(target, workload)
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if document["accepted"] else 1


def _load_fault(
    path: Path,
    target: str,
    workload: Workload,
    fault_id: str,
    *,
    allow_reference: bool = False,
) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError("fault spec has unsupported schema_version")
    if document.get("target") != target:
        raise ValidationError("fault spec target does not match --target")
    if document.get("reference_only") is True and not allow_reference:
        raise ValidationError(
            "reference-only fault data must be copied and reviewed against inventory"
        )
    if "workloads" in document:
        workloads = document.get("workloads")
        if not isinstance(workloads, dict) or workload.id not in workloads:
            raise ValidationError("fault spec does not define --workload")
        workload_document = workloads[workload.id]
    else:
        if document.get("workload") != workload.id:
            raise ValidationError("fault spec workload does not match --workload")
        workload_document = document
    if not isinstance(workload_document, dict):
        raise ValidationError("fault workload policy must be an object")
    if document.get("review_required") is not False:
        raise ValidationError("fault spec must set review_required=false after review")
    faults = workload_document.get("faults", [])
    if not isinstance(faults, list):
        raise ValidationError("faults must be a list")
    matches = [
        fault
        for fault in faults
        if isinstance(fault, dict) and fault.get("id") == fault_id
    ]
    if len(matches) != 1:
        raise ValidationError(f"fault spec must define exactly one {fault_id!r}")
    fault = matches[0]
    if fault.get("family") not in _fault_families(target, workload):
        raise ValidationError("fault family is not admitted by the workload manifest")
    environment = fault.get("environment")
    if not isinstance(environment, dict) or not environment:
        raise ValidationError("fault environment must be a non-empty object")
    if any(
        not isinstance(key, str)
        or not key.startswith("RJ_CONSAN_FAULT_")
        or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValidationError(
            "fault environment may contain only string RJ_CONSAN_FAULT_* values"
        )
    mutations = [
        key
        for key, value in environment.items()
        if value == "1"
        and key
        in {
            "RJ_CONSAN_FAULT_DROP_BARRIER",
            "RJ_CONSAN_FAULT_MOVE_BARRIER",
            "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER",
            "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_SCOPE",
            "RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS",
        }
    ]
    if len(mutations) != 1:
        raise ValidationError("fault spec must enable exactly one mutation family")
    if "RJ_CONSAN_FAULT_SITE_IDENTITY" not in environment:
        raise ValidationError("fault spec must select an exact site identity")
    if any("REPLACE_FROM_INVENTORY" in value for value in environment.values()):
        raise ValidationError("fault spec still contains an inventory placeholder")
    return fault


def _fault_trials(fault: dict, profile: str) -> tuple[dict, list[dict[str, str]]]:
    profiles = fault.get("profiles", {})
    policy = profiles.get(profile, {}) if isinstance(profiles, dict) else {}
    if not isinstance(policy, dict):
        raise ValidationError(f"invalid profile policy for {profile}")
    policy_environment = policy.get("environment", {})
    if not isinstance(policy_environment, dict) or any(
        not isinstance(key, str)
        or not key.startswith("RJ_CONSAN_")
        or not isinstance(value, str)
        for key, value in policy_environment.items()
    ):
        raise ValidationError(f"invalid profile environment for {profile}")
    unset = policy.get("unset", [])
    if not isinstance(unset, list) or any(
        not isinstance(name, str) or not name.startswith("RJ_CONSAN_") for name in unset
    ):
        raise ValidationError(f"invalid profile unset list for {profile}")
    if "trials" in policy and "trial_axis" in policy:
        raise ValidationError(f"{profile} may define trials or trial_axis, not both")
    if "trial_axis" in policy:
        axis = policy["trial_axis"]
        if not isinstance(axis, dict) or len(axis) != 1:
            raise ValidationError(f"fault trial_axis for {profile} needs one setting")
        name, bounds = next(iter(axis.items()))
        if (
            not isinstance(name, str)
            or not name.startswith("RJ_CONSAN_")
            or not isinstance(bounds, dict)
            or not isinstance(bounds.get("start"), int)
            or not isinstance(bounds.get("stop"), int)
            or bounds["start"] >= bounds["stop"]
            or bounds["stop"] - bounds["start"] > 256
        ):
            raise ValidationError(f"invalid trial_axis for {profile}")
        trials = [
            {name: str(value)} for value in range(bounds["start"], bounds["stop"])
        ]
    else:
        trials = policy.get("trials", [{}])
    if not isinstance(trials, list) or not trials:
        raise ValidationError(f"fault trials for {profile} must be a non-empty list")
    for trial in trials:
        if not isinstance(trial, dict) or any(
            not isinstance(key, str)
            or not key.startswith("RJ_CONSAN_")
            or not isinstance(value, str)
            for key, value in trial.items()
        ):
            raise ValidationError(f"invalid trial environment for {profile}")
    return policy, trials


def _fault_trial_environment(
    profile: str,
    workload: Workload,
    hook: Path,
    target: str,
    fault: dict,
    policy: dict,
    trial: dict[str, str],
) -> dict[str, str]:
    environment = _clean_environment(profile, workload, hook, target)
    environment["CTEST_PARALLEL_LEVEL"] = "1"
    environment.update(fault["environment"])
    if policy.get("detector") in {"detected", "statistical"}:
        environment.pop("RJ_CONSAN_MOI_FORBID_DIAGNOSTICS", None)
    environment.update(policy.get("environment", {}))
    for name in policy.get("unset", []):
        environment.pop(name, None)
    environment.update(trial)
    environment["RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE"] = "1"
    return environment


def _faults_from_spec(
    path: Path,
    target: str,
    workload: Workload,
    *,
    allow_reference: bool,
) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("target") != target:
        raise ValidationError("fault spec target does not match --target")
    if "workloads" in document:
        workload_document = document.get("workloads", {}).get(workload.id)
    else:
        workload_document = (
            document if document.get("workload") == workload.id else None
        )
    if workload_document is None:
        return []
    faults = workload_document.get("faults", [])
    if not isinstance(faults, list):
        raise ValidationError("faults must be a list")
    loaded = []
    for fault in faults:
        if not isinstance(fault, dict) or not isinstance(fault.get("id"), str):
            raise ValidationError("every fault in the spec must have a string id")
        loaded.append(
            _load_fault(
                path,
                target,
                workload,
                fault["id"],
                allow_reference=allow_reference,
            )
        )
    return loaded


def _required_diagnostic(policy: dict) -> str:
    disposition = policy.get("disposition")
    if disposition == "not-applicable":
        return "not-applicable"
    detector = policy.get("detector")
    if detector == "detected":
        return "one attributable ConSan detection in every trial"
    if detector == "statistical":
        minimum = policy.get("minimum_detections", "REVIEW_REQUIRED")
        return f"at least {minimum} attributable ConSan detections across all trials"
    if detector == "not_detected":
        return "no ConSan detection; this is a precommitted qualified miss"
    return "REVIEW_REQUIRED"


def _fault_audit(
    workspace: Path,
    target: str,
    workload: Workload,
    fault: dict,
    profiles: tuple[str, ...],
    source: str,
) -> dict:
    hook = _hook_path(workspace)
    command = _workload_command(
        workspace,
        target,
        workload,
        "fault",
        Path("$ARTIFACT_ROOT") / workload.id / "fault" / "unused.json",
    )
    if workload.kind == "sharktank":
        command.append("--allow-oracle-failure")
    expectations = []
    for profile in profiles:
        contract = _fault_qualification_contract_for_profile(workload, profile)
        if contract is not None:
            policy = {
                "disposition": "not-applicable",
                "reason": contract.fault_qualification_withheld_reason,
                "tracking_issue": contract.tracking_issue,
            }
            trials = []
        else:
            policy, trials = _fault_trials(fault, profile)
        trial_audits = []
        if policy.get("disposition") != "not-applicable":
            for index, trial in enumerate(trials):
                environment = _fault_trial_environment(
                    profile, workload, hook, target, fault, policy, trial
                )
                trial_audits.append(
                    {
                        "index": index,
                        "overrides": _audited_settings(trial),
                        "effective_settings": _audited_settings(environment),
                        "implicit_runtime_defaults": _profile_runtime_defaults(
                            profile, environment
                        ),
                    }
                )
        expectations.append(
            {
                "profile": profile,
                "disposition": policy.get("disposition", "applicable"),
                "detector": policy.get("detector", "REVIEW_REQUIRED"),
                "oracle": policy.get("oracle", "any"),
                "required_diagnostic": _required_diagnostic(policy),
                "minimum_detections": policy.get("minimum_detections"),
                "reason": policy.get("reason"),
                "tracking_issue": policy.get("tracking_issue"),
                "policy_settings": _audited_settings(policy.get("environment", {})),
                "policy_unsets": _audited_unsets(policy.get("unset", [])),
                "trial_count": len(trial_audits),
                "trials": trial_audits,
            }
        )
    return {
        "id": fault["id"],
        "family": fault["family"],
        "source": source,
        "payload_argv": command,
        "validator_argv_template": [
            sys.executable,
            str(Path(__file__).resolve()),
            "--target",
            target,
            "fault",
            "--workload",
            workload.id,
            "--profile",
            "all" if len(profiles) > 1 else profiles[0],
            "--spec",
            "$FAULT_SPEC",
            "--fault",
            fault["id"],
            "--artifact-root",
            "$ARTIFACT_ROOT",
            "--allow-destructive",
        ],
        "mutation_settings": _audited_settings(fault["environment"]),
        "profile_expectations": expectations,
    }


def _explain_contract(
    workspace: Path,
    target: str,
    workload_ids: tuple[str, ...],
    profiles: tuple[str, ...],
    spec_path: Path | None,
    *,
    allow_reference: bool,
) -> dict:
    if allow_reference and spec_path is None:
        raise ValidationError("--allow-reference requires --spec")
    spec_document = None
    spec_metadata = None
    if spec_path is not None:
        spec_path = spec_path.resolve()
        spec_document = json.loads(spec_path.read_text(encoding="utf-8"))
        spec_metadata = {
            "path": str(spec_path),
            "sha256": sha256_file(spec_path),
            "reference_only": spec_document.get("reference_only") is True,
            "review_required": spec_document.get("review_required"),
        }
    workloads = []
    script = str(Path(__file__).resolve())
    for workload_id in workload_ids:
        workload = _workload_for_target(target, workload_id)
        effective_workload = _effective_workload(target, workload)
        output_root = Path("$ARTIFACT_ROOT") / workload.id
        commands = {}
        for phase in ("clean", "overhead"):
            profile_artifact_roots = {
                profile: output_root / _result_phase(phase, profile, workload) / profile
                for profile in profiles
            }
            result_phases = {
                _result_phase(phase, profile, workload) for profile in profiles
            }
            commands[phase] = {
                "payload_argv": (
                    _workload_command(
                        workspace,
                        target,
                        workload,
                        phase,
                        output_root
                        / next(iter(result_phases))
                        / "$PROFILE"
                        / "benchmark-0.json",
                    )
                    if len(result_phases) == 1
                    else None
                ),
                "processes": _outer_repetitions(target, phase, workload),
                "profile_artifact_roots": {
                    profile: str(root)
                    for profile, root in profile_artifact_roots.items()
                },
                "payload_argv_by_profile": {
                    profile: _workload_command(
                        workspace,
                        target,
                        workload,
                        phase,
                        root / "benchmark-0.json",
                    )
                    for profile, root in profile_artifact_roots.items()
                },
                "validator_argv_template": [
                    sys.executable,
                    script,
                    "--target",
                    target,
                    "run",
                    "--workload",
                    workload.id,
                    "--profile",
                    "all" if len(profiles) > 1 else profiles[0],
                    "--phase",
                    phase,
                    "--include-baseline",
                    "--artifact-root",
                    "$ARTIFACT_ROOT",
                ],
            }
        profile_audits = []
        for profile in profiles:
            result_phase = _result_phase("clean", profile, workload)
            environment = _run_environment(
                profile, workload, _hook_path(workspace), target, "clean"
            )
            inherited = _clean_environment(
                None, workload, _hook_path(workspace), target
            )
            harness_environment = {
                name: value
                for name, value in environment.items()
                if name == "HIP_TARGET"
                or name not in inherited
                or inherited[name] != value
            }
            settings = _audited_settings(harness_environment)
            runtime_defaults = _profile_runtime_defaults(profile, environment)
            profile_audits.append(
                {
                    "id": profile,
                    "flavor": PROFILES[profile].flavor,
                    "engine": PROFILES[profile].engine,
                    "clean_result_phase": result_phase,
                    "clean_artifact_root": str(output_root / result_phase / profile),
                    "settings": settings,
                    "implicit_runtime_defaults": runtime_defaults,
                    "usability_exceptions": [
                        setting
                        for setting in settings
                        if setting["usability_exception"]
                    ],
                }
            )
        if spec_path is None:
            fault_source = "unreviewed-template"
            faults = _fault_template(target, workload)["faults"]
        else:
            fault_source = (
                "reference-only"
                if spec_document.get("reference_only")
                else "reviewed-spec"
            )
            faults = _faults_from_spec(
                spec_path,
                target,
                workload,
                allow_reference=allow_reference,
            )
        workloads.append(
            {
                **asdict(effective_workload),
                "commands": commands,
                "profiles": profile_audits,
                "faults": [
                    _fault_audit(
                        workspace, target, workload, fault, profiles, fault_source
                    )
                    for fault in faults
                ],
                "fault_spec_status": (
                    fault_source if faults else "workload-not-present-in-spec"
                ),
            }
        )
    workload_tuning = []
    explicit_event_family_overrides = []
    forbidden_present = []
    fault_policy_exceptions = []
    for workload in workloads:
        for profile in workload["profiles"]:
            names = {setting["name"] for setting in profile["settings"]}
            forbidden_present.extend(
                {
                    "workload": workload["id"],
                    "profile": profile["id"],
                    "setting": name,
                }
                for name in sorted(names & set(ORDINARY_FORBIDDEN_ENVIRONMENT))
            )
            tuned = [
                setting["name"]
                for setting in profile["settings"]
                if setting["usability_exception"]
            ]
            if tuned:
                workload_tuning.append(
                    {
                        "workload": workload["id"],
                        "profile": profile["id"],
                        "settings": tuned,
                    }
                )
            selected = [
                setting["name"]
                for setting in profile["settings"]
                if "usability_note" in setting
                and setting["category"] == "instrumentation-selection"
            ]
            if selected:
                explicit_event_family_overrides.append(
                    {
                        "workload": workload["id"],
                        "profile": profile["id"],
                        "settings": selected,
                    }
                )
        for fault in workload["faults"]:
            for expectation in fault["profile_expectations"]:
                if expectation["policy_unsets"]:
                    fault_policy_exceptions.append(
                        {
                            "workload": workload["id"],
                            "fault": fault["id"],
                            "profile": expectation["profile"],
                            "unsets": [
                                setting["name"]
                                for setting in expectation["policy_unsets"]
                            ],
                        }
                    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": "consan-real-workload-validation-audit-v1",
        "target": target,
        "workspace": str(workspace),
        "setting_categories": SETTING_CATEGORIES,
        "ordinary_forbidden_environment": list(ORDINARY_FORBIDDEN_ENVIRONMENT),
        "usability_audit": {
            "coverage_limiting_controls_present": forbidden_present,
            "workload_specific_tuning": workload_tuning,
            "automatic_event_family_defaults": (
                [
                    {
                        "profiles": [
                            profile
                            for profile in profiles
                            if PROFILES[profile].flavor == "moi"
                        ],
                        "settings": sorted(ORDINARY_MOI_RUNTIME_DEFAULTS),
                    }
                ]
                if any(PROFILES[profile].flavor == "moi" for profile in profiles)
                else []
            ),
            "automatic_profile_defaults": [
                {
                    "profile": profile,
                    "settings": {
                        setting["name"]: setting["value"]
                        for setting in _profile_runtime_defaults(profile)
                    },
                }
                for profile in profiles
                if PROFILES[profile].flavor == "moi"
            ],
            "explicit_event_family_overrides": explicit_event_family_overrides,
            "fault_policy_exceptions": fault_policy_exceptions,
            "fault_qualification_exceptions": [
                {
                    "workload": workload["id"],
                    "profile": workload["coverage_output_contract"]["profile"],
                    "reason": workload["coverage_output_contract"][
                        "fault_qualification_withheld_reason"
                    ],
                    "tracking_issue": workload["coverage_output_contract"][
                        "tracking_issue"
                    ],
                }
                for workload in workloads
                if workload["coverage_output_contract"] is not None
                and workload["coverage_output_contract"]["withhold_fault_qualification"]
            ],
            "coverage_output_contracts": [
                {
                    "workload": workload["id"],
                    "contract": workload["coverage_output_contract"],
                }
                for workload in workloads
                if workload["coverage_output_contract"] is not None
            ],
        },
        "fault_spec": spec_metadata,
        "workloads": workloads,
    }


def _print_explain(document: dict) -> None:
    print(f"target: {document['target']}")
    print(f"workspace: {document['workspace']}")
    if document["fault_spec"] is None:
        print("fault expectations: REVIEW_REQUIRED templates (no --spec supplied)")
    else:
        source = (
            "reference-only" if document["fault_spec"]["reference_only"] else "reviewed"
        )
        print(f"fault expectations: {source} {document['fault_spec']['path']}")
    usability = document["usability_audit"]
    print(
        "ordinary coverage-limiting controls: "
        + ("PRESENT" if usability["coverage_limiting_controls_present"] else "none")
    )
    print(
        "workload-specific tuning: "
        + (
            ", ".join(
                f"{item['workload']}/{item['profile']}"
                for item in usability["workload_specific_tuning"]
            )
            if usability["workload_specific_tuning"]
            else "none"
        )
    )
    print(
        "coverage-output contracts: "
        + (
            ", ".join(
                f"{item['workload']}/{item['contract']['profile']}"
                f" ({item['contract']['tracking_issue']})"
                for item in usability["coverage_output_contracts"]
            )
            if usability["coverage_output_contracts"]
            else "none"
        )
    )
    print("exact selectors and effective per-trial environments: use --json")
    for workload in document["workloads"]:
        print(f"\n{workload['priority']} {workload['id']}")
        for phase in ("clean", "overhead"):
            phase_command = workload["commands"][phase]
            processes = phase_command["processes"]
            if phase_command["payload_argv"] is not None:
                command = shlex.join(phase_command["payload_argv"])
                print(f"  {phase} ({processes} process(es)): {command}")
            else:
                for profile in workload["profiles"]:
                    profile_id = profile["id"]
                    command = shlex.join(
                        phase_command["payload_argv_by_profile"][profile_id]
                    )
                    print(
                        f"  {phase}/{profile_id} "
                        f"({processes} process(es)): {command}"
                    )
        for profile in workload["profiles"]:
            controls = ", ".join(
                f"{setting['name']}={setting['value']} [{setting['category']}]"
                for setting in profile["settings"]
            )
            defaults = ", ".join(
                f"{setting['name']}={setting['value']}"
                for setting in profile["implicit_runtime_defaults"]
            )
            marker = " USABILITY EXCEPTION" if profile["usability_exceptions"] else ""
            print(f"  {profile['id']}{marker}: {controls}")
            if defaults:
                print(f"    automatic runtime defaults: {defaults}")
        for fault in workload["faults"]:
            outcomes = ", ".join(
                f"{item['profile']}={item['detector']}/{item['oracle']}"
                f" ({item['trial_count']} trial(s))"
                for item in fault["profile_expectations"]
            )
            print(f"  fault {fault['id']} [{fault['source']}]: {outcomes}")


def _fault_acceptance(result: dict, policy: dict) -> tuple[bool, list[str]]:
    reasons = []
    mutation = result.get("mutation", {})
    if mutation.get("requested") != 1:
        reasons.append(f"requested={mutation.get('requested')}")
    if mutation.get("planned") != 1:
        reasons.append(f"planned={mutation.get('planned')}")
    if mutation.get("applied") != 1:
        reasons.append(f"applied={mutation.get('applied')}")
    accounting_schema_version = mutation.get("accounting_schema_version")
    if accounting_schema_version != 2:
        reasons.append(
            "accounting_schema_version="
            f"{accounting_schema_version}, expected=2; rerun required"
        )
    elif mutation.get("installation_evidence_complete") is not True:
        reasons.append(
            "installation_evidence_complete="
            f"{mutation.get('installation_evidence_complete')}"
        )
    if mutation.get("discarded_applied", 0):
        reasons.append(f"discarded_applied={mutation['discarded_applied']}")
    expected_detector = policy.get("detector")
    actual_detector = result.get("sanitizer", {}).get("outcome")
    if expected_detector == "statistical":
        pass
    elif expected_detector not in {"detected", "not_detected"}:
        reasons.append(
            "profile policy lacks detector=detected|not_detected|statistical"
        )
    elif actual_detector != expected_detector:
        reasons.append(f"detector={actual_detector}, expected={expected_detector}")
    expected_oracle = policy.get("oracle", "any")
    actual_oracle = result.get("oracle", {}).get("outcome")
    if expected_oracle not in {"any", "pass", "fail"}:
        reasons.append(f"invalid expected oracle={expected_oracle}")
    elif expected_oracle != "any" and actual_oracle != expected_oracle:
        reasons.append(f"oracle={actual_oracle}, expected={expected_oracle}")
    execution = result.get("execution", {})
    if execution.get("timed_out"):
        reasons.append("timed out")
    execution_outcome = execution.get("outcome")
    if execution_outcome in {
        "signal",
        "queue_timeout",
        "device_lost",
        "preflight_device_unhealthy",
        "preflight_device_quarantined",
    }:
        reasons.append(f"invalid execution outcome={execution_outcome}")
    if execution_outcome == "trap" and actual_detector != "detected":
        reasons.append("unattributed trap is not a detection")
    for name in ("health_before", "health_after"):
        health = execution.get(name)
        if not isinstance(health, dict) or not health.get("healthy"):
            reasons.append(f"{name} failed")
    return not reasons, reasons


def _fault(args: argparse.Namespace) -> int:
    selection = _resolve_workload_selection(args, allow_all=False)
    target = selection.target
    workload = selection.require_workload()
    workspace = _workspace_from_environment()
    if not _doctor(workspace, target, (workload.id,))["ok"]:
        raise ValidationError("workspace doctor failed; run the doctor subcommand")
    if not args.allow_destructive:
        raise ValidationError("fault execution requires --allow-destructive")
    spec_path = args.spec.resolve()
    fault = _load_fault(spec_path, target, workload, args.fault)
    profiles = PROFILE_IDS if args.profile == "all" else (args.profile,)
    hook = _hook_path(workspace)
    fault_root = args.artifact_root.resolve() / workload.id / "faults" / fault["id"]
    fault_root.mkdir(parents=True, exist_ok=False)
    provenance = _write_provenance(workspace, target, workload, fault_root)
    root = fault_root / "rows"
    root.mkdir()
    smoke = _health_smoke_command(
        workspace, target, workload, root / "health-smoke.json"
    )
    if args.smoke_command_json is not None:
        smoke = args.smoke_command_json
    health_command = args.health_command_json or [
        shutil.which("rocminfo") or "rocminfo"
    ]
    runner = Path(__file__).with_name("consan_fault_runner.py")
    summaries = []
    profile_summaries = []
    for profile in profiles:
        contract = _fault_qualification_contract_for_profile(workload, profile)
        if contract is not None:
            row = {
                "profile": profile,
                "accepted": True,
                "disposition": "not-applicable",
                "reason": contract.fault_qualification_withheld_reason,
                "tracking_issue": contract.tracking_issue,
            }
            summaries.append(row)
            profile_summaries.append(
                {
                    "profile": profile,
                    "accepted": True,
                    "disposition": "not-applicable",
                    "reason": contract.fault_qualification_withheld_reason,
                    "tracking_issue": contract.tracking_issue,
                }
            )
            continue
        policy, trials = _fault_trials(fault, profile)
        if policy.get("disposition") == "not-applicable":
            row = {
                "profile": profile,
                "accepted": True,
                "disposition": "not-applicable",
                "reason": policy.get("reason"),
                "tracking_issue": policy.get("tracking_issue"),
            }
            summaries.append(row)
            profile_summaries.append(dict(row))
            continue
        profile_rows = []
        for index, trial in enumerate(trials):
            name = f"{fault['id']}-{profile}-{index}"
            environment = _fault_trial_environment(
                profile, workload, hook, target, fault, policy, trial
            )
            enabled_mutations = [
                key
                for key in (
                    "RJ_CONSAN_FAULT_DROP_BARRIER",
                    "RJ_CONSAN_FAULT_MOVE_BARRIER",
                    "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER",
                    "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_SCOPE",
                    "RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS",
                )
                if environment.get(key) == "1"
            ]
            if len(enabled_mutations) != 1:
                raise ValidationError(
                    f"trial {profile}/{index} enables {len(enabled_mutations)} mutations"
                )
            command = _workload_command(
                workspace, target, workload, "fault", root / "unused.json"
            )
            if workload.kind == "sharktank":
                command.append("--allow-oracle-failure")
            identities = sorted(
                value
                for key, value in environment.items()
                if key.startswith("RJ_CONSAN_FAULT_") and key.endswith("IDENTITY")
            )
            invocation = [
                sys.executable,
                str(runner),
                "--artifact-root",
                str(root),
                "--name",
                name,
                "--row-role",
                "fault",
                "--corpus",
                workload.corpus,
                "--workload",
                workload.id,
                "--flavor",
                PROFILES[profile].flavor,
                "--engine",
                PROFILES[profile].engine,
                "--fault-family",
                fault["family"],
                "--timeout",
                str(args.timeout),
                "--health-timeout",
                str(args.health_timeout),
                "--destructive",
                "--allow-destructive",
                "--serialize-gpu",
                "--health-command-json",
                json.dumps(health_command),
                "--smoke-command-json",
                json.dumps(smoke),
                "--revision-root",
                str(workspace / workload.corpus),
                "--hash-file",
                f"hook={hook}",
            ]
            for key, value in _controlled_environment(environment).items():
                invocation.extend(["--env", f"{key}={value}"])
            for identity in identities:
                invocation.extend(["--site-id", identity])
            invocation.extend(["--", *command])
            child_environment = _clean_environment(None, workload, hook, target)
            child_environment["CTEST_PARALLEL_LEVEL"] = "1"
            subprocess.run(invocation, env=child_environment, check=False)
            result_path = root / name / "result.json"
            if not result_path.is_file():
                row = {
                    "profile": profile,
                    "trial": index,
                    "accepted": False,
                    "reasons": ["fault runner produced no result.json"],
                    "detector": None,
                }
                summaries.append(row)
                profile_rows.append(row)
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            accepted, reasons = _fault_acceptance(result, policy)
            row = {
                "profile": profile,
                "trial": index,
                "accepted": accepted,
                "reasons": reasons,
                "detector": result.get("sanitizer", {}).get("outcome"),
                "oracle": result.get("oracle", {}).get("outcome"),
                "result": str(result_path),
            }
            summaries.append(row)
            profile_rows.append(row)
        detected = sum(row.get("detector") == "detected" for row in profile_rows)
        expected_detector = policy.get("detector")
        profile_reasons = []
        if expected_detector == "statistical":
            minimum = policy.get("minimum_detections")
            if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
                profile_reasons.append(
                    "statistical policy needs minimum_detections >= 1"
                )
            elif detected < minimum:
                profile_reasons.append(
                    f"detections={detected}/{len(profile_rows)}, minimum={minimum}"
                )
        profile_summaries.append(
            {
                "profile": profile,
                "accepted": all(row["accepted"] for row in profile_rows)
                and not profile_reasons,
                "detector_policy": expected_detector,
                "detections": detected,
                "trials": len(profile_rows),
                "reasons": profile_reasons,
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "workload": workload.id,
        "fault": fault["id"],
        "fault_spec": {
            "path": str(spec_path),
            "sha256": sha256_file(spec_path),
        },
        "provenance": str(provenance),
        "rows": summaries,
        "profiles": profile_summaries,
        "accepted": all(profile["accepted"] for profile in profile_summaries),
    }
    summary_path = fault_root / "summary.json"
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["accepted"] else 1


def _run(args: argparse.Namespace) -> int:
    selection = _resolve_workload_selection(args, allow_all=False)
    target = selection.target
    workload = selection.require_workload()
    workspace = _workspace_from_environment()
    timeout = args.timeout if args.timeout is not None else workload.run_timeout_seconds
    doctor = _doctor(workspace, target, (workload.id,))
    if not doctor["ok"]:
        raise ValidationError("workspace doctor failed; run the doctor subcommand")
    artifact_root = args.artifact_root.resolve()
    launcher = _launcher_from_json(args.launcher_json)
    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_provenance(
        workspace,
        target,
        workload,
        _workload_provenance_path(artifact_root, workload).parent,
    )
    profiles = PROFILE_IDS if args.profile == "all" else (args.profile,)
    if args.phase == "overhead" and args.include_baseline:
        selections = (
            ((None, "baseline-before"),)
            + tuple((profile, None) for profile in profiles)
            + ((None, "baseline-after"),)
        )
    else:
        selected = (None, *profiles) if args.include_baseline else profiles
        selections = tuple((profile, None) for profile in selected)
    results = []
    baseline_before_failed = False
    for profile, row_label in selections:
        result = _run_profile(
            workspace,
            target,
            workload,
            profile,
            args.phase,
            artifact_root,
            timeout,
            row_label,
            launcher,
        )
        results.append(result)
        if (
            args.phase == "overhead"
            and args.include_baseline
            and row_label == "baseline-before"
            and not result["accepted"]
        ):
            baseline_before_failed = True
            break
    if args.phase == "overhead" and args.include_baseline:
        if baseline_before_failed:
            summary = {
                "schema_version": SCHEMA_VERSION,
                "baseline_policy": "mean-of-before-and-after-medians",
                "paired_baseline_median_ms": {},
                "profiles": {},
                "accepted": False,
                "reasons": ["baseline-before rejected; profile phases skipped"],
            }
        else:
            summary = _overhead_summary(results)
        summary_path = artifact_root / workload.id / "overhead" / "summary.json"
        atomic_write_json(summary_path, summary)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(result["accepted"] for result in results) else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help=f"gfx target; defaults to {TARGET_ENV}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="validate tools and workspace layout")
    doctor.add_argument(
        "--workload", choices=(*tuple(WORKLOAD_BY_ID), "all"), default="all"
    )
    doctor.add_argument("--json", action="store_true")

    manifest = subparsers.add_parser("manifest", help="print the executable matrix")
    manifest.add_argument("--json", action="store_true")

    explain = subparsers.add_parser(
        "explain", help="expand commands, settings, and fault expectations"
    )
    explain.add_argument(
        "--workload", choices=(*tuple(WORKLOAD_BY_ID), "all"), default="all"
    )
    explain.add_argument("--profile", choices=(*PROFILE_IDS, "all"), default="all")
    explain.add_argument(
        "--spec", type=Path, help="reviewed fault spec to include in the audit"
    )
    explain.add_argument(
        "--allow-reference",
        action="store_true",
        help="audit, but never execute, a reference-only historical spec",
    )
    explain.add_argument("--json", action="store_true")

    run = subparsers.add_parser("run", help="run clean correctness or overhead rows")
    run.add_argument("--workload", choices=tuple(WORKLOAD_BY_ID), required=True)
    run.add_argument("--profile", choices=(*PROFILE_IDS, "all"), default="all")
    run.add_argument("--phase", choices=("clean", "overhead"), required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument(
        "--timeout",
        type=int,
        help="override the workload timeout declared by the executable manifest",
    )
    run.add_argument("--include-baseline", action="store_true")
    run.add_argument(
        "--launcher-json",
        help="JSON argv prefix used to launch each workload process",
    )

    inventory = subparsers.add_parser(
        "inventory", help="record target-specific fault sites without mutation"
    )
    inventory.add_argument("--workload", choices=tuple(WORKLOAD_BY_ID), required=True)
    inventory.add_argument("--artifact-root", type=Path, required=True)
    inventory.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)

    fault = subparsers.add_parser(
        "fault", help="run a reviewed exact fault spec with health containment"
    )
    fault.add_argument("--workload", choices=tuple(WORKLOAD_BY_ID), required=True)
    fault.add_argument("--profile", choices=(*PROFILE_IDS, "all"), default="all")
    fault.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="reviewed JSON spec generated from the current inventory",
    )
    fault.add_argument("--fault", required=True, help="fault id in the JSON spec")
    fault.add_argument("--artifact-root", type=Path, required=True)
    fault.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)
    fault.add_argument(
        "--health-timeout",
        type=float,
        default=30.0,
        help="deadline in seconds for each retained discovery and smoke probe",
    )
    fault.add_argument("--allow-destructive", action="store_true")
    fault.add_argument(
        "--health-command-json",
        type=_command_json,
        help="explicit retained health-discovery command",
    )
    fault.add_argument(
        "--smoke-command-json",
        type=_command_json,
        help="explicit retained target-dispatch smoke command",
    )
    args = parser.parse_args(argv)
    timeout = getattr(args, "timeout", None)
    if timeout is not None and timeout <= 0:
        parser.error("--timeout must be positive")
    if getattr(args, "health_timeout", 1) <= 0:
        parser.error("--health-timeout must be positive")
    if (getattr(args, "health_command_json", None) is None) != (
        getattr(args, "smoke_command_json", None) is None
    ):
        parser.error(
            "--health-command-json and --smoke-command-json must be provided together"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        selection = _resolve_workload_selection(args, allow_all=True)
        target = selection.target
        # Reject cheap target/input mismatches before requiring a configured
        # workspace. Handlers reuse the same resolver for direct entry calls.
        if args.command == "manifest":
            result = _manifest(target)
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                for workload in _workloads_for_target(target):
                    faults = ",".join(_fault_families(target, workload))
                    print(f"{workload.priority} {workload.id}: {faults}")
            return 0
        workspace = _workspace_from_environment()
        if args.command == "explain":
            if selection.is_all:
                workload_ids = tuple(
                    workload.id for workload in _workloads_for_target(target)
                )
            else:
                workload_ids = (selection.require_workload().id,)
            profiles = PROFILE_IDS if args.profile == "all" else (args.profile,)
            result = _explain_contract(
                workspace,
                target,
                workload_ids,
                profiles,
                args.spec,
                allow_reference=args.allow_reference,
            )
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                _print_explain(result)
            return 0
        if args.command == "doctor":
            result = _doctor(workspace, target, selection.selected_ids())
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print(f"workspace: {result['workspace']}")
                print(f"target: {result['target']}")
                for label, item in result["paths"].items():
                    state = "ok" if item["present"] else "MISSING"
                    print(f"{state:7} {label}: {item['path']}")
                for tool, path in result["tools"].items():
                    print(
                        f"{'ok' if path else 'MISSING':7} PATH tool {tool}: {path or '-'}"
                    )
                for runtime, item in result.get("runtimes", {}).items():
                    state = "ok" if item["ok"] else "BROKEN"
                    print(
                        f"{state:7} {runtime} runtime {item['python']}: "
                        f"{json.dumps(item['detail'], sort_keys=True)}"
                    )
                    for reason in item.get("reasons", ()):
                        print(f"        reason: {reason}")
            return 0 if result["ok"] else 1
        if args.command == "inventory":
            return _inventory(args)
        if args.command == "fault":
            return _fault(args)
        return _run(args)
    except (OSError, ValidationError, ValueError, json.JSONDecodeError) as error:
        print(f"validation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
