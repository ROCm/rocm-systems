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
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
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
    FAULT_RESERVATION_QUALIFIED,
    SITE_KINDS,
    atomic_write_json,
    fault_reservation_qualification,
    git_identity,
    sha256_file,
)

SCHEMA_VERSION = 2
EMPIRICAL_CAMPAIGN_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 2
WORKSPACE_ENV = "CONSAN_VALIDATION_WORKSPACE_DIR"
TARGET_ENV = "CONSAN_VALIDATION_TARGET"
PYTORCH_PYTHON_ENV = "CONSAN_VALIDATION_PYTORCH_PYTHON"
SHARKTANK_PYTHON_ENV = "CONSAN_VALIDATION_SHARKTANK_PYTHON"
TENSILE_PYTHON_ENV = "CONSAN_VALIDATION_TENSILE_PYTHON"
RDNA4_MATMUL_DIR_ENV = "CONSAN_VALIDATION_RDNA4_MATMUL_DIR"
LLAMA_BUILD_DIR_ENV = "CONSAN_VALIDATION_LLAMA_BUILD_DIR"
LLVM_READELF_ENV = "CONSAN_VALIDATION_LLVM_READELF"
TIMEOUT_SECONDS = 30
EMPIRICAL_DEFAULT_ROUNDS = 10
EMPIRICAL_DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
EMPIRICAL_DEFAULT_BASELINE_DRIFT_LIMIT = 0.05
EMPIRICAL_MINIMUM_TIMED_MS = 250.0
EMPIRICAL_MAX_INNER_REPETITIONS = 1_000_000
PROCESS_OUTPUT_DRAIN_SECONDS = 2
PROCESS_TERMINATION_GRACE_SECONDS = 5
NATIVE_CDNA_TARGETS = frozenset(("gfx942", "gfx950"))
SINGLE_REPETITION_TARGETS = frozenset(("gfx942", "gfx950", "gfx1250"))
QWEN_OVERHEAD_REPETITIONS = {target: 1 for target in SINGLE_REPETITION_TARGETS}
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
    self_timed_device_minimum_ms: float | None = None
    warm_timing_mode: str | None = None


@dataclass(frozen=True)
class RetiredWorkloadCoverage:
    id: str
    tracking_issue: str
    coverage_successors: tuple[str, ...]
    remaining_gap: str


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
COVERAGE_OUTPUT_ARTIFACT_SCHEMA_VERSION = 1
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
RETIRED_WORKLOAD_COVERAGE = {
    "gfx1201": (
        RetiredWorkloadCoverage(
            id="pytorch-rdna4-sdpa",
            tracking_issue="bd-1w9.26",
            coverage_successors=("d128-block", "d128-pressure", "wmma-attention"),
            remaining_gap="real-framework causal attention",
        ),
    ),
}


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
    for target, retired_rows in RETIRED_WORKLOAD_COVERAGE.items():
        for retired in retired_rows:
            if retired.id in WORKLOAD_BY_ID:
                raise RuntimeError(f"retired workload remains registered: {retired.id}")
            if not retired.tracking_issue.startswith("bd-"):
                raise RuntimeError(
                    f"retired workload needs a tracking bead: {retired.id}"
                )
            for successor_id in retired.coverage_successors:
                successor = WORKLOAD_BY_ID.get(successor_id)
                if successor is None or (
                    successor.targets is not None and target not in successor.targets
                ):
                    raise RuntimeError(
                        f"{target} retired workload has unavailable coverage successor: "
                        f"{retired.id} -> {successor_id}"
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


def _jakub_override(target: _NativeGtestTarget) -> dict[str, str]:
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

    # The producer-skew case is a schedule discriminator, not a timing row.
    # Keep clean coverage broad, measure only ordinary variants, and mutate
    # only the case whose exact oracle is designed to expose a missing handoff.
    return {
        "relative_path": relative_path,
        "clean_filter": f"{oracle_prefix}/*",
        "overhead_filter": (
            f"{oracle_prefix}/CooperativeLdsK32:"
            f"{oracle_prefix}/DoubleBufferedLdsK128"
        ),
        "fault_filter": f"{oracle_prefix}/ProducerSkewLdsK128",
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


def _rdna4_matmul_root(workspace: Path) -> Path:
    configured = os.environ.get(RDNA4_MATMUL_DIR_ENV)
    if configured:
        return Path(os.path.abspath(Path(configured).expanduser()))
    return workspace / "rdna4_matmul"


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
    if any(workload.corpus == "rdna4-matmul" for workload in workloads):
        paths["rdna4-matmul"] = _rdna4_matmul_root(workspace)
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
    candidates = []
    configured = os.environ.get(LLAMA_BUILD_DIR_ENV)
    if configured:
        candidates.append(Path(os.path.abspath(Path(configured).expanduser())) / name)
    candidates.extend(
        (
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
    )
    return next(
        (candidate for candidate in candidates if candidate.is_file()), candidates[0]
    )


def _llama_runtime_files(executable: Path) -> dict[str, Path]:
    build_root = executable.parents[2]
    source_root = build_root / "third_party" / "llama.cpp" / "ggml" / "src"
    return {
        "ggml": source_root / "libggml.so.0",
        "ggml-base": source_root / "libggml-base.so.0",
        "ggml-cpu": source_root / "libggml-cpu.so.0",
        "ggml-hip": source_root / "ggml-hip" / "libggml-hip.so.0",
    }


def _input_files(workspace: Path, target: str, workload: Workload) -> dict[str, Path]:
    workload = _resolved_workload(target, workload)
    if workload.kind == "pytorch":
        return {
            "python": _pytorch_python(workspace),
            "workload-source": Path(__file__).with_name(workload.relative_path),
        }
    if workload.kind == "tensile":
        paths = resolve_tensile_validation_paths(workspace, target)
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
        executable = _llama_executable(workspace, target, workload.relative_path)
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
            "executable": executable,
            **_llama_runtime_files(executable),
        }
    if workload.kind == "rdna4-matmul":
        root = _rdna4_matmul_root(workspace)
        return {
            "workload-source": Path(__file__).with_name(
                "consan_rdna4_matmul_validation.py"
            ),
            "project-source": root / "rdna4_matmul.hip",
            "project-build-script": root / "build_and_test.sh",
            "executable": root / workload.relative_path,
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
        "retired_workloads": [
            asdict(workload) for workload in RETIRED_WORKLOAD_COVERAGE.get(target, ())
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
    if workload.kind == "llama" and os.environ.get(LLAMA_BUILD_DIR_ENV):
        executable = (
            Path(os.path.abspath(Path(os.environ[LLAMA_BUILD_DIR_ENV]).expanduser()))
            / workload.relative_path
        )
        library_directories = list(
            dict.fromkeys(
                str(path.parent) for path in _llama_runtime_files(executable).values()
            )
        )
        existing = environment.get("LD_LIBRARY_PATH")
        if existing:
            library_directories.append(existing)
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_directories)
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
    if workload.kind in {"pytorch", "llama", "rdna4-matmul"}:
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
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_OVERRIDE_GFX_VERSION",
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
    if profile == "record-replay":
        defaults.update(RECORD_REPLAY_STANDARD_RUNTIME_DEFAULTS)
    elif profile == "sampled":
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
    workspace: Path,
    target: str,
    overhead: bool,
    output: Path,
    repetitions_override: int | None = None,
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
        repetitions = repetitions_override or QWEN_OVERHEAD_REPETITIONS.get(target, 10)
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
    qwen_command = _qwen_command(workspace, target, False, output)
    if shutil.which(qwen_command[0]) and all(
        path.is_file() for path in _input_files(workspace, target, qwen).values()
    ):
        return qwen_command
    # A workload-scoped doctor permits an independently ready row to proceed
    # when unrelated Qwen artifacts are absent. Its destructive health gate
    # must honor the same contract instead of manufacturing an unhealthy GPU
    # result from a missing universal smoke file.
    return _workload_command(workspace, target, workload, "clean", output)


def _inner_repetitions(target: str, phase: str, workload: Workload) -> int:
    if phase != "overhead" or target in SINGLE_REPETITION_TARGETS:
        return 1
    # A workload either collects warm in-process samples or declares multiple
    # isolated outer processes. Do not multiply the two repetition axes.
    return 1 if workload.overhead_processes > 1 else 10


def _workload_command(
    workspace: Path,
    target: str,
    workload: Workload,
    phase: str,
    output: Path,
    inner_repetitions_override: int | None = None,
) -> list[str]:
    workload = _resolved_workload(target, workload)
    overhead = phase == "overhead"
    if workload.kind == "qwen":
        return _qwen_command(
            workspace,
            target,
            overhead,
            output,
            inner_repetitions_override,
        )
    if workload.kind == "sharktank":
        # The active architecture campaigns use one end-to-end repetition.
        # Keep both the outer process count and this inner suite count at one.
        repetitions = inner_repetitions_override or _inner_repetitions(
            target, phase, workload
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
        # Large rows may declare isolated outer processes because repeated
        # instrumented dispatches accumulate bounded report state. Small rows
        # retain warm in-process timing, and simulator targets stay single-shot.
        repetitions = inner_repetitions_override or _inner_repetitions(
            target, phase, workload
        )
        return [
            str(_pytorch_python(workspace)),
            str(Path(__file__).with_name(workload.relative_path)),
            "--workload",
            workload.id.removeprefix("pytorch-"),
            "--repetitions",
            str(repetitions),
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
            "--gpu-target",
            target,
            "--output-dir",
            str(output.parent / "tensile-work"),
            "--repetitions",
            "1",
            "--minimum-timed-ms",
            str(EMPIRICAL_MINIMUM_TIMED_MS),
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
    if workload.kind == "rdna4-matmul":
        command = [
            sys.executable,
            str(Path(__file__).with_name("consan_rdna4_matmul_validation.py")),
            "--executable",
            str(_rdna4_matmul_root(workspace) / workload.relative_path),
            "--workload",
            workload.id.removeprefix("rdna4-matmul-"),
            "--phase",
            "clean" if phase in {"clean", "fault"} else "warm",
            "--repetitions",
            "1",
            "--minimum-timed-ms",
            str(workload.self_timed_device_minimum_ms),
            "--label",
            f"{workload.id}-{phase}",
        ]
        if inner_repetitions_override is not None:
            command.extend(["--fixed-iterations", str(inner_repetitions_override)])
        return command
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
    files.update(_runtime_library_paths())
    llvm_readelf = _llvm_readelf()
    if llvm_readelf is not None and llvm_readelf.is_file():
        files["llvm-readelf"] = llvm_readelf
    document = {
        "schema_version": SCHEMA_VERSION,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
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
        "environment_selectors": {
            name: {"present": name in os.environ, "value": os.environ.get(name)}
            for name in (
                "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL",
                "HSA_OVERRIDE_GFX_VERSION",
            )
        },
        "machine": _machine_identity(target),
        "runtime_tools": _runtime_tool_identities(llvm_readelf, hook),
        "workload_runtime": _workload_runtime_identity(workspace, workload),
        "observations": _empirical_observation_snapshot(),
    }
    normalized_document = json.loads(json.dumps(document))
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValidationError(
                f"cannot read existing provenance {path}: {error}"
            ) from error
        stable_existing = {
            key: value for key, value in existing.items() if key != "observations"
        }
        stable_document = {
            key: value
            for key, value in normalized_document.items()
            if key != "observations"
        }
        if stable_existing != stable_document:
            raise ValidationError(
                f"provenance conflicts with existing artifact: {path}"
            )
        return path
    atomic_write_json(path, document)
    return path


def _read_identity_file(path: Path) -> dict[str, object]:
    try:
        value = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as error:
        return {"available": False, "reason": str(error)}
    return {"available": True, "value": value}


def _command_identity(command: list[str], timeout: int = 10) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "command": command, "reason": str(error)}
    output = completed.stdout or ""
    limit = 65536
    return {
        "available": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "output": output[:limit],
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "output_truncated": len(output) > limit,
    }


def _workload_runtime_identity(
    workspace: Path, workload: Workload
) -> dict[str, object]:
    if workload.kind != "pytorch":
        return {"kind": workload.kind}
    python = _pytorch_python(workspace)
    script = (
        "import json, torch, triton; "
        "print(json.dumps({"
        "'torch_version': torch.__version__, "
        "'torch_hip_version': torch.version.hip, "
        "'torch_file': torch.__file__, "
        "'triton_version': getattr(triton, '__version__', None), "
        "'triton_file': triton.__file__}, sort_keys=True))"
    )
    return {
        "kind": workload.kind,
        "python_packages": _command_identity([str(python), "-c", script]),
    }


def _gfx_target_version(target: str) -> int | None:
    match = re.fullmatch(r"gfx([0-9]{2})([0-9])([0-9])", target)
    if match is None:
        return None
    major, minor, stepping = (int(value) for value in match.groups())
    return major * 10000 + minor * 100 + stepping


def _machine_identity(target: str) -> dict[str, object]:
    topology = {}
    topology_root = Path("/sys/class/kfd/kfd/topology/nodes")
    if topology_root.is_dir():
        for node in sorted(topology_root.iterdir(), key=lambda path: path.name):
            if not node.is_dir():
                continue
            topology[node.name] = {
                name: _read_identity_file(node / name)
                for name in ("gpu_id", "name", "properties")
            }
    pci_devices = {}
    for device in sorted(Path("/sys/bus/pci/devices").glob("*")):
        vendor = _read_identity_file(device / "vendor")
        if vendor.get("value") != "0x1002":
            continue
        pci_devices[device.name] = {
            name: _read_identity_file(device / name)
            for name in (
                "vendor",
                "device",
                "subsystem_vendor",
                "subsystem_device",
                "revision",
            )
        }
        driver = device / "driver"
        try:
            pci_devices[device.name]["driver"] = driver.resolve().name
        except OSError:
            pci_devices[device.name]["driver"] = None
    target_version = _gfx_target_version(target)
    selected_kfd_nodes = []
    if target_version is not None:
        marker = f"gfx_target_version {target_version}"
        selected_kfd_nodes = [
            node
            for node, identity in topology.items()
            if marker in str(identity["properties"].get("value", "")).splitlines()
        ]
    return {
        "uname": platform.uname()._asdict(),
        "os_release": _read_identity_file(Path("/etc/os-release")),
        "kfd_topology": topology,
        "selected_kfd_nodes": selected_kfd_nodes,
        "amd_pci_devices": pci_devices,
        "amdgpu_module_version": _read_identity_file(
            Path("/sys/module/amdgpu/version")
        ),
        "amdgpu_module_source_version": _read_identity_file(
            Path("/sys/module/amdgpu/srcversion")
        ),
    }


def _runtime_library_paths() -> dict[str, Path]:
    roots = [
        Path(value)
        for value in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if value
    ]
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path:
        roots.append(Path(rocm_path) / "lib")
    libraries = {}
    for label, name in (
        ("hsa-runtime", "libhsa-runtime64.so"),
        ("hip-runtime", "libamdhip64.so"),
    ):
        match = next((root / name for root in roots if (root / name).is_file()), None)
        if match is not None:
            libraries[label] = match
    return libraries


def _unavailable_command_identity(name: str) -> dict[str, object]:
    return {"available": False, "command": [name], "reason": "tool is unavailable"}


def _empirical_observation_snapshot() -> dict[str, object]:
    rocm_smi = shutil.which("rocm-smi")
    amd_smi = shutil.which("amd-smi")
    return {
        "schema_version": 1,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clocks_temperature_and_utilization": (
            _command_identity(
                [
                    rocm_smi,
                    "--showtemp",
                    "--showclocks",
                    "--showuse",
                    "--json",
                ]
            )
            if rocm_smi
            else _unavailable_command_identity("rocm-smi")
        ),
        "competing_gpu_processes": (
            _command_identity([amd_smi, "process", "--json"])
            if amd_smi
            else _unavailable_command_identity("amd-smi")
        ),
        "firmware": (
            _command_identity([amd_smi, "firmware", "--json"])
            if amd_smi
            else _unavailable_command_identity("amd-smi")
        ),
        "amd_smi_metrics": (
            _command_identity([amd_smi, "metric", "--json"])
            if amd_smi
            else _unavailable_command_identity("amd-smi")
        ),
    }


def _runtime_tool_identities(
    llvm_readelf: Path | None, hook: Path
) -> dict[str, object]:
    commands = {
        "python": [sys.executable, "--version"],
        "rocminfo": [shutil.which("rocminfo") or "rocminfo"],
    }
    if llvm_readelf is not None:
        commands["llvm-readelf"] = [str(llvm_readelf), "--version"]
    commands["hook-linkage"] = [shutil.which("ldd") or "ldd", str(hook)]
    rocm_sdk = shutil.which("rocm-sdk")
    if rocm_sdk:
        commands["rocm-sdk"] = [rocm_sdk, "path", "--root"]
    amdclang = shutil.which("amdclang++")
    if amdclang:
        commands["amdclang++"] = [amdclang, "--version"]
    return {name: _command_identity(command) for name, command in commands.items()}


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
    if workload.kind == "rdna4-matmul":
        roots.append(_rdna4_matmul_root(workspace))
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
                    ("reader", identity[0]),
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


def _benchmark_samples(path: Path) -> list[float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("benchmarks", [])
    iterations = [
        row
        for row in rows
        if row.get("run_type") == "iteration"
        and str(row.get("name", "")).startswith("BM_main/")
    ]
    selected = iterations
    if not selected:
        selected = [
            row
            for row in rows
            if row.get("aggregate_name") == "median"
            and str(row.get("name", "")).startswith("BM_main/")
        ]
    if not selected:
        raise ValidationError(f"expected Qwen benchmark timing rows in {path}")
    benchmark_names = {str(row.get("name", "")) for row in selected}
    if len(benchmark_names) != 1:
        raise ValidationError(
            f"expected one Qwen benchmark identity in {path}, found "
            f"{sorted(benchmark_names)}"
        )
    scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
    try:
        return [float(row["real_time"]) * scale[row["time_unit"]] for row in selected]
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationError(f"malformed Qwen benchmark timing in {path}") from error


def _benchmark_median(path: Path) -> float:
    return statistics.median(_benchmark_samples(path))


def _json_timing_samples(log_text: str, workload_kind: str) -> dict[str, list[float]]:
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
    timings = {}
    for key, value in document.items():
        if not isinstance(value, dict):
            continue
        if "median_ms" in value:
            samples = value.get("samples_ms")
            if samples is None:
                samples = [value["median_ms"]]
            if (
                not isinstance(samples, list)
                or not samples
                or any(
                    not isinstance(sample, (int, float))
                    or not math.isfinite(float(sample))
                    or sample <= 0
                    for sample in samples
                )
            ):
                raise ValidationError(
                    f"invalid {workload_kind} timing samples for {key}"
                )
            timings[key] = [float(sample) for sample in samples]
        device_samples = value.get("device_samples_ms")
        if device_samples is None and "device_median_ms" in value:
            device_samples = [value["device_median_ms"]]
        if device_samples is not None:
            if (
                not isinstance(device_samples, list)
                or not device_samples
                or any(
                    not isinstance(sample, (int, float))
                    or not math.isfinite(float(sample))
                    or sample <= 0
                    for sample in device_samples
                )
            ):
                raise ValidationError(
                    f"invalid {workload_kind} device timing samples for {key}"
                )
            timings[f"{key}:device"] = [float(sample) for sample in device_samples]
    return timings


def _json_measurements(log_text: str, workload_kind: str) -> dict[str, dict]:
    documents = []
    for line in log_text.splitlines():
        if line.startswith("{"):
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if len(documents) != 1:
        raise ValidationError(f"expected one {workload_kind} JSON result")
    measurements = {
        key: value
        for key, value in documents[0].items()
        if isinstance(key, str) and isinstance(value, dict)
    }
    if not measurements:
        raise ValidationError(f"expected {workload_kind} measurement rows")
    return measurements


def _json_medians(log_text: str, workload_kind: str) -> dict[str, float]:
    return {
        key: statistics.median(values)
        for key, values in _json_timing_samples(log_text, workload_kind).items()
    }


def _sharktank_medians(log_text: str) -> dict[str, float]:
    return _json_medians(log_text, "Sharktank")


def _gtest_timing_samples(log_texts: list[str]) -> dict[str, list[float]]:
    pattern = re.compile(r"\[==========\].*\(([0-9]+) ms total\)")
    values = []
    for log_text in log_texts:
        matches = pattern.findall(log_text)
        if not matches:
            raise ValidationError("missing GTest total latency")
        values.append(float(matches[-1]))
    return {"process": values}


def _gtest_median(log_texts: list[str]) -> dict[str, float]:
    return {
        mode: statistics.median(values)
        for mode, values in _gtest_timing_samples(log_texts).items()
    }


def _discard_first_sample_per_process(
    per_run: list[dict[str, list[float]]],
) -> list[dict[str, list[float]]]:
    discarded = []
    for item in per_run:
        if any(len(values) < 2 for values in item.values()):
            raise ValidationError(
                "cannot discard one warmup sample from a process with fewer than "
                "two timing samples"
            )
        discarded.append({key: values[1:] for key, values in item.items()})
    return discarded


def _nonnegative_float(fields: dict[str, str], name: str) -> float | None:
    try:
        value = float(fields[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


def _empirical_structural_metrics(log_text: str) -> dict[str, object]:
    readers: dict[int, dict[str, object]] = {}
    process_memory: dict[str, int] = {}
    reasons = []

    def parse(line: str, marker: str, context: str) -> dict[str, str] | None:
        return _parse_log_fields(line, marker, context, reasons)

    for line in log_text.splitlines():
        fields = None
        if "ConSan waitcheck timing reader=" in line:
            fields = parse(line, "ConSan waitcheck timing ", "waitcheck timing")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                elapsed = _nonnegative_float(fields, "elapsed_ms")
                if reader is None or elapsed is None:
                    reasons.append("malformed waitcheck timing")
                else:
                    readers.setdefault(reader, {})["waitcheck_ms"] = elapsed
        elif "ConSan MOI inventory end reader=" in line:
            fields = parse(line, "ConSan MOI inventory end ", "MOI inventory timing")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                elapsed = _nonnegative_float(fields, "elapsed_ms")
                if reader is None or elapsed is None:
                    reasons.append("malformed MOI inventory timing")
                else:
                    readers.setdefault(reader, {})["inventory_ms"] = elapsed
        elif "ConSan patch begin reader=" in line:
            fields = parse(line, "ConSan patch begin ", "patch begin")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                byte_count = _unsigned(fields, "bytes")
                if reader is None or byte_count is None:
                    reasons.append("malformed patch begin")
                else:
                    readers.setdefault(reader, {})["original_bytes"] = byte_count
        elif "ConSan patch end reader=" in line:
            fields = parse(line, "ConSan patch end ", "patch end")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                elapsed = _nonnegative_float(fields, "patch_ms")
                patches = _unsigned(fields, "patches")
                if reader is None or elapsed is None or patches is None:
                    reasons.append("patch end lacks empirical timing fields")
                else:
                    record = readers.setdefault(reader, {})
                    record["patch_ms"] = elapsed
                    record["patches"] = patches
                    record["modified"] = _boolean(fields, "modified")
                    record["outcome"] = fields.get("outcome")
        elif "ConSan replacement reader=" in line:
            fields = parse(line, "ConSan replacement ", "replacement image")
            if fields is not None:
                reader = _unsigned(fields, "original_reader")
                byte_count = _unsigned(fields, "bytes")
                if reader is None or byte_count is None:
                    reasons.append("malformed replacement image")
                else:
                    readers.setdefault(reader, {})["patched_bytes"] = byte_count
        elif "ConSan MOI resources reader=" in line:
            fields = parse(line, "ConSan MOI resources ", "MOI resources")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                if reader is None:
                    reasons.append("malformed MOI resources")
                    continue
                resource_names = (
                    "explicit",
                    "dead",
                    "descriptor_growth",
                    "spill",
                    "unsupported",
                    "planned_spill_slot_bytes",
                    "emitted_spill_patches",
                    "emitted_spill_slot_bytes",
                    "alternative_attempts",
                    "alternative_selected",
                    "alternative_rejected",
                    "alternative_superseded",
                    "alternative_contributed",
                    "alternative_vetoed",
                )
                resources = {name: _unsigned(fields, name) for name in resource_names}
                if any(value is None for value in resources.values()):
                    reasons.append("malformed MOI resources")
                else:
                    readers.setdefault(reader, {})["resources"] = resources
        elif "ConSan MOI report memory required_bytes=" in line:
            fields = parse(line, "ConSan MOI report memory ", "report memory")
            if fields is not None:
                for source, destination in (
                    ("required_bytes", "report_required_bytes"),
                    ("allocated_bytes", "report_allocated_bytes"),
                    ("peak_live_bytes", "report_peak_live_bytes"),
                    ("allocation_failures", "report_allocation_failures"),
                    ("capacity_failures", "report_capacity_failures"),
                    ("cleanup_failures", "report_cleanup_failures"),
                ):
                    value = _unsigned(fields, source)
                    if value is None:
                        reasons.append(f"malformed report memory field {source}")
                    else:
                        process_memory[destination] = value
        elif "ConSan transform admission memory live_bytes=" in line:
            fields = parse(
                line,
                "ConSan transform admission memory ",
                "transform admission memory",
            )
            if fields is not None:
                value = _unsigned(fields, "peak_reserved_bytes")
                if value is None:
                    reasons.append("malformed transform admission memory")
                else:
                    process_memory["transform_peak_reserved_bytes"] = value
        elif "ConSan patched-image memory live_bytes=" in line:
            fields = parse(line, "ConSan patched-image memory ", "patched image memory")
            if fields is not None:
                value = _unsigned(fields, "peak_image_bytes")
                if value is None:
                    reasons.append("malformed patched image memory")
                else:
                    process_memory["patched_image_peak_bytes"] = value
        elif "ConSan patched-image growth memory live_bytes=" in line:
            fields = parse(
                line,
                "ConSan patched-image growth memory ",
                "patched image growth memory",
            )
            if fields is not None:
                value = _unsigned(fields, "peak_growth_bytes")
                if value is None:
                    reasons.append("malformed patched image growth memory")
                else:
                    process_memory["patched_image_peak_growth_bytes"] = value

    code_objects = []
    for reader, record in sorted(readers.items()):
        original = record.get("original_bytes")
        patched = record.get("patched_bytes", original)
        record["reader"] = reader
        record["patched_bytes"] = patched
        if isinstance(original, int) and isinstance(patched, int):
            record["growth_bytes"] = patched - original
            record["growth_ratio"] = patched / original if original else None
        code_objects.append(record)
    patched = [
        record["patch_ms"]
        for record in code_objects
        if isinstance(record.get("patch_ms"), float)
    ]
    return {
        "accepted": not reasons and bool(patched),
        "reasons": reasons,
        "code_objects": code_objects,
        "total_patch_ms": sum(patched),
        "process_memory": process_memory,
    }


def _empirical_structural_totals(result: dict) -> dict[str, float]:
    if result.get("accepted") is not True:
        raise ValidationError("empirical row was rejected")
    runs = result.get("structural_metrics_runs")
    if not isinstance(runs, list) or not runs:
        raise ValidationError("empirical row has no structural metrics")
    if any(run.get("accepted") is not True for run in runs):
        raise ValidationError("empirical row has rejected structural metrics")
    totals = {"patch_ms": 0.0, "waitcheck_ms": 0.0, "inventory_ms": 0.0}
    for run in runs:
        totals["patch_ms"] += float(run["total_patch_ms"])
        for record in run["code_objects"]:
            for source, destination in (
                ("waitcheck_ms", "waitcheck_ms"),
                ("inventory_ms", "inventory_ms"),
            ):
                value = record.get(source)
                if isinstance(value, (int, float)):
                    totals[destination] += float(value)
    return totals


def _parse_amdgpu_kernel_metadata(text: str) -> dict[str, object]:
    integer_fields = {
        "group_segment_fixed_size",
        "private_segment_fixed_size",
        "sgpr_count",
        "sgpr_spill_count",
        "vgpr_count",
        "vgpr_spill_count",
        "agpr_count",
        "accum_offset",
        "wavefront_size",
        "max_flat_workgroup_size",
    }
    kernels = []
    current: dict[str, object] | None = None
    for line in text.splitlines():
        start = re.match(r"^  - \.([a-z_]+):\s*(.*?)\s*$", line)
        if start is not None:
            if current:
                kernels.append(current)
            current = {}
            name, value = start.groups()
            if name in integer_fields and value:
                try:
                    current[name] = int(value, 0)
                except ValueError:
                    current[name] = None
        if current is None:
            continue
        match = re.match(r"^    \.([a-z_]+):\s*(.*?)\s*$", line)
        if match is None:
            continue
        name, value = match.groups()
        if name == "name":
            current[name] = value
        elif name in integer_fields:
            try:
                current[name] = int(value, 0)
            except ValueError:
                current[name] = None
    if current:
        kernels.append(current)
    return {
        "kernels": kernels,
        "kernel_count": len(kernels),
        "fields": sorted(
            {name for kernel in kernels for name in kernel if name != "name"}
        ),
    }


def _llvm_readelf() -> Path | None:
    configured = os.environ.get(LLVM_READELF_ENV)
    if configured:
        return Path(os.path.abspath(Path(configured).expanduser()))
    discovered = shutil.which("llvm-readelf")
    return Path(discovered) if discovered else None


def _amdgpu_kernel_metadata(path: Path) -> dict[str, object]:
    tool = _llvm_readelf()
    if tool is None:
        return {
            "accepted": False,
            "reason": "llvm-readelf is unavailable",
            "tool": None,
        }
    try:
        completed = subprocess.run(
            [str(tool), "--notes", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"accepted": False, "reason": str(error), "tool": str(tool)}
    if completed.returncode != 0:
        return {
            "accepted": False,
            "reason": completed.stderr.strip() or "llvm-readelf failed",
            "tool": str(tool),
        }
    metadata = _parse_amdgpu_kernel_metadata(completed.stdout)
    return {
        "accepted": metadata["kernel_count"] > 0,
        "reason": None if metadata["kernel_count"] > 0 else "no AMDGPU kernels found",
        "tool": str(tool),
        **metadata,
    }


def _retained_code_object_inventory(row_dir: Path) -> dict[str, object]:
    records: dict[tuple[str, str], dict[str, object]] = {}
    for path in sorted(row_dir.glob("code-objects-*/*.hsaco")):
        match = _COVERAGE_DUMP_NAME.fullmatch(path.name)
        if match is None:
            continue
        key = (match.group("dump_id"), match.group("reader"))
        record = records.setdefault(
            key,
            {"dump_id": key[0], "reader": int(key[1])},
        )
        record[match.group("kind")] = {
            "path": _retained_relative_path(row_dir, path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "metadata": _amdgpu_kernel_metadata(path),
        }
    pairs = []
    for record in records.values():
        original = record.get("original")
        patched = record.get("patched")
        if isinstance(original, dict) and isinstance(patched, dict):
            record["growth_bytes"] = patched["size"] - original["size"]
            record["growth_ratio"] = (
                patched["size"] / original["size"] if original["size"] else None
            )
            original_kernels = {
                kernel.get("name"): kernel
                for kernel in original["metadata"].get("kernels", [])
                if isinstance(kernel.get("name"), str)
            }
            patched_kernels = {
                kernel.get("name"): kernel
                for kernel in patched["metadata"].get("kernels", [])
                if isinstance(kernel.get("name"), str)
            }
            original_names = sorted(original_kernels)
            patched_names = sorted(patched_kernels)
            resource_fields = (
                "group_segment_fixed_size",
                "private_segment_fixed_size",
                "sgpr_count",
                "sgpr_spill_count",
                "vgpr_count",
                "vgpr_spill_count",
                "agpr_count",
            )
            record["kernel_metadata_delta"] = {
                "original_names": original_names,
                "patched_names": patched_names,
                "name_sets_match": original_names == patched_names,
                "kernels": {
                    name: {
                        field: patched_kernels[name][field]
                        - original_kernels[name][field]
                        for field in resource_fields
                        if isinstance(original_kernels[name].get(field), int)
                        and isinstance(patched_kernels[name].get(field), int)
                    }
                    for name in sorted(set(original_kernels) & set(patched_kernels))
                },
            }
        pairs.append(record)
    return {
        "pairs": pairs,
        "complete_pairs": sum("growth_bytes" in row for row in pairs),
        "metadata_complete_pairs": sum(
            "growth_bytes" in row
            and row["original"]["metadata"]["accepted"]
            and row["patched"]["metadata"]["accepted"]
            for row in pairs
        ),
    }


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


def _launcher_argument(value: str) -> list[str]:
    try:
        return _launcher_from_json(value)
    except ValidationError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _with_launcher(launcher: list[str], command: list[str]) -> list[str]:
    return [*launcher, *command]


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


_COVERAGE_DUMP_NAME = re.compile(
    r"rj-dbi-(?P<dump_id>[0-9]{6,})-reader-(?P<reader>[0-9]+)-"
    r"(?P<kind>original|patched)\.hsaco"
)


def _retained_relative_path(row_dir: Path, path: Path) -> str:
    row = row_dir.resolve()
    workload_root = row.parents[1]
    resolved = path.resolve()
    if not resolved.is_relative_to(workload_root):
        raise ValidationError(
            f"retained coverage-output path escapes workload artifacts: {resolved}"
        )
    return os.path.relpath(resolved, row)


def _retained_file_record(row_dir: Path, path: Path) -> dict:
    if not path.is_file():
        raise ValidationError(f"missing retained coverage-output file: {path}")
    return {
        "path": _retained_relative_path(row_dir, path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _fnv1a64_file(path: Path) -> str:
    # The runtime reports FNV-1a identities for loaded code objects. SHA-256
    # remains the integrity hash; this value only joins a retained original to
    # the diagnostic contract emitted by the runtime.
    value = 14695981039346656037
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            for byte in chunk:
                value ^= byte
                value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{value:016x}"


def _coverage_dump_inventory(row_dir: Path, dump_directories: list[Path]) -> list[dict]:
    records = []
    identities = set()
    for run_index, directory in enumerate(dump_directories):
        if not directory.is_dir():
            raise ValidationError(
                f"missing retained coverage-output code-object directory: {directory}"
            )
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                raise ValidationError(
                    f"unexpected entry in coverage-output code-object directory: {path}"
                )
            match = _COVERAGE_DUMP_NAME.fullmatch(path.name)
            if match is None:
                raise ValidationError(
                    f"unexpected retained coverage-output code-object name: {path.name}"
                )
            identity = (
                run_index,
                match.group("dump_id"),
                match.group("reader"),
                match.group("kind"),
            )
            if identity in identities:
                raise ValidationError(
                    f"duplicate retained coverage-output code object: {path.name}"
                )
            identities.add(identity)
            file_record = {
                **_retained_file_record(row_dir, path),
                "run": run_index,
                "dump_id": match.group("dump_id"),
                "reader": match.group("reader"),
                "kind": match.group("kind"),
            }
            if match.group("kind") == "original":
                file_record["content_fingerprint"] = _fnv1a64_file(path)
            records.append(file_record)
    return records


def _coverage_output_artifact_payload(
    row_dir: Path,
    log_paths: list[Path],
    provenance_path: Path,
) -> dict:
    dump_directories = [
        row_dir / f"code-objects-{index}" for index in range(len(log_paths))
    ]
    dump_records = _coverage_dump_inventory(row_dir, dump_directories)
    return {
        "schema_version": COVERAGE_OUTPUT_ARTIFACT_SCHEMA_VERSION,
        "command_logs": [_retained_file_record(row_dir, path) for path in log_paths],
        "workload_provenance": _retained_file_record(row_dir, provenance_path),
        "code_objects": {
            "directories": [
                _retained_relative_path(row_dir, path) for path in dump_directories
            ],
            "files": dump_records,
        },
    }


def _coverage_output_contract_from_document(value: object) -> CoverageOutputContract:
    if not isinstance(value, dict):
        raise ValidationError("coverage-output provenance has no diagnostic contract")
    required = {
        "profile",
        "diagnostics",
        "max_diagnostics",
        "instruction_groups",
        "code_object_fingerprint",
        "tracking_issue",
        "withhold_fault_qualification",
        "fault_qualification_withheld_reason",
    }
    if set(value) != required:
        raise ValidationError(
            "coverage-output provenance has a malformed diagnostic contract"
        )
    diagnostics = value["diagnostics"]
    groups = value["instruction_groups"]
    if (
        not isinstance(value["profile"], str)
        or not isinstance(diagnostics, list)
        or not diagnostics
        or any(not isinstance(item, str) for item in diagnostics)
        or type(value["max_diagnostics"]) is not int
        or not isinstance(groups, list)
        or not groups
        or any(
            not isinstance(group, list)
            or not group
            or any(type(instruction) is not int for instruction in group)
            for group in groups
        )
        or not isinstance(value["code_object_fingerprint"], str)
        or not isinstance(value["tracking_issue"], str)
        or type(value["withhold_fault_qualification"]) is not bool
        or not isinstance(value["fault_qualification_withheld_reason"], str)
    ):
        raise ValidationError(
            "coverage-output provenance has a malformed diagnostic contract"
        )
    contract = CoverageOutputContract(
        profile=value["profile"],
        diagnostics=tuple(diagnostics),
        max_diagnostics=value["max_diagnostics"],
        instruction_groups=tuple(tuple(group) for group in groups),
        code_object_fingerprint=value["code_object_fingerprint"],
        tracking_issue=value["tracking_issue"],
        withhold_fault_qualification=value["withhold_fault_qualification"],
        fault_qualification_withheld_reason=value[
            "fault_qualification_withheld_reason"
        ],
    )
    try:
        _validate_coverage_output_contract(
            replace(
                WORKLOADS[0], id="retained-artifact", coverage_output_contract=contract
            )
        )
    except RuntimeError as error:
        raise ValidationError(str(error)) from error
    return contract


def _coverage_output_contract_from_provenance(
    provenance: object, result: dict
) -> CoverageOutputContract:
    if not isinstance(provenance, dict):
        raise ValidationError("coverage-output provenance must be an object")
    if provenance.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError("coverage-output provenance schema is unsupported")
    manifest = provenance.get("manifest")
    if not isinstance(manifest, dict):
        raise ValidationError("coverage-output provenance has no executable manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get(
        "target"
    ) != result.get("target"):
        raise ValidationError(
            "coverage-output provenance manifest identity does not match the result"
        )
    workloads = manifest.get("workloads")
    if not isinstance(workloads, list):
        raise ValidationError("coverage-output provenance manifest has no workloads")
    matches = [
        workload
        for workload in workloads
        if isinstance(workload, dict) and workload.get("id") == result.get("workload")
    ]
    if len(matches) != 1:
        raise ValidationError(
            "coverage-output provenance does not identify exactly one workload"
        )
    contract = _coverage_output_contract_from_document(
        matches[0].get("coverage_output_contract")
    )
    if result.get("profile") != contract.profile:
        raise ValidationError(
            "coverage-output result profile does not match workload provenance"
        )
    return contract


def _row_runtime_acceptance(
    returncodes: object,
    gtest_counts: object,
    coverage_runs: object,
    profile: str | None,
    expected_runs: int,
) -> bool:
    returncodes_valid = (
        isinstance(returncodes, list)
        and len(returncodes) == expected_runs
        and all(type(code) is int and code == 0 for code in returncodes)
    )
    gtest_valid = gtest_counts is None or (
        isinstance(gtest_counts, list)
        and len(gtest_counts) == expected_runs
        and all(type(count) is int and count > 0 for count in gtest_counts)
    )
    coverage_valid = profile is None or (
        isinstance(coverage_runs, list)
        and len(coverage_runs) == expected_runs
        and bool(coverage_runs)
        and all(
            isinstance(item, dict) and item.get("accepted") is True
            for item in coverage_runs
        )
    )
    return returncodes_valid and gtest_valid and coverage_valid


def _source_identity_is_complete(source: object) -> bool:
    if not isinstance(source, dict) or not isinstance(source.get("root"), str):
        return False
    head = source.get("head")
    dirty = source.get("dirty")
    return (head is None and dirty is None) or (
        re.fullmatch(r"[0-9a-f]{40,64}", str(head)) is not None and type(dirty) is bool
    )


def _verify_retained_file_record(
    row_dir: Path, record: object, expected_path: Path, label: str
) -> list[str]:
    reasons = []
    if not isinstance(record, dict):
        return [f"coverage-output {label} record is malformed"]
    try:
        expected_reference = _retained_relative_path(row_dir, expected_path)
    except ValidationError as error:
        return [str(error)]
    if record.get("path") != expected_reference:
        reasons.append(f"coverage-output {label} path is not relocatable")
    if type(record.get("size")) is not int or record["size"] < 0:
        reasons.append(f"coverage-output {label} size is malformed")
    if re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))) is None:
        reasons.append(f"coverage-output {label} hash is malformed")
    try:
        if not expected_path.is_file():
            reasons.append(f"coverage-output {label} is missing")
        else:
            if record.get("size") != expected_path.stat().st_size:
                reasons.append(f"coverage-output {label} size does not match storage")
            if record.get("sha256") != sha256_file(expected_path):
                reasons.append(f"coverage-output {label} hash does not match storage")
    except OSError as error:
        reasons.append(f"cannot verify coverage-output {label}: {error}")
    return reasons


def _coverage_output_artifact_reasons(
    row_dir: Path,
    result: dict,
    log_paths: list[Path],
    provenance_path: Path,
    replayed: list[dict | None],
    contract: CoverageOutputContract | None,
) -> list[str]:
    reasons = []
    artifacts = result.get("retained_artifacts")
    if not isinstance(artifacts, dict):
        return ["coverage-output result predates or lacks retained artifact inventory"]
    if artifacts.get("schema_version") != COVERAGE_OUTPUT_ARTIFACT_SCHEMA_VERSION:
        reasons.append("coverage-output retained artifact schema is unsupported")
    collection_error = artifacts.get("collection_error")
    if collection_error is not None:
        if isinstance(collection_error, str):
            reasons.append(
                f"coverage-output artifact collection failed: {collection_error}"
            )
        else:
            reasons.append("coverage-output artifact collection error is malformed")

    command_logs = artifacts.get("command_logs")
    if not isinstance(command_logs, list) or len(command_logs) != len(log_paths):
        reasons.append("coverage-output retained command-log inventory is malformed")
    else:
        for index, (record, path) in enumerate(zip(command_logs, log_paths)):
            reasons.extend(
                _verify_retained_file_record(row_dir, record, path, f"run-{index} log")
            )
    reasons.extend(
        _verify_retained_file_record(
            row_dir,
            artifacts.get("workload_provenance"),
            provenance_path,
            "workload provenance",
        )
    )

    code_objects = artifacts.get("code_objects")
    if not isinstance(code_objects, dict):
        reasons.append("coverage-output retained code-object inventory is malformed")
        return reasons
    dump_directories = [
        row_dir / f"code-objects-{index}" for index in range(len(log_paths))
    ]
    expected_directories = [
        _retained_relative_path(row_dir, path) for path in dump_directories
    ]
    if code_objects.get("directories") != expected_directories:
        reasons.append("coverage-output code-object directory inventory is malformed")
    records = code_objects.get("files")
    if not isinstance(records, list):
        reasons.append("coverage-output retained code-object files are malformed")
        return reasons

    actual_paths = set()
    for run_index, directory in enumerate(dump_directories):
        try:
            entries = sorted(directory.iterdir()) if directory.is_dir() else []
        except OSError as error:
            reasons.append(
                f"cannot inspect coverage-output code-object run {run_index}: {error}"
            )
            continue
        if not directory.is_dir():
            reasons.append(
                f"coverage-output code-object directory is missing for run {run_index}"
            )
            continue
        files = [path for path in entries if path.is_file()]
        if len(files) != len(entries):
            reasons.append(
                f"coverage-output code-object directory has non-files for run {run_index}"
            )
        if not files:
            reasons.append(
                f"coverage-output hook retained no code-object dumps for run {run_index}"
            )
        actual_paths.update(_retained_relative_path(row_dir, path) for path in files)

    recorded_paths = set()
    valid_records = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            reasons.append(f"coverage-output code-object record {index} is malformed")
            continue
        reference = record["path"]
        if reference in recorded_paths:
            reasons.append(
                f"coverage-output code-object record is duplicated: {reference}"
            )
            continue
        recorded_paths.add(reference)
        path = (row_dir / reference).resolve()
        workload_root = row_dir.resolve().parents[1]
        if not path.is_relative_to(workload_root):
            reasons.append(
                f"coverage-output code-object path escapes artifacts: {reference}"
            )
            continue
        match = _COVERAGE_DUMP_NAME.fullmatch(path.name)
        run = record.get("run")
        if (
            match is None
            or type(run) is not int
            or run < 0
            or run >= len(dump_directories)
            or path.parent != dump_directories[run].resolve()
            or record.get("dump_id") != (match.group("dump_id") if match else None)
            or record.get("reader") != (match.group("reader") if match else None)
            or record.get("kind") != (match.group("kind") if match else None)
        ):
            reasons.append(
                f"coverage-output code-object identity is malformed: {reference}"
            )
            continue
        reasons.extend(
            _verify_retained_file_record(
                row_dir, record, path, f"code object {reference}"
            )
        )
        fingerprint = record.get("content_fingerprint")
        if record["kind"] == "original":
            if re.fullmatch(r"fnv1a64:[0-9a-f]{16}", str(fingerprint)) is None:
                reasons.append(
                    f"coverage-output original fingerprint is malformed: {reference}"
                )
        elif fingerprint is not None:
            reasons.append(
                f"coverage-output patched object has an unexpected fingerprint: {reference}"
            )
        if path.is_file():
            valid_records.append(record)
    if recorded_paths != actual_paths:
        reasons.append(
            "coverage-output code-object file inventory does not match storage"
        )

    pairs: dict[tuple[int, str, str], set[str]] = {}
    for record in valid_records:
        key = (record["run"], record["dump_id"], record["reader"])
        pairs.setdefault(key, set()).add(record["kind"])
    for (run, dump_id, reader), kinds in sorted(pairs.items()):
        if kinds != {"original", "patched"}:
            missing = "patched" if kinds == {"original"} else "original"
            reasons.append(
                "coverage-output code-object pair is incomplete: "
                f"run={run}, dump_id={dump_id}, reader={reader}, missing={missing}"
            )

    if contract is not None:
        for run_index, coverage in enumerate(replayed):
            if not isinstance(coverage, dict):
                continue
            diagnostics = coverage.get("diagnostics")
            readers = (
                diagnostics.get("readers") if isinstance(diagnostics, dict) else None
            )
            matching_readers = (
                {
                    str(source.get("reader"))
                    for source in readers.values()
                    if isinstance(readers, dict)
                    and isinstance(source, dict)
                    and type(source.get("reader")) is int
                    and source.get("code_object_fingerprint")
                    == contract.code_object_fingerprint
                }
                if isinstance(readers, dict)
                else set()
            )
            if not matching_readers:
                reasons.append(
                    f"coverage-output run {run_index} has no contracted diagnostic source"
                )
                continue
            for reader in sorted(matching_readers):
                reader_originals = [
                    record
                    for record in valid_records
                    if record["run"] == run_index
                    and record["reader"] == reader
                    and record["kind"] == "original"
                ]
                if not reader_originals:
                    reasons.append(
                        "coverage-output retained no original object for diagnostic "
                        f"source: run={run_index}, reader={reader}"
                    )
                elif not any(
                    record.get("content_fingerprint")
                    == contract.code_object_fingerprint
                    for record in reader_originals
                ):
                    reasons.append(
                        "coverage-output retained objects do not match the contracted "
                        f"fingerprint: run={run_index}, reader={reader}"
                    )
    return reasons


def _derive_coverage_output_verification(
    result_path: Path, result: dict
) -> tuple[dict, bool]:
    reasons = []
    runtime_accepted = False
    row_dir = result_path.resolve().parent
    commands = result.get("commands")
    coverage_runs = result.get("coverage_runs")
    if not isinstance(commands, list) or not commands:
        reasons.append("coverage-output result has no retained commands")
        commands = []
    elif any(
        not isinstance(command, list)
        or not command
        or any(not isinstance(argument, str) for argument in command)
        for command in commands
    ):
        reasons.append("coverage-output retained commands are malformed")
    if not isinstance(coverage_runs, list) or len(coverage_runs) != len(commands):
        reasons.append("coverage-output command and diagnostic-run counts disagree")
        coverage_runs = []
    log_paths = [row_dir / f"run-{index}.log" for index in range(len(commands))]
    provenance_path = row_dir.parents[1] / "provenance.json"

    provenance = None
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        reasons.append(f"cannot read coverage-output workload provenance: {error}")
    if isinstance(provenance, dict):
        if provenance.get("workload") != result.get("workload"):
            reasons.append("coverage-output workload provenance mismatch")
        if provenance.get("target") != result.get("target"):
            reasons.append("coverage-output target provenance mismatch")

    contract = None
    try:
        contract = _coverage_output_contract_from_provenance(provenance, result)
    except ValidationError as error:
        reasons.append(str(error))

    replayed: list[dict | None] = [None] * len(commands)
    if contract is not None and len(coverage_runs) == len(commands):
        canonical_contract = json.loads(json.dumps(asdict(contract)))
        for index, (log_path, recorded) in enumerate(zip(log_paths, coverage_runs)):
            normalized_recorded = (
                json.loads(json.dumps(recorded)) if isinstance(recorded, dict) else None
            )
            diagnostics = (
                normalized_recorded.get("diagnostics")
                if isinstance(normalized_recorded, dict)
                else None
            )
            if not isinstance(diagnostics, dict):
                reasons.append(
                    f"coverage-output run {index} has a malformed diagnostic summary"
                )
                continue
            if diagnostics.get("contract") != canonical_contract:
                reasons.append(
                    f"coverage-output run {index} contract differs from provenance"
                )
            try:
                replay = _coverage_summary(
                    log_path.read_text(encoding="utf-8", errors="replace"),
                    profile=contract.profile,
                    coverage_output_contract=contract,
                )
                replay = json.loads(json.dumps(replay))
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                ValidationError,
                TypeError,
                ValueError,
                AttributeError,
            ) as error:
                reasons.append(
                    f"cannot replay coverage-output diagnostics for run {index}: {error}"
                )
                continue
            replayed[index] = replay
            if replay != normalized_recorded:
                reasons.append(
                    f"coverage-output diagnostic decision does not replay for run {index}"
                )
    if replayed and all(isinstance(item, dict) for item in replayed):
        normalized_coverage = json.loads(json.dumps(result.get("coverage")))
        if normalized_coverage != replayed[-1]:
            reasons.append(
                "coverage-output summary does not match the final command log"
            )

    runtime_accepted = _row_runtime_acceptance(
        result.get("returncodes"),
        result.get("gtest_test_counts"),
        replayed,
        contract.profile if contract is not None else result.get("profile"),
        len(commands),
    )
    if result.get("coverage_acceptance") is not runtime_accepted:
        reasons.append("coverage-output runtime acceptance is not reproducible")

    if isinstance(provenance, dict):
        provenance_files = provenance.get("files")
        result_files = result.get("files")
        provenance_hook = (
            provenance_files.get("hook") if isinstance(provenance_files, dict) else None
        )
        hook = result_files.get("hook") if isinstance(result_files, dict) else None
        hook_matches = (
            isinstance(hook, dict)
            and isinstance(provenance_hook, dict)
            and hook.get("path") == provenance_hook.get("path")
            and hook.get("sha256") == provenance_hook.get("sha256")
        )
        if not hook_matches:
            reasons.append("coverage-output hook identity mismatch")
        elif (
            not isinstance(hook.get("path"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(hook.get("sha256"))) is None
        ):
            reasons.append("coverage-output hook identity is incomplete")
        sources = result.get("sources")
        if sources != provenance.get("sources"):
            reasons.append("coverage-output source revisions mismatch")
        if (
            not isinstance(sources, list)
            or not sources
            or any(not _source_identity_is_complete(source) for source in sources)
        ):
            reasons.append("coverage-output source revisions are incomplete")

    reasons.extend(
        _coverage_output_artifact_reasons(
            row_dir, result, log_paths, provenance_path, replayed, contract
        )
    )
    verification = {
        "schema_version": COVERAGE_OUTPUT_ARTIFACT_SCHEMA_VERSION,
        "accepted": not reasons,
        "reasons": reasons,
    }
    return verification, runtime_accepted


def verify_coverage_output_result(result_path: Path) -> dict:
    result_path = result_path.resolve()
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {
            "schema_version": COVERAGE_OUTPUT_ARTIFACT_SCHEMA_VERSION,
            "accepted": False,
            "reasons": [str(error)],
        }
    if not isinstance(result, dict) or result.get("schema_version") != SCHEMA_VERSION:
        return {
            "schema_version": COVERAGE_OUTPUT_ARTIFACT_SCHEMA_VERSION,
            "accepted": False,
            "reasons": [
                f"coverage-output result must use schema version {SCHEMA_VERSION}"
            ],
        }
    if result.get("phase") != "coverage-output":
        return {
            "schema_version": COVERAGE_OUTPUT_ARTIFACT_SCHEMA_VERSION,
            "accepted": False,
            "reasons": ["result phase is not coverage-output"],
        }
    verification, runtime_accepted = _derive_coverage_output_verification(
        result_path, result
    )
    reasons = list(verification["reasons"])
    if result.get("artifact_verification") != verification:
        reasons.append("stored coverage-output artifact verification is stale")
    expected_row_acceptance = runtime_accepted and verification["accepted"]
    if result.get("accepted") is not expected_row_acceptance:
        reasons.append("coverage-output final acceptance is not reproducible")
    return {
        "schema_version": COVERAGE_OUTPUT_ARTIFACT_SCHEMA_VERSION,
        "accepted": not reasons,
        "reasons": reasons,
    }


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
    *,
    row_dir_override: Path | None = None,
    repetitions_override: int | None = None,
    inner_repetitions_override: int | None = None,
    discard_first_timing_sample: bool = False,
    retain_code_objects: bool = False,
    collect_structural_metrics: bool = False,
) -> dict:
    profile_id = profile or "baseline"
    result_phase = _result_phase(phase, profile, workload)
    row_dir = row_dir_override or (
        artifact_root / workload.id / result_phase / (row_label or profile_id)
    )
    row_dir.mkdir(parents=True, exist_ok=False)
    hook = _hook_path(workspace)
    repetitions = (
        repetitions_override
        if repetitions_override is not None
        else _outer_repetitions(target, phase, workload)
    )
    if repetitions <= 0:
        raise ValidationError("profile repetitions must be positive")
    if inner_repetitions_override is not None and inner_repetitions_override <= 0:
        raise ValidationError("inner repetitions must be positive")
    coverage_contract = _coverage_contract_for_profile(workload, profile)
    logs = []
    commands = []
    recorded_environment = None
    returncodes = []
    elapsed_seconds = []
    qwen_json_paths = []
    for index in range(repetitions):
        benchmark_path = row_dir / f"benchmark-{index}.json"
        command = _workload_command(
            workspace,
            target,
            workload,
            phase,
            benchmark_path,
            inner_repetitions_override,
        )
        command = _with_launcher(launcher or [], command)
        log_path = row_dir / f"run-{index}.log"
        environment = _run_environment(profile, workload, hook, target, phase)
        if profile is not None and (
            result_phase == "coverage-output" or retain_code_objects
        ):
            dump_dir = row_dir / f"code-objects-{index}"
            dump_dir.mkdir()
            environment["RJ_CONSAN_DUMP_DIR"] = str(dump_dir.resolve())
        returncode, elapsed, output = _run_process(
            command, environment, log_path, timeout
        )
        commands.append(command)
        recorded_environment = _controlled_environment(environment)
        # The retained code-object inventory records relocatable per-run dump
        # directories. Do not also publish the execution machine's absolute
        # directory through the compatibility environment summary.
        recorded_environment.pop("RJ_CONSAN_DUMP_DIR", None)
        returncodes.append(returncode)
        elapsed_seconds.append(elapsed)
        logs.append(output)
        if workload.kind == "qwen" and phase == "overhead":
            qwen_json_paths.append(benchmark_path)

    timing = None
    timing_samples = None
    measurement_runs = None
    if phase == "overhead" and all(code == 0 for code in returncodes):
        if workload.kind == "qwen":
            per_run_samples = [
                {"dispatch": _benchmark_samples(path)} for path in qwen_json_paths
            ]
            if discard_first_timing_sample:
                per_run_samples = _discard_first_sample_per_process(per_run_samples)
            timing_samples = {
                "dispatch": [
                    value for item in per_run_samples for value in item["dispatch"]
                ]
            }
        elif workload.kind in {
            "sharktank",
            "pytorch",
            "tensile",
            "llama",
            "rdna4-matmul",
        }:
            per_run = [
                _json_timing_samples(log, workload.kind.capitalize()) for log in logs
            ]
            key_sets = [set(item) for item in per_run]
            if any(keys != key_sets[0] for keys in key_sets[1:]):
                raise ValidationError(
                    f"{workload.id} timing metric schema differs across processes"
                )
            keys = key_sets[0]
            if discard_first_timing_sample:
                per_run = _discard_first_sample_per_process(per_run)
            timing_samples = {
                key: [value for item in per_run for value in item[key]]
                for key in sorted(keys)
            }
            if workload.kind == "rdna4-matmul":
                measurement_runs = [
                    _json_measurements(log, "Rdna4-matmul") for log in logs
                ]
        else:
            timing_samples = _gtest_timing_samples(logs)
        timing = {
            key: statistics.median(values) for key, values in timing_samples.items()
        }

    coverage = None
    coverage_runs = None
    if profile is not None and logs:
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
    runtime_accepted = _row_runtime_acceptance(
        returncodes,
        gtest_test_counts,
        coverage_runs,
        profile,
        len(commands),
    )
    provenance_path = _workload_provenance_path(artifact_root, workload)
    structural_metrics_runs = (
        [_empirical_structural_metrics(log) for log in logs]
        if profile is not None and collect_structural_metrics
        else None
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "workload": workload.id,
        "profile": profile_id,
        "phase": result_phase,
        "target": target,
        "commands": commands,
        "environment": recorded_environment,
        "returncodes": returncodes,
        "elapsed_seconds": elapsed_seconds,
        "timeout_seconds": timeout,
        "repetition_policy": {
            "empirical_row_schema_version": 1,
            "outer_processes": repetitions,
            "inner_repetitions_override": inner_repetitions_override,
            "discarded_first_timing_sample": discard_first_timing_sample,
            "discarded_timing_samples_per_process": int(discard_first_timing_sample),
            "retained_code_objects": retain_code_objects,
            "collected_structural_metrics": collect_structural_metrics,
        },
        "timing_median_ms": timing,
        "timing_samples_ms": timing_samples,
        "timing_statistic": (
            "median-of-raw-google-benchmark-iterations-single-identity"
            if workload.kind == "qwen" and timing_samples is not None
            else (
                "median-of-retained-raw-samples" if timing_samples is not None else None
            )
        ),
        "measurement_runs": measurement_runs,
        "structural_metrics_runs": structural_metrics_runs,
        "coverage": coverage,
        "coverage_runs": coverage_runs,
        "gtest_test_counts": gtest_test_counts,
        "accepted": runtime_accepted if result_phase != "coverage-output" else False,
        "files": {
            "hook": {
                "path": str(hook),
                "sha256": sha256_file(hook),
            }
        },
        "sources": _source_identities(workspace, workload),
        "provenance": str(provenance_path),
    }
    if profile is not None and retain_code_objects:
        result["retained_code_objects"] = _retained_code_object_inventory(row_dir)
    if result_phase == "coverage-output":
        assert coverage_runs is not None
        result["coverage_acceptance"] = runtime_accepted
        try:
            result["retained_artifacts"] = _coverage_output_artifact_payload(
                row_dir,
                [row_dir / f"run-{index}.log" for index in range(repetitions)],
                provenance_path,
            )
        except (OSError, UnicodeError, ValidationError) as error:
            result["retained_artifacts"] = {
                "schema_version": COVERAGE_OUTPUT_ARTIFACT_SCHEMA_VERSION,
                "collection_error": str(error),
            }
    result_path = row_dir / "result.json"
    if result_phase == "coverage-output":
        verification, reproduced_runtime_acceptance = (
            _derive_coverage_output_verification(result_path, result)
        )
        result["artifact_verification"] = verification
        result["accepted"] = reproduced_runtime_acceptance and verification["accepted"]
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


def _linear_quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValidationError("cannot summarize an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ValidationError("quantile probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _bootstrap_median_interval(
    values: list[float], *, resamples: int, seed: int
) -> dict[str, float]:
    if not values:
        raise ValidationError("cannot bootstrap an empty sample")
    if resamples <= 0:
        raise ValidationError("bootstrap resamples must be positive")
    if len(values) == 1:
        lower = upper = float(values[0])
    else:
        generator = random.Random(seed)
        medians = [
            statistics.median(generator.choices(values, k=len(values)))
            for _ in range(resamples)
        ]
        lower = _linear_quantile(medians, 0.025)
        upper = _linear_quantile(medians, 0.975)
    return {"lower": lower, "upper": upper}


def _sample_summary(
    values: list[float], *, bootstrap_resamples: int, bootstrap_seed: int
) -> dict[str, object]:
    if not values:
        raise ValidationError("cannot summarize an empty sample")
    q1 = _linear_quantile(values, 0.25)
    q3 = _linear_quantile(values, 0.75)
    return {
        "count": len(values),
        "minimum": min(values),
        "q1": q1,
        "median": statistics.median(values),
        "q3": q3,
        "maximum": max(values),
        "iqr": q3 - q1,
        "bootstrap_median_95": _bootstrap_median_interval(
            values,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
    }


def _bootstrap_stream_seed(seed: int, *labels: str) -> int:
    digest = hashlib.sha256("\0".join((str(seed), *labels)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _empirical_row_metrics(
    result: dict,
    *,
    include_process: bool = True,
    include_workload: bool = True,
    prefix: str = "",
) -> dict[str, float]:
    elapsed = result.get("elapsed_seconds")
    if not isinstance(elapsed, list) or not elapsed:
        raise ValidationError("empirical row has no process elapsed samples")
    metrics = {}
    if include_process:
        process_name = f"{prefix}:process" if prefix else "process"
        metrics[process_name] = statistics.median(elapsed) * 1000.0
    timing = result.get("timing_median_ms")
    if include_workload and isinstance(timing, dict):
        metrics.update(
            {
                f"{prefix + ':' if prefix else ''}workload:{mode}": float(value)
                for mode, value in timing.items()
                if isinstance(mode, str)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                and value > 0.0
            }
        )
    return metrics


def _empirical_round_summary(
    round_index: int,
    order: list[str],
    baseline_before: dict,
    profile_results: dict[str, dict],
    baseline_after: dict,
    *,
    baseline_drift_limit: float,
    include_process_metric: bool = True,
    include_workload_metrics: bool = True,
    metric_prefix: str = "",
) -> dict[str, object]:
    rows = [
        baseline_before,
        *(profile_results[profile] for profile in order),
        baseline_after,
    ]
    reasons = []
    for label, result in (
        ("baseline-before", baseline_before),
        *((profile, profile_results[profile]) for profile in order),
        ("baseline-after", baseline_after),
    ):
        if result.get("accepted") is not True:
            returncodes = result.get("returncodes")
            if isinstance(returncodes, list) and 124 in returncodes:
                timeout_seconds = result.get("timeout_seconds")
                suffix = (
                    f" after {timeout_seconds}s"
                    if isinstance(timeout_seconds, int)
                    else ""
                )
                reasons.append(f"{label} row timed out{suffix}")
            else:
                reasons.append(f"{label} row rejected with returncodes={returncodes!r}")

    row_metrics = [
        _empirical_row_metrics(
            result,
            include_process=include_process_metric,
            include_workload=include_workload_metrics,
            prefix=metric_prefix,
        )
        for result in rows
    ]
    metric_schemas = [set(row) for row in row_metrics]
    expected_metrics = metric_schemas[0]
    if not expected_metrics:
        reasons.append("rows have no timing metrics")
    if any(schema != expected_metrics for schema in metric_schemas[1:]):
        reasons.append(
            "row timing metric schemas differ: "
            + "; ".join(
                f"row-{index}={sorted(schema)}"
                for index, schema in enumerate(metric_schemas)
            )
        )
    metrics = {}
    denominator = len(order) + 1
    rows_accepted = not reasons
    for metric in sorted(expected_metrics if not reasons else set()):
        before = row_metrics[0][metric]
        after = row_metrics[-1][metric]
        mean_baseline = statistics.mean((before, after))
        drift = abs(after - before) / mean_baseline if mean_baseline > 0.0 else math.inf
        metric_accepted = rows_accepted and drift <= baseline_drift_limit
        if drift > baseline_drift_limit:
            reasons.append(
                f"{metric} baseline drift {drift:.6f} exceeds "
                f"{baseline_drift_limit:.6f}"
            )
        profiles = {}
        for position, profile in enumerate(order, start=1):
            measured = row_metrics[position][metric]
            fraction = position / denominator
            interpolated = before + fraction * (after - before)
            profiles[profile] = {
                "position": position,
                "timing_ms": measured,
                "interpolated_baseline_ms": interpolated,
                "slowdown": measured / interpolated if interpolated > 0.0 else None,
            }
        metrics[metric] = {
            "accepted": metric_accepted,
            "baseline_before_ms": before,
            "baseline_after_ms": after,
            "baseline_drift_fraction": drift,
            "profiles": profiles,
        }
    metric_acceptance = [metric["accepted"] for metric in metrics.values()]
    return {
        "round": round_index,
        "profile_order": order,
        "rows_accepted": rows_accepted,
        "usable": any(metric_acceptance),
        "fully_accepted": bool(metric_acceptance) and all(metric_acceptance),
        "reasons": reasons,
        "metrics": metrics,
    }


def _combine_empirical_round_summaries(
    round_index: int,
    order: list[str],
    schedules: dict[str, dict[str, object]],
) -> dict[str, object]:
    if not schedules:
        raise ValidationError("empirical round has no timing schedules")
    metrics = {}
    for name, schedule in schedules.items():
        if (
            schedule.get("round") != round_index
            or schedule.get("profile_order") != order
        ):
            raise ValidationError(
                f"empirical {name} schedule does not match its parent round"
            )
        for metric, value in schedule["metrics"].items():
            if metric in metrics:
                raise ValidationError(
                    f"empirical schedules duplicate timing metric {metric}"
                )
            metrics[metric] = value
    reasons = [
        f"{name}: {reason}"
        for name, schedule in schedules.items()
        for reason in schedule["reasons"]
    ]
    metric_acceptance = [metric["accepted"] for metric in metrics.values()]
    return {
        "round": round_index,
        "profile_order": order,
        "rows_accepted": all(
            bool(schedule["rows_accepted"]) for schedule in schedules.values()
        ),
        "usable": any(metric_acceptance),
        "fully_accepted": bool(metric_acceptance) and all(metric_acceptance),
        "reasons": reasons,
        "metrics": metrics,
        "schedules": schedules,
    }


def _empirical_campaign_summary(
    rounds: list[dict[str, object]],
    profiles: tuple[str, ...],
    *,
    required_accepted_rounds: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    require_structural_metrics: bool = False,
) -> dict[str, object]:
    metric_names = sorted(
        {metric for round_result in rounds for metric in round_result["metrics"]}
    )
    profile_summaries = {}
    insufficient = []
    for profile in profiles:
        profile_metrics = {}
        for metric in metric_names:
            samples = [
                round_result["metrics"][metric]["profiles"][profile]
                for round_result in rounds
                if metric in round_result["metrics"]
                and round_result["metrics"][metric]["accepted"]
                and profile in round_result["metrics"][metric]["profiles"]
            ]
            if not samples:
                insufficient.append(
                    f"{profile}/{metric}: 0/{required_accepted_rounds} accepted rounds"
                )
                continue
            seed = _bootstrap_stream_seed(bootstrap_seed, profile, metric, "timing")
            profile_metrics[metric] = {
                "timing_ms": _sample_summary(
                    [sample["timing_ms"] for sample in samples],
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=seed,
                ),
                "paired_baseline_ms": _sample_summary(
                    [sample["interpolated_baseline_ms"] for sample in samples],
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=_bootstrap_stream_seed(
                        bootstrap_seed, profile, metric, "baseline"
                    ),
                ),
                "slowdown": _sample_summary(
                    [sample["slowdown"] for sample in samples],
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=_bootstrap_stream_seed(
                        bootstrap_seed, profile, metric, "slowdown"
                    ),
                ),
            }
            if len(samples) < required_accepted_rounds:
                insufficient.append(
                    f"{profile}/{metric}: {len(samples)}/{required_accepted_rounds} "
                    "accepted rounds"
                )
        structural_samples: dict[str, list[float]] = {}
        for round_result in rounds:
            cold = round_result.get("schedules", {}).get("cold", {})
            profile_structural = cold.get("structural_metrics", {}).get(profile)
            if not isinstance(profile_structural, dict):
                continue
            for name, value in profile_structural.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    structural_samples.setdefault(name, []).append(float(value))
        structural_metrics = {
            name: _sample_summary(
                values,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=_bootstrap_stream_seed(
                    bootstrap_seed, profile, name, "structural"
                ),
            )
            for name, values in sorted(structural_samples.items())
        }
        for name, values in sorted(structural_samples.items()):
            if require_structural_metrics and len(values) < required_accepted_rounds:
                insufficient.append(
                    f"{profile}/structural:{name}: {len(values)}/"
                    f"{required_accepted_rounds} accepted rounds"
                )
        if require_structural_metrics and not structural_metrics:
            insufficient.append(f"{profile}: no structural metrics")
        profile_summaries[profile] = {
            "metrics": profile_metrics,
            "structural_metrics": structural_metrics,
        }
    if not metric_names:
        insufficient.append("campaign has no timing metrics")
    return {
        "schema_version": EMPIRICAL_CAMPAIGN_SCHEMA_VERSION,
        "required_accepted_rounds": required_accepted_rounds,
        "attempted_rounds": len(rounds),
        "usable_rounds": sum(bool(round_result["usable"]) for round_result in rounds),
        "fully_accepted_rounds": sum(
            bool(round_result["fully_accepted"]) for round_result in rounds
        ),
        "accepted": not insufficient,
        "reasons": insufficient,
        "profiles": profile_summaries,
    }


def _inventory_records(
    log_text: str, family: str | None = None
) -> dict[str, list[str]]:
    event_kind = FAULT_FAMILY_SITE_KINDS.get(family) if family else None
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
    site_kind = FAULT_FAMILY_SITE_KINDS[family]
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
    command = _with_launcher(args.launcher, command)
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
            "RJ_CONSAN_FAULT_LDS_WRONG_ADDRESS",
        }
    ]
    if len(mutations) != 1:
        raise ValidationError("fault spec must enable exactly one mutation family")
    if "RJ_CONSAN_FAULT_SITE_IDENTITY" not in environment:
        raise ValidationError("fault spec must select an exact site identity")
    if any("REPLACE_FROM_INVENTORY" in value for value in environment.values()):
        raise ValidationError("fault spec still contains an inventory placeholder")
    site_provenance = fault.get("site_provenance")
    if site_provenance is not None:
        required_keys = {"corpus_commit", "executable", "inventory_run"}
        if (
            not isinstance(site_provenance, dict)
            or set(site_provenance) != required_keys
            or re.fullmatch(
                r"[0-9a-f]{40}", str(site_provenance.get("corpus_commit", ""))
            )
            is None
            or not isinstance(site_provenance.get("executable"), str)
            or not site_provenance["executable"]
            or Path(site_provenance["executable"]).name != site_provenance["executable"]
            or not isinstance(site_provenance.get("inventory_run"), str)
            or not site_provenance["inventory_run"]
        ):
            raise ValidationError("fault site_provenance is invalid")
    reach_witness = fault.get("reach_witness")
    if reach_witness is not None:
        if (
            not isinstance(reach_witness, dict)
            or set(reach_witness) != {"kind", "evidence"}
            or reach_witness.get("kind")
            not in {"reviewed-unconditional-final-isa", "runtime-access-counter"}
            or not isinstance(reach_witness.get("evidence"), str)
            or not reach_witness["evidence"].strip()
        ):
            raise ValidationError("fault reach_witness is invalid")
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


def _wilson_detection_interval(detections: int, trials: int) -> dict[str, float]:
    if (
        type(detections) is not int
        or type(trials) is not int
        or trials <= 0
        or detections < 0
        or detections > trials
    ):
        raise ValidationError("invalid detection count for Wilson interval")
    z = 1.959963984540054
    proportion = detections / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return {
        "confidence": 0.95,
        "lower": 0.0 if detections == 0 else max(0.0, center - radius),
        "upper": 1.0 if detections == trials else min(1.0, center + radius),
    }


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
    reservation_status, reservation_reasons = fault_reservation_qualification(
        mutation.get("reservation")
    )
    if reservation_status != FAULT_RESERVATION_QUALIFIED:
        reasons.extend(reservation_reasons)
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


def _fault_admission_and_reach(
    result: dict, reach_witness: dict | None
) -> tuple[bool, bool, str | None, list[str]]:
    reasons = []
    mutation = result.get("mutation", {})
    if (
        mutation.get("accounting_schema_version") != 2
        or mutation.get("installation_evidence_complete") is not True
        or mutation.get("requested") != 1
        or mutation.get("planned") != 1
        or mutation.get("applied") != 1
        or mutation.get("discarded_applied", 0) != 0
    ):
        reasons.append("mutation installation was not admitted")
    reservation_status, reservation_reasons = fault_reservation_qualification(
        mutation.get("reservation")
    )
    if reservation_status != FAULT_RESERVATION_QUALIFIED:
        reasons.extend(reservation_reasons)
    execution = result.get("execution", {})
    health_before = execution.get("health_before")
    if not isinstance(health_before, dict) or not health_before.get("healthy"):
        reasons.append("pre-execution health check failed")
    admitted = not reasons
    sanitizer = result.get("sanitizer", {})
    sanitizer_outcome = sanitizer.get("outcome")
    runtime_diagnostic_count = sum(
        int(sanitizer.get(name, 0))
        for name in (
            "inline_diagnostics",
            "replay_diagnostics",
            "sampled_conflicts",
            "sampled_immediate_conflicts",
            "supercollider_diagnostics",
        )
        if isinstance(sanitizer.get(name, 0), int)
    )
    command_ran = execution.get("command_ran") is True
    completed = execution.get("completed") is True
    oracle_outcome = result.get("oracle", {}).get("outcome")
    witness_outcome = None
    if (
        admitted
        and command_ran
        and sanitizer_outcome == "detected"
        and runtime_diagnostic_count > 0
    ):
        witness_outcome = "detector-owned-runtime-diagnostic"
    elif admitted and command_ran and oracle_outcome == "fail":
        witness_outcome = "independent-oracle-manifestation"
    elif admitted and command_ran and completed and reach_witness is not None:
        witness_outcome = str(reach_witness["kind"])
    reached = witness_outcome is not None
    if admitted and not reached:
        reasons.append(
            "trial lacks a detector/oracle runtime witness or reviewed reach proof"
        )
    return admitted, reached, witness_outcome, reasons


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
    launcher = args.launcher
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
    health_command = (
        args.health_command_json
        if args.health_command_json is not None
        else [shutil.which("rocminfo") or "rocminfo"]
    )
    if args.smoke_command_json is None:
        smoke = _with_launcher(launcher, smoke)
    if args.health_command_json is None:
        health_command = _with_launcher(launcher, health_command)
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
                    "RJ_CONSAN_FAULT_LDS_WRONG_ADDRESS",
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
            command = _with_launcher(launcher, command)
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
                    "admitted": False,
                    "reached": False,
                    "reach_outcome": None,
                }
                summaries.append(row)
                profile_rows.append(row)
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            accepted, reasons = _fault_acceptance(result, policy)
            admitted, reached, reach_outcome, reach_reasons = (
                _fault_admission_and_reach(result, fault.get("reach_witness"))
            )
            row = {
                "profile": profile,
                "trial": index,
                "accepted": accepted,
                "reasons": reasons,
                "detector": result.get("sanitizer", {}).get("outcome"),
                "oracle": result.get("oracle", {}).get("outcome"),
                "admitted": admitted,
                "reached": reached,
                "reach_outcome": reach_outcome,
                "reach_reasons": reach_reasons,
                "result": str(result_path),
            }
            summaries.append(row)
            profile_rows.append(row)
        reached_rows = [row for row in profile_rows if row.get("reached") is True]
        detected = sum(row.get("detector") == "detected" for row in reached_rows)
        oracle_manifestations = sum(row.get("oracle") == "fail" for row in reached_rows)
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
                    f"detections={detected}/{len(reached_rows)}, minimum={minimum}"
                )
        if not reached_rows:
            profile_reasons.append("no admitted trial reached workload execution")
        detection_interval = (
            _wilson_detection_interval(detected, len(reached_rows))
            if reached_rows
            else None
        )
        oracle_interval = (
            _wilson_detection_interval(oracle_manifestations, len(reached_rows))
            if reached_rows
            else None
        )
        profile_summaries.append(
            {
                "profile": profile,
                "accepted": all(row["accepted"] for row in profile_rows)
                and not profile_reasons,
                "detector_policy": expected_detector,
                "attempted_trials": len(profile_rows),
                "admitted_trials": sum(
                    row.get("admitted") is True for row in profile_rows
                ),
                "reached_trials": len(reached_rows),
                "detections": detected,
                "trials": len(reached_rows),
                "detection_rate": (
                    detected / len(reached_rows) if reached_rows else None
                ),
                "detection_wilson_95": detection_interval,
                "oracle_manifestations": oracle_manifestations,
                "oracle_manifestation_rate": (
                    oracle_manifestations / len(reached_rows) if reached_rows else None
                ),
                "oracle_manifestation_wilson_95": oracle_interval,
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
        "launcher": launcher,
        "provenance": str(provenance),
        "rows": summaries,
        "profiles": profile_summaries,
        "accepted": all(profile["accepted"] for profile in profile_summaries),
    }
    if "site_provenance" in fault:
        summary["site_provenance"] = fault["site_provenance"]
    if "reach_witness" in fault:
        summary["reach_witness"] = fault["reach_witness"]
    summary_path = fault_root / "summary.json"
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["accepted"] else 1


def _load_empirical_row(
    row_dir: Path,
    *,
    target: str,
    workload: Workload,
    profile: str | None,
    phase: str,
    inner_repetitions_override: int | None,
    discard_first_timing_sample: bool,
    retain_code_objects: bool,
    collect_structural_metrics: bool,
) -> dict:
    result_path = row_dir / "result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(
            f"cannot resume empirical row {result_path}: {error}"
        ) from error
    expected = {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "workload": workload.id,
        "profile": profile or "baseline",
        "phase": _result_phase(phase, profile, workload),
    }
    mismatches = [
        f"{key}={result.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if result.get(key) != value
    ]
    if mismatches:
        raise ValidationError(
            f"empirical row conflicts with campaign {result_path}: "
            + "; ".join(mismatches)
        )
    expected_repetition_policy = {
        "empirical_row_schema_version": 1,
        "outer_processes": 1,
        "inner_repetitions_override": inner_repetitions_override,
        "discarded_first_timing_sample": discard_first_timing_sample,
        "discarded_timing_samples_per_process": int(discard_first_timing_sample),
        "retained_code_objects": retain_code_objects,
        "collected_structural_metrics": collect_structural_metrics,
    }
    repetition_policy = result.get("repetition_policy")
    if (
        not isinstance(repetition_policy, dict)
        or repetition_policy.get("empirical_row_schema_version") != 1
    ):
        raise ValidationError(
            f"empirical row predates or lacks empirical row schema version 1: "
            f"{result_path}"
        )
    if repetition_policy != expected_repetition_policy:
        raise ValidationError(
            f"empirical row repetition policy conflicts with campaign {result_path}"
        )
    return result


def _preserve_incomplete_empirical_row(row_dir: Path) -> None:
    suffix = 1
    while True:
        destination = row_dir.with_name(f"{row_dir.name}.incomplete-{suffix}")
        if not destination.exists():
            row_dir.rename(destination)
            return
        suffix += 1


def _run_or_resume_empirical_row(
    workspace: Path,
    target: str,
    workload: Workload,
    profile: str | None,
    phase: str,
    artifact_root: Path,
    timeout: int,
    launcher: list[str],
    row_dir: Path,
    *,
    resume: bool,
    inner_repetitions_override: int | None = None,
    discard_first_timing_sample: bool = False,
    retain_code_objects: bool = False,
    collect_structural_metrics: bool = False,
) -> dict:
    result_path = row_dir / "result.json"
    if result_path.is_file():
        if not resume:
            raise ValidationError(f"empirical row already exists: {result_path}")
        return _load_empirical_row(
            row_dir,
            target=target,
            workload=workload,
            profile=profile,
            phase=phase,
            inner_repetitions_override=inner_repetitions_override,
            discard_first_timing_sample=discard_first_timing_sample,
            retain_code_objects=retain_code_objects,
            collect_structural_metrics=collect_structural_metrics,
        )
    if row_dir.exists():
        if not resume:
            raise ValidationError(f"empirical row directory already exists: {row_dir}")
        _preserve_incomplete_empirical_row(row_dir)
    return _run_profile(
        workspace,
        target,
        workload,
        profile,
        phase,
        artifact_root,
        timeout,
        launcher=launcher,
        row_dir_override=row_dir,
        repetitions_override=1,
        inner_repetitions_override=inner_repetitions_override,
        discard_first_timing_sample=discard_first_timing_sample,
        retain_code_objects=retain_code_objects,
        collect_structural_metrics=collect_structural_metrics,
    )


def _empirical_supports_warm_timing(target: str, workload: Workload) -> bool:
    return (
        target not in SINGLE_REPETITION_TARGETS
        and workload.overhead_processes == 1
        and workload.warm_timing_mode in {"host-json", "device-fixed"}
    )


def _empirical_timing_protocol(
    target: str, workload: Workload, calibration: dict | None
) -> dict[str, object]:
    if not _empirical_supports_warm_timing(target, workload):
        return {
            "kind": "cold-process",
            "minimum_timed_aggregate_ms": None,
            "timed_inner_repetitions": None,
            "command_inner_repetitions": None,
            "discard_first_timing_sample": False,
        }
    if calibration is None or calibration.get("accepted") is not True:
        raise ValidationError(
            "warm empirical timing requires an accepted calibration row"
        )
    timing = calibration.get("timing_median_ms")
    if not isinstance(timing, dict) or not timing:
        raise ValidationError("warm empirical calibration has no workload timing")
    values = [float(value) for value in timing.values()]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValidationError(
            "warm empirical calibration timing must be finite and positive"
        )
    if workload.self_timed_device_minimum_ms is not None:
        if workload.self_timed_device_minimum_ms < EMPIRICAL_MINIMUM_TIMED_MS:
            raise ValidationError(
                "self-timed empirical workload does not meet the minimum timed "
                "aggregate"
            )
        measurement_runs = calibration.get("measurement_runs")
        if not isinstance(measurement_runs, list) or len(measurement_runs) != 1:
            raise ValidationError(
                "self-timed empirical calibration lacks one measurement record"
            )
        measurements = measurement_runs[0]
        if not isinstance(measurements, dict) or len(measurements) != 1:
            raise ValidationError(
                "self-timed empirical calibration must identify one benchmark"
            )
        measurement = next(iter(measurements.values()))
        fixed_iterations = measurement.get("benchmark_iterations")
        aggregate_ms = measurement.get("timed_aggregate_ms")
        if (
            not isinstance(fixed_iterations, int)
            or isinstance(fixed_iterations, bool)
            or fixed_iterations <= 0
            or not isinstance(aggregate_ms, (int, float))
            or not math.isfinite(float(aggregate_ms))
            or aggregate_ms < workload.self_timed_device_minimum_ms
        ):
            raise ValidationError(
                "self-timed empirical calibration lacks a valid fixed iteration "
                "count and timed aggregate"
            )
        return {
            "kind": "warm-device-self-timed",
            "minimum_timed_aggregate_ms": workload.self_timed_device_minimum_ms,
            "calibration_timing_median_ms": timing,
            "calibration_timed_aggregate_ms": float(aggregate_ms),
            "timed_inner_repetitions": fixed_iterations,
            "command_inner_repetitions": fixed_iterations,
            "discard_first_timing_sample": False,
        }
    timed_repetitions = max(
        1,
        math.ceil(EMPIRICAL_MINIMUM_TIMED_MS / min(values)),
    )
    discard_first = workload.kind == "pytorch"
    command_repetitions = timed_repetitions + int(discard_first)
    if command_repetitions > EMPIRICAL_MAX_INNER_REPETITIONS:
        raise ValidationError(
            "warm empirical calibration exceeds the inner-repetition safety bound: "
            f"required={command_repetitions}, "
            f"maximum={EMPIRICAL_MAX_INNER_REPETITIONS}"
        )
    return {
        "kind": "warm-host",
        "minimum_timed_aggregate_ms": EMPIRICAL_MINIMUM_TIMED_MS,
        "calibration_timing_median_ms": timing,
        "timed_inner_repetitions": timed_repetitions,
        "command_inner_repetitions": command_repetitions,
        "discard_first_timing_sample": discard_first,
    }


def _empirical_config(
    args: argparse.Namespace,
    target: str,
    workload: Workload,
    profiles: tuple[str, ...],
    max_rounds: int,
    timeout: int,
) -> dict[str, object]:
    return {
        "schema_version": EMPIRICAL_CAMPAIGN_SCHEMA_VERSION,
        "protocol": "consan-gfx1201-empirical-v1",
        "target": target,
        "workload": workload.id,
        "profiles": list(profiles),
        "required_accepted_rounds": args.rounds,
        "max_rounds": max_rounds,
        "randomization_seed": args.seed,
        "baseline_drift_limit": args.baseline_drift_limit,
        "bootstrap_resamples": args.bootstrap_resamples,
        "minimum_timed_aggregate_ms": EMPIRICAL_MINIMUM_TIMED_MS,
        "maximum_inner_repetitions": EMPIRICAL_MAX_INNER_REPETITIONS,
        "timeout_seconds": timeout,
        "launcher": args.launcher,
    }


def _write_or_verify_empirical_config(path: Path, config: dict[str, object]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValidationError(
                f"cannot read empirical campaign config {path}: {error}"
            ) from error
        if existing != config:
            raise ValidationError(f"empirical campaign config conflicts with {path}")
        return
    atomic_write_json(path, config)


def _empirical_campaign(args: argparse.Namespace) -> int:
    selection = _resolve_workload_selection(args, allow_all=False)
    target = selection.target
    if target != "gfx1201":
        raise ValidationError(
            "the empirical study command currently requires physical gfx1201"
        )
    workload = selection.require_workload()
    workspace = _workspace_from_environment()
    timeout = args.timeout if args.timeout is not None else workload.run_timeout_seconds
    doctor = _doctor(workspace, target, (workload.id,))
    if not doctor["ok"]:
        raise ValidationError("workspace doctor failed; run the doctor subcommand")
    profiles = PROFILE_IDS if args.profile == "all" else (args.profile,)
    max_rounds = args.max_rounds if args.max_rounds is not None else args.rounds * 2
    if max_rounds < args.rounds:
        raise ValidationError("--max-rounds must be at least --rounds")

    artifact_root = args.artifact_root.resolve()
    campaign_root = artifact_root / workload.id / "empirical-campaign"
    if campaign_root.exists() and not args.resume:
        raise ValidationError(
            f"empirical campaign already exists; pass --resume or use a new root: {campaign_root}"
        )
    campaign_root.mkdir(parents=True, exist_ok=True)
    _write_provenance(
        workspace,
        target,
        workload,
        _workload_provenance_path(artifact_root, workload).parent,
    )
    config = _empirical_config(args, target, workload, profiles, max_rounds, timeout)
    _write_or_verify_empirical_config(campaign_root / "config.json", config)

    admission_results = {}
    admission_root = campaign_root / "admission"
    for profile in (None, *profiles):
        profile_id = profile or "baseline"
        admission_results[profile_id] = _run_or_resume_empirical_row(
            workspace,
            target,
            workload,
            profile,
            "clean",
            artifact_root,
            timeout,
            args.launcher,
            admission_root / profile_id,
            resume=args.resume,
            retain_code_objects=profile is not None,
            collect_structural_metrics=profile is not None,
        )
    admission_row_acceptance = {}
    for profile, result in admission_results.items():
        accepted = result.get("accepted") is True
        if profile != "baseline":
            structural_runs = result.get("structural_metrics_runs")
            retained = result.get("retained_code_objects")
            accepted = (
                accepted
                and isinstance(structural_runs, list)
                and bool(structural_runs)
                and all(run.get("accepted") is True for run in structural_runs)
                and isinstance(retained, dict)
                and retained.get("complete_pairs", 0) > 0
                and retained.get("metadata_complete_pairs", 0) > 0
            )
        admission_row_acceptance[profile] = accepted
    admission_accepted = all(admission_row_acceptance.values())
    admission = {
        "accepted": admission_accepted,
        "rows": {
            profile: {
                "accepted": admission_row_acceptance[profile],
                "runtime_accepted": result.get("accepted") is True,
                "result": str(
                    (admission_root / profile / "result.json").relative_to(
                        artifact_root
                    )
                ),
            }
            for profile, result in admission_results.items()
        },
    }
    if not admission_accepted:
        campaign = {
            **config,
            "admission": admission,
            "rounds": [],
            "summary": {
                "schema_version": EMPIRICAL_CAMPAIGN_SCHEMA_VERSION,
                "accepted": False,
                "reasons": ["clean admission rejected"],
            },
        }
        atomic_write_json(campaign_root / "campaign.json", campaign)
        print(json.dumps(campaign, indent=2, sort_keys=True))
        return 1

    calibration = None
    calibration_record = None
    if _empirical_supports_warm_timing(target, workload):
        calibration_root = campaign_root / "calibration" / "baseline"
        calibration = _run_or_resume_empirical_row(
            workspace,
            target,
            workload,
            None,
            "overhead",
            artifact_root,
            timeout,
            args.launcher,
            calibration_root,
            resume=args.resume,
        )
        calibration_record = {
            "accepted": calibration.get("accepted") is True,
            "result": str(
                (calibration_root / "result.json").relative_to(artifact_root)
            ),
        }
    timing_protocol = _empirical_timing_protocol(target, workload, calibration)
    inner_repetitions = timing_protocol["command_inner_repetitions"]
    discard_first_timing_sample = timing_protocol["discard_first_timing_sample"]

    rounds = []
    campaign = None
    for round_index in range(max_rounds):
        if campaign is not None and campaign["summary"]["accepted"]:
            break
        order = list(profiles)
        random.Random(args.seed + round_index).shuffle(order)
        round_root = campaign_root / "rounds" / f"round-{round_index:03d}"

        def run_schedule(
            name: str,
            schedule_inner_repetitions: int | None,
            discard_first: bool,
            include_process: bool,
            include_workload: bool,
        ) -> dict[str, object]:
            schedule_root = round_root / name
            row_phase = (
                "clean"
                if name == "cold" and workload.self_timed_device_minimum_ms is not None
                else "overhead"
            )
            before = _run_or_resume_empirical_row(
                workspace,
                target,
                workload,
                None,
                row_phase,
                artifact_root,
                timeout,
                args.launcher,
                schedule_root / "00-baseline-before",
                resume=args.resume,
                inner_repetitions_override=schedule_inner_repetitions,
                discard_first_timing_sample=discard_first,
            )
            profile_results = {}
            for position, profile in enumerate(order, start=1):
                profile_results[profile] = _run_or_resume_empirical_row(
                    workspace,
                    target,
                    workload,
                    profile,
                    row_phase,
                    artifact_root,
                    timeout,
                    args.launcher,
                    schedule_root / f"{position:02d}-{profile}",
                    resume=args.resume,
                    inner_repetitions_override=schedule_inner_repetitions,
                    discard_first_timing_sample=discard_first,
                    collect_structural_metrics=name == "cold",
                )
            after_position = len(order) + 1
            after = _run_or_resume_empirical_row(
                workspace,
                target,
                workload,
                None,
                row_phase,
                artifact_root,
                timeout,
                args.launcher,
                schedule_root / f"{after_position:02d}-baseline-after",
                resume=args.resume,
                inner_repetitions_override=schedule_inner_repetitions,
                discard_first_timing_sample=discard_first,
            )
            schedule = _empirical_round_summary(
                round_index,
                order,
                before,
                profile_results,
                after,
                baseline_drift_limit=args.baseline_drift_limit,
                include_process_metric=include_process,
                include_workload_metrics=include_workload,
                metric_prefix=name,
            )
            schedule["structural_metrics"] = {}
            if name == "cold":
                for profile, result in profile_results.items():
                    try:
                        schedule["structural_metrics"][profile] = (
                            _empirical_structural_totals(result)
                        )
                    except ValidationError as error:
                        schedule["reasons"].append(
                            f"{profile}: structural metrics rejected: {error}"
                        )
            return schedule

        schedules = {
            "cold": run_schedule(
                "cold",
                1,
                False,
                True,
                workload.self_timed_device_minimum_ms is None,
            ),
        }
        if _empirical_supports_warm_timing(target, workload):
            schedules["warm"] = run_schedule(
                "warm",
                inner_repetitions,
                discard_first_timing_sample,
                False,
                True,
            )
        round_summary = _combine_empirical_round_summaries(
            round_index,
            order,
            schedules,
        )
        round_summary["row_results"] = [
            str(path.relative_to(artifact_root))
            for path in sorted(round_root.rglob("result.json"))
        ]
        atomic_write_json(round_root / "round.json", round_summary)
        rounds.append(round_summary)
        summary = _empirical_campaign_summary(
            rounds,
            profiles,
            required_accepted_rounds=args.rounds,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.seed,
            require_structural_metrics=True,
        )
        campaign = {
            **config,
            "admission": admission,
            "calibration": calibration_record,
            "timing_protocol": timing_protocol,
            "rounds": rounds,
            "summary": summary,
        }
        atomic_write_json(campaign_root / "campaign.json", campaign)

    assert campaign is not None
    print(json.dumps(campaign, indent=2, sort_keys=True))
    return 0 if campaign["summary"]["accepted"] else 1


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
    launcher = args.launcher
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
        dest="launcher",
        type=_launcher_argument,
        default=[],
        help="JSON argv prefix used to launch each workload process",
    )

    study = subparsers.add_parser(
        "study",
        help="run a reproducible physical-gfx1201 empirical overhead campaign",
    )
    study.add_argument("--workload", choices=tuple(WORKLOAD_BY_ID), required=True)
    study.add_argument("--profile", choices=(*PROFILE_IDS, "all"), default="all")
    study.add_argument("--artifact-root", type=Path, required=True)
    study.add_argument(
        "--rounds",
        type=int,
        default=EMPIRICAL_DEFAULT_ROUNDS,
        help="required number of accepted independently bracketed rounds",
    )
    study.add_argument(
        "--max-rounds",
        type=int,
        help="maximum attempted rounds; defaults to twice --rounds",
    )
    study.add_argument("--seed", type=int, default=0)
    study.add_argument(
        "--baseline-drift-limit",
        type=float,
        default=EMPIRICAL_DEFAULT_BASELINE_DRIFT_LIMIT,
    )
    study.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=EMPIRICAL_DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    study.add_argument(
        "--timeout",
        type=int,
        help="override the workload timeout declared by the executable manifest",
    )
    study.add_argument(
        "--launcher-json",
        dest="launcher",
        type=_launcher_argument,
        default=[],
        help="JSON argv prefix used to launch each workload process",
    )
    study.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete rows and preserve then retry interrupted rows",
    )

    verify_coverage = subparsers.add_parser(
        "verify-coverage-output",
        help="replay and verify one retained coverage-output result",
    )
    verify_coverage.add_argument(
        "--result", type=Path, required=True, help="path to the retained result.json"
    )

    inventory = subparsers.add_parser(
        "inventory", help="record target-specific fault sites without mutation"
    )
    inventory.add_argument("--workload", choices=tuple(WORKLOAD_BY_ID), required=True)
    inventory.add_argument("--artifact-root", type=Path, required=True)
    inventory.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)
    inventory.add_argument(
        "--launcher-json",
        dest="launcher",
        type=_launcher_argument,
        default=[],
        help="JSON argv prefix used to launch the workload process",
    )

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
        "--launcher-json",
        dest="launcher",
        type=_launcher_argument,
        default=[],
        help=(
            "JSON argv prefix used for the workload and default health/smoke "
            "commands; explicit paired health/smoke commands remain verbatim"
        ),
    )
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
    if getattr(args, "rounds", 1) <= 0:
        parser.error("--rounds must be positive")
    max_rounds = getattr(args, "max_rounds", None)
    if max_rounds is not None and max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    if getattr(args, "bootstrap_resamples", 1) <= 0:
        parser.error("--bootstrap-resamples must be positive")
    drift_limit = getattr(args, "baseline_drift_limit", 0.05)
    if not 0.0 <= drift_limit < 1.0:
        parser.error("--baseline-drift-limit must be in [0, 1)")
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
        if args.command == "verify-coverage-output":
            result = verify_coverage_output_result(args.result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["accepted"] else 1
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
        if args.command == "study":
            return _empirical_campaign(args)
        return _run(args)
    except (OSError, ValidationError, ValueError, json.JSONDecodeError) as error:
        print(f"validation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
