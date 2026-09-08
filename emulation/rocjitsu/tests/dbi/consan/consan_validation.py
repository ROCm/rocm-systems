#!/usr/bin/env python3
"""Runs ConSan's portable real-workload validation matrix.

The required CONSAN_VALIDATION_WORKSPACE_DIR contains external repositories,
their build outputs, and a rocJITsu build. IREE command-line tools and rocminfo
are resolved from PATH. Run `consan_validation.py doctor` before GPU work and
`consan_validation.py explain` to audit commands, settings, and fault policy.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
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
import threading
import time

import consan_cdna_hip_moi_registry as cdna_hip_moi_registry
from consan_coverage_gate import CoverageParseError, parse_coverage_evidence
from consan_tensile_support import (
    TensileValidationPaths,
    resolve_tensile_validation_paths,
    tensile_python_environment,
)
from consan_validation_catalog import (
    CONTROLLED_ENV_PREFIX,
    EMPIRICAL_CAMPAIGN_SCHEMA_VERSION,
    EMPIRICAL_DEFAULT_BASELINE_DRIFT_LIMIT,
    EMPIRICAL_DEFAULT_BOOTSTRAP_RESAMPLES,
    EMPIRICAL_DEFAULT_ROUNDS,
    EMPIRICAL_MAX_INNER_REPETITIONS,
    EMPIRICAL_MINIMUM_TIMED_MS,
    FAULT_FAMILY_ENVIRONMENTS,
    FAULT_FAMILY_SITE_KINDS,
    HIP_MOI_GPU_BENCHMARK_ITERATIONS_ENV,
    HIP_MOI_GPU_BENCHMARK_WARMUP_ITERATIONS_ENV,
    HSA_TOOL_ENVIRONMENT,
    LLAMA_BUILD_DIR_ENV,
    LLVM_READELF_ENV,
    MOI_DIAGNOSTIC_KINDS,
    MOI_SHADOW_ACCESS_WRITE,
    NATIVE_GTEST_TARGETS,
    NATIVE_GTEST_WORKLOAD_IDS,
    NATIVE_GTEST_WORKLOAD_OVERRIDES,
    ORDINARY_FORBIDDEN_ENVIRONMENT,
    ORDINARY_MOI_RUNTIME_DEFAULTS,
    PROCESS_OUTPUT_DRAIN_SECONDS,
    PROCESS_TERMINATION_GRACE_SECONDS,
    PROFILE_IDS,
    PROFILES,
    PROVENANCE_SCHEMA_VERSION,
    PYTORCH_OVERHEAD_PROCESSES,
    PYTORCH_PYTHON_ENV,
    QWEN_BUILD_MANIFEST_SCHEMA_VERSION,
    QWEN_COMPILE_OPTIONS,
    QWEN_OVERHEAD_REPETITIONS,
    RDNA4_MATMUL_DIR_ENV,
    RECORD_REPLAY_STANDARD_RUNTIME_DEFAULTS,
    SAMPLED_STANDARD_RUNTIME_DEFAULTS,
    SCHEMA_VERSION,
    SETTING_CATEGORIES,
    SHARKTANK_PYTHON_ENV,
    SINGLE_REPETITION_TARGETS,
    SOFTWARE_MODEL_ENVIRONMENT,
    STREAMK_FAULT_FAMILIES,
    STREAMK_WORKLOAD_IDS,
    STREAMK_WORKLOAD_SHAPES,
    TARGET_ENV,
    TARGET_WORKLOAD_OVERRIDES,
    TENSILE_PYTHON_ENV,
    TENSILE_SHARD_TIMING_CANARY_MS,
    TIMEOUT_SECONDS,
    TOOLS,
    ValidationError,
    Workload,
    WORKLOAD_BY_ID,
    WORKLOADS,
    WORKSPACE_ENV,
    _attention_override,
    _cdna_gtest_target,
    _fault_families,
    _fault_family_environment,
    _jakub_override,
    _native_gtest_overrides,
    _native_gtest_path,
    _resolved_workload,
    _single_oracle_override,
    _streamk_overrides,
    _target_fault_families,
    _validate_exact_keys,
    _validate_tensile_sharding,
    _validate_workload_manifest,
    _workload_for_target,
    _workloads_for_target,
    Profile,
    resolved_workload_relative_path,
)
from consan_validation_commands import (
    LlamaRuntime,
    WorkloadSelection,
    _audited_settings,
    _audited_unsets,
    _clean_environment,
    _command_identity,
    _command_json,
    _controlled_environment,
    _corpus_root,
    _doctor,
    _effective_workload,
    _empirical_observation_snapshot,
    _fault_workload_command,
    _gfx_target_version,
    _health_smoke_command,
    _hook_path,
    _inner_repetitions,
    _input_files,
    _launcher_argument,
    _launcher_from_json,
    _llama_executable,
    _llama_library_environment,
    _llama_runtime,
    _llama_runtime_files,
    _machine_identity,
    _manifest,
    _native_runtime_identity,
    _normalize_dynamic_loader_output,
    _outer_repetitions,
    _prepare_qwen,
    _profile_runtime_defaults,
    _pytorch_python,
    _pytorch_runtime_probe,
    _qwen_build_check,
    _qwen_build_manifest,
    _qwen_command,
    _qwen_compile_options,
    _rdna4_matmul_root,
    _read_identity_file,
    _required_paths,
    _resolve_workload_selection,
    _run_environment,
    _runtime_libraries_from_ldd_output,
    _runtime_library_records,
    _runtime_tool_identities,
    _setting_metadata,
    _sharktank_python,
    _source_identities,
    _target,
    _target_outer_repetitions,
    _tensile_python,
    _tensile_runtime_probe,
    _unavailable_command_identity,
    _with_launcher,
    _workload_command,
    _workload_commands,
    _workload_provenance_path,
    _workload_runtime_identity,
    _workspace_from_environment,
    _write_provenance,
)
from consan_validation_diagnostics import (
    DIAGNOSTIC_OUTPUT_PARSERS,
    DiagnosticPolicy,
    DiagnosticRecord,
    DiagnosticSourceSummary,
    ParsedDiagnosticOutput,
    ReplayIdentity,
    _amdgpu_kernel_metadata,
    _benchmark_median,
    _benchmark_samples,
    _boolean,
    _bool_label,
    _code_object_fingerprint,
    _coverage_summary,
    _DiagnosticFieldsError,
    _diagnostic_output_summary,
    _diagnostic_record_result,
    _diagnostic_source_result,
    _discard_first_sample_per_process,
    _empirical_structural_metrics,
    _empirical_structural_totals,
    _evaluate_diagnostic_output,
    _gtest_device_measurement,
    _gtest_median,
    _gtest_test_count,
    _gtest_timing_samples,
    _identity_label,
    _instruction_label,
    _json_measurements,
    _json_medians,
    _json_timing_samples,
    _lds_range,
    _llvm_readelf,
    _log_fields,
    _nonnegative_float,
    _parse_amdgpu_kernel_metadata,
    _parse_log_fields,
    _parse_record_replay_diagnostic_output,
    _ReplayDiagnosticRecord,
    _replay_diagnostic_record,
    _replay_identity,
    _ReplayReport,
    _ReplaySkipped,
    _ReplaySummary,
    _retained_code_object_inventory,
    _retained_relative_path,
    _sharktank_medians,
    _unsigned,
)
from consan_validation_empirical import (
    _empirical_campaign,
    _empirical_config,
    _empirical_device_timed_minimum,
    _empirical_supports_warm_timing,
    _empirical_timing_protocol,
    _empirical_uses_device_timing,
    _load_empirical_row,
    _preserve_incomplete_empirical_row,
    _run_or_resume_empirical_row,
    _write_or_verify_empirical_config,
)
from consan_validation_execution import (
    _bounded_process_output,
    _row_runtime_acceptance,
    _run_process,
    _run_process_batch,
    _run_profile,
    _stop_process_group,
)
from consan_validation_faults import (
    _explain_contract,
    _fault,
    _fault_acceptance,
    _fault_admission_and_reach,
    _fault_audit,
    _fault_inventory_environment,
    _fault_template,
    _fault_trial_environment,
    _fault_trials,
    _faults_from_spec,
    _inventory,
    _inventory_collection_complete,
    _inventory_line_completes,
    _inventory_records,
    _load_fault,
    _load_resumable_fault_result,
    _print_explain,
    _required_diagnostic,
    _run_inventory_process,
    _wilson_detection_interval,
)
from consan_validation_statistics import (
    _bootstrap_median_interval,
    _bootstrap_stream_seed,
    _combine_empirical_round_summaries,
    _empirical_campaign_summary,
    _empirical_round_summary,
    _empirical_row_metrics,
    _linear_quantile,
    _overhead_summary,
    _sample_summary,
)
from consan_validation_support import (
    FAULT_RESERVATION_QUALIFIED,
    RESULT_SCHEMA_VERSION,
    SITE_KINDS,
    atomic_write_json,
    fault_reservation_qualification,
    git_identity,
    sha256_file,
)








def _run(args: argparse.Namespace) -> int:
    selection = _resolve_workload_selection(args, allow_all=False)
    target = selection.target
    workload = _resolved_workload(target, selection.require_workload())
    workspace = _workspace_from_environment()
    timeout = args.timeout if args.timeout is not None else workload.run_timeout_seconds
    doctor = _doctor(workspace, target, (workload.id,), args.launcher)
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
        launcher,
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
    doctor.add_argument(
        "--launcher-json",
        dest="launcher",
        type=_launcher_argument,
        default=[],
        help="JSON argv prefix used for target runtime probes",
    )
    doctor.add_argument("--json", action="store_true")

    manifest = subparsers.add_parser("manifest", help="print the executable matrix")
    manifest.add_argument("--json", action="store_true")

    prepare = subparsers.add_parser(
        "prepare", help="build a canonical external workload artifact"
    )
    prepare.add_argument(
        "--workload", choices=("qwen-prefill",), required=True
    )

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
    fault.add_argument(
        "--timeout",
        type=int,
        help="override the workload timeout declared by the executable manifest",
    )
    fault.add_argument(
        "--health-timeout",
        type=float,
        default=30.0,
        help="deadline in seconds for each retained discovery and smoke probe",
    )
    fault.add_argument("--allow-destructive", action="store_true")
    fault.add_argument(
        "--resume",
        action="store_true",
        help="reuse only complete fault rows whose execution contract still matches",
    )
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
        if args.command == "prepare":
            result = _prepare_qwen(workspace, target)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
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
            )
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                _print_explain(result)
            return 0
        if args.command == "doctor":
            result = _doctor(
                workspace, target, selection.selected_ids(), args.launcher
            )
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
                for artifact, item in result.get("artifacts", {}).items():
                    state = "ok" if item["ok"] else "STALE"
                    print(f"{state:7} artifact {artifact}: {item['manifest']}")
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
