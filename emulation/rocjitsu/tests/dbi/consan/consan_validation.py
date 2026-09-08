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
from consan_validation_support import (
    FAULT_RESERVATION_QUALIFIED,
    RESULT_SCHEMA_VERSION,
    SITE_KINDS,
    atomic_write_json,
    fault_reservation_qualification,
    git_identity,
    sha256_file,
)



def _workspace_from_environment() -> Path:
    value = os.environ.get(WORKSPACE_ENV)
    if not value:
        raise ValidationError(f"{WORKSPACE_ENV} is required")
    workspace = Path(value).expanduser().resolve()
    if not workspace.is_dir():
        raise ValidationError(f"{WORKSPACE_ENV} is not a directory: {workspace}")
    return workspace


def _corpus_root(workspace: Path, corpus: str) -> Path:
    """Resolve a validation corpus in both standalone and TheRock layouts."""
    canonical = workspace / corpus
    if canonical.exists() or corpus != "rocm-systems":
        return canonical
    therock = workspace / "TheRock" / "rocm-systems"
    return therock if therock.is_dir() else canonical


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
    if any(workload.corpus == "rocm-systems" for workload in workloads):
        paths["rocm-systems"] = _corpus_root(workspace, "rocm-systems")
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
    python: Path,
    hook: Path,
    target: str,
    workload: Workload,
    workspace: Path,
    launcher: list[str] | None = None,
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
    environment = _clean_environment("supercollider", workload, hook, target, workspace)
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
        command = [str(python), "-c", probe_source, str(hook.resolve())]
        probe = subprocess.run(
            _with_launcher(launcher or [], command),
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


def _tensile_runtime_probe(
    python: Path, paths: TensileValidationPaths
) -> dict:
    """Proves that the Tensile driver and its native client are loadable."""
    environment = tensile_python_environment(paths)
    try:
        import_probe = subprocess.run(
            [str(python), "-P", "-c", "from Tensile import Tensile"],
            check=False,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "python": str(python), "detail": str(error)}
    ldd = shutil.which("ldd")
    if ldd is None:
        return {
            "ok": False,
            "python": str(python),
            "detail": "ldd is missing",
            "reasons": ["cannot verify Tensile client runtime dependencies"],
        }
    try:
        client_probe = subprocess.run(
            [ldd, str(paths.client)],
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
            "reasons": ["cannot inspect Tensile client runtime dependencies"],
        }
    import_detail = (import_probe.stderr or import_probe.stdout).strip()
    client_detail = (client_probe.stdout or client_probe.stderr).strip()
    missing_libraries = [
        line.strip()
        for line in client_detail.splitlines()
        if re.search(r"\s=>\s+not found\s*$", line)
    ]
    reasons = []
    if import_probe.returncode != 0:
        reasons.append(f"Tensile import exited with status {import_probe.returncode}")
    if client_probe.returncode != 0:
        reasons.append(f"Tensile client ldd exited with status {client_probe.returncode}")
    if missing_libraries:
        reasons.append(
            "Tensile client has missing runtime libraries: "
            + ", ".join(missing_libraries)
        )
    linkage_detail = "\n".join(missing_libraries)
    if client_probe.returncode != 0 and not linkage_detail:
        linkage_detail = client_detail
    detail = "\n".join(
        part for part in (import_detail, linkage_detail) if part
    ) or "Tensile import and client runtime closure passed"
    return {
        "ok": not reasons,
        "python": str(python),
        "detail": detail,
        "reasons": reasons,
    }


@dataclass(frozen=True)
class LlamaRuntime:
    build_root: Path
    executable: Path
    libraries: dict[str, Path]


def _llama_runtime_files(build_root: Path) -> dict[str, Path]:
    source_root = build_root / "third_party" / "llama.cpp" / "ggml" / "src"
    return {
        "ggml": source_root / "libggml.so",
        "ggml-base": source_root / "libggml-base.so",
        "ggml-cpu": source_root / "libggml-cpu.so",
        "ggml-hip": source_root / "ggml-hip" / "libggml-hip.so",
    }


def _llama_runtime(workspace: Path, target: str, name: str) -> LlamaRuntime:
    configured = os.environ.get(LLAMA_BUILD_DIR_ENV)
    if configured:
        build_roots = (Path(os.path.abspath(Path(configured).expanduser())),)
    else:
        build_roots = (
            workspace / "rocjitsu-test-corpus-build" / "kernels" / target,
            workspace
            / "rocjitsu-test-corpus"
            / ".pytest-artifacts-rdna4-llama-baseline"
            / "_suite_shards"
            / "kernels_shard_0"
            / "kernels"
            / target
            / "build",
        )

    failures = []
    for build_root in build_roots:
        executable = build_root / "cases" / "llama.cpp" / name
        libraries = _llama_runtime_files(build_root)
        missing = [
            path for path in (executable, *libraries.values()) if not path.is_file()
        ]
        if not missing:
            return LlamaRuntime(build_root, executable, libraries)
        failures.append((build_root, missing))

    details = "; ".join(
        f"{build_root}: missing {', '.join(str(path) for path in missing)}"
        for build_root, missing in failures
    )
    if configured:
        raise ValidationError(
            f"{LLAMA_BUILD_DIR_ENV} must name a complete llama build root; {details}"
        )
    raise ValidationError(f"cannot locate a complete llama runtime; checked {details}")


def _llama_executable(workspace: Path, target: str, name: str) -> Path:
    return _llama_runtime(workspace, target, name).executable


def _llama_library_environment(
    runtime: LlamaRuntime, environment: dict[str, str]
) -> None:
    library_directories = list(
        dict.fromkeys(str(path.parent) for path in runtime.libraries.values())
    )
    existing = environment.get("LD_LIBRARY_PATH")
    if existing:
        library_directories.append(existing)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_directories)


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
            "config": _corpus_root(workspace, workload.corpus)
            / workload.relative_path,
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
        runtime = _llama_runtime(workspace, target, workload.relative_path)
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
            "executable": runtime.executable,
            **runtime.libraries,
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
            "build-manifest": root / target / "qwen3-600m.consan-build.json",
            "source": workspace
            / "iree-test-suites/torch_models/qwen3-600m/model.mlir",
            "parameters": data / "real_weights.irpa",
            "input": data / "inference_input.0.bin",
            "expected": data / "inference_output.0.bin",
        }
    if workload.kind == "gtest":
        return {"executable": workspace / workload.relative_path}
    if workload.kind == "native-executable":
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


def _qwen_compile_options(target: str, encoder_output: Path) -> tuple[str, ...]:
    return (
        f"--iree-rocm-target={target}",
        *QWEN_COMPILE_OPTIONS,
        f"--iree-parameter-encoder-output-file={encoder_output}",
    )


def _qwen_build_manifest(
    target: str,
    source: Path,
    vmfb: Path,
    compiler: Path,
    encoder_output: Path,
) -> dict[str, object]:
    return {
        "schema_version": QWEN_BUILD_MANIFEST_SCHEMA_VERSION,
        "workload": "qwen-prefill",
        "target": target,
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "compiler": {"path": str(compiler), "sha256": sha256_file(compiler)},
        "compile_options": list(_qwen_compile_options(target, encoder_output)),
        "vmfb": {"path": str(vmfb), "sha256": sha256_file(vmfb)},
    }


def _qwen_build_check(workspace: Path, target: str) -> dict[str, object]:
    inputs = _input_files(workspace, target, WORKLOAD_BY_ID["qwen-prefill"])
    manifest_path = inputs["build-manifest"]
    result: dict[str, object] = {
        "ok": False,
        "manifest": str(manifest_path),
        "reasons": [],
    }
    reasons = result["reasons"]
    assert isinstance(reasons, list)
    if not manifest_path.is_file():
        reasons.append("missing canonical Qwen build manifest; run prepare")
        return result
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        reasons.append(f"cannot read canonical Qwen build manifest: {error}")
        return result
    if not isinstance(document, dict):
        reasons.append("canonical Qwen build manifest is not a JSON object")
        return result
    expected_options = list(
        _qwen_compile_options(
            target, manifest_path.with_name("qwen3-600m.parameter-encoder.mlir")
        )
    )
    scalar_expectations = {
        "schema_version": QWEN_BUILD_MANIFEST_SCHEMA_VERSION,
        "workload": "qwen-prefill",
        "target": target,
        "compile_options": expected_options,
    }
    for field, expected in scalar_expectations.items():
        if document.get(field) != expected:
            reasons.append(f"Qwen build manifest has stale {field}")
    for field, path in (("source", inputs["source"]), ("vmfb", inputs["vmfb"])):
        record = document.get(field)
        if (
            not path.is_file()
            or not isinstance(record, dict)
            or record.get("sha256") != sha256_file(path)
        ):
            reasons.append(f"Qwen build manifest {field} hash does not match")
    result["ok"] = not reasons
    return result


def _prepare_qwen(workspace: Path, target: str) -> dict[str, object]:
    workload = _workload_for_target(target, "qwen-prefill")
    inputs = _input_files(workspace, target, workload)
    source = inputs["source"]
    if not source.is_file():
        raise ValidationError(f"missing Qwen source MLIR: {source}")
    compiler_name = shutil.which("iree-compile")
    if compiler_name is None:
        raise ValidationError("iree-compile is required to prepare Qwen")
    compiler = Path(compiler_name).resolve()
    vmfb = inputs["vmfb"]
    vmfb.parent.mkdir(parents=True, exist_ok=True)
    candidate = vmfb.with_name(f"qwen3-600m.{os.getpid()}.next.vmfb")
    encoder_output = vmfb.with_name("qwen3-600m.parameter-encoder.mlir")
    command = [
        str(compiler),
        str(source),
        *_qwen_compile_options(target, encoder_output),
        "-o",
        str(candidate),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise ValidationError(
            f"Qwen compilation failed with exit code {completed.returncode}"
        )
    if not candidate.is_file() or candidate.stat().st_size == 0:
        raise ValidationError("Qwen compilation produced no VMFB")
    os.replace(candidate, vmfb)
    document = _qwen_build_manifest(
        target, source, vmfb, compiler, encoder_output
    )
    atomic_write_json(inputs["build-manifest"], document)
    return document


def _doctor(
    workspace: Path,
    target: str,
    workload_ids: tuple[str, ...] | None = None,
    launcher: list[str] | None = None,
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
            executable = (
                workload.kind == "native-executable" and label == "executable"
            ) or (
                workload.kind == "tensile"
                and label
                in {
                    "python",
                    "client",
                    "wrapper",
                    "rocjitsu",
                    "llvm-readelf",
                    "amdclang++",
                }
            )
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
                workspace,
                launcher,
            )
        else:
            runtimes["pytorch"] = {
                "ok": False,
                "python": str(python),
                "detail": "interpreter is missing",
            }
    tensile_workloads = tuple(
        workload for workload in workloads if workload.kind == "tensile"
    )
    if tensile_workloads:
        python = _tensile_python()
        tensile_paths = resolve_tensile_validation_paths(workspace, target)
        if (
            python.is_file()
            and os.access(python, os.X_OK)
            and tensile_paths.tensilelite.is_dir()
        ):
            runtimes["tensile"] = _tensile_runtime_probe(python, tensile_paths)
        else:
            runtimes["tensile"] = {
                "ok": False,
                "python": str(python),
                "detail": "interpreter or TensileLite package is missing",
            }
    artifacts = {}
    if any(workload.kind == "qwen" for workload in workloads):
        artifacts["qwen-prefill"] = _qwen_build_check(workspace, target)
    ok = (
        all(item["present"] for item in path_checks.values())
        and all(tools.values())
        and all(item["ok"] for item in runtimes.values())
        and all(item["ok"] for item in artifacts.values())
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "workspace": str(workspace),
        "target": target,
        "workloads": list(selected_ids),
        "paths": path_checks,
        "runtimes": runtimes,
        "artifacts": artifacts,
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
    hook: Path | None,
    target: str | None,
    workspace: Path,
) -> dict[str, str]:
    if target is not None:
        workload = _resolved_workload(target, workload)
    ignored_environment = HSA_TOOL_ENVIRONMENT | {
        "HIP_TARGET",
        HIP_MOI_GPU_BENCHMARK_ITERATIONS_ENV,
        HIP_MOI_GPU_BENCHMARK_WARMUP_ITERATIONS_ENV,
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(CONTROLLED_ENV_PREFIX)
        and key not in ignored_environment
        and key not in SOFTWARE_MODEL_ENVIRONMENT
    }
    environment.update(dict(workload.command_environment))
    if target is not None:
        environment["HIP_TARGET"] = target
    if workload.kind == "llama":
        if target is None:
            raise ValidationError("llama runtime environment requires a target")
        runtime = _llama_runtime(workspace, target, workload.relative_path)
        _llama_library_environment(runtime, environment)
    if target == "gfx1250":
        # ROCr loads the gfx1250 A0 HotSwap helper by soname when it is not
        # installed beside libhsa-runtime64. Standalone RocJITsu builds keep
        # that companion DSO beside the ConSan hook, so make the complete hook
        # bundle visible to validation payloads and health probes.
        companion_dir = str((hook or _hook_path(workspace)).parent)
        existing = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            (companion_dir, existing) if existing else (companion_dir,)
        )
    if profile is None:
        return environment
    if hook is None:
        raise ValidationError("instrumented runtime environment requires a hook")
    config = PROFILES[profile]
    environment.update(config.environment)
    environment.update(
        {
            "HSA_TOOLS_LIB": str(hook),
            "RJ_CONSAN_LOG": "1",
        }
    )
    if (
        profile == "record-replay"
        and workload.record_replay_runtime_sample_stride is not None
    ):
        stride = workload.record_replay_runtime_sample_stride
        if stride <= 0 or stride & (stride - 1):
            raise ValidationError(
                f"{workload.id} has a non-power-of-two Record/Replay sampling stride: {stride}"
            )
        environment["RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE"] = str(stride)
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
    workspace: Path,
) -> dict[str, str]:
    if phase not in {"clean", "overhead"}:
        raise ValidationError(f"unsupported validation phase: {phase}")
    return _clean_environment(profile, workload, hook, target, workspace)


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
        HIP_MOI_GPU_BENCHMARK_ITERATIONS_ENV,
        HIP_MOI_GPU_BENCHMARK_WARMUP_ITERATIONS_ENV,
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
        "RJ_CONSAN_MOI_TRACK_BARRIERS",
        "RJ_CONSAN_MOI_TRACK_ATOMICS",
    }:
        category = "instrumentation-selection"
    elif name in ORDINARY_FORBIDDEN_ENVIRONMENT or name in {
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
    # Qwen is a useful universal smoke on hardware targets, but merely having
    # its files does not make it a bounded software-emulator health probe.  The
    # gfx1250 campaign already qualifies the single D128-pressure exact oracle
    # as its independent, sub-30-second target-dispatch denominator.
    if target == "gfx1250":
        smoke_workload = WORKLOAD_BY_ID["d128-pressure"]
        if all(
            path.is_file()
            for path in _input_files(workspace, target, smoke_workload).values()
        ):
            return _workload_command(
                workspace, target, smoke_workload, "overhead", output
            )
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
    *,
    tensile_exact_problem_sizes: tuple[tuple[int, ...], ...] | None = None,
    tensile_expected_numeric_rows: int | None = None,
    tensile_expected_client_passes: int | None = None,
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
        command = [
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
        if workload.sharktank_skip_warmup:
            command.append("--skip-warmup")
        return command
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
        minimum_timed_ms = workload.tensile_minimum_timed_ms
        if tensile_exact_problem_sizes is not None:
            # Each leaf proves a positive timing canary. The parent result
            # aggregates all shards and enforces the workload-level minimum.
            minimum_timed_ms = min(
                minimum_timed_ms, TENSILE_SHARD_TIMING_CANARY_MS
            )
        command = [
            str(_tensile_python()),
            str(Path(__file__).with_name("consan_tensile_validation.py")),
            "--workspace",
            str(workspace),
            "--config",
            str(_corpus_root(workspace, workload.corpus) / workload.relative_path),
            "--gpu-target",
            target,
            "--output-dir",
            str(output.parent / "tensile-work"),
            "--repetitions",
            "1",
            "--minimum-timed-ms",
            str(minimum_timed_ms),
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
        expected_numeric_rows = (
            tensile_expected_numeric_rows
            if tensile_expected_numeric_rows is not None
            else workload.tensile_expected_numeric_rows
        )
        if expected_numeric_rows is not None:
            command.extend(
                (
                    "--expect-numeric-rows",
                    str(expected_numeric_rows),
                )
            )
        expected_client_passes = (
            tensile_expected_client_passes
            if tensile_expected_client_passes is not None
            else workload.tensile_expected_client_passes
        )
        if expected_client_passes is not None:
            command.extend(
                (
                    "--expect-client-passes",
                    str(expected_client_passes),
                )
            )
        if tensile_exact_problem_sizes is not None:
            source_blocks = (
                workload.tensile_expected_source_exact_problem_size_blocks
            )
            source_problem_sizes = tuple(
                dict.fromkeys(
                    size
                    for shard in workload.tensile_exact_problem_size_shards
                    for size in shard
                )
            )
            if not source_problem_sizes:
                raise ValidationError(
                    f"{workload.id} command selected a problem-size shard "
                    "without a source inventory"
                )
            command.extend(
                (
                    "--exact-problem-sizes-json",
                    json.dumps(tensile_exact_problem_sizes, separators=(",", ":")),
                )
            )
            if source_blocks:
                command.extend(
                    (
                        "--expect-source-exact-problem-size-blocks-json",
                        json.dumps(source_blocks, separators=(",", ":")),
                    )
                )
            else:
                command.extend(
                    (
                        "--expect-source-exact-problem-sizes-json",
                        json.dumps(source_problem_sizes, separators=(",", ":")),
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
        command = [
            sys.executable,
            str(Path(__file__).with_name("consan_llama_validation.py")),
            "--executable",
            str(_llama_executable(workspace, target, workload.relative_path)),
            "--workload",
            llama_workload,
            "--output-dir",
            str(output.parent / f"{output.stem}-llama-work"),
        ]
        if overhead and workload.id == "llama-rdna4-mul-mat-vec-q":
            command.extend(
                [
                    "--n-embd",
                    "1024",
                    "--benchmark-iterations",
                    str(inner_repetitions_override or 40_000),
                    "--benchmark-warmup-iterations",
                    "5",
                    "--minimum-timed-ms",
                    str(workload.self_timed_device_minimum_ms),
                ]
            )
        return command
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
    if workload.kind == "native-executable":
        return [str(workspace / workload.relative_path), *workload.command_arguments]
    executable = workspace / workload.relative_path
    selected_filter = (
        workload.fault_filter or workload.clean_filter
        if phase == "fault"
        else workload.overhead_filter if overhead else workload.clean_filter
    )
    return [str(executable), f"--gtest_filter={selected_filter}"]


def _workload_commands(
    workspace: Path,
    target: str,
    workload: Workload,
    phase: str,
    output: Path,
    inner_repetitions_override: int | None = None,
) -> list[list[str]]:
    workload = _resolved_workload(target, workload)
    shards = workload.tensile_exact_problem_size_shards
    if not shards:
        return [
            _workload_command(
                workspace,
                target,
                workload,
                phase,
                output,
                inner_repetitions_override,
            )
        ]
    expected_rows = workload.tensile_expected_numeric_rows_per_shard or (
        None,
    ) * len(shards)
    expected_clients = workload.tensile_expected_client_passes_per_shard or (
        None,
    ) * len(shards)
    return [
        _workload_command(
            workspace,
            target,
            workload,
            phase,
            output,
            inner_repetitions_override,
            tensile_exact_problem_sizes=shard,
            tensile_expected_numeric_rows=shard_expected_rows,
            tensile_expected_client_passes=shard_expected_clients,
        )
        for shard, shard_expected_rows, shard_expected_clients in zip(
            shards,
            expected_rows,
            expected_clients,
            strict=True,
        )
    ]


def _fault_workload_command(
    workspace: Path,
    target: str,
    workload: Workload,
    output: Path,
) -> list[str]:
    """Build one bounded command for inventory and exact-one fault execution."""
    workload = _resolved_workload(target, workload)
    index = workload.tensile_fault_shard_index
    if index is None:
        return _workload_command(workspace, target, workload, "fault", output)
    return _workload_command(
        workspace,
        target,
        workload,
        "fault",
        output,
        tensile_exact_problem_sizes=workload.tensile_exact_problem_size_shards[index],
        tensile_expected_numeric_rows=(
            workload.tensile_expected_numeric_rows_per_shard[index]
            if workload.tensile_expected_numeric_rows_per_shard
            else None
        ),
        tensile_expected_client_passes=(
            workload.tensile_expected_client_passes_per_shard[index]
            if workload.tensile_expected_client_passes_per_shard
            else None
        ),
    )


def _write_provenance(
    workspace: Path,
    target: str,
    workload: Workload,
    workload_root: Path,
    launcher: list[str] | None = None,
) -> Path:
    workload_root.mkdir(parents=True, exist_ok=True)
    path = workload_root / "provenance.json"
    hook = _hook_path(workspace)
    files = {"hook": hook, **_input_files(workspace, target, workload)}
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
        "workload_runtime": _workload_runtime_identity(
            workspace, target, workload, launcher
        ),
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
        existing_schema = existing.get("provenance_schema_version")
        if existing_schema != PROVENANCE_SCHEMA_VERSION:
            raise ValidationError(
                "provenance schema changed "
                f"from {existing_schema!r} to {PROVENANCE_SCHEMA_VERSION}; "
                f"use a new artifact root instead of resuming {path}"
            )
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


def _command_identity(
    command: list[str],
    timeout: int = 10,
    environment: dict[str, str] | None = None,
    output_normalizer: Callable[[str], str] | None = None,
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "command": command, "reason": str(error)}
    output = completed.stdout or ""
    if output_normalizer is not None:
        output = output_normalizer(output)
    limit = 65536
    return {
        "available": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "output": output[:limit],
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "output_truncated": len(output) > limit,
    }


_DYNAMIC_LOADER_ADDRESS = re.compile(
    r"[ \t]+\(0x[0-9a-fA-F]+\)[ \t]*(?=\r?$)", re.MULTILINE
)


def _normalize_dynamic_loader_output(output: str) -> str:
    """Remove per-process load addresses while retaining the loader closure."""
    return _DYNAMIC_LOADER_ADDRESS.sub("", output)


def _runtime_library_records(paths: dict[str, Path]) -> dict[str, object]:
    records = {}
    for label, path in paths.items():
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ValidationError(
                f"cannot resolve loaded {label} runtime {path}: {error}"
            ) from error
        records[label] = {
            "path": str(path),
            "resolved_path": str(resolved),
            "size": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    return records


def _runtime_libraries_from_ldd_output(output: str) -> dict[str, Path]:
    prefixes = {
        "hip-runtime": "libamdhip64.so",
        "hsa-runtime": "libhsa-runtime64.so",
    }
    matches: dict[str, set[Path]] = {label: set() for label in prefixes}
    for line in output.splitlines():
        match = re.match(r"^\s*(\S+)\s+=>\s+(\S+)", line)
        if match is None or match.group(2) == "not":
            continue
        soname, loaded = match.groups()
        for label, prefix in prefixes.items():
            if soname.startswith(prefix):
                matches[label].add(Path(loaded))
    ambiguous = {label: paths for label, paths in matches.items() if len(paths) > 1}
    if ambiguous:
        raise ValidationError(
            "dynamic loader reported multiple runtime libraries: "
            + "; ".join(
                f"{label}={','.join(sorted(str(path) for path in paths))}"
                for label, paths in sorted(ambiguous.items())
            )
        )
    return {label: next(iter(paths)) for label, paths in matches.items() if paths}


def _native_runtime_identity(
    executable: Path, environment: dict[str, str]
) -> dict[str, object]:
    ldd = shutil.which("ldd")
    if ldd is None:
        raise ValidationError("cannot verify native runtime closure: ldd is missing")
    identity = _command_identity(
        [ldd, str(executable)],
        timeout=TIMEOUT_SECONDS,
        environment=environment,
        output_normalizer=_normalize_dynamic_loader_output,
    )
    if not identity.get("available"):
        raise ValidationError(
            "cannot verify native runtime closure: "
            + json.dumps(identity, sort_keys=True)
        )
    libraries = _runtime_libraries_from_ldd_output(str(identity.get("output", "")))
    missing = {"hip-runtime", "hsa-runtime"} - set(libraries)
    if missing:
        raise ValidationError(
            "native runtime closure is missing " + ", ".join(sorted(missing))
        )
    return {
        "loader": identity,
        "loaded_runtime_libraries": _runtime_library_records(libraries),
    }


def _llama_runtime_identity(
    workspace: Path, target: str, workload: Workload
) -> dict[str, object]:
    runtime = _llama_runtime(workspace, target, workload.relative_path)
    ldd = shutil.which("ldd")
    if ldd is None:
        raise ValidationError("cannot verify llama runtime closure: ldd is missing")
    environment = _clean_environment(None, workload, None, target, workspace)
    try:
        completed = subprocess.run(
            [ldd, str(runtime.executable)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValidationError(
            f"cannot verify llama runtime closure with ldd: {error}"
        ) from error
    output = completed.stdout or ""
    if completed.returncode != 0:
        raise ValidationError(
            "cannot verify llama runtime closure: "
            f"ldd exited with status {completed.returncode}: {output.strip()}"
        )

    loaded_paths = []
    missing_names = []
    for line in output.splitlines():
        match = re.match(r"^\s*(\S+)\s+=>\s+(\S+)", line)
        if match is None or not match.group(1).startswith("libggml"):
            continue
        if match.group(2) == "not":
            missing_names.append(match.group(1))
            continue
        loaded_paths.append(Path(match.group(2)))
    if missing_names:
        raise ValidationError(
            "llama runtime closure has unresolved libraries: "
            + ", ".join(sorted(missing_names))
        )

    libraries = {}
    unmatched = list(loaded_paths)
    for label, expected in runtime.libraries.items():
        expected_real = expected.resolve(strict=True)
        match = next(
            (
                candidate
                for candidate in unmatched
                if candidate.resolve(strict=True) == expected_real
            ),
            None,
        )
        if match is None:
            raise ValidationError(
                "llama loader did not resolve the recorded runtime library "
                f"{label} from {expected}"
            )
        unmatched.remove(match)
        libraries[label] = {
            "recorded_path": str(expected),
            "loader_path": str(match),
            "resolved_path": str(expected_real),
        }
    if unmatched:
        raise ValidationError(
            "llama loader resolved unrecorded ggml libraries: "
            + ", ".join(str(path) for path in unmatched)
        )
    return {
        "kind": workload.kind,
        "identity_source": "validated dynamic-loader closure",
        "executable": str(runtime.executable),
        "libraries": libraries,
        "loaded_runtime_libraries": _runtime_library_records(
            _runtime_libraries_from_ldd_output(output)
        ),
    }


def _workload_runtime_identity(
    workspace: Path,
    target: str,
    workload: Workload,
    launcher: list[str] | None = None,
) -> dict[str, object]:
    if workload.kind == "llama":
        return _llama_runtime_identity(workspace, target, workload)
    if workload.kind in {"gtest", "native-executable", "rdna4-matmul"}:
        executable = _input_files(workspace, target, workload)["executable"]
        environment = _clean_environment(None, workload, None, target, workspace)
        return {
            "kind": workload.kind,
            "identity_source": "validated dynamic-loader closure",
            "executable": str(executable),
            **_native_runtime_identity(executable, environment),
        }
    if workload.kind != "pytorch":
        return {
            "kind": workload.kind,
            "identity_source": "hashed provenance files",
        }
    python = _pytorch_python(workspace)
    script = """
import json
import pathlib

import torch
import triton

torch.ones(1, device="cuda")
torch.cuda.synchronize()
mapped = sorted({
    fields[-1]
    for line in pathlib.Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    if len(fields := line.split()) >= 6 and fields[-1].startswith("/")
})
prefixes = {
    "hip-runtime": "libamdhip64.so",
    "hsa-runtime": "libhsa-runtime64.so",
}
runtime_libraries = {
    label: [path for path in mapped if pathlib.Path(path).name.startswith(prefix)]
    for label, prefix in prefixes.items()
}
print(json.dumps({
    "torch_version": torch.__version__,
    "torch_hip_version": torch.version.hip,
    "torch_file": torch.__file__,
    "triton_version": getattr(triton, "__version__", None),
    "triton_file": triton.__file__,
    "runtime_libraries": runtime_libraries,
}, sort_keys=True))
"""
    packages = _command_identity(
        _with_launcher(launcher or [], [str(python), "-c", script]),
        timeout=TIMEOUT_SECONDS,
        environment=_clean_environment(None, workload, None, target, workspace),
    )
    if not packages.get("available"):
        raise ValidationError(
            "cannot record required PyTorch/Triton runtime identity: "
            + json.dumps(packages, sort_keys=True)
        )
    try:
        package_document = json.loads(str(packages.get("output", "")))
    except json.JSONDecodeError as error:
        raise ValidationError(
            "PyTorch/Triton runtime identity did not emit valid JSON"
        ) from error
    required = {
        "torch_version",
        "torch_hip_version",
        "torch_file",
        "triton_version",
        "triton_file",
        "runtime_libraries",
    }
    if not isinstance(package_document, dict) or any(
        not package_document.get(name) for name in required
    ):
        raise ValidationError(
            "PyTorch/Triton runtime identity is missing required fields: "
            + ", ".join(
                sorted(
                    required
                    - set(
                        package_document if isinstance(package_document, dict) else ()
                    )
                )
            )
        )
    runtime_candidates = package_document["runtime_libraries"]
    if not isinstance(runtime_candidates, dict):
        raise ValidationError("PyTorch runtime identity has invalid library closure")
    runtime_paths = {}
    for label in ("hip-runtime", "hsa-runtime"):
        candidates = runtime_candidates.get(label)
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValidationError(
                f"PyTorch runtime identity needs exactly one loaded {label}: "
                f"{candidates!r}"
            )
        runtime_paths[label] = Path(candidates[0])
    return {
        "kind": workload.kind,
        "identity_source": "framework probe and loaded process mappings",
        "python_packages": packages,
        "package_document": package_document,
        "loaded_runtime_libraries": _runtime_library_records(runtime_paths),
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
    rocm_sdk = shutil.which("rocm-sdk")
    if rocm_sdk:
        commands["rocm-sdk"] = [rocm_sdk, "path", "--root"]
    amdclang = shutil.which("amdclang++")
    if amdclang:
        commands["amdclang++"] = [amdclang, "--version"]
    identities = {
        name: _command_identity(command) for name, command in commands.items()
    }
    identities["hook-linkage"] = _command_identity(
        [shutil.which("ldd") or "ldd", str(hook)],
        output_normalizer=_normalize_dynamic_loader_output,
    )
    return identities


def _source_identities(workspace: Path, workload: Workload) -> list[dict | None]:
    roots = [
        workspace / "iree-test-suites",
        workspace / "hip-moi",
        Path(__file__).resolve().parents[5],
    ]
    if workload.corpus == "rocjitsu-test-corpus":
        roots.append(workspace / "rocjitsu-test-corpus")
    if workload.kind == "tensile":
        paths = resolve_tensile_validation_paths(workspace)
        roots.append(paths.tensilelite)
    if workload.kind == "llama":
        roots.append(workspace / "rocjitsu-test-corpus")
    if workload.kind == "rdna4-matmul":
        roots.append(_rdna4_matmul_root(workspace))
    if workload.kind == "pytorch":
        roots.append(workspace / "pytorch")
    return [git_identity(root) for root in roots]



def _run_process(
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    timeout: int,
    *,
    active_processes: set[subprocess.Popen[str]] | None = None,
    active_processes_lock: threading.Lock | None = None,
) -> tuple[int, float, str]:
    if (active_processes is None) != (active_processes_lock is None):
        raise ValidationError("active process registry and lock must be paired")
    start = time.monotonic()
    process = subprocess.Popen(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if active_processes is not None:
        assert active_processes_lock is not None
        with active_processes_lock:
            active_processes.add(process)
    try:
        try:
            output, _ = process.communicate(timeout=timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
            _stop_process_group(process, signal.SIGTERM)
            try:
                output, _ = process.communicate(
                    timeout=PROCESS_TERMINATION_GRACE_SECONDS
                )
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
    finally:
        if active_processes is not None:
            assert active_processes_lock is not None
            with active_processes_lock:
                active_processes.discard(process)


def _run_process_batch(
    runs: list[tuple[list[str], dict[str, str], Path, int]],
    max_parallelism: int,
) -> list[tuple[int, float, str]]:
    if not runs:
        return []
    if max_parallelism < 1 or max_parallelism > len(runs):
        raise ValidationError(
            "process-batch parallelism must be positive and no larger than its "
            "run count"
        )
    if max_parallelism == 1:
        return [_run_process(*run) for run in runs]
    active_processes: set[subprocess.Popen[str]] = set()
    active_processes_lock = threading.Lock()
    previous_handlers: dict[int, signal.Handlers] = {}

    def terminate_active_processes(signum: int, _frame: object) -> None:
        with active_processes_lock:
            processes = tuple(active_processes)
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, terminate_active_processes)
    try:
        with ThreadPoolExecutor(max_workers=max_parallelism) as executor:
            # executor.map preserves declaration order even when a later shard
            # finishes first, keeping commands, logs, coverage, and oracle evidence
            # joined by one stable index in the retained result.
            return list(
                executor.map(
                    lambda run: _run_process(
                        *run,
                        active_processes=active_processes,
                        active_processes_lock=active_processes_lock,
                    ),
                    runs,
                )
            )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


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


def _workload_provenance_path(artifact_root: Path, workload: Workload) -> Path:
    return artifact_root / workload.id / "provenance.json"



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
    collect_gtest_device_timing: bool = False,
    minimum_device_timed_aggregate_ms: float | None = None,
) -> dict:
    profile_id = profile or "baseline"
    row_dir = row_dir_override or (
        artifact_root / workload.id / phase / (row_label or profile_id)
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
    if collect_gtest_device_timing and (
        phase != "overhead" or workload.warm_timing_mode != "device-gtest"
    ):
        raise ValidationError(
            "GTest device timing requires an overhead row with native event support"
        )
    if collect_gtest_device_timing and workload.device_timing_warmup_iterations < 0:
        raise ValidationError("GTest device timing warmup count must be nonnegative")
    if minimum_device_timed_aggregate_ms is not None and (
        not math.isfinite(minimum_device_timed_aggregate_ms)
        or minimum_device_timed_aggregate_ms <= 0.0
    ):
        raise ValidationError("minimum device timed aggregate must be positive")
    logs = []
    commands = []
    recorded_environment = None
    returncodes = []
    elapsed_seconds = []
    qwen_json_paths = []
    process_runs = []
    resolved_workload = _resolved_workload(target, workload)
    if resolved_workload.tensile_exact_problem_size_shards and repetitions != 1:
        raise ValidationError(
            "Tensile problem-size sharding requires one outer validation repetition"
        )
    for index in range(repetitions):
        benchmark_path = row_dir / f"benchmark-{index}.json"
        row_commands = _workload_commands(
            workspace,
            target,
            workload,
            phase,
            benchmark_path,
            inner_repetitions_override,
        )
        for row_command in row_commands:
            run_index = len(process_runs)
            command = _with_launcher(launcher or [], row_command)
            log_path = row_dir / f"run-{run_index}.log"
            environment = _run_environment(
                profile, workload, hook, target, phase, workspace
            )
            if collect_gtest_device_timing:
                environment[HIP_MOI_GPU_BENCHMARK_ITERATIONS_ENV] = str(
                    inner_repetitions_override or 1
                )
                environment[HIP_MOI_GPU_BENCHMARK_WARMUP_ITERATIONS_ENV] = str(
                    workload.device_timing_warmup_iterations
                )
            if profile is not None and retain_code_objects:
                dump_dir = row_dir / f"code-objects-{run_index}"
                dump_dir.mkdir()
                environment["RJ_CONSAN_DUMP_DIR"] = str(dump_dir.resolve())
            commands.append(command)
            process_runs.append((command, environment, log_path, timeout))
            recorded_environment = _controlled_environment(environment)
            # The retained code-object inventory records relocatable per-run dump
            # directories. Do not also publish the execution machine's absolute
            # directory through the compatibility environment summary.
            recorded_environment.pop("RJ_CONSAN_DUMP_DIR", None)
        if workload.kind == "qwen" and phase == "overhead":
            qwen_json_paths.append(benchmark_path)

    parallelism = (
        resolved_workload.tensile_shard_parallelism
        if resolved_workload.tensile_exact_problem_size_shards
        else 1
    )
    for returncode, elapsed, output in _run_process_batch(
        process_runs, parallelism
    ):
        returncodes.append(returncode)
        elapsed_seconds.append(elapsed)
        logs.append(output)

    timing = None
    timing_samples = None
    measurement_runs = None
    if phase == "overhead" and all(code == 0 for code in returncodes):
        if collect_gtest_device_timing:
            expected_dispatches = inner_repetitions_override or 1
            parsed = [
                _gtest_device_measurement(log, workload.id, expected_dispatches)
                for log in logs
            ]
            timing_samples = {"target-dispatch:device": [value for value, _ in parsed]}
            measurement_runs = [
                {"target-dispatch": measurement} for _, measurement in parsed
            ]
        elif workload.kind == "qwen":
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
            if workload.self_timed_device_minimum_ms is not None:
                measurement_runs = [
                    _json_measurements(log, workload.kind.capitalize()) for log in logs
                ]
        elif workload.kind == "native-executable":
            timing_samples = {
                "process": [elapsed * 1_000.0 for elapsed in elapsed_seconds]
            }
        else:
            timing_samples = _gtest_timing_samples(logs)
        timing = {
            key: statistics.median(values) for key, values in timing_samples.items()
        }

    device_timed_aggregates_ms = None
    timing_acceptance_reasons = []
    if minimum_device_timed_aggregate_ms is not None:
        if repetitions != 1:
            raise ValidationError(
                "device timed aggregate validation requires one outer process"
            )
        if workload.warm_timing_mode in {"device-fixed", "device-gtest"}:
            aggregates = {}
            if isinstance(measurement_runs, list) and len(measurement_runs) == 1:
                for name, measurement in measurement_runs[0].items():
                    if isinstance(measurement, dict):
                        value = measurement.get("timed_aggregate_ms")
                        if (
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(float(value))
                            and value > 0.0
                        ):
                            aggregates[f"{name}:device"] = float(value)
        else:
            aggregates = {
                mode: sum(values)
                for mode, values in (timing_samples or {}).items()
                if mode.endswith(":device")
            }
        device_timed_aggregates_ms = aggregates
        if not aggregates:
            timing_acceptance_reasons.append("no GPU timed aggregate was recorded")
        for mode, aggregate_ms in aggregates.items():
            if aggregate_ms < minimum_device_timed_aggregate_ms:
                timing_acceptance_reasons.append(
                    f"{mode} timed aggregate {aggregate_ms} ms is below "
                    f"{minimum_device_timed_aggregate_ms} ms"
                )

    coverage = None
    coverage_runs = None
    if profile is not None and logs:
        coverage_runs = [
            _coverage_summary(log, profile=profile)
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
        "phase": phase,
        "target": target,
        "commands": commands,
        "command_batch": {
            "max_parallelism": parallelism,
            "processes": len(commands),
            "tensile_exact_problem_size_shards": [
                [list(size) for size in shard]
                for shard in resolved_workload.tensile_exact_problem_size_shards
            ],
            "tensile_expected_numeric_rows_per_shard": list(
                resolved_workload.tensile_expected_numeric_rows_per_shard
            ),
            "tensile_expected_source_exact_problem_size_blocks": [
                [list(size) for size in block]
                for block in resolved_workload.tensile_expected_source_exact_problem_size_blocks
            ],
            "tensile_expected_client_passes_per_shard": list(
                resolved_workload.tensile_expected_client_passes_per_shard
            ),
        },
        "environment": recorded_environment,
        "returncodes": returncodes,
        "elapsed_seconds": elapsed_seconds,
        "timeout_seconds": timeout,
        "repetition_policy": {
            "empirical_row_schema_version": 2,
            "outer_processes": repetitions,
            "inner_repetitions_override": inner_repetitions_override,
            "discarded_first_timing_sample": discard_first_timing_sample,
            "discarded_timing_samples_per_process": int(discard_first_timing_sample),
            "retained_code_objects": retain_code_objects,
            "collected_structural_metrics": collect_structural_metrics,
            "collected_gtest_device_timing": collect_gtest_device_timing,
            "minimum_device_timed_aggregate_ms": minimum_device_timed_aggregate_ms,
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
        "device_timed_aggregates_ms": device_timed_aggregates_ms,
        "timing_acceptance_reasons": timing_acceptance_reasons,
        "structural_metrics_runs": structural_metrics_runs,
        "coverage": coverage,
        "coverage_runs": coverage_runs,
        "gtest_test_counts": gtest_test_counts,
        "accepted": runtime_accepted and not timing_acceptance_reasons,
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
    device_workload_only: bool = False,
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
                and (not device_workload_only or mode.endswith(":device"))
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
    device_workload_only: bool = False,
    qualifying: bool = True,
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
            device_workload_only=device_workload_only,
            prefix=metric_prefix,
        )
        for result in rows
    ]
    metric_schemas = [set(row) for row in row_metrics]
    expected_metrics = metric_schemas[0]
    if qualifying and not expected_metrics:
        reasons.append("rows have no timing metrics")
    if any(schema != expected_metrics for schema in metric_schemas[1:]):
        if qualifying:
            reasons.append(
                "row timing metric schemas differ: "
                + "; ".join(
                    f"row-{index}={sorted(schema)}"
                    for index, schema in enumerate(metric_schemas)
                )
            )
        else:
            expected_metrics = set.intersection(*metric_schemas)
    metrics = {}
    denominator = len(order) + 1
    rows_accepted = not reasons
    for metric in sorted(expected_metrics if not reasons else set()):
        before = row_metrics[0][metric]
        after = row_metrics[-1][metric]
        mean_baseline = statistics.mean((before, after))
        drift = abs(after - before) / mean_baseline if mean_baseline > 0.0 else math.inf
        metric_accepted = rows_accepted and (
            not qualifying or drift <= baseline_drift_limit
        )
        if qualifying and drift > baseline_drift_limit:
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
            "qualifying": qualifying,
            "baseline_before_ms": before,
            "baseline_after_ms": after,
            "baseline_drift_fraction": drift,
            "profiles": profiles,
        }
    metric_acceptance = [
        metric["accepted"]
        for metric in metrics.values()
        if metric.get("qualifying") is True
    ]
    return {
        "round": round_index,
        "profile_order": order,
        "rows_accepted": rows_accepted,
        "usable": rows_accepted and (any(metric_acceptance) if qualifying else True),
        "fully_accepted": rows_accepted
        and (
            bool(metric_acceptance) and all(metric_acceptance) if qualifying else True
        ),
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
    metric_acceptance = [
        metric["accepted"]
        for metric in metrics.values()
        if metric.get("qualifying") is True
    ]
    rows_accepted = all(
        bool(schedule["rows_accepted"]) for schedule in schedules.values()
    )
    return {
        "round": round_index,
        "profile_order": order,
        "rows_accepted": rows_accepted,
        "usable": rows_accepted and any(metric_acceptance),
        "fully_accepted": (
            rows_accepted and bool(metric_acceptance) and all(metric_acceptance)
        ),
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
        {
            metric
            for round_result in rounds
            for metric, value in round_result["metrics"].items()
            if value.get("qualifying") is True
        }
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
            for schedule in round_result.get("schedules", {}).values():
                if schedule.get("collects_structural_metrics") is not True:
                    continue
                profile_structural = schedule.get("structural_metrics", {}).get(profile)
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
    profile_policies = {
        profile: {"detector": "REVIEW_REQUIRED", "oracle": "any"}
        for profile in PROFILE_IDS
    }
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
    if not _doctor(workspace, target, (workload.id,), args.launcher)["ok"]:
        raise ValidationError("workspace doctor failed; run the doctor subcommand")
    root = args.artifact_root.resolve() / workload.id / "inventory"
    root.mkdir(parents=True, exist_ok=False)
    provenance = _write_provenance(
        workspace, target, workload, root, args.launcher
    )
    hook = _hook_path(workspace)
    command = _fault_workload_command(
        workspace, target, workload, root / "unused.json"
    )
    command = _with_launcher(args.launcher, command)
    family_runs = []
    aggregate_records = {"sites": set(), "sequences": set(), "destinations": set()}
    for family in _fault_families(target, workload):
        environment = _clean_environment(
            "supercollider", workload, hook, target, workspace
        )
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
) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError("fault spec has unsupported schema_version")
    if document.get("target") != target:
        raise ValidationError("fault spec target does not match --target")
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
            or reach_witness.get("kind") != "reviewed-unconditional-final-isa"
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
    workspace: Path,
) -> dict[str, str]:
    environment = _clean_environment(profile, workload, hook, target, workspace)
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
            _load_fault(path, target, workload, fault["id"])
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
    command = _fault_workload_command(
        workspace,
        target,
        workload,
        Path("$ARTIFACT_ROOT") / workload.id / "fault" / "unused.json",
    )
    if workload.kind == "sharktank":
        command.append("--allow-oracle-failure")
    expectations = []
    for profile in profiles:
        policy, trials = _fault_trials(fault, profile)
        trial_audits = []
        if policy.get("disposition") != "not-applicable":
            for index, trial in enumerate(trials):
                environment = _fault_trial_environment(
                    profile,
                    workload,
                    hook,
                    target,
                    fault,
                    policy,
                    trial,
                    workspace,
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
) -> dict:
    spec_document = None
    spec_metadata = None
    if spec_path is not None:
        spec_path = spec_path.resolve()
        spec_document = json.loads(spec_path.read_text(encoding="utf-8"))
        spec_metadata = {
            "path": str(spec_path),
            "sha256": sha256_file(spec_path),
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
                profile: output_root / phase / profile
                for profile in profiles
            }
            commands[phase] = {
                "payload_argv": _workload_command(
                    workspace,
                    target,
                    workload,
                    phase,
                    output_root / phase / "$PROFILE" / "benchmark-0.json",
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
            environment = _run_environment(
                profile,
                workload,
                _hook_path(workspace),
                target,
                "clean",
                workspace,
            )
            inherited = _clean_environment(
                None, workload, _hook_path(workspace), target, workspace
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
                    "clean_result_phase": "clean",
                    "clean_artifact_root": str(output_root / "clean" / profile),
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
            fault_source = "reviewed-spec"
            faults = _faults_from_spec(spec_path, target, workload)
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
                tuned_settings = [
                    setting["name"]
                    for setting in expectation["policy_settings"]
                    if setting["usability_exception"]
                ]
                policy_unsets = [
                    setting["name"] for setting in expectation["policy_unsets"]
                ]
                if tuned_settings or policy_unsets:
                    fault_policy_exceptions.append(
                        {
                            "workload": workload["id"],
                            "fault": fault["id"],
                            "profile": expectation["profile"],
                            "settings": tuned_settings,
                            "unsets": policy_unsets,
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
        print(f"fault expectations: reviewed {document['fault_spec']['path']}")
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


def _load_resumable_fault_result(
    result_path: Path,
    *,
    name: str,
    command: list[str],
    environment: dict[str, str],
    identities: list[str],
    workload: Workload,
    profile: str,
    fault: dict,
) -> dict:
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(
            f"cannot resume fault row {result_path}: {error}"
        ) from error
    mismatches = []
    expected_scalars = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "state": "complete",
        "name": name,
    }
    mismatches.extend(
        f"{key}={result.get(key)!r}, expected={expected!r}"
        for key, expected in expected_scalars.items()
        if result.get(key) != expected
    )
    if result.get("command") != command:
        mismatches.append("payload command changed")
    if result.get("site_identities") != identities:
        mismatches.append("fault site identities changed")
    expected_spec = {
        "corpus": workload.corpus,
        "workload": workload.id,
        "flavor": PROFILES[profile].flavor,
        "engine": PROFILES[profile].engine,
        "fault_family": fault["family"],
        "row_role": "fault",
    }
    actual_spec = result.get("spec")
    if not isinstance(actual_spec, dict):
        mismatches.append("fault row has no run specification")
    else:
        mismatches.extend(
            f"spec.{key}={actual_spec.get(key)!r}, expected={expected!r}"
            for key, expected in expected_spec.items()
            if actual_spec.get(key) != expected
        )
    expected_environment = _controlled_environment(environment)
    actual_environment = result.get("environment")
    if not isinstance(actual_environment, dict):
        mismatches.append("fault row has no controlled environment")
    else:
        actual_controlled = _controlled_environment(actual_environment)
        # The runner owns this row-local output path rather than the campaign.
        actual_controlled.pop("RJ_CONSAN_DUMP_DIR", None)
        if actual_controlled != expected_environment:
            mismatches.append("controlled environment changed")
    if mismatches:
        raise ValidationError(
            f"fault row conflicts with resume request {result_path}: "
            + "; ".join(mismatches)
        )
    return result


def _fault(args: argparse.Namespace) -> int:
    selection = _resolve_workload_selection(args, allow_all=False)
    target = selection.target
    workload = _resolved_workload(target, selection.require_workload())
    timeout = args.timeout if args.timeout is not None else workload.run_timeout_seconds
    workspace = _workspace_from_environment()
    if not _doctor(workspace, target, (workload.id,), args.launcher)["ok"]:
        raise ValidationError("workspace doctor failed; run the doctor subcommand")
    if not args.allow_destructive:
        raise ValidationError("fault execution requires --allow-destructive")
    spec_path = args.spec.resolve()
    fault = _load_fault(spec_path, target, workload, args.fault)
    profiles = PROFILE_IDS if args.profile == "all" else (args.profile,)
    launcher = args.launcher
    hook = _hook_path(workspace)
    fault_root = args.artifact_root.resolve() / workload.id / "faults" / fault["id"]
    fault_root.mkdir(parents=True, exist_ok=args.resume)
    provenance = _write_provenance(
        workspace, target, workload, fault_root, launcher
    )
    root = fault_root / "rows"
    root.mkdir(exist_ok=args.resume)
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
                profile,
                workload,
                hook,
                target,
                fault,
                policy,
                trial,
                workspace,
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
            command = _fault_workload_command(
                workspace, target, workload, root / "unused.json"
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
                str(timeout),
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
                str(_corpus_root(workspace, workload.corpus)),
                "--hash-file",
                f"hook={hook}",
            ]
            for key, value in _controlled_environment(environment).items():
                invocation.extend(["--env", f"{key}={value}"])
            for identity in identities:
                invocation.extend(["--site-id", identity])
            invocation.extend(["--", *command])
            child_environment = _clean_environment(
                None, workload, hook, target, workspace
            )
            child_environment["CTEST_PARALLEL_LEVEL"] = "1"
            result_path = root / name / "result.json"
            if args.resume and result_path.is_file():
                result = _load_resumable_fault_result(
                    result_path,
                    name=name,
                    command=command,
                    environment=environment,
                    identities=identities,
                    workload=workload,
                    profile=profile,
                    fault=fault,
                )
            else:
                if args.resume and result_path.parent.exists():
                    raise ValidationError(
                        "cannot resume incomplete fault row; use a new artifact "
                        f"root instead of overwriting {result_path.parent}"
                    )
                subprocess.run(invocation, env=child_environment, check=False)
                if result_path.is_file():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    result = None
            if result is None:
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
        reach_outcomes = {}
        for row in reached_rows:
            outcome = str(row.get("reach_outcome"))
            reach_outcomes[outcome] = reach_outcomes.get(outcome, 0) + 1
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
                "reach_outcomes": reach_outcomes,
                "detections": detected,
                "trials": len(profile_rows),
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
    collect_gtest_device_timing: bool,
    minimum_device_timed_aggregate_ms: float | None,
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
        "phase": phase,
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
        "empirical_row_schema_version": 2,
        "outer_processes": 1,
        "inner_repetitions_override": inner_repetitions_override,
        "discarded_first_timing_sample": discard_first_timing_sample,
        "discarded_timing_samples_per_process": int(discard_first_timing_sample),
        "retained_code_objects": retain_code_objects,
        "collected_structural_metrics": collect_structural_metrics,
        "collected_gtest_device_timing": collect_gtest_device_timing,
        "minimum_device_timed_aggregate_ms": minimum_device_timed_aggregate_ms,
    }
    repetition_policy = result.get("repetition_policy")
    if (
        not isinstance(repetition_policy, dict)
        or repetition_policy.get("empirical_row_schema_version") != 2
    ):
        raise ValidationError(
            f"empirical row predates or lacks empirical row schema version 2: "
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
    collect_gtest_device_timing: bool = False,
    minimum_device_timed_aggregate_ms: float | None = None,
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
            collect_gtest_device_timing=collect_gtest_device_timing,
            minimum_device_timed_aggregate_ms=minimum_device_timed_aggregate_ms,
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
        collect_gtest_device_timing=collect_gtest_device_timing,
        minimum_device_timed_aggregate_ms=minimum_device_timed_aggregate_ms,
    )


def _empirical_supports_warm_timing(target: str, workload: Workload) -> bool:
    return target not in SINGLE_REPETITION_TARGETS and workload.warm_timing_mode in {
        "host-json",
        "device-json",
        "device-fixed",
        "device-gtest",
    }


def _empirical_uses_device_timing(workload: Workload) -> bool:
    return workload.warm_timing_mode in {
        "device-json",
        "device-fixed",
        "device-gtest",
    }


def _empirical_device_timed_minimum(workload: Workload) -> float:
    minimum = (
        workload.empirical_device_timed_minimum_ms
        if workload.empirical_device_timed_minimum_ms is not None
        else EMPIRICAL_MINIMUM_TIMED_MS
    )
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValidationError("empirical device timed minimum must be positive")
    return minimum


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
    qualifying_timing = (
        {
            mode: value
            for mode, value in timing.items()
            if isinstance(mode, str) and mode.endswith(":device")
        }
        if _empirical_uses_device_timing(workload)
        else timing
    )
    if not qualifying_timing:
        raise ValidationError("warm empirical calibration has no qualifying timing")
    values = [float(value) for value in qualifying_timing.values()]
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
            "calibration_timing_median_ms": qualifying_timing,
            "calibration_timed_aggregate_ms": float(aggregate_ms),
            "timed_inner_repetitions": fixed_iterations,
            "command_inner_repetitions": fixed_iterations,
            "discard_first_timing_sample": False,
        }
    calibration_floor = dict(qualifying_timing)
    timing_samples = calibration.get("timing_samples_ms")
    if isinstance(timing_samples, dict):
        for mode in qualifying_timing:
            samples = timing_samples.get(mode)
            if isinstance(samples, list) and samples:
                sample_values = [float(value) for value in samples]
                if any(
                    not math.isfinite(value) or value <= 0.0 for value in sample_values
                ):
                    raise ValidationError(
                        "warm empirical calibration samples must be finite and "
                        "positive"
                    )
                calibration_floor[mode] = min(sample_values)
    device_minimum = _empirical_device_timed_minimum(workload)
    if (
        not math.isfinite(workload.device_timing_aggregate_headroom)
        or workload.device_timing_aggregate_headroom < 1.0
    ):
        raise ValidationError("device timing aggregate headroom must be at least one")
    timed_repetitions = max(
        1,
        math.ceil(
            device_minimum
            * workload.device_timing_aggregate_headroom
            / min(calibration_floor.values())
        ),
    )
    if (
        workload.device_timing_max_iterations is not None
        and timed_repetitions > workload.device_timing_max_iterations
    ):
        raise ValidationError(
            "warm empirical calibration exceeds the workload's device iteration "
            f"limit: required={timed_repetitions}, "
            f"maximum={workload.device_timing_max_iterations}"
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
        "kind": (
            "warm-device-gtest"
            if workload.warm_timing_mode == "device-gtest"
            else (
                "warm-device-json"
                if workload.warm_timing_mode == "device-json"
                else "warm-host"
            )
        ),
        "minimum_timed_aggregate_ms": device_minimum,
        "timed_aggregate_headroom": workload.device_timing_aggregate_headroom,
        "calibration_timing_median_ms": qualifying_timing,
        "calibration_timing_floor_ms": calibration_floor,
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
        "protocol": f"consan-{target}-empirical-v3",
        "target": target,
        "workload": workload.id,
        "profiles": list(profiles),
        "admission_policy": "time every admitted requested profile",
        "required_accepted_rounds": args.rounds,
        "max_rounds": max_rounds,
        "randomization_seed": args.seed,
        "baseline_drift_limit": args.baseline_drift_limit,
        "bootstrap_resamples": args.bootstrap_resamples,
        "minimum_timed_aggregate_ms": (
            _empirical_device_timed_minimum(workload)
            if _empirical_uses_device_timing(workload)
            else EMPIRICAL_MINIMUM_TIMED_MS
        ),
        "timing_acceptance_source": (
            "gpu-timestamps"
            if _empirical_uses_device_timing(workload)
            else "host-timing"
        ),
        "process_timing_role": (
            "secondary-diagnostic"
            if _empirical_uses_device_timing(workload)
            else "qualifying"
        ),
        "maximum_inner_repetitions": EMPIRICAL_MAX_INNER_REPETITIONS,
        "workload_maximum_device_iterations": workload.device_timing_max_iterations,
        "device_timing_calibration_iterations": (
            workload.device_timing_calibration_iterations
        ),
        "device_timing_aggregate_headroom": workload.device_timing_aggregate_headroom,
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
    if target not in {"gfx950", "gfx1201"}:
        raise ValidationError(
            "the empirical study command requires physical gfx950 or gfx1201"
        )
    workload = _resolved_workload(target, selection.require_workload())
    workspace = _workspace_from_environment()
    timeout = args.timeout if args.timeout is not None else workload.run_timeout_seconds
    doctor = _doctor(workspace, target, (workload.id,), args.launcher)
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
        args.launcher,
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
    baseline_accepted = admission_row_acceptance["baseline"]
    admitted_profiles = tuple(
        profile for profile in profiles if admission_row_acceptance[profile]
    )
    rejected_profiles = tuple(
        profile for profile in profiles if not admission_row_acceptance[profile]
    )
    timing_eligible = baseline_accepted and bool(admitted_profiles)
    admission = {
        "accepted": timing_eligible,
        "all_requested_profiles_accepted": not rejected_profiles,
        "admitted_profiles": list(admitted_profiles),
        "rejected_profiles": list(rejected_profiles),
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
    if not timing_eligible:
        reasons = []
        if not baseline_accepted:
            reasons.append("baseline clean admission rejected")
        if not admitted_profiles:
            reasons.append("no requested profile passed clean admission")
        campaign = {
            **config,
            "admission": admission,
            "timed_profiles": [],
            "rounds": [],
            "summary": {
                "schema_version": EMPIRICAL_CAMPAIGN_SCHEMA_VERSION,
                "accepted": False,
                "reasons": reasons,
                "requested_profiles": list(profiles),
                "timed_profiles": [],
                "rejected_profiles": list(rejected_profiles),
            },
        }
        atomic_write_json(campaign_root / "campaign.json", campaign)
        print(json.dumps(campaign, indent=2, sort_keys=True))
        return 1

    requested_profiles = profiles
    profiles = admitted_profiles

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
            inner_repetitions_override=(workload.device_timing_calibration_iterations),
            discard_first_timing_sample=(workload.kind == "pytorch"),
            collect_gtest_device_timing=(workload.warm_timing_mode == "device-gtest"),
            minimum_device_timed_aggregate_ms=(workload.self_timed_device_minimum_ms),
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
        order_seed = _bootstrap_stream_seed(
            args.seed, workload.id, "profile-order", str(round_index)
        )
        random.Random(order_seed).shuffle(order)
        round_root = campaign_root / "rounds" / f"round-{round_index:03d}"

        def run_schedule(
            name: str,
            schedule_inner_repetitions: int | None,
            discard_first: bool,
            include_process: bool,
            include_workload: bool,
            collect_structural_metrics: bool,
            qualifying: bool,
            device_workload_only: bool,
            collect_gtest_device_timing: bool,
            minimum_device_timed_aggregate_ms: float | None,
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
                collect_gtest_device_timing=collect_gtest_device_timing,
                minimum_device_timed_aggregate_ms=(minimum_device_timed_aggregate_ms),
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
                    collect_structural_metrics=collect_structural_metrics,
                    collect_gtest_device_timing=collect_gtest_device_timing,
                    minimum_device_timed_aggregate_ms=(
                        minimum_device_timed_aggregate_ms
                    ),
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
                collect_gtest_device_timing=collect_gtest_device_timing,
                minimum_device_timed_aggregate_ms=(minimum_device_timed_aggregate_ms),
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
                device_workload_only=device_workload_only,
                qualifying=qualifying,
                metric_prefix=name,
            )
            schedule["structural_metrics"] = {}
            schedule["collects_structural_metrics"] = collect_structural_metrics
            if collect_structural_metrics:
                for profile, result in profile_results.items():
                    try:
                        schedule["structural_metrics"][profile] = (
                            _empirical_structural_totals(result)
                        )
                    except ValidationError as error:
                        schedule["reasons"].append(
                            f"{profile}: structural metrics rejected: {error}"
                        )
                        schedule["rows_accepted"] = False
                        schedule["usable"] = False
                        schedule["fully_accepted"] = False
            return schedule

        schedules = {
            "cold": run_schedule(
                "cold",
                1,
                False,
                True,
                workload.self_timed_device_minimum_ms is None,
                True,
                not _empirical_uses_device_timing(workload),
                False,
                False,
                None,
            ),
        }
        if _empirical_supports_warm_timing(target, workload):
            schedules["warm"] = run_schedule(
                "warm",
                inner_repetitions,
                discard_first_timing_sample,
                False,
                True,
                False,
                True,
                _empirical_uses_device_timing(workload),
                workload.warm_timing_mode == "device-gtest",
                (
                    float(timing_protocol["minimum_timed_aggregate_ms"])
                    if _empirical_uses_device_timing(workload)
                    else None
                ),
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
        summary["requested_profiles"] = list(requested_profiles)
        summary["timed_profiles"] = list(profiles)
        summary["rejected_profiles"] = list(rejected_profiles)
        campaign = {
            **config,
            "admission": admission,
            "timed_profiles": list(profiles),
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
