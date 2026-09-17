#!/usr/bin/env python3
"""Benchmark bounded Aorta workloads natively and under every ConSan mode."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

from consan_coverage_gate import CoverageParseError, parse_coverage_evidence
from consan_validation_catalog import (
    CONTROLLED_ENV_PREFIX,
    HSA_TOOL_ENVIRONMENT,
    PROFILES,
    SOFTWARE_MODEL_ENVIRONMENT,
)
from consan_validation_support import SITE_KINDS

SCHEMA_VERSION = 3
AORTA_DIR_ENV = "CONSAN_BENCHMARK_AORTA_DIR"
PYTHON_ENV = "CONSAN_BENCHMARK_PYTHON"
HOOK_ENV = "CONSAN_BENCHMARK_HOOK"
ROCPROFV3_ENV = "CONSAN_BENCHMARK_ROCPROFV3"
HIPBLASLT_BENCH_ENV = "CONSAN_BENCHMARK_HIPBLASLT_BENCH"
RESULT_MARKER = "CONSAN_BENCHMARK_RESULT="
SUPPORTED_TARGETS = ("gfx950", "gfx1201")
MODE_LABELS = {
    "default": "Default Mode",
    "default-high": "Default Mode (high)",
    "supercollider": "SuperCollider",
}
PROFILE_IDS = tuple(MODE_LABELS)


class BenchmarkError(RuntimeError):
    """The benchmark contract could not be satisfied."""


@dataclass(frozen=True)
class Workload:
    id: str
    description: str
    primary_metric: str
    config: dict[str, Any]
    payload: str = "aorta"


def _model_config(*, num_experts: int = 1) -> dict[str, Any]:
    return {
        "kind": "decoder_transformer",
        "hidden_size": 512,
        "num_layers": 4,
        "num_heads": 8,
        "ffn_size": 2048,
        "vocab_size": 32_000,
        "num_experts": num_experts,
    }


WORKLOADS = (
    Workload(
        id="pytorch-dense-prefill",
        description="PyTorch synthetic dense prefill (32-token prompt)",
        primary_metric="prefill_latency_ms",
        config={
            "mode": "offline_batch",
            "device": "cuda",
            "dtype": "bfloat16",
            "seed": 1234,
            "warmup_steps": 0,
            "steps": 1,
            "model": _model_config(),
            "request": {"batch_size": 1, "prompt_len": 32, "generate_tokens": 0},
            "serving": {"kv_cache": True},
            "checks": {
                "fail_on_nan_logits": True,
                "fail_on_nonfinite_output": True,
                "compare_logits_checksum": True,
            },
        },
    ),
    Workload(
        id="pytorch-synthetic-decode",
        description="PyTorch synthetic dense decode (one continuous-batch tick)",
        primary_metric="decode_latency_ms",
        config={
            "mode": "continuous_batch",
            "device": "cuda",
            "dtype": "bfloat16",
            "seed": 1234,
            "warmup_steps": 0,
            "steps": 1,
            "model": _model_config(),
            "request": {"batch_size": 1, "prompt_len": 16, "generate_tokens": 1},
            "serving": {
                "kv_cache": True,
                "continuous_batch": {
                    "enabled": True,
                    "max_active_requests": 1,
                    "arrival_pattern": "fixed",
                },
            },
            "checks": {
                "fail_on_nan_logits": True,
                "fail_on_nonfinite_output": True,
                "compare_logits_checksum": True,
            },
        },
    ),
    Workload(
        id="pytorch-top1-moe-prefill",
        description="PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt)",
        primary_metric="prefill_latency_ms",
        config={
            "mode": "offline_batch",
            "device": "cuda",
            "dtype": "bfloat16",
            "seed": 1234,
            "warmup_steps": 0,
            "steps": 1,
            "model": _model_config(num_experts=4),
            "request": {"batch_size": 1, "prompt_len": 16, "generate_tokens": 0},
            "serving": {"kv_cache": True},
            "checks": {
                "fail_on_nan_logits": True,
                "fail_on_nonfinite_output": True,
                "compare_logits_checksum": True,
            },
        },
    ),
    Workload(
        id="gluon-shared-roundtrip",
        description="Gluon verified shared-memory round trip (1024 elements)",
        primary_metric="latency_ms",
        config={"size": 1024},
        payload="gluon",
    ),
    Workload(
        id="hipblaslt-tensile-gemm",
        description="hipBLASLt/Tensile verified FP16 GEMM (512×512×512)",
        primary_metric="latency_ms",
        config={"m": 512, "n": 512, "k": 512, "iterations": 10},
        payload="hipblaslt",
    ),
)


GFX950_ADDITIONS = tuple(
    Workload(id, description, "latency_ms", config, payload="tokenspeed")
    for id, description, config in (
        (
            "tokenspeed-bf16-gemm-mediumm",
            "TokenSpeed Gluon BF16 GEMM medium-M (128×4096×4096)",
            {"operation": "gemm", "variant": "mediumm", "m": 128, "n": 4096, "k": 4096},
        ),
        (
            "tokenspeed-bf16-gemm-largem",
            "TokenSpeed Gluon BF16 GEMM large-M (4096×4096×4096)",
            {"operation": "gemm", "variant": "largem", "m": 4096, "n": 4096, "k": 4096},
        ),
        (
            "tokenspeed-attention-prefill",
            "TokenSpeed Gluon attention prefill",
            {"operation": "attention-prefill"},
        ),
        (
            "tokenspeed-attention-decode",
            "TokenSpeed Gluon attention decode",
            {"operation": "attention-decode"},
        ),
        (
            "tokenspeed-qwen-prefill",
            "TokenSpeed Qwen3-0.6B prefill",
            {"operation": "qwen-prefill"},
        ),
        (
            "tokenspeed-qwen-decode",
            "TokenSpeed Qwen3-0.6B real cached decode",
            {"operation": "qwen-decode"},
        ),
        (
            "tokenspeed-fp8-blockscale-gemm",
            "TokenSpeed Triton FP8 block-scaled GEMM",
            {"operation": "fp8-gemm", "m": 16, "n": 256, "k": 256},
        ),
        ("tokenspeed-bf16-moe", "TokenSpeed Gluon BF16 MoE", {"operation": "moe"}),
    )
)


def _target_workloads(target: str) -> tuple[Workload, ...]:
    return WORKLOADS + (GFX950_ADDITIONS if target == "gfx950" else ())


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=SUPPORTED_TARGETS, required=True)
    parser.add_argument(
        "--aorta-dir",
        type=Path,
        default=os.environ.get(AORTA_DIR_ENV),
        help=f"external Aorta checkout (default: ${AORTA_DIR_ENV})",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=os.environ.get(PYTHON_ENV, sys.executable),
        help=f"Python with ROCm PyTorch (default: ${PYTHON_ENV} or this Python)",
    )
    parser.add_argument(
        "--hook",
        type=Path,
        default=os.environ.get(HOOK_ENV),
        help=f"librocjitsu_dbi_hooks.so (default: ${HOOK_ENV})",
    )
    parser.add_argument(
        "--rocprofv3",
        type=Path,
        default=os.environ.get(ROCPROFV3_ENV),
        help=f"rocprofv3 from the tested ROCm distribution (default: ${ROCPROFV3_ENV})",
    )
    parser.add_argument(
        "--hipblaslt-bench",
        type=Path,
        default=os.environ.get(HIPBLASLT_BENCH_ENV),
        help=(
            "hipblaslt-bench from the tested ROCm distribution "
            f"(default: ${HIPBLASLT_BENCH_ENV})"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--qwen-model",
        type=Path,
        default=os.environ.get("CONSAN_BENCHMARK_QWEN_MODEL"),
        help="local Qwen3-0.6B checkpoint for the two Qwen rows",
    )
    parser.add_argument("--status", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse fingerprint-matched completed cells from the output directory",
    )
    parser.add_argument(
        "--audit-sites",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-audit-sites",
        dest="audit_sites",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--workload",
        choices=tuple(workload.id for workload in WORKLOADS + GFX950_ADDITIONS),
        action="append",
        help="run only this workload (repeatable)",
    )
    return parser.parse_args(argv)


def _git_identity(checkout: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ("git", "-C", str(checkout), *args),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    # Stable across commits of the same code and across documentation updates.
    # Start with index blob identities, then substitute actual unstaged content.
    entries = {}
    for line in git("ls-files", "--stage").splitlines():
        metadata, name = line.split("\t", 1)
        if not name.endswith(".md"):
            entries[name] = metadata.split()[:2]
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    for name in set(git("diff", "--name-only").splitlines() + untracked):
        if name.endswith(".md"):
            continue
        path = checkout / name
        if not path.exists() and not path.is_symlink():
            entries.pop(name, None)
        elif path.is_file() or path.is_symlink():
            content = (
                os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
            )
            blob = hashlib.sha1(
                b"blob " + str(len(content)).encode() + b"\0" + content
            ).hexdigest()
            mode = (
                "120000"
                if path.is_symlink()
                else ("100755" if path.stat().st_mode & 0o111 else "100644")
            )
            entries[name] = [mode, blob]
    return {
        "path": str(checkout.resolve()),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "execution_tree_sha256": hashlib.sha256(
            json.dumps(entries, sort_keys=True).encode()
        ).hexdigest(),
        "diff_sha256": hashlib.sha256(
            git("diff", "--binary", "HEAD").encode()
        ).hexdigest(),
        "untracked_sha256": {
            name: _sha256(checkout / name)
            for name in git("ls-files", "--others", "--exclude-standard").splitlines()
            if (checkout / name).is_file()
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workload_identity(identity: dict[str, Any], workload: Workload) -> dict[str, Any]:
    """Executable inputs, independent of the selected subset and ledger commits.

    Source checkout metadata remains in provenance. The loaded hook's content
    identifies compiled ConSan; Python sources and environment have their own
    content identities. A documentation commit cannot change these executables.
    """
    result = dict(identity)
    if "source" in result:
        result["source"] = {
            "execution_tree_sha256": result["source"]["execution_tree_sha256"]
        }
    if "payload_sha256" in result:
        result["payload_sha256"] = {
            workload.payload: result["payload_sha256"][workload.payload]
        }
    if workload.payload != "aorta":
        result.pop("aorta", None)
    if workload.payload != "hipblaslt":
        result.pop("hipblaslt_bench_sha256", None)
    result["workload"] = {"id": workload.id, "config": workload.config}
    return result


def _python_environment_identity(python: Path) -> dict[str, Any]:
    # Read installed metadata without importing GPU libraries or initializing HSA.
    script = """
import hashlib, importlib.metadata, json, sys
packages = {}
for dist in importlib.metadata.distributions():
    name = dist.metadata['Name'].lower().replace('_', '-')
    record = dist.read_text('RECORD') or ''
    packages[name] = {'version': dist.version, 'record_sha256': hashlib.sha256(record.encode()).hexdigest()}
print(json.dumps({'executable': sys.executable, 'version': sys.version, 'packages': packages}))
"""
    result = subprocess.run(
        (str(python), "-c", script), check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)


def _clean_environment(
    target: str,
    hook: Path | None,
    mode: str | None,
    audit_sites: bool,
    kernel_allowlist_file: Path | None = None,
    epoch_analysis: str | None = None,
) -> dict[str, str]:
    removed = HSA_TOOL_ENVIRONMENT | SOFTWARE_MODEL_ENVIRONMENT | {"HIP_TARGET"}
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(CONTROLLED_ENV_PREFIX) and name not in removed
    }
    environment["HIP_TARGET"] = target
    if mode is None:
        return environment
    assert hook is not None
    profile_environment = dict(
        PROFILES["default" if mode == "default-high" else mode].environment
    )
    if mode in ("default", "default-high"):
        profile_environment["RJ_CONSAN_PRESET"] = (
            "high" if mode == "default-high" else "default"
        )
    # Validation profiles require exhaustive dynamic evidence and reject
    # reported races. A performance benchmark instead gates on static site
    # instrumentation and numerical correctness. Keep dynamic diagnostics and
    # bounded-capture saturation visible in artifacts without aborting the
    # measured process.
    profile_environment.update(
        {
            "RJ_CONSAN_REQUIRE_RECORDS": "0",
            "RJ_CONSAN_FORBID_DIAGNOSTICS": "0",
            "RJ_CONSAN_FORBID_OVERFLOW": "0",
        }
    )
    environment.update(profile_environment)
    environment.update(
        {
            "HSA_TOOLS_LIB": str(hook),
            "HSA_TOOLS_ROCPROFILER_V1_TOOLS": "1",
            "RJ_CONSAN_LOG": "3" if audit_sites else "0",
        }
    )
    if kernel_allowlist_file is not None:
        environment["RJ_CONSAN_KERNEL_ALLOWLIST_FILE"] = str(kernel_allowlist_file)
    if epoch_analysis is not None and mode != "supercollider":
        environment["RJ_CONSAN_EPOCH_ANALYSIS"] = epoch_analysis
    return environment


def _parse_payload(output: str) -> dict[str, Any]:
    payloads = [
        line[len(RESULT_MARKER) :]
        for line in output.splitlines()
        if line.startswith(RESULT_MARKER)
    ]
    if len(payloads) != 1:
        raise BenchmarkError(
            f"expected exactly one {RESULT_MARKER!r} record, found {len(payloads)}"
        )
    try:
        payload = json.loads(payloads[0])
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"malformed benchmark result: {error}") from error
    if not payload.get("result", {}).get("passed"):
        raise BenchmarkError("benchmark workload numerical oracle failed")
    return payload


def _coverage_summary(output: str) -> dict[str, Any]:
    try:
        evidence = parse_coverage_evidence(output)
    except CoverageParseError as error:
        raise BenchmarkError(f"site audit evidence is invalid: {error}") from error
    verdict = evidence.verdict
    applicable = tuple(record for record in evidence.coverage if record.applicable)
    reasons: list[str] = []
    if not verdict.applicable:
        reasons.append("verdict applicable=false")
    if not verdict.static_complete:
        reasons.append("verdict static_complete=false")
    if verdict.applicable_code_objects == 0:
        reasons.append("no applicable code objects")
    if verdict.incomplete_code_objects != 0:
        reasons.append(f"incomplete code objects: {verdict.incomplete_code_objects}")
    for record in applicable:
        if record.expert_limit:
            reasons.append(f"reader {record.reader} used an expert patch limit")
        if not record.analysis_complete:
            reasons.append(f"reader {record.reader} static analysis incomplete")
        for kind in SITE_KINDS:
            patched = record.counts[f"{kind}_patched"]
            supported = record.counts[f"{kind}_supported"]
            if patched != supported:
                reasons.append(
                    f"reader {record.reader} {kind} patched/supported mismatch: "
                    f"{patched}/{supported}"
                )
            for category in (
                "unsupported",
                "resource_failed",
                "placement_or_lowering_failed",
                "expert_limit_omitted",
            ):
                count = record.counts[f"{kind}_{category}"]
                if count:
                    reasons.append(f"reader {record.reader} {kind} {category}: {count}")
    if reasons:
        raise BenchmarkError("site audit failed: " + "; ".join(reasons))
    counts: dict[str, int] = {}
    for field in ("discovered", "selected", "patched", "supported", "unsupported"):
        counts[field] = sum(
            record.counts[f"{kind}_{field}"]
            for record in applicable
            for kind in SITE_KINDS
        )
    counts["checked"] = counts["discovered"]
    counts["missed"] = counts["supported"] - counts["patched"]
    return {
        "accepted": True,
        "applicable_code_objects": verdict.applicable_code_objects,
        "dynamic_complete": verdict.dynamic_complete,
        **counts,
    }


def _payload_program(workload: Workload) -> Path:
    filenames = {
        "aorta": "consan_aorta_benchmark_workload.py",
        "gluon": "consan_gluon_benchmark_workload.py",
        "hipblaslt": "consan_hipblaslt_benchmark_workload.py",
        "tokenspeed": "consan_tokenspeed_benchmark_workload.py",
    }
    try:
        filename = filenames[workload.payload]
    except KeyError as error:
        raise BenchmarkError(
            f"unknown workload payload kind: {workload.payload}"
        ) from error
    return Path(__file__).with_name(filename)


def _payload_command(args: argparse.Namespace, workload: Workload) -> list[str]:
    command = [
        str(args.python),
        str(_payload_program(workload)),
        "--aorta-dir",
        str(args.aorta_dir),
        "--config-json",
        json.dumps(workload.config, separators=(",", ":"), sort_keys=True),
    ]
    if workload.payload == "hipblaslt":
        if args.hipblaslt_bench is None:
            raise BenchmarkError(
                f"set ${HIPBLASLT_BENCH_ENV} or pass --hipblaslt-bench for {workload.id}"
            )
        command.extend(("--hipblaslt-bench", str(args.hipblaslt_bench)))
    return command


def _archive_existing(path: Path) -> None:
    if not path.exists():
        return
    archived = path.with_name(f"{path.name}.previous-{time.time_ns()}")
    path.rename(archived)


def _run_one_impl(
    *,
    args: argparse.Namespace,
    workload: Workload,
    mode: str | None,
    audit_sites: bool,
    label: str,
    kernel_allowlist_file: Path | None = None,
    profile_directory: Path | None = None,
) -> dict[str, Any]:
    command = _payload_command(args, workload)
    if profile_directory is not None:
        if args.rocprofv3 is None:
            raise BenchmarkError(f"set ${ROCPROFV3_ENV} or pass --rocprofv3")
        command = [
            str(args.rocprofv3),
            "--kernel-trace",
            "--output-format",
            "csv",
            "--output-directory",
            str(profile_directory),
            "--",
            *command,
        ]
    environment = _clean_environment(
        args.target,
        args.hook,
        mode,
        audit_sites,
        kernel_allowlist_file,
        epoch_analysis=(
            "manual"
            if workload.payload in ("aorta", "gluon", "tokenspeed")
            else "nth:1"
        ),
    )
    log_path = args.output_dir / f"{workload.id}--{label}.log"
    checkpoint_path = args.output_dir / f"{workload.id}--{label}.json"
    controlled_environment = {
        name: value
        for name, value in environment.items()
        if name.startswith(CONTROLLED_ENV_PREFIX)
        or name in HSA_TOOL_ENVIRONMENT
        or name == "HIP_TARGET"
    }
    fingerprint_payload = {
        "run_identity": _workload_identity(args.run_identity, workload),
        "command": command,
        "environment": controlled_environment,
        "kernel_allowlist_sha256": (
            _sha256(kernel_allowlist_file)
            if kernel_allowlist_file is not None
            else None
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if args.resume and checkpoint_path.is_file() and log_path.is_file():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            checkpoint = None
        profile_available = profile_directory is None or profile_directory.is_dir()
        if (
            isinstance(checkpoint, dict)
            and checkpoint.get("fingerprint") == fingerprint
            and profile_available
        ):
            result = checkpoint.get("result")
            if isinstance(result, dict):
                print(f"resume {workload.id} {label}", flush=True)
                _cell_progress(args, workload, label, mode, "resumed", result)
                return result

    if checkpoint_path.exists():
        _cell_progress(args, workload, label, mode, "invalidated")
    _cell_progress(args, workload, label, mode, "running")
    if profile_directory is not None:
        _archive_existing(profile_directory)
    print(f"run {workload.id} {label}", flush=True)
    start = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as error:

        def timeout_text(value: str | bytes | None) -> str:
            if value is None:
                return ""
            return (
                value.decode("utf-8", errors="replace")
                if isinstance(value, bytes)
                else value
            )

        output = timeout_text(error.stdout) + timeout_text(error.stderr)
        log_path.write_text(output, encoding="utf-8")
        raise BenchmarkError(
            f"{workload.id} {label} timed out after {args.timeout} seconds; see {log_path}"
        ) from error
    wall_ms = (time.perf_counter() - start) * 1000.0
    output = process.stdout + process.stderr
    log_path.write_text(output, encoding="utf-8")
    if process.returncode != 0:
        raise BenchmarkError(
            f"{workload.id} {label} exited {process.returncode}; see {log_path}"
        )
    result = {
        "label": label,
        "mode": mode,
        "audit_sites": audit_sites,
        "wall_ms": wall_ms,
        "payload": _parse_payload(output),
        "log": str(log_path),
    }
    if mode is not None and audit_sites:
        result["coverage"] = _coverage_summary(output)
    _atomic_write(
        checkpoint_path,
        json.dumps(
            {
                "fingerprint": fingerprint,
                "inputs": fingerprint_payload,
                "provenance": args.run_identity,
                "result": result,
            },
            indent=2,
        )
        + "\n",
    )
    print(f"pass {workload.id} {label} wall_ms={wall_ms:.1f}", flush=True)
    _cell_progress(args, workload, label, mode, "accepted", result)
    return result


def _run_one(**kwargs: Any) -> dict[str, Any]:
    try:
        return _run_one_impl(**kwargs)
    except (BenchmarkError, OSError, subprocess.SubprocessError) as error:
        _cell_progress(
            kwargs["args"],
            kwargs["workload"],
            kwargs["label"],
            kwargs["mode"],
            "failed",
            error=str(error),
        )
        raise


def _write_progress(args: argparse.Namespace) -> None:
    projection = getattr(args, "status_projection", None)
    if projection is None:
        return
    _atomic_write(
        args.output_dir / "progress.json",
        json.dumps(
            {"run_identity": args.run_identity, "projection": projection}, indent=2
        )
        + "\n",
    )
    if args.status is not None:
        _atomic_write(args.status, _render_status(projection))


def _cell_progress(
    args: argparse.Namespace,
    workload: Workload,
    label: str,
    mode: str | None,
    state: str,
    result: dict[str, Any] | None = None,
    *,
    error: str | None = None,
) -> None:
    projection = getattr(args, "status_projection", None)
    if projection is None:
        return
    row = next(row for row in projection["workloads"] if row["id"] == workload.id)
    log = str(args.output_dir / f"{workload.id}--{label}.log")
    row["progress"] = {"label": label, "state": state, "log": log, "error": error}
    row.setdefault("cell_states", {})[mode or "native"] = state
    if mode is not None:
        row["modes"].pop(mode, None)
        if result is not None:
            row["modes"][mode] = _summarize_mode(
                result, tuple(row["native"]["runtime_ms"])
            )
    _write_progress(args)


def _generate_kernel_allowlist(
    trace_directory: Path, output: Path, log: Path
) -> tuple[str, ...]:
    converter = (
        Path(__file__).resolve().parents[3] / "scripts" / "rocjitsu_consan_allowlist.py"
    )
    process = subprocess.run(
        (sys.executable, str(converter), "--output", str(output), str(trace_directory)),
        capture_output=True,
        text=True,
    )
    converter_output = process.stdout + process.stderr
    log.write_text(converter_output, encoding="utf-8")
    if process.returncode != 0:
        raise BenchmarkError(
            f"rocprofv3 allowlist conversion exited {process.returncode}; see {log}"
        )
    try:
        names = tuple(
            line for line in output.read_text(encoding="utf-8").splitlines() if line
        )
    except OSError as error:
        raise BenchmarkError(f"could not read generated allowlist: {error}") from error
    if not names or len(names) != len(set(names)):
        raise BenchmarkError(
            "generated kernel allowlist is empty or contains duplicates"
        )
    if any("\n" in name or "\r" in name for name in names):
        raise BenchmarkError(
            "generated kernel allowlist contains an invalid kernel name"
        )
    return names


def _median(values: list[float]) -> float:
    if not values:
        raise BenchmarkError("cannot summarize an empty sample set")
    return statistics.median(values)


def _operation(run: dict[str, Any], index: int) -> dict[str, Any]:
    operations = run["payload"].get("runs")
    if not isinstance(operations, list) or len(operations) != 2:
        raise BenchmarkError("benchmark payload must contain exactly two runs")
    operation = operations[index]
    if not isinstance(operation, dict):
        raise BenchmarkError(f"benchmark run {index + 1} is malformed")
    return operation


def _metric(run: dict[str, Any], name: str, index: int) -> float:
    value = _operation(run, index).get("result", {}).get("metrics", {}).get(name)
    if not isinstance(value, (int, float)) or value <= 0:
        raise BenchmarkError(f"invalid primary metric {name}: {value!r}")
    return float(value)


def _instrumentation_ms(run: dict[str, Any], name: str) -> float:
    value = run["payload"]["instrumentation_ms"].get(name)
    if not isinstance(value, (int, float)) or value < 0:
        raise BenchmarkError(f"invalid instrumentation latency {name}: {value!r}")
    return float(value)


def _runtime_ms(run: dict[str, Any], index: int) -> float:
    operation = _operation(run, index)
    direct = operation.get("runtime_ms")
    if direct is not None:
        if not isinstance(direct, (int, float)) or direct <= 0:
            raise BenchmarkError(f"invalid direct runtime latency: {direct!r}")
        return float(direct)
    phase = operation.get("phase_ms")
    instrumentation = operation.get("instrumentation_ms")
    if not isinstance(phase, (int, float)) or phase <= 0:
        raise BenchmarkError(f"invalid run {index + 1} phase latency: {phase!r}")
    if not isinstance(instrumentation, (int, float)) or instrumentation < 0:
        raise BenchmarkError(
            f"invalid run {index + 1} instrumentation latency: {instrumentation!r}"
        )
    runtime = float(phase) - float(instrumentation)
    if runtime <= 0:
        raise BenchmarkError("instrumentation time exceeds timed runtime")
    return runtime


def _summarize_mode(
    run: dict[str, Any], native_runtime: tuple[float, float]
) -> dict[str, Any]:
    runtime_ms = tuple(_runtime_ms(run, index) for index in range(2))
    instrumentation_ms = _instrumentation_ms(run, "total")
    result: dict[str, Any] = {
        "runtime_ms": list(runtime_ms),
        "runtime_ratio": [
            runtime_ms[index] / native_runtime[index] for index in range(2)
        ],
        # Startup is the complete cold-path latency, not merely the portion
        # that happens to fall inside the hook's instrumentation clock.
        "startup_ms": instrumentation_ms + runtime_ms[0],
        "instrumentation_ms": instrumentation_ms,
        "run_ms": runtime_ms[1],
        "run_ratio": runtime_ms[1] / native_runtime[1],
        "peak_device_memory": run["payload"]["peak_device_memory"],
    }
    if "coverage" in run:
        result["coverage"] = run["coverage"]
    return result


def _summarize_workload(
    workload: Workload,
    native_reference: list[dict[str, Any]],
    native_validation: dict[str, Any],
    modes: dict[str, dict[str, Any]],
    kernel_allowlist: tuple[str, ...],
) -> dict[str, Any]:
    native_latency = tuple(
        _median(
            [
                _metric(run, workload.primary_metric, run_index)
                for run in native_reference
            ]
        )
        for run_index in range(2)
    )
    native_runtime = tuple(
        _median([_runtime_ms(run, run_index) for run in native_reference])
        for run_index in range(2)
    )
    validation_runtime = tuple(
        _runtime_ms(native_validation, run_index) for run_index in range(2)
    )
    payload_metrics = native_reference[0]["payload"]["result"]["metrics"]
    result: dict[str, Any] = {
        "id": workload.id,
        "description": workload.description,
        "primary_metric": workload.primary_metric,
        "config": workload.config,
        "native_latency_ms": list(native_latency),
        "native_samples_ms": [
            [_metric(run, workload.primary_metric, run_index) for run_index in range(2)]
            for run in native_reference
        ],
        "native_runtime_ms": list(native_runtime),
        "native_runtime_samples_ms": [
            [_runtime_ms(run, run_index) for run_index in range(2)]
            for run in native_reference
        ],
        "native_validation_runtime_ms": list(validation_runtime),
        "native_validation_drift_ratio": [
            validation_runtime[index] / native_runtime[index] - 1.0
            for index in range(2)
        ],
        "payload_metrics": payload_metrics,
        "runtime": native_reference[0]["payload"]["runtime"],
        "kernel_inventory": list(kernel_allowlist),
        "kernel_allowlist": list(kernel_allowlist),
        "modes": {},
    }
    if "parameter_count" in payload_metrics:
        result["parameter_count"] = payload_metrics["parameter_count"]
    for mode in PROFILE_IDS:
        run = modes[mode]
        mode_result = _summarize_mode(run, native_runtime)
        mode_result["latency_ms"] = [
            _metric(run, workload.primary_metric, run_index) for run_index in range(2)
        ]
        result["modes"][mode] = mode_result
    return result


def _format_three_significant_digits(value: float) -> str:
    if not math.isfinite(value) or value <= 0:
        raise BenchmarkError(f"cannot format invalid benchmark value: {value!r}")
    decimal_places = 2 - math.floor(math.log10(value))
    if decimal_places > 0:
        return f"{value:,.{decimal_places}f}".rstrip("0").rstrip(".")
    return f"{round(value, decimal_places):,.0f}"


def _render_status(summary: dict[str, Any]) -> str:
    columns = tuple(PROFILE_IDS)
    live = any("progress" in row for row in summary["workloads"])
    lines = [
        f"# ConSan `{summary['target']}` benchmark status",
        "",
        "For each mode, **Startup** is the total latency through the first synchronized",
        "run and its selected evidence checkpoints, including instrumentation, loading,",
        "binding, and warm-up; **Run** is the absolute second-run latency followed",
        "by its ratio to the matching uninstrumented second run.",
        "",
        "| Workload | Uninstrumented Startup | Uninstrumented Run | "
        + " | ".join(
            column
            for mode in columns
            for column in (
                f"{MODE_LABELS[mode]} Startup",
                f"{MODE_LABELS[mode]} Run",
            )
        )
        + (" | Progress |" if live else " |"),
        "| --- | "
        + " | ".join("---:" for _ in range(2 + 2 * len(columns)))
        + (" | --- |" if live else " |"),
    ]
    for workload in summary["workloads"]:
        native_runtime = workload.get("native_runtime_ms")
        if native_runtime is None:
            native_runtime = workload.get("native", {}).get("runtime_ms")
        if not isinstance(native_runtime, (list, tuple)) or len(native_runtime) != 2:
            state = workload.get("cell_states", {}).get("native", "pending")
            # A single accepted native sample is not yet a baseline median.
            state = (
                state
                if state in ("running", "failed", "invalidated", "blocked")
                else "pending"
            )
            native_cells = [state, state]
        else:
            native_cells = [
                f"{_format_three_significant_digits(native_runtime[0] / 1000.0)} s",
                f"{_format_three_significant_digits(native_runtime[1] / 1000.0)} s (1×)",
            ]
        cells = native_cells
        for mode in columns:
            result = workload["modes"].get(mode)
            if result is None:
                state = workload.get("cell_states", {}).get(mode, "pending")
                cells.extend((state, state))
            else:
                cells.extend(
                    (
                        f"{_format_three_significant_digits(result['startup_ms'] / 1000.0)} s",
                        f"{_format_three_significant_digits(result['run_ms'] / 1000.0)} s "
                        f"({_format_three_significant_digits(result['run_ratio'])}×)",
                    )
                )
        if live:
            progress = workload.get("progress", {})
            detail = f"{progress.get('label', 'not started')}: {progress.get('state', 'pending')}"
            if progress.get("log"):
                detail = f"[{detail}](<{progress['log']}>)"
            cells.append(detail)
        lines.append(f"| {workload['description']} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if args.aorta_dir is None:
        raise BenchmarkError(f"set ${AORTA_DIR_ENV} or pass --aorta-dir")
    args.aorta_dir = args.aorta_dir.resolve()
    if args.hook is None:
        raise BenchmarkError(f"set ${HOOK_ENV} or pass --hook")
    args.hook = args.hook.resolve()
    if args.rocprofv3 is None:
        raise BenchmarkError(f"set ${ROCPROFV3_ENV} or pass --rocprofv3")
    args.rocprofv3 = args.rocprofv3.resolve()
    available = _target_workloads(args.target)
    selected = [
        workload
        for workload in available
        if args.workload is None or workload.id in args.workload
    ]
    if not selected:
        raise BenchmarkError("no workloads selected")
    if args.workload and set(args.workload) - {workload.id for workload in selected}:
        raise BenchmarkError("selected workload is not supported on this target")
    if args.qwen_model is not None or any(
        w.config.get("operation", "").startswith("qwen-") for w in selected
    ):
        if args.qwen_model is None or not args.qwen_model.is_dir():
            raise BenchmarkError("pass --qwen-model with a local Qwen3-0.6B checkpoint")
        model = args.qwen_model.resolve()
        model_hashes = {
            str(path.relative_to(model)): _sha256(path)
            for path in sorted(model.rglob("*"))
            if path.is_file() and ".cache" not in path.relative_to(model).parts
        }
        available = [
            (
                Workload(
                    w.id,
                    w.description,
                    w.primary_metric,
                    {**w.config, "model": str(model), "model_sha256": model_hashes},
                    w.payload,
                )
                if w.config.get("operation", "").startswith("qwen-")
                else w
            )
            for w in available
        ]
        selected_ids = {w.id for w in selected}
        selected = [w for w in available if w.id in selected_ids]
    if args.hipblaslt_bench is not None:
        args.hipblaslt_bench = args.hipblaslt_bench.resolve()
    for path, description in (
        (args.python, "benchmark Python"),
        (args.hook, "ConSan hook"),
        (args.rocprofv3, "rocprofv3"),
    ):
        if not path.is_file():
            raise BenchmarkError(f"{description} does not exist: {path}")
    if any(workload.payload == "hipblaslt" for workload in selected):
        if args.hipblaslt_bench is None:
            raise BenchmarkError(
                f"set ${HIPBLASLT_BENCH_ENV} or pass --hipblaslt-bench"
            )
        if not args.hipblaslt_bench.is_file():
            raise BenchmarkError(
                f"hipblaslt-bench does not exist: {args.hipblaslt_bench}"
            )
    aorta_identity = _git_identity(args.aorta_dir)
    source_root = Path(__file__).resolve().parents[5]
    source_identity = _git_identity(source_root)
    hook_sha256 = _sha256(args.hook)
    converter = (
        Path(__file__).resolve().parents[3] / "scripts" / "rocjitsu_consan_allowlist.py"
    )
    args.run_identity = {
        "source": source_identity,
        "aorta": aorta_identity,
        "hook_sha256": hook_sha256,
        "rocprofv3_sha256": _sha256(args.rocprofv3),
        "runner_sha256": _sha256(Path(__file__)),
        "converter_sha256": _sha256(converter),
        "python_environment": _python_environment_identity(args.python),
        "runtime_environment": {
            name: value
            for name, value in os.environ.items()
            if name
            in (
                "PATH",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "ROCM_PATH",
                "HIP_PATH",
                "TRITON_LIBHIP_PATH",
                "HSA_OVERRIDE_GFX_VERSION",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "payload_sha256": {
            workload.payload: _sha256(_payload_program(workload))
            for workload in _target_workloads(args.target)
        },
        "python_support_sha256": {
            path.name: _sha256(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    if args.hipblaslt_bench is not None:
        args.run_identity["hipblaslt_bench_sha256"] = _sha256(args.hipblaslt_bench)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.status is not None:
        args.status = args.status.resolve()

    suite_start = time.perf_counter()
    status_projection = {
        "target": args.target,
        "workloads": [
            {
                "id": workload.id,
                "description": workload.description,
                "modes": {},
                "progress": {},
            }
            for workload in _target_workloads(args.target)
        ],
    }
    progress_path = args.output_dir / "progress.json"
    if args.resume and progress_path.is_file():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        previous_rows = {row["id"]: row for row in previous["projection"]["workloads"]}
        current_workloads = {w.id: w for w in available}
        for index, row in enumerate(status_projection["workloads"]):
            old = previous_rows.get(row["id"])
            workload = current_workloads[row["id"]]
            if old and old.get("identity") == _workload_identity(
                args.run_identity, workload
            ):
                status_projection["workloads"][index] = old
    for row in status_projection["workloads"]:
        workload = next((w for w in selected if w.id == row["id"]), None)
        if workload is not None:
            row["identity"] = _workload_identity(args.run_identity, workload)
    args.status_projection = status_projection
    _write_progress(args)
    status_rows = {
        workload["id"]: workload for workload in status_projection["workloads"]
    }
    for workload in selected:
        profile_directory = args.output_dir / f"{workload.id}--kernel-profile"
        _run_one(
            args=args,
            workload=workload,
            mode=None,
            audit_sites=False,
            label="kernel-inventory",
            profile_directory=profile_directory,
        )
        kernel_allowlist_file = args.output_dir / f"{workload.id}--kernel-allowlist.txt"
        kernel_allowlist = _generate_kernel_allowlist(
            profile_directory,
            kernel_allowlist_file,
            args.output_dir / f"{workload.id}--kernel-allowlist.log",
        )
        native_reference = [
            _run_one(
                args=args,
                workload=workload,
                mode=None,
                audit_sites=False,
                label=f"native-reference-{sample}",
            )
            for sample in (1, 2)
        ]
        native_runtime = tuple(
            _median([_runtime_ms(run, run_index) for run in native_reference])
            for run_index in range(2)
        )
        status_rows[workload.id]["native"] = {"runtime_ms": list(native_runtime)}
        _write_progress(args)
        modes: dict[str, dict[str, Any]] = {}
        for mode in PROFILE_IDS:
            modes[mode] = _run_one(
                args=args,
                workload=workload,
                mode=mode,
                audit_sites=args.audit_sites,
                label=f"{mode}--audit-{'on' if args.audit_sites else 'off'}",
                kernel_allowlist_file=kernel_allowlist_file,
            )
            status_rows[workload.id]["modes"][mode] = _summarize_mode(
                modes[mode], native_runtime
            )
            if args.status is not None:
                _atomic_write(args.status, _render_status(status_projection))
        native_validation = _run_one(
            args=args,
            workload=workload,
            mode=None,
            audit_sites=False,
            label="native-validation",
        )
        completed = _summarize_workload(
            workload,
            native_reference,
            native_validation,
            modes,
            kernel_allowlist,
        )
        completed["provenance"] = args.run_identity
        status_rows[workload.id]["summary"] = completed
        _write_progress(args)

    suite_seconds = time.perf_counter() - suite_start
    summary = {
        "schema_version": SCHEMA_VERSION,
        "target": args.target,
        "audit_sites": args.audit_sites,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suite_wall_seconds": suite_seconds,
        "under_30_minute_target": suite_seconds < 30 * 60,
        "artifact_dir": str(args.output_dir),
        "provenance": {
            "aorta": aorta_identity,
            "rocm_systems": source_identity,
            "hook": {"path": str(args.hook), "sha256": hook_sha256},
            "python": str(args.python),
            "rocprofv3": {
                "path": str(args.rocprofv3),
                "sha256": args.run_identity["rocprofv3_sha256"],
            },
        },
        "workloads": [
            row["summary"] for row in status_projection["workloads"] if "summary" in row
        ],
    }
    _atomic_write(
        args.output_dir / "summary.json", json.dumps(summary, indent=2) + "\n"
    )
    _write_progress(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except (BenchmarkError, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
