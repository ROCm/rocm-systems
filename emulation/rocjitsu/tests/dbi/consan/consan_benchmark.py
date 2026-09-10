#!/usr/bin/env python3
"""Benchmark bounded Aorta workloads natively and under every ConSan mode."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

from consan_coverage_gate import CoverageParseError, acceptance_decision
from consan_validation_catalog import (
    CONTROLLED_ENV_PREFIX,
    HSA_TOOL_ENVIRONMENT,
    PROFILE_IDS,
    PROFILES,
    SOFTWARE_MODEL_ENVIRONMENT,
)
from consan_validation_support import SITE_KINDS


SCHEMA_VERSION = 1
AORTA_DIR_ENV = "CONSAN_BENCHMARK_AORTA_DIR"
PYTHON_ENV = "CONSAN_BENCHMARK_PYTHON"
HOOK_ENV = "CONSAN_BENCHMARK_HOOK"
RESULT_MARKER = "CONSAN_BENCHMARK_RESULT="
SUPPORTED_TARGETS = ("gfx950", "gfx1201")
MODE_LABELS = {
    "supercollider": "SuperCollider",
    "record-replay": "Record/Replay",
    "sampled": "Sampled",
    "inline-shadow": "Inline Shadow",
}


class BenchmarkError(RuntimeError):
    """The benchmark contract could not be satisfied."""


@dataclass(frozen=True)
class Workload:
    id: str
    description: str
    primary_metric: str
    config: dict[str, Any]


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
        description="PyTorch synthetic dense prefill (long prompt, one output token)",
        primary_metric="prefill_latency_ms",
        config={
            "mode": "offline_batch",
            "device": "cuda",
            "dtype": "bfloat16",
            "seed": 1234,
            "warmup_steps": 1,
            "steps": 5,
            "model": _model_config(),
            "request": {"batch_size": 1, "prompt_len": 128, "generate_tokens": 1},
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
        description="PyTorch synthetic dense decode (short prompt, repeated token loop)",
        primary_metric="decode_latency_ms",
        config={
            "mode": "offline_batch",
            "device": "cuda",
            "dtype": "bfloat16",
            "seed": 1234,
            "warmup_steps": 1,
            "steps": 4,
            "model": _model_config(),
            "request": {"batch_size": 4, "prompt_len": 16, "generate_tokens": 16},
            "serving": {"kv_cache": True},
            "checks": {
                "fail_on_nan_logits": True,
                "fail_on_nonfinite_output": True,
                "compare_logits_checksum": True,
            },
        },
    ),
    Workload(
        id="pytorch-top1-moe-prefill",
        description="PyTorch synthetic four-expert top-1 MoE prefill",
        primary_metric="prefill_latency_ms",
        config={
            "mode": "offline_batch",
            "device": "cuda",
            "dtype": "bfloat16",
            "seed": 1234,
            "warmup_steps": 1,
            "steps": 4,
            "model": _model_config(num_experts=4),
            "request": {"batch_size": 1, "prompt_len": 64, "generate_tokens": 1},
            "serving": {"kv_cache": True},
            "checks": {
                "fail_on_nan_logits": True,
                "fail_on_nonfinite_output": True,
                "compare_logits_checksum": True,
            },
        },
    ),
)


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--audit-sites", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--workload",
        choices=tuple(workload.id for workload in WORKLOADS),
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

    return {
        "path": str(checkout.resolve()),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_environment(
    target: str,
    hook: Path | None,
    mode: str | None,
    audit_sites: bool,
    kernel_allowlist_file: Path | None = None,
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
    environment.update(PROFILES[mode].environment)
    environment.update(
        {
            "HSA_TOOLS_LIB": str(hook),
            "HSA_TOOLS_ROCPROFILER_V1_TOOLS": "1",
            "RJ_CONSAN_LOG": "3" if audit_sites else "0",
        }
    )
    if kernel_allowlist_file is not None:
        environment["RJ_CONSAN_KERNEL_ALLOWLIST_FILE"] = str(kernel_allowlist_file)
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
        raise BenchmarkError("Aorta workload numerical oracle failed")
    return payload


def _coverage_summary(output: str) -> dict[str, Any]:
    try:
        decision = acceptance_decision(output)
    except CoverageParseError as error:
        raise BenchmarkError(f"site audit evidence is invalid: {error}") from error
    if not decision.accepted:
        raise BenchmarkError("site audit failed: " + "; ".join(decision.reasons))
    applicable = tuple(record for record in decision.evidence.coverage if record.applicable)
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
        "applicable_code_objects": decision.evidence.verdict.applicable_code_objects,
        "dynamic_complete": decision.evidence.verdict.dynamic_complete,
        **counts,
    }


def _run_one(
    *,
    args: argparse.Namespace,
    workload: Workload,
    mode: str | None,
    audit_sites: bool,
    label: str,
    kernel_allowlist_file: Path | None = None,
    profile_kernels: bool = False,
) -> dict[str, Any]:
    payload_program = Path(__file__).with_name("consan_aorta_benchmark_workload.py")
    command = [
        str(args.python),
        str(payload_program),
        "--aorta-dir",
        str(args.aorta_dir),
        "--config-json",
        json.dumps(workload.config, separators=(",", ":"), sort_keys=True),
    ]
    if profile_kernels:
        command.append("--profile-kernels")
    start = time.perf_counter()
    process = subprocess.run(
        command,
        env=_clean_environment(
            args.target, args.hook, mode, audit_sites, kernel_allowlist_file
        ),
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    output = process.stdout + process.stderr
    log_path = args.output_dir / f"{workload.id}--{label}.log"
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
    return result


def _select_dispatched_kernels(inventory: dict[str, Any]) -> tuple[str, ...]:
    names = inventory["payload"].get("kernel_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise BenchmarkError("native kernel inventory is absent or malformed")
    # Instrument the complete observed end-to-end dispatch set. The file-backed
    # exact-name interface accommodates demangled names containing commas.
    selected = tuple(sorted(set(names)))
    if not selected:
        raise BenchmarkError("native workload dispatched no GPU kernel")
    if any("\n" in name or "\r" in name for name in selected):
        raise BenchmarkError("selected kernel name cannot be encoded in the allowlist file")
    return selected


def _write_kernel_allowlist(path: Path, kernel_names: tuple[str, ...]) -> None:
    path.write_text("".join(f"{name}\n" for name in kernel_names), encoding="utf-8")


def _median(values: list[float]) -> float:
    if not values:
        raise BenchmarkError("cannot summarize an empty sample set")
    return statistics.median(values)


def _metric(run: dict[str, Any], name: str) -> float:
    value = run["payload"]["result"]["metrics"].get(name)
    if not isinstance(value, (int, float)) or value <= 0:
        raise BenchmarkError(f"invalid primary metric {name}: {value!r}")
    return float(value)


def _summarize_workload(
    workload: Workload,
    native: list[dict[str, Any]],
    modes: dict[str, dict[str, dict[str, Any] | None]],
    inventory: dict[str, Any],
    kernel_allowlist: tuple[str, ...],
) -> dict[str, Any]:
    native_latency = _median([_metric(run, workload.primary_metric) for run in native])
    result: dict[str, Any] = {
        "id": workload.id,
        "description": workload.description,
        "primary_metric": workload.primary_metric,
        "config": workload.config,
        "native_latency_ms": native_latency,
        "native_samples_ms": [_metric(run, workload.primary_metric) for run in native],
        "parameter_count": native[0]["payload"]["result"]["metrics"]["parameter_count"],
        "runtime": native[0]["payload"]["runtime"],
        "kernel_inventory": inventory["payload"]["kernel_names"],
        "kernel_allowlist": list(kernel_allowlist),
        "modes": {},
    }
    for mode in PROFILE_IDS:
        quick = modes[mode]["quick"]
        assert quick is not None
        latency = _metric(quick, workload.primary_metric)
        audit = modes[mode]["audit"]
        mode_result: dict[str, Any] = {
            "latency_ms": latency,
            "ratio": latency / native_latency,
            "peak_device_memory": quick["payload"]["peak_device_memory"],
        }
        if audit is not None:
            quick_wall = float(quick["wall_ms"])
            audit_wall = float(audit["wall_ms"])
            quick_run = float(quick["payload"]["phase_ms"]["run"])
            audit_run = float(audit["payload"]["phase_ms"]["run"])
            mode_result["site_audit"] = {
                "quick_wall_ms": quick_wall,
                "audit_wall_ms": audit_wall,
                "wall_delta_ms": audit_wall - quick_wall,
                "wall_delta_percent": (audit_wall / quick_wall - 1.0) * 100.0,
                "quick_run_ms": quick_run,
                "audit_run_ms": audit_run,
                "run_delta_ms": audit_run - quick_run,
                "run_delta_percent": (audit_run / quick_run - 1.0) * 100.0,
                "coverage": audit["coverage"],
            }
        result["modes"][mode] = mode_result
    return result


def _format_bytes(value: int) -> str:
    return f"{value / (1024**2):.1f} MiB"


def _render_status(summary: dict[str, Any]) -> str:
    columns = tuple(PROFILE_IDS)
    lines = [
        "# ConSan `gfx1201` benchmark status",
        "",
        f"Last measured: {summary['completed_at']}.",
        "",
        "## Instrumentation overhead",
        "",
        "Each cell is **audit-disabled instrumented latency / native latency**. "
        "Site-audit cost is excluded and reported separately below.",
        "",
        "| Workload | " + " | ".join(MODE_LABELS[mode] for mode in columns) + " |",
        "| --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for workload in summary["workloads"]:
        cells = [f"{workload['modes'][mode]['ratio']:.3f}×" for mode in columns]
        lines.append(f"| {workload['description']} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Absolute latency",
        "",
        "Milliseconds in the workload's synchronized warm phase. Native is the "
        "median of the measurements bracketing the four-mode matrix.",
        "",
        "| Workload | Native | "
        + " | ".join(MODE_LABELS[mode] for mode in columns)
        + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for workload in summary["workloads"]:
        cells = [f"{workload['modes'][mode]['latency_ms']:.3f}" for mode in columns]
        lines.append(
            f"| {workload['id']} | {workload['native_latency_ms']:.3f} | "
            + " | ".join(cells)
            + " |"
        )

    if summary["audit_sites"]:
        lines += [
            "",
            "## Site-audit overhead",
            "",
            "This is the separately measured process-wall delta from enabling "
            "detailed site evidence. Negative deltas are retained rather than clipped.",
            "",
            "| Workload | Mode | Audit off | Audit on | Delta | Warm-run delta |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for workload in summary["workloads"]:
            for mode in columns:
                audit = workload["modes"][mode]["site_audit"]
                lines.append(
                    f"| {workload['id']} | {MODE_LABELS[mode]} | "
                    f"{audit['quick_wall_ms']:.1f} ms | {audit['audit_wall_ms']:.1f} ms | "
                    f"{audit['wall_delta_ms']:+.1f} ms ({audit['wall_delta_percent']:+.2f}%) | "
                    f"{audit['run_delta_ms']:+.1f} ms ({audit['run_delta_percent']:+.2f}%) |"
                )
        lines += [
            "",
            "### Site-audit coverage",
            "",
            "`Checked` counts every discovered site examined by the audit; `missed` "
            "is supported minus patched. Every row also passed the final static and "
            "dynamic completeness verdict.",
            "",
            "| Workload | Mode | Objects | Selected | Instrumented | Checked | Unsupported | Missed |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for workload in summary["workloads"]:
            for mode in columns:
                coverage = workload["modes"][mode]["site_audit"]["coverage"]
                lines.append(
                    f"| {workload['id']} | {MODE_LABELS[mode]} | "
                    f"{coverage['applicable_code_objects']} | {coverage['selected']} | "
                    f"{coverage['patched']} | {coverage['checked']} | "
                    f"{coverage['unsupported']} | {coverage['missed']} |"
                )

    lines += [
        "",
        "## Workloads and memory",
        "",
        "| Workload | Parameters | Peak allocated (largest mode) | Primary metric |",
        "| --- | ---: | ---: | --- |",
    ]
    for workload in summary["workloads"]:
        peak = max(
            workload["modes"][mode]["peak_device_memory"]["allocated_bytes"]
            for mode in columns
        )
        lines.append(
            f"| {workload['description']} | {workload['parameter_count']:,} | "
            f"{_format_bytes(peak)} | `{workload['primary_metric']}` |"
        )

    aorta = summary["provenance"]["aorta"]
    source = summary["provenance"]["rocm_systems"]
    lines += [
        "",
        "## Provenance and scope",
        "",
        f"- Target: `{summary['target']}`; device: "
        f"`{summary['workloads'][0]['runtime']['device_name']}` "
        f"(`{summary['workloads'][0]['runtime']['architecture']}`).",
        f"- ROCm Systems commit: `{source['commit']}`"
        + (" (dirty benchmark implementation)" if source["dirty"] else "")
        + ".",
        f"- Aorta commit: `{aorta['commit']}`"
        + (" (dirty checkout)" if aorta["dirty"] else "")
        + ".",
        f"- Hook SHA-256: `{summary['provenance']['hook']['sha256']}`.",
        f"- PyTorch: `{summary['workloads'][0]['runtime']['torch']}`; HIP: "
        f"`{summary['workloads'][0]['runtime']['hip']}`.",
        f"- Complete matrix wall time: {summary['suite_wall_seconds']:.1f} seconds.",
        "- TokenSpeed is not selected on gfx1201 because its pinned Aorta container "
        "rejects this target. Portable Gluon and explicitly pinned "
        "hipBLASLt/Tensile end-to-end cells remain future corpus additions.",
        "- These synthetic Aorta models provide PyTorch prefill, synthetic decode, "
        "and top-1 MoE coverage; they are not real Qwen serving workloads.",
        "- Each workload first inventories its native dispatches. Every exact observed "
        "kernel entry forms the ConSan allowlist; colocated but undispatched library "
        "kernels are outside the measurement policy.",
        "",
        "The complete machine-readable samples and logs are retained in the artifact "
        f"directory `{summary['artifact_dir']}`. See [BENCHMARK.md](BENCHMARK.md) "
        "for the measurement and admission contract.",
        "",
    ]
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
    for path, description in (
        (args.python, "benchmark Python"),
        (args.hook, "ConSan hook"),
    ):
        if not path.is_file():
            raise BenchmarkError(f"{description} does not exist: {path}")
    aorta_identity = _git_identity(args.aorta_dir)
    source_root = Path(__file__).resolve().parents[5]
    source_identity = _git_identity(source_root)
    selected = [
        workload
        for workload in WORKLOADS
        if args.workload is None or workload.id in args.workload
    ]
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suite_start = time.perf_counter()
    workload_summaries = []
    for workload in selected:
        inventory = _run_one(
            args=args,
            workload=workload,
            mode=None,
            audit_sites=False,
            label="kernel-inventory",
            profile_kernels=True,
        )
        kernel_allowlist = _select_dispatched_kernels(inventory)
        kernel_allowlist_file = args.output_dir / f"{workload.id}--kernel-allowlist.txt"
        _write_kernel_allowlist(kernel_allowlist_file, kernel_allowlist)
        native = [
            _run_one(
                args=args,
                workload=workload,
                mode=None,
                audit_sites=False,
                label="native-before",
            )
        ]
        modes: dict[str, dict[str, dict[str, Any] | None]] = {}
        for mode in PROFILE_IDS:
            quick = _run_one(
                args=args,
                workload=workload,
                mode=mode,
                audit_sites=False,
                label=f"{mode}--audit-off",
                kernel_allowlist_file=kernel_allowlist_file,
            )
            audit = None
            if args.audit_sites:
                audit = _run_one(
                    args=args,
                    workload=workload,
                    mode=mode,
                    audit_sites=True,
                    label=f"{mode}--audit-on",
                    kernel_allowlist_file=kernel_allowlist_file,
                )
            modes[mode] = {"quick": quick, "audit": audit}
        native.append(
            _run_one(
                args=args,
                workload=workload,
                mode=None,
                audit_sites=False,
                label="native-after",
            )
        )
        workload_summaries.append(
            _summarize_workload(
                workload, native, modes, inventory, kernel_allowlist
            )
        )

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
            "hook": {"path": str(args.hook), "sha256": _sha256(args.hook)},
            "python": str(args.python),
        },
        "workloads": workload_summaries,
    }
    _atomic_write(args.output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    if args.status is not None:
        if args.target != "gfx1201":
            raise BenchmarkError("the current status renderer is specific to gfx1201")
        _atomic_write(args.status.resolve(), _render_status(summary))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except (BenchmarkError, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
