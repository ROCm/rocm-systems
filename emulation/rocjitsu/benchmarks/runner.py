# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the fixed Rocjitsu performance benchmark matrix."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parent
ROCJITSU_ROOT = BENCHMARK_ROOT.parent
DEFAULT_MANIFEST = BENCHMARK_ROOT / "suites" / "nightly.toml"
TARGET_CONFIGS = {
    "gfx950": ROCJITSU_ROOT / "configs" / "gfx950_mi355x_kmd.json",
    "gfx1250": ROCJITSU_ROOT / "configs" / "gfx1250_mi455x_kmd.json",
}
MANIFEST_FIELDS = {"name", "targets", "cases", "warmups", "samples", "timeout_seconds"}
PACKAGE_NAMES = (
    "rocm-sdk-devel",
    "rocm-sdk-libraries",
    "rocm-sdk-device-gfx950",
    "rocm-sdk-device-gfx1250",
    "torch",
    "triton",
    "amd-torch-device-gfx950",
    "amd-torch-device-gfx1250",
)
CASE_ID = re.compile(r"^(hip|triton|tensile)\.[a-z0-9_]+(?:\.[a-z0-9_]+)*$")
WORKLOAD_FIELDS = {"schema", "case", "target", "provider", "parameters", "timings_ns"}


class RunnerError(ValueError):
    """A user-facing configuration or execution error."""


@dataclasses.dataclass(frozen=True)
class Suite:
    name: str
    targets: tuple[str, ...]
    cases: tuple[str, ...]
    warmups: int
    samples: int
    timeout_seconds: float


@dataclasses.dataclass(frozen=True)
class Cell:
    case: str
    target: str

    @property
    def provider(self) -> str:
        return provider_for(self.case)


@dataclasses.dataclass(frozen=True)
class PreparedCommand:
    argv: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]
    workload_path: Path


@dataclasses.dataclass(frozen=True)
class CaseMetadata:
    suite: str
    name: str
    operation: str
    data_type: str | None


@dataclasses.dataclass(frozen=True)
class TargetMetadata:
    exec_mode: str
    num_threads: int


CASE_METADATA = {
    "hip.launch_noop": CaseMetadata("HIP", "Launch overhead", "Launch", None),
    "hip.copy_fp32_32m": CaseMetadata("HIP", "32 MiB FP32 copy", "Copy", "fp32"),
    "hip.vector_add_fp32_tail": CaseMetadata(
        "HIP", "FP32 vector add with tail", "Vector add", "fp32"
    ),
    "triton.softmax_fp16_aligned": CaseMetadata(
        "Triton", "Aligned FP16 softmax", "Softmax", "fp16"
    ),
    "triton.softmax_fp16_boundary": CaseMetadata(
        "Triton", "Boundary FP16 softmax", "Softmax", "fp16"
    ),
    "triton.rmsnorm_bf16": CaseMetadata("Triton", "BF16 RMSNorm", "RMSNorm", "bf16"),
    "triton.gemm_bf16_aligned": CaseMetadata(
        "Triton", "Aligned BF16 GEMM", "GEMM", "bf16"
    ),
    "triton.gemm_bf16_ragged": CaseMetadata(
        "Triton", "Ragged BF16 GEMM", "GEMM", "bf16"
    ),
    "triton.attention_fp16": CaseMetadata(
        "Triton", "FP16 attention", "Attention", "fp16"
    ),
    "tensile.gemm_fp16": CaseMetadata("TensileLite", "FP16 GEMM", "GEMM", "fp16"),
    "tensile.gemm_bf16_batched": CaseMetadata(
        "TensileLite", "Batched BF16 GEMM", "GEMM", "bf16"
    ),
    "tensile.gemm_fp8_scaled": CaseMetadata(
        "TensileLite", "Scaled FP8 GEMM", "GEMM", "fp8"
    ),
}


def provider_for(case_id: str) -> str:
    """Derive the public provider name from a built-in case ID."""

    match = CASE_ID.fullmatch(case_id)
    if match is None:
        raise RunnerError(f"invalid benchmark case ID {case_id!r}")
    return match.group(1)


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RunnerError(f"{field} must be a non-empty array of strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise RunnerError(f"{field} must be a non-empty array of strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise RunnerError(f"{field} must not contain duplicates")
    return result


def _integer(value: Any, field: str, *, allow_zero: bool) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise RunnerError(f"{field} must be a {qualifier} integer")
    return value


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> Suite:
    """Read and validate the intentionally small suite manifest."""

    manifest = Path(path).expanduser().resolve()
    try:
        value = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RunnerError(f"cannot read manifest {manifest}: {error}") from error
    fields = set(value)
    if fields != MANIFEST_FIELDS:
        missing = sorted(MANIFEST_FIELDS - fields)
        extra = sorted(fields - MANIFEST_FIELDS)
        raise RunnerError(f"manifest fields differ: missing={missing}, extra={extra}")
    name = value["name"]
    if not isinstance(name, str) or not name:
        raise RunnerError("name must be a non-empty string")
    targets = _string_list(value["targets"], "targets")
    unknown_targets = sorted(set(targets) - set(TARGET_CONFIGS))
    if unknown_targets:
        raise RunnerError(f"unknown targets: {unknown_targets}")
    cases = _string_list(value["cases"], "cases")
    for case_id in cases:
        provider_for(case_id)
        if case_id not in CASE_METADATA:
            raise RunnerError(f"benchmark case has no metadata: {case_id!r}")
    timeout = value["timeout_seconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise RunnerError("timeout_seconds must be a positive number")
    return Suite(
        name=name,
        targets=targets,
        cases=cases,
        warmups=_integer(value["warmups"], "warmups", allow_zero=True),
        samples=_integer(value["samples"], "samples", allow_zero=False),
        timeout_seconds=float(timeout),
    )


def select_matrix(
    suite: Suite,
    *,
    targets: Sequence[str] = (),
    cases: Sequence[str] = (),
) -> tuple[Cell, ...]:
    """Select cells while retaining manifest case-major ordering."""

    requested_targets = set(targets)
    requested_cases = set(cases)
    unknown_targets = sorted(requested_targets - set(suite.targets))
    unknown_cases = sorted(requested_cases - set(suite.cases))
    if unknown_targets:
        raise RunnerError(f"selected targets are not in the suite: {unknown_targets}")
    if unknown_cases:
        raise RunnerError(f"selected cases are not in the suite: {unknown_cases}")
    chosen_targets = tuple(
        target
        for target in suite.targets
        if not requested_targets or target in requested_targets
    )
    chosen_cases = tuple(
        case for case in suite.cases if not requested_cases or case in requested_cases
    )
    return tuple(
        Cell(case, target) for case in chosen_cases for target in chosen_targets
    )


def _program(build_dir: Path, cell: Cell) -> tuple[str, ...]:
    if cell.provider == "hip":
        program = build_dir / "benchmarks" / f"rocjitsu-benchmark-native-{cell.target}"
        return (str(program),)
    if cell.provider == "triton":
        return (sys.executable, "-m", "benchmarks.workloads.triton_workloads")
    return (str(build_dir / "benchmarks" / "rocjitsu-benchmark-hipblaslt"),)


def prepare_command(
    build_dir: str | Path,
    output: str | Path,
    cell: Cell,
    *,
    warmups: int,
    samples: int,
) -> PreparedCommand:
    """Construct one shell-free Rocjitsu workload command."""

    build = Path(build_dir).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    workload_path = output_root / "cases" / cell.case / cell.target / "workload.json"
    payload = _program(build, cell) + (
        "--case",
        cell.case,
        "--target",
        cell.target,
        "--warmups",
        str(warmups),
        "--samples",
        str(samples),
        "--output",
        str(workload_path),
    )
    environment = dict(os.environ)
    # Official runs always use the device libraries installed with the selected
    # SDK, never a caller-provided source-build or development override.
    environment.pop("HIPBLASLT_TENSILE_LIBPATH", None)
    environment["PYTHONHASHSEED"] = "0"
    environment["TRITON_CACHE_DIR"] = str(
        output_root / "cache" / "triton" / cell.target
    )
    return PreparedCommand(
        argv=(
            str(build / "tools" / "rocjitsu" / "rocjitsu"),
            "--config",
            str(TARGET_CONFIGS[cell.target]),
            "--",
            *payload,
        ),
        cwd=ROCJITSU_ROOT,
        environment=environment,
        workload_path=workload_path,
    )


def _require_file(path: Path, description: str, *, executable: bool = False) -> None:
    if not path.is_file():
        raise RunnerError(f"missing {description}: {path}")
    if executable and not os.access(path, os.X_OK):
        raise RunnerError(f"{description} is not executable: {path}")


def validate_build(build_dir: str | Path, matrix: Sequence[Cell]) -> str:
    """Require a Release build and every file needed by the selected matrix."""

    build = Path(build_dir).expanduser().resolve()
    cache = build / "CMakeCache.txt"
    _require_file(cache, "CMake cache")
    build_type = None
    source_dir = None
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CMAKE_BUILD_TYPE:"):
            build_type = line.partition("=")[2]
        elif line.startswith("CMAKE_HOME_DIRECTORY:"):
            source_dir = line.partition("=")[2]
    if build_type != "Release":
        raise RunnerError(f"benchmark build must be Release, got {build_type!r}")
    if source_dir is None or Path(source_dir).resolve() != ROCJITSU_ROOT.resolve():
        raise RunnerError(
            f"benchmark build belongs to {source_dir!r}, expected {str(ROCJITSU_ROOT)!r}"
        )
    _require_file(
        build / "tools" / "rocjitsu" / "rocjitsu", "rocjitsu", executable=True
    )
    for target in {cell.target for cell in matrix}:
        _require_file(TARGET_CONFIGS[target], f"{target} configuration")
    for cell in matrix:
        program = Path(_program(build, cell)[0])
        if cell.provider != "triton":
            _require_file(program, f"{cell.provider} workload", executable=True)
    if any(cell.provider == "triton" for cell in matrix):
        _require_file(
            BENCHMARK_ROOT / "workloads" / "triton_workloads.py", "Triton workload"
        )
    return build_type


def _target_metadata(target: str) -> TargetMetadata:
    path = TARGET_CONFIGS[target]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunnerError(
            f"cannot read target configuration {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise RunnerError(f"target configuration must be a JSON object: {path}")
    exec_mode = value.get("exec_mode")
    num_threads = value.get("num_threads")
    if not isinstance(exec_mode, str) or not exec_mode:
        raise RunnerError(f"target configuration has invalid exec_mode: {path}")
    if (
        isinstance(num_threads, bool)
        or not isinstance(num_threads, int)
        or num_threads <= 0
    ):
        raise RunnerError(f"target configuration has invalid num_threads: {path}")
    return TargetMetadata(exec_mode=exec_mode, num_threads=num_threads)


def _camel_case(key: str) -> str:
    head, *tail = key.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


def _dashboard_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RunnerError("workload parameter keys must be strings")
            dashboard_key = _camel_case(key)
            if dashboard_key in normalized:
                raise RunnerError(
                    f"workload parameter keys collide as {dashboard_key!r}"
                )
            normalized[dashboard_key] = _dashboard_value(item)
        return normalized
    if isinstance(value, list):
        return [_dashboard_value(item) for item in value]
    return value


def validate_workload(path: Path, cell: Cell, samples: int) -> dict[str, Any]:
    """Validate and aggregate one workload's small JSON contract."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            reject_constant(value)
        return parsed

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise RunnerError(f"cannot read workload result: {error}") from error
    if not isinstance(value, Mapping):
        raise RunnerError("workload result must be a JSON object")
    if set(value) != WORKLOAD_FIELDS:
        raise RunnerError("workload result must contain exactly the schema fields")
    expected = {
        "schema": "rocjitsu.benchmark.workload.v1",
        "case": cell.case,
        "target": cell.target,
        "provider": cell.provider,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise RunnerError(
                f"workload {field} is {value.get(field)!r}, expected {expected_value!r}"
            )
    parameters = value.get("parameters")
    if not isinstance(parameters, dict):
        raise RunnerError("workload parameters must be an object")
    timings = value.get("timings_ns")
    if not isinstance(timings, list) or len(timings) != samples:
        actual = len(timings) if isinstance(timings, list) else "not a list"
        raise RunnerError(f"workload has {actual} timings, expected {samples}")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in timings
    ):
        raise RunnerError("workload timings must be positive integers")
    median = statistics.median(timings)
    return {
        "problem": _dashboard_value(parameters),
        "durationSeconds": median / 1_000_000_000,
        "timing": {
            "unit": "ns",
            "samples": timings,
            "minimum": min(timings),
            "median": median,
            "maximum": max(timings),
        },
    }


def _utc_now() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    )


def _normalize_timestamp(value: str) -> str | None:
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _source_info() -> dict[str, Any]:
    revision = None
    commit_timestamp = None
    dirty = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROCJITSU_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    if revision is not None:
        try:
            value = subprocess.check_output(
                ["git", "show", "-s", "--format=%cI", revision],
                cwd=ROCJITSU_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            commit_timestamp = _normalize_timestamp(value)
        except (OSError, subprocess.CalledProcessError):
            pass
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ROCJITSU_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "commit_sha": revision,
        "commit_timestamp": commit_timestamp,
        "dirty": dirty,
    }


def _environment_info() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "packages": packages,
    }


def _provenance(
    source: Mapping[str, Any], environment: Mapping[str, Any], build_type: str
) -> dict[str, Any]:
    packages = environment["packages"]
    return {
        "rocjitsuCommitSha": source["commit_sha"],
        "rocjitsuCommitTimestamp": source["commit_timestamp"],
        "dirty": source["dirty"],
        "buildType": build_type,
        "rocmSdkVersion": packages["rocm-sdk-devel"],
        "pythonVersion": environment["python"],
        "torchVersion": packages["torch"],
        "tritonVersion": packages["triton"],
        "tritonCommitSha": None,
        "tensileLiteCommitSha": None,
        "packages": packages,
    }


def _write_run(output: Path, run: dict[str, Any]) -> None:
    temporary = output / ".run.json.tmp"
    temporary.write_text(
        json.dumps(run, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output / "run.json")


def _captured_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Run one cell and leave no descendant processes behind."""

    def terminate_group() -> None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()

    def terminate() -> tuple[str, str]:
        terminate_group()
        return process.communicate()

    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stdout, stderr = terminate()
        raise subprocess.TimeoutExpired(
            argv, timeout, output=stdout, stderr=stderr
        ) from None
    except BaseException:
        try:
            terminate()
        except BaseException:
            pass
        raise
    if os.name == "posix":
        terminate_group()
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _failed_test(cell: Cell, target: TargetMetadata, error: str) -> dict[str, Any]:
    metadata = CASE_METADATA[cell.case]
    base = f"cases/{cell.case}/{cell.target}"
    return {
        "testId": f"{cell.target}:{cell.case}",
        "logicalTestId": cell.case,
        "suite": metadata.suite,
        "name": metadata.name,
        "target": cell.target,
        "operation": metadata.operation,
        "dataType": metadata.data_type,
        "problem": None,
        "execMode": target.exec_mode,
        "numThreads": target.num_threads,
        "durationSeconds": None,
        "timing": {
            "unit": "ns",
            "samples": [],
            "minimum": None,
            "median": None,
            "maximum": None,
        },
        "status": "failed",
        "exitCode": None,
        "timedOut": False,
        "error": error,
        "artifacts": {
            "workload": f"{base}/workload.json",
            "stdout": f"{base}/stdout.txt",
            "stderr": f"{base}/stderr.txt",
        },
    }


def run_suite(
    suite: Suite,
    matrix: Sequence[Cell],
    *,
    build_dir: str | Path,
    output: str | Path,
    warmups: int | None = None,
    samples: int | None = None,
) -> dict[str, Any]:
    """Run all selected cells, preserving partial results after failures."""

    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise RunnerError(f"output already exists: {output_path}")
    selected_warmups = (
        suite.warmups
        if warmups is None
        else _integer(warmups, "warmups", allow_zero=True)
    )
    selected_samples = (
        suite.samples
        if samples is None
        else _integer(samples, "samples", allow_zero=False)
    )
    build = Path(build_dir).expanduser().resolve()
    build_type = validate_build(build, matrix)
    targets = tuple(dict.fromkeys(cell.target for cell in matrix))
    target_metadata = {target: _target_metadata(target) for target in targets}
    started = time.monotonic()
    timestamp = _utc_now()
    source = _source_info()
    environment = _environment_info()
    output_path.mkdir(parents=True)
    run: dict[str, Any] = {
        "schemaVersion": 1,
        "timestamp": timestamp,
        "finishedAt": None,
        "status": "running",
        "wallTimeSeconds": 0.0,
        "benchmarkSuite": suite.name,
        "targets": list(targets),
        "measurement": {
            "warmups": selected_warmups,
            "samples": selected_samples,
            "timeoutSeconds": suite.timeout_seconds,
        },
        "provenance": _provenance(source, environment, build_type),
        "environment": {
            key: environment[key] for key in ("hostname", "platform", "kernel", "cpu")
        },
        "tests": [],
    }
    _write_run(output_path, run)

    for cell in matrix:
        command = prepare_command(
            build, output_path, cell, warmups=selected_warmups, samples=selected_samples
        )
        cell_dir = command.workload_path.parent
        cell_dir.mkdir(parents=True)
        (output_path / "cache" / "triton" / cell.target).mkdir(
            parents=True, exist_ok=True
        )
        stdout = ""
        stderr = ""
        result = _failed_test(
            cell, target_metadata[cell.target], "workload did not run"
        )
        try:
            completed = _run_command(
                command.argv,
                cwd=command.cwd,
                env=command.environment,
                timeout=suite.timeout_seconds,
            )
            stdout = _captured_text(completed.stdout)
            stderr = _captured_text(completed.stderr)
            result["exitCode"] = completed.returncode
            if completed.returncode != 0:
                raise RunnerError(f"command exited with status {completed.returncode}")
            aggregate = validate_workload(command.workload_path, cell, selected_samples)
            result.update(aggregate)
            result["status"] = "completed"
            result["error"] = None
        except subprocess.TimeoutExpired as error:
            stdout = _captured_text(error.stdout)
            stderr = _captured_text(error.stderr)
            result["status"] = "timeout"
            result["timedOut"] = True
            result["error"] = (
                f"command timed out after {suite.timeout_seconds:g} seconds"
            )
        except (OSError, RunnerError) as error:
            result["error"] = str(error)
        if not command.workload_path.is_file():
            result["artifacts"]["workload"] = None
        (cell_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
        (cell_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
        run["tests"].append(result)
        run["wallTimeSeconds"] = time.monotonic() - started
        _write_run(output_path, run)

    run["status"] = (
        "completed"
        if all(item["status"] == "completed" for item in run["tests"])
        else "failed"
    )
    run["finishedAt"] = _utc_now()
    run["wallTimeSeconds"] = time.monotonic() - started
    _write_run(output_path, run)
    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        suite = load_manifest(arguments.manifest)
        matrix = select_matrix(suite, targets=arguments.target, cases=arguments.case)
        if arguments.list:
            for cell in matrix:
                print(f"{cell.case}\t{cell.target}")
            return 0
        if arguments.build_dir is None or arguments.output is None:
            raise RunnerError(
                "--build-dir and --output are required unless --list is used"
            )
        run = run_suite(
            suite,
            matrix,
            build_dir=arguments.build_dir,
            output=arguments.output,
            warmups=arguments.warmups,
            samples=arguments.samples,
        )
        artifact = arguments.output.expanduser().resolve() / "run.json"
        print(f"run {run['status']}: {artifact}")
        return 0 if run["status"] == "completed" else 1
    except RunnerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
