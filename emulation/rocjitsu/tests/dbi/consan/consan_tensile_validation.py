#!/usr/bin/env python3
"""Runs one bounded, numerically validated Tensile workload."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

from consan_tensile_support import (
    DEFAULT_TARGET,
    TensileValidationPaths,
    resolve_tensile_validation_paths,
)

DEFAULT_TIMEOUT_SECONDS = 55
CODE_OBJECT_BUDGET_SECONDS = 5
OUTPUT_DRAIN_SECONDS = 2
TERMINATION_GRACE_SECONDS = 3
_TENSILE_DRIVER = """
from Tensile import Tensile as tensile
import sys
tensile.Tensile(sys.argv[1:])
"""


def _prepend(environment: dict[str, str], name: str, value: Path) -> None:
    previous = environment.get(name)
    environment[name] = f"{value}{os.pathsep}{previous}" if previous else str(value)


def _oracle_payload(outcome: str, detail: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "oracle": outcome,
        "detail": detail,
        "source_diagnostics": {
            "outcome": "not_applicable",
            "count": None,
            "expectation": "not_applicable",
            "detail": "Tensile numeric validation has no separate diagnostic channel",
        },
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_oracle_result(
    outcome: str, detail: object, *, retained_path: Path | None = None
) -> None:
    payload = _oracle_payload(outcome, detail)
    result_path = os.environ.get("CONSAN_ROW_RESULT_PATH") or os.environ.get(
        "CONSAN_WORKLOAD_RESULT_PATH"
    )
    paths = ([Path(result_path)] if result_path else []) + (
        [retained_path] if retained_path is not None else []
    )
    for path in dict.fromkeys(paths):
        _write_json_atomic(path, payload)


def _resolve_config(workspace: Path, configured: Path) -> Path:
    candidate = configured.expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    config = candidate.resolve()
    try:
        config.relative_to(workspace)
    except ValueError as error:
        raise ValueError(
            f"config is outside the validation workspace: {config}"
        ) from error
    if not config.is_file():
        raise ValueError(f"config is not a file: {config}")
    return config


def _prerequisite_errors(
    paths: TensileValidationPaths,
    config: Path,
    *,
    streamk_fixed_grid: int | None = None,
) -> list[str]:
    checks = (
        (
            "Tensile Python package",
            paths.tensilelite / "Tensile",
            "directory",
            "CONSAN_VALIDATION_TENSILELITE_ROOT",
        ),
        ("ROCm root", paths.rocm, "directory", "CONSAN_VALIDATION_ROCM_ROOT"),
        (
            "amdclang++",
            paths.rocm / "bin" / "amdclang++",
            "executable",
            "CONSAN_VALIDATION_ROCM_ROOT",
        ),
        (
            "Tensile client",
            paths.client,
            "executable",
            "CONSAN_VALIDATION_TENSILE_CLIENT",
        ),
        (
            "Tensile launcher wrapper",
            paths.wrapper,
            "executable",
            "CONSAN_VALIDATION_TENSILE_WRAPPER",
        ),
        (
            "RocJITsu launcher",
            paths.rocjitsu,
            "executable",
            "CONSAN_VALIDATION_ROCJITSU_EXE",
        ),
        (
            "RocJITsu target config",
            paths.rocjitsu_config,
            "file",
            "CONSAN_VALIDATION_ROCJITSU_CONFIG",
        ),
        (
            "llvm-readelf",
            paths.llvm_readelf,
            "executable",
            "CONSAN_VALIDATION_LLVM_READELF",
        ),
        ("Tensile workload config", config, "file", None),
    )
    errors = []
    for label, path, kind, override in checks:
        present = (
            path.is_dir()
            if kind == "directory"
            else path.is_file() and (kind != "executable" or os.access(path, os.X_OK))
        )
        if not present:
            suffix = f" (override with {override})" if override else ""
            errors.append(f"missing {label}: {path}{suffix}")
    if streamk_fixed_grid is not None:
        hardware_header = paths.tensilelite / "include" / "Tensile" / "AMDGPU.hpp"
        try:
            hardware_source = hardware_header.read_text(encoding="utf-8")
        except OSError as error:
            errors.append(
                "cannot verify TENSILE_STREAMK_FIXED_GRID support in "
                f"{hardware_header}: {error}"
            )
        else:
            if 'std::getenv("TENSILE_STREAMK_FIXED_GRID")' not in hardware_source:
                errors.append(
                    "TensileLite does not expose TENSILE_STREAMK_FIXED_GRID in "
                    f"{hardware_header}"
                )
    return errors


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=OUTPUT_DRAIN_SECONDS)
        except subprocess.TimeoutExpired:
            pass


def _bounded_communicate(process: subprocess.Popen[str]) -> str:
    try:
        output, _ = process.communicate(timeout=OUTPUT_DRAIN_SECONDS)
        return output
    except subprocess.TimeoutExpired as error:
        if process.stdout is not None:
            process.stdout.close()
        output = error.output or ""
        return output.decode(errors="replace") if isinstance(output, bytes) else output


def _run_command(
    command: list[str], environment: dict[str, str], timeout_seconds: int
) -> tuple[int, str, bool]:
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    previous_handlers: dict[int, signal.Handlers] = {}

    def forward_termination(signum: int, _frame: object) -> None:
        _terminate_process_group(process)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, forward_termination)
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
        return process.returncode, output, False
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        output = _bounded_communicate(process)
        return process.returncode, output, True
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _numeric_validation_errors(
    output: str,
    *,
    expected_result_count: int | None = None,
    required_streamk_mode: int | None = None,
) -> tuple[int, list[str]]:
    errors = []
    if "WRONG_HARDWARE" in output:
        errors.append("Tensile selected incompatible hardware")

    result_count = 0
    header_width: int | None = None
    validation_index: int | None = None
    solution_index: int | None = None
    for line_number, line in enumerate(output.splitlines(), start=1):
        try:
            row = next(csv.reader([line], strict=True))
        except csv.Error:
            if header_width is not None and line.lstrip()[:1].isdigit():
                errors.append(f"line {line_number}: malformed numeric row")
            continue
        stripped = [field.strip() for field in row]
        if stripped and stripped[0] == "run" and "validation" in stripped:
            header_width = len(stripped)
            validation_index = stripped.index("validation")
            solution_index = (
                stripped.index("solution") if "solution" in stripped else None
            )
            continue
        if header_width is None or not stripped or not stripped[0].isdigit():
            continue
        if len(stripped) != header_width:
            errors.append(
                f"line {line_number}: malformed numeric row "
                f"(expected {header_width} columns, found {len(stripped)})"
            )
            continue
        result_count += 1
        assert validation_index is not None
        validation = stripped[validation_index]
        if validation != "PASSED":
            errors.append(
                f"line {line_number}: numeric validation is "
                f"{validation or '<empty>'}, not PASSED"
            )
        if required_streamk_mode is not None:
            expected = f"_SK{required_streamk_mode}_"
            if solution_index is None or expected not in stripped[solution_index]:
                errors.append(
                    f"line {line_number}: selected solution is not "
                    f"Stream-K mode {required_streamk_mode}"
                )
    if expected_result_count is not None and result_count != expected_result_count:
        errors.append(
            f"expected {expected_result_count} numeric result rows, "
            f"found {result_count}"
        )
    elif result_count == 0:
        errors.append("numeric validation produced no result rows")
    return result_count, errors


def _code_object_errors(
    output_dir: Path,
    llvm_readelf: Path,
    *,
    target: str = DEFAULT_TARGET,
    deadline: float | None = None,
) -> tuple[list[Path], list[str]]:
    discovered = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix in {".co", ".hsaco"}
    )
    if not discovered:
        return [], ["Tensile produced no AMDGPU code objects"]

    errors = []
    verified = []
    target_pattern = re.compile(rf"^  Flags:.*\b{re.escape(target)}\b", re.MULTILINE)
    for artifact in discovered:
        remaining = 10.0 if deadline is None else deadline - time.monotonic()
        if remaining <= 0:
            errors.append("code-object verification exceeded its budget")
            break
        try:
            header = subprocess.run(
                [str(llvm_readelf), "--file-header", str(artifact)],
                check=False,
                capture_output=True,
                text=True,
                timeout=min(10.0, remaining),
            )
        except subprocess.TimeoutExpired:
            errors.append(f"code-object check timed out: {artifact}")
            continue
        except OSError as error:
            errors.append(f"cannot inspect {artifact}: {error}")
            continue
        if header.returncode != 0:
            errors.append(f"llvm-readelf rejected {artifact}")
            continue
        valid = True
        if "Machine:" not in header.stdout or "EM_AMDGPU" not in header.stdout:
            errors.append(f"code object is not AMDGPU ELF: {artifact}")
            valid = False
        if target_pattern.search(header.stdout) is None:
            errors.append(f"code object does not declare {target}: {artifact}")
            valid = False
        if valid:
            verified.append(artifact)
    return verified, errors


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _gpu_target(value: str) -> str:
    if re.fullmatch(r"gfx[0-9a-z]+", value) is None:
        raise argparse.ArgumentTypeError("must name a gfx target")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-target", type=_gpu_target, default=DEFAULT_TARGET)
    parser.add_argument("--repetitions", type=int, choices=(1,), default=1)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--streamk-fixed-grid", type=_positive_int)
    parser.add_argument(
        "--require-streamk-mode",
        type=int,
        choices=range(1, 5),
    )
    parser.add_argument("--expect-numeric-rows", type=_positive_int)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    try:
        config = _resolve_config(workspace, args.config)
    except ValueError as error:
        parser.error(str(error))
    paths = resolve_tensile_validation_paths(workspace, args.gpu_target)
    prerequisite_errors = _prerequisite_errors(
        paths,
        config,
        streamk_fixed_grid=args.streamk_fixed_grid,
    )
    if prerequisite_errors:
        detail = {"config": str(config), "reasons": prerequisite_errors}
        _write_oracle_result("fail", detail)
        for error in prerequisite_errors:
            print(f"error: {error}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.label).strip("-") or "tensile"
    work_dir = Path(
        tempfile.mkdtemp(prefix=f"{label_slug}-", dir=args.output_dir.resolve())
    )
    transcript = work_dir / "validation.log"
    oracle_artifact = work_dir / "oracle.json"

    environment = os.environ.copy()
    environment.update(
        {
            "ROCM_PATH": str(paths.rocm),
            "HIP_PATH": str(paths.rocm),
            "ROCJITSU_EXE": str(paths.rocjitsu),
            "ROCJITSU_CONFIG": str(paths.rocjitsu_config),
            "TENSILE_CLIENT_EXE": str(paths.client),
        }
    )
    if args.streamk_fixed_grid is not None:
        environment["TENSILE_STREAMK_FIXED_GRID"] = str(args.streamk_fixed_grid)
    _prepend(environment, "PATH", paths.rocm / "bin")
    _prepend(environment, "LD_LIBRARY_PATH", paths.rocm / "lib")
    _prepend(environment, "PYTHONPATH", paths.tensilelite)

    command = [
        sys.executable,
        "-P",
        "-c",
        _TENSILE_DRIVER,
        str(config),
        str(work_dir),
        "--gpu-targets",
        args.gpu_target,
        "--prebuilt-client",
        str(paths.wrapper),
        "--global-parameters",
        "NumBenchmarks=1",
        "SyncsPerBenchmark=1",
        "EnqueuesPerSync=1",
        "NumWarmups=0",
    ]
    started = time.monotonic()
    try:
        returncode, output, timed_out = _run_command(
            command, environment, args.timeout_seconds
        )
    except OSError as error:
        detail = {"config": str(config), "reason": str(error)}
        _write_oracle_result("fail", detail, retained_path=oracle_artifact)
        print(f"error: cannot run Tensile: {error}", file=sys.stderr)
        return 2
    transcript.write_text(output, encoding="utf-8")
    if output:
        print(output, end="" if output.endswith("\n") else "\n")

    result_count, numeric_errors = _numeric_validation_errors(
        output,
        expected_result_count=args.expect_numeric_rows,
        required_streamk_mode=args.require_streamk_mode,
    )
    execution_elapsed_seconds = time.monotonic() - started
    artifacts: list[Path] = []
    artifact_errors: list[str] = []
    if not timed_out:
        artifacts, artifact_errors = _code_object_errors(
            work_dir,
            paths.llvm_readelf,
            target=args.gpu_target,
            deadline=time.monotonic() + CODE_OBJECT_BUDGET_SECONDS,
        )
    elapsed_seconds = time.monotonic() - started
    errors = []
    if timed_out:
        errors.append(
            f"Tensile exceeded its {args.timeout_seconds}-second execution budget"
        )
    elif returncode != 0:
        errors.append(f"Tensile exited with status {returncode}")
    if execution_elapsed_seconds > args.timeout_seconds and not timed_out:
        errors.append(
            f"Tensile execution exceeded its {args.timeout_seconds}-second budget"
        )
    errors.extend(numeric_errors)
    errors.extend(artifact_errors)
    detail = {
        "config": str(config),
        "elapsed_seconds": elapsed_seconds,
        "execution_elapsed_seconds": execution_elapsed_seconds,
        "expected_numeric_rows": args.expect_numeric_rows,
        "label": args.label,
        "numeric_rows": result_count,
        "rocjitsu_config": str(paths.rocjitsu_config),
        "rocjitsu_executable": str(paths.rocjitsu),
        "required_streamk_mode": args.require_streamk_mode,
        "requested_streamk_fixed_grid": args.streamk_fixed_grid,
        "target": args.gpu_target,
        "timeout_seconds": args.timeout_seconds,
        "transcript": str(transcript),
        "verified_code_objects": [str(path) for path in artifacts],
        "wrapper": str(paths.wrapper),
    }
    if errors:
        detail["reasons"] = errors
        _write_oracle_result("fail", detail, retained_path=oracle_artifact)
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                args.label: {
                    "median_ms": execution_elapsed_seconds * 1000.0,
                    "oracle_passed": True,
                    "repetitions": args.repetitions,
                }
            },
            sort_keys=True,
        )
    )
    _write_oracle_result("pass", detail, retained_path=oracle_artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
