"""Bounded ConSan workload execution and per-profile result assembly."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import threading
import time

from consan_validation_catalog import (
    HIP_MOI_GPU_BENCHMARK_ITERATIONS_ENV,
    HIP_MOI_GPU_BENCHMARK_WARMUP_ITERATIONS_ENV,
    PROCESS_OUTPUT_DRAIN_SECONDS,
    PROCESS_TERMINATION_GRACE_SECONDS,
    SCHEMA_VERSION,
    ValidationError,
    Workload,
    _resolved_workload,
)
from consan_validation_commands import (
    _controlled_environment,
    _hook_path,
    _outer_repetitions,
    _run_environment,
    _source_identities,
    _with_launcher,
    _workload_commands,
    _workload_provenance_path,
)
from consan_validation_diagnostics import (
    _benchmark_samples,
    _coverage_summary,
    _discard_first_sample_per_process,
    _empirical_structural_metrics,
    _gtest_device_measurement,
    _gtest_test_count,
    _gtest_timing_samples,
    _json_measurements,
    _json_timing_samples,
    _retained_code_object_inventory,
)
from consan_validation_support import atomic_write_json, sha256_file


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
