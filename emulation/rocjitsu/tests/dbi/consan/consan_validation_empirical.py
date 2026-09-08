"""ConSan resumable empirical-campaign orchestration and timing protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

from consan_validation_catalog import (
    EMPIRICAL_CAMPAIGN_SCHEMA_VERSION,
    EMPIRICAL_MAX_INNER_REPETITIONS,
    EMPIRICAL_MINIMUM_TIMED_MS,
    PROFILE_IDS,
    SCHEMA_VERSION,
    SINGLE_REPETITION_TARGETS,
    ValidationError,
    Workload,
    _resolved_workload,
)
from consan_validation_commands import (
    _doctor,
    _resolve_workload_selection,
    _workload_provenance_path,
    _workspace_from_environment,
    _write_provenance,
)
from consan_validation_diagnostics import _empirical_structural_totals
from consan_validation_execution import _run_profile
from consan_validation_statistics import (
    _bootstrap_stream_seed,
    _combine_empirical_round_summaries,
    _empirical_campaign_summary,
    _empirical_round_summary,
)
from consan_validation_support import atomic_write_json


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
