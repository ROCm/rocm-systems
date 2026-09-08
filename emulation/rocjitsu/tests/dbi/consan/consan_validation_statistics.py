"""ConSan overhead and empirical-campaign statistics."""

from __future__ import annotations

import hashlib
import math
import random
import statistics

from consan_validation_catalog import (
    EMPIRICAL_CAMPAIGN_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ValidationError,
)


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
