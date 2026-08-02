#!/usr/bin/env python3
"""Verify and render a cross-workload ConSan empirical study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

PROFILE_ORDER = (
    "supercollider",
    "record-replay",
    "sampled",
    "inline-shadow",
)
PROFILE_LABELS = {
    "supercollider": "SuperCollider",
    "record-replay": "Record/Replay",
    "sampled": "Sampled",
    "inline-shadow": "Inline Shadow",
}
TEXT_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hip", ".hpp", ".inc", ".json", ".py"}


class StudyError(RuntimeError):
    """Raised when retained evidence does not satisfy the frozen study."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise StudyError(f"JSON document must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StudyError(message)


def _discover_one(root: Path, pattern: str, kind: str) -> Path:
    matches = sorted(root.glob(pattern))
    _require(
        len(matches) == 1,
        f"expected exactly one {kind} below {root}, found {len(matches)}",
    )
    return matches[0]


def discover_documents(
    manifest: dict[str, Any], artifact_base: Path
) -> tuple[
    list[tuple[str, Path, dict[str, Any]]], list[tuple[str, Path, dict[str, Any]]]
]:
    performance: list[tuple[str, Path, dict[str, Any]]] = []
    detection: list[tuple[str, Path, dict[str, Any]]] = []
    for name in manifest.get("performance_artifacts", []):
        root = artifact_base / name
        path = _discover_one(
            root, "*/empirical-campaign/campaign.json", "campaign.json"
        )
        performance.append((name, path, _read_json(path)))
    for name in manifest.get("detection_artifacts", []):
        root = artifact_base / name
        path = _discover_one(root, "*/faults/*/summary.json", "fault summary")
        detection.append((name, path, _read_json(path)))
    _require(performance, "manifest contains no performance artifacts")
    _require(detection, "manifest contains no detection artifacts")
    return performance, detection


def _source_head(provenance: dict[str, Any], repository_name: str) -> str:
    matches = [
        source.get("head")
        for source in provenance.get("sources", [])
        if isinstance(source, dict)
        and Path(str(source.get("root", ""))).name == repository_name
    ]
    _require(
        len(matches) == 1 and isinstance(matches[0], str),
        f"missing unique {repository_name} source head",
    )
    return matches[0]


def _file_hash(provenance: dict[str, Any], name: str) -> str:
    value = provenance.get("files", {}).get(name, {}).get("sha256")
    _require(
        isinstance(value, str) and len(value) == 64,
        f"missing provenance hash for {name}",
    )
    return value


def _available_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get("available") is True:
        return value.get("value")
    return value


def provenance_signature(
    provenance: dict[str, Any], required_runtime_tools: Iterable[str] = ()
) -> dict[str, Any]:
    machine = provenance.get("machine", {})
    selected_nodes = machine.get("selected_kfd_nodes", [])
    topology = machine.get("kfd_topology", {})
    selected_topology = {
        node: {
            "gpu_id": _available_value(topology.get(node, {}).get("gpu_id")),
            "name": _available_value(topology.get(node, {}).get("name")),
            "properties": _available_value(topology.get(node, {}).get("properties")),
        }
        for node in selected_nodes
    }
    runtime_tools = provenance.get("runtime_tools", {})
    tool_outputs = {}
    for name in required_runtime_tools:
        identity = runtime_tools.get(name, {})
        output_hash = identity.get("output_sha256")
        _require(
            identity.get("available") is True
            and isinstance(output_hash, str)
            and len(output_hash) == 64,
            f"required runtime tool {name} has no usable identity",
        )
        tool_outputs[name] = output_hash
    return {
        "target": provenance.get("target"),
        "source_head": _source_head(provenance, "rocm-systems"),
        "hook": _file_hash(provenance, "hook"),
        "llvm_readelf": _file_hash(provenance, "llvm-readelf"),
        "selectors": provenance.get("environment_selectors"),
        "selected_topology": selected_topology,
        "uname": machine.get("uname"),
        "driver_source": machine.get("amdgpu_module_source_version"),
        "tool_outputs": tool_outputs,
    }


def _source_heads(provenance: dict[str, Any]) -> dict[str, str]:
    output = {}
    for source in provenance.get("sources", []):
        if not isinstance(source, dict):
            continue
        name = Path(str(source.get("root", ""))).name
        head = source.get("head")
        _require(
            isinstance(head, str) and len(head) == 40,
            f"source {name} has no commit identity",
        )
        _require(source.get("dirty") is False, f"source {name} was dirty")
        output[name] = head
    return output


def workload_provenance_signature(
    provenance: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    file_hashes = {
        label: _file_hash(provenance, label) for label in config.get("file_labels", [])
    }
    runtime = provenance.get("workload_runtime", {})
    libraries = runtime.get("loaded_runtime_libraries", {})
    runtime_hashes = {}
    for label in config.get("required_runtime_libraries", []):
        value = libraries.get(label, {}).get("sha256")
        _require(
            isinstance(value, str) and len(value) == 64,
            f"{provenance.get('workload')}: missing loaded {label} identity",
        )
        runtime_hashes[label] = value
    observed_sources = _source_heads(provenance)
    expected_sources = config.get("source_heads", {})
    for name, head in expected_sources.items():
        _require(
            observed_sources.get(name) == head,
            f"{provenance.get('workload')}: source {name} mismatch",
        )
    return {
        "files": file_hashes,
        "runtime_libraries": runtime_hashes,
        "package_document": runtime.get("package_document"),
        "sources": {name: observed_sources[name] for name in expected_sources},
    }


def _performance_provenance_path(campaign_path: Path) -> Path:
    return campaign_path.parents[1] / "provenance.json"


def _detection_provenance_path(summary_path: Path) -> Path:
    return summary_path.parent / "provenance.json"


def validate_common_provenance(
    manifest: dict[str, Any],
    performance: list[tuple[str, Path, dict[str, Any]]],
    detection: list[tuple[str, Path, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    provenances: dict[str, dict[str, Any]] = {}
    workload_by_artifact = {}
    for name, path, campaign in performance:
        provenances[name] = _read_json(_performance_provenance_path(path))
        workload_by_artifact[name] = campaign.get("workload")
    for name, path, summary in detection:
        provenances[name] = _read_json(_detection_provenance_path(path))
        workload_by_artifact[name] = summary.get("workload")
        spec_hash = summary.get("fault_spec", {}).get("sha256")
        _require(
            spec_hash == manifest.get("fault_spec_sha256"),
            f"{name}: fault-spec hash mismatch",
        )

    first_name = next(iter(provenances))
    required_tools = manifest.get("required_runtime_tools", [])
    for name, provenance in provenances.items():
        _require(
            provenance.get("provenance_schema_version")
            == manifest.get("provenance_schema_version"),
            f"{name}: provenance schema mismatch",
        )
    reference = provenance_signature(provenances[first_name], required_tools)
    for name, provenance in provenances.items():
        signature = provenance_signature(provenance, required_tools)
        _require(
            signature == reference,
            f"{name}: stable provenance differs from {first_name}",
        )
    _require(
        reference["target"] == manifest.get("target"), "provenance target mismatch"
    )
    _require(
        reference["source_head"] == manifest.get("source_head"),
        "RocJITsu source head mismatch",
    )
    _require(
        reference["hook"] == manifest.get("hook_sha256"), "ConSan hook hash mismatch"
    )
    workload_configs = manifest.get("workload_identities", {})
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for name, provenance in provenances.items():
        workload = workload_by_artifact[name]
        config = workload_configs.get(workload)
        _require(isinstance(config, dict), f"missing identity lock for {workload}")
        grouped.setdefault(str(workload), []).append(
            (name, workload_provenance_signature(provenance, config))
        )
    for workload, identities in grouped.items():
        _require(
            len(identities) >= 2,
            f"{workload}: performance and detection identities are not both present",
        )
        reference_name, workload_reference = identities[0]
        for name, identity in identities[1:]:
            _require(
                identity == workload_reference,
                f"{workload}: runtime identity differs between {reference_name} and {name}",
            )
    return reference, provenances


def _result_path(campaign_path: Path, relative: str) -> Path:
    return campaign_path.parents[2] / relative


def _admission_result(
    campaign_path: Path, campaign: dict[str, Any], profile: str
) -> dict[str, Any]:
    row = campaign.get("admission", {}).get("rows", {}).get(profile)
    _require(
        isinstance(row, dict),
        f"{campaign.get('workload')}: missing admission row for {profile}",
    )
    relative = row.get("result")
    _require(
        isinstance(relative, str),
        f"{campaign.get('workload')}/{profile}: missing admission result",
    )
    return _read_json(_result_path(campaign_path, relative))


def _coverage_reason(result: dict[str, Any]) -> str:
    reasons = result.get("coverage", {}).get("reasons", [])
    if isinstance(reasons, list) and reasons:
        return "; ".join(str(reason) for reason in reasons)
    returncodes = result.get("returncodes", [])
    if isinstance(returncodes, list) and any(code != 0 for code in returncodes):
        return (
            "runtime rejected (return codes "
            + ",".join(str(code) for code in returncodes)
            + ")"
        )
    return "clean admission rejected"


def _metric_kind(metric: str) -> str:
    if metric == "cold:process":
        return "cold_process"
    if metric.startswith("cold:workload:"):
        return "cold_device" if metric.endswith(":device") else "cold_workload"
    if metric.startswith("warm:workload:") and metric.endswith(":device"):
        return "warm_device"
    if metric.startswith("warm:workload:"):
        return "warm_host"
    raise StudyError(f"unmapped empirical metric {metric}")


def extract_metric_cells(
    campaign: dict[str, Any], allowed_unqualified: set[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    workload = campaign.get("workload")
    required = campaign.get("required_accepted_rounds")
    _require(isinstance(workload, str), "campaign workload is missing")
    _require(
        isinstance(required, int) and required >= 10,
        f"{workload}: invalid required round count",
    )
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for profile, profile_summary in (
        campaign.get("summary", {}).get("profiles", {}).items()
    ):
        metrics = profile_summary.get("metrics", {})
        profile_cells: dict[str, dict[str, Any]] = {}
        for metric_name, metric in metrics.items():
            kind = _metric_kind(metric_name)
            _require(
                kind not in profile_cells,
                f"{workload}/{profile}: duplicate {kind} metric",
            )
            slowdown = metric.get("slowdown", {})
            timing = metric.get("timing_ms", {})
            count = slowdown.get("count")
            key = f"{workload}/{profile}/{metric_name}"
            qualified = isinstance(count, int) and count >= required
            if not qualified:
                _require(
                    key in allowed_unqualified,
                    f"unexpected underqualified metric {key}: {count}/{required}",
                )
            interval = slowdown.get("bootstrap_median_95", {})
            profile_cells[kind] = {
                "metric": metric_name,
                "qualified": qualified,
                "count": count,
                "required": required,
                "slowdown": slowdown.get("median"),
                "timing_ms": timing.get("median"),
                "lower": interval.get("lower"),
                "upper": interval.get("upper"),
            }
        cells[profile] = profile_cells
    return cells


def _median(values: list[float | int]) -> float | None:
    return float(statistics.median(values)) if values else None


def _round_results(campaign_path: Path) -> Iterable[dict[str, Any]]:
    for round_path in sorted(
        (campaign_path.parent / "rounds").glob("round-*/round.json")
    ):
        round_document = _read_json(round_path)
        for relative in round_document.get("row_results", []):
            if isinstance(relative, str):
                yield _read_json(_result_path(campaign_path, relative))


def _run_count(result: dict[str, Any]) -> int:
    return max(1, len(result.get("returncodes") or []))


def _unexpected_diagnostic_count(result: dict[str, Any]) -> int:
    total = 0
    coverage_runs = result.get("coverage_runs") or [result.get("coverage", {})]
    for coverage in coverage_runs:
        if not isinstance(coverage, dict):
            continue
        value = coverage.get("diagnostics", {}).get("diagnostic_count", 0)
        if isinstance(value, int):
            total += value
    return total


def structural_metrics(
    campaign_path: Path, campaign: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    profiles = set(campaign.get("timed_profiles", []))
    samples: dict[str, list[dict[str, float]]] = {profile: [] for profile in profiles}
    clean_gate_runs: dict[str, int] = {profile: 0 for profile in profiles}
    clean_gate_rejections: dict[str, int] = {profile: 0 for profile in profiles}
    unexpected_diagnostics: dict[str, int] = {profile: 0 for profile in profiles}
    metadata_nonzero: dict[str, int] = {profile: 0 for profile in profiles}
    metadata_compared: dict[str, int] = {profile: 0 for profile in profiles}
    for profile in profiles:
        admission = _admission_result(campaign_path, campaign, profile)
        clean_gate_runs[profile] += _run_count(admission)
        unexpected_diagnostics[profile] += _unexpected_diagnostic_count(admission)
        if admission.get("accepted") is not True:
            clean_gate_rejections[profile] += _run_count(admission)
        for pair in admission.get("retained_code_objects", {}).get("pairs", []):
            kernels = pair.get("kernel_metadata_delta", {}).get("kernels", {})
            for delta in kernels.values():
                metadata_compared[profile] += sum(
                    1 for value in delta.values() if isinstance(value, (int, float))
                )
                metadata_nonzero[profile] += sum(
                    1
                    for value in delta.values()
                    if isinstance(value, (int, float)) and value != 0
                )
    for result in _round_results(campaign_path):
        profile = result.get("profile")
        if profile not in profiles:
            continue
        clean_gate_runs[profile] += _run_count(result)
        unexpected_diagnostics[profile] += _unexpected_diagnostic_count(result)
        if result.get("accepted") is not True:
            clean_gate_rejections[profile] += _run_count(result)
            continue
        for structural in result.get("structural_metrics_runs") or []:
            if (
                not isinstance(structural, dict)
                or structural.get("accepted") is not True
            ):
                continue
            modified = [
                item
                for item in structural.get("code_objects", [])
                if isinstance(item, dict) and item.get("modified") is True
            ]
            if not modified:
                continue
            original = sum(int(item.get("original_bytes", 0)) for item in modified)
            patched = sum(int(item.get("patched_bytes", 0)) for item in modified)
            resources = [item.get("resources", {}) for item in modified]
            memory = structural.get("process_memory", {})
            all_code_objects = [
                item
                for item in structural.get("code_objects", [])
                if isinstance(item, dict)
            ]
            samples[profile].append(
                {
                    "original_bytes": float(original),
                    "patched_bytes": float(patched),
                    "growth_ratio": patched / original if original else math.nan,
                    "report_peak_live_bytes": float(
                        memory.get("report_peak_live_bytes", 0)
                    ),
                    "transform_peak_reserved_bytes": float(
                        memory.get("transform_peak_reserved_bytes", 0)
                    ),
                    "descriptor_growth": float(
                        sum(
                            int(value.get("descriptor_growth", 0))
                            for value in resources
                        )
                    ),
                    "spill": float(
                        sum(int(value.get("spill", 0)) for value in resources)
                    ),
                    "unsupported": float(
                        sum(int(value.get("unsupported", 0)) for value in resources)
                    ),
                    "patch_ms": float(structural.get("total_patch_ms", 0)),
                    "waitcheck_ms": float(
                        sum(
                            float(item.get("waitcheck_ms", 0))
                            for item in all_code_objects
                        )
                    ),
                    "inventory_ms": float(
                        sum(
                            float(item.get("inventory_ms", 0))
                            for item in all_code_objects
                        )
                    ),
                }
            )

    required = int(campaign.get("required_accepted_rounds", 0))
    output: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        values = samples[profile]
        _require(
            len(values) >= required,
            f"{campaign.get('workload')}/{profile}: only {len(values)} structural samples",
        )
        medians = {
            key: _median([sample[key] for sample in values]) for key in values[0]
        }
        medians["samples"] = len(values)
        medians["clean_gate_runs"] = clean_gate_runs[profile]
        medians["clean_gate_rejections"] = clean_gate_rejections[profile]
        medians["unexpected_diagnostics"] = unexpected_diagnostics[profile]
        medians["metadata_nonzero_fields"] = metadata_nonzero[profile]
        medians["metadata_compared_fields"] = metadata_compared[profile]
        output[profile] = medians
    return output


def collect_performance(
    manifest: dict[str, Any], performance: list[tuple[str, Path, dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, int]]]:
    allowed = set(manifest.get("allowed_unqualified_metrics", []))
    expected_workloads = set(manifest.get("performance_workloads", []))
    found_workloads = {campaign.get("workload") for _, _, campaign in performance}
    _require(found_workloads == expected_workloads, "performance workload set mismatch")
    rows: list[dict[str, Any]] = []
    clean_gates: dict[tuple[str, str], dict[str, int]] = {}
    observed_rejected_campaigns = set()
    for artifact, path, campaign in performance:
        workload = campaign.get("workload")
        _require(
            campaign.get("schema_version") == 2, f"{workload}: campaign schema mismatch"
        )
        _require(
            campaign.get("protocol") == manifest.get("campaign_protocol"),
            f"{workload}: campaign protocol mismatch",
        )
        _require(
            campaign.get("target") == manifest.get("target"),
            f"{workload}: campaign target mismatch",
        )
        _require(
            campaign.get("randomization_seed") == manifest.get("randomization_seed"),
            f"{workload}: seed mismatch",
        )
        _require(
            campaign.get("baseline_drift_limit")
            == manifest.get("baseline_drift_limit"),
            f"{workload}: drift limit mismatch",
        )
        _require(
            campaign.get("bootstrap_resamples") == manifest.get("bootstrap_resamples"),
            f"{workload}: bootstrap count mismatch",
        )
        admitted = set(campaign.get("admission", {}).get("admitted_profiles", []))
        timed = set(campaign.get("timed_profiles", []))
        _require(
            admitted == timed,
            f"{workload}: timed profiles differ from admitted profiles",
        )
        cells = extract_metric_cells(campaign, allowed)
        campaign_summary = campaign.get("summary", {})
        campaign_reasons = campaign_summary.get("reasons", [])
        expected_reasons = manifest.get("allowed_rejected_campaigns", {}).get(workload)
        if campaign_summary.get("accepted") is not True:
            _require(
                isinstance(expected_reasons, list)
                and campaign_reasons == expected_reasons,
                f"{workload}: rejected campaign is not exactly allowlisted",
            )
            observed_rejected_campaigns.add(workload)
        else:
            _require(
                expected_reasons is None,
                f"{workload}: manifest allowlists an accepted campaign",
            )
        derived_reasons = sorted(
            f"{profile}/{cell['metric']}: {cell['count']}/{cell['required']} accepted rounds"
            for profile, profile_cells in cells.items()
            for cell in profile_cells.values()
            if not cell["qualified"]
        )
        _require(
            sorted(campaign_reasons) == derived_reasons,
            f"{workload}: campaign rejection reasons do not match underqualified metrics",
        )
        structural = structural_metrics(path, campaign)
        for profile in PROFILE_ORDER:
            admission = _admission_result(path, campaign, profile)
            is_admitted = profile in admitted
            _require(
                (admission.get("accepted") is True) == is_admitted,
                f"{workload}/{profile}: admission mismatch",
            )
            row = {
                "artifact": artifact,
                "workload": workload,
                "profile": profile,
                "admitted": is_admitted,
                "admission_reason": (
                    None if is_admitted else _coverage_reason(admission)
                ),
                "metrics": cells.get(profile, {}),
                "structural": structural.get(profile),
            }
            if is_admitted:
                gate = structural[profile]
                _require(
                    gate["clean_gate_rejections"] == 0,
                    f"{workload}/{profile}: clean gate rejected a retained run",
                )
                _require(
                    gate["unexpected_diagnostics"] == 0,
                    f"{workload}/{profile}: clean gate observed diagnostics",
                )
                _require(
                    gate["metadata_compared_fields"] > 0,
                    f"{workload}/{profile}: no metadata fields were compared",
                )
                clean_gates[(workload, profile)] = {
                    "runs": int(gate["clean_gate_runs"]),
                    "rejections": int(gate["clean_gate_rejections"]),
                    "diagnostics": int(gate["unexpected_diagnostics"]),
                }
            rows.append(row)
    _require(
        not (
            allowed
            - {
                f"{row['workload']}/{row['profile']}/{cell['metric']}"
                for row in rows
                for cell in row["metrics"].values()
                if not cell["qualified"]
            }
        ),
        "manifest names an underqualified cell that was not observed",
    )
    _require(
        observed_rejected_campaigns
        == set(manifest.get("allowed_rejected_campaigns", {})),
        "manifest rejected-campaign allowlist does not match observed campaigns",
    )
    return rows, clean_gates


def collect_detection(
    manifest: dict[str, Any], detection: list[tuple[str, Path, dict[str, Any]]]
) -> list[dict[str, Any]]:
    expected_cases = {
        (item["workload"], item["fault"])
        for item in manifest.get("detection_cases", [])
    }
    found_cases = {
        (summary.get("workload"), summary.get("fault")) for _, _, summary in detection
    }
    _require(found_cases == expected_cases, "detection case set mismatch")
    rows: list[dict[str, Any]] = []
    for artifact, _, summary in detection:
        workload = summary.get("workload")
        fault = summary.get("fault")
        _require(
            summary.get("schema_version") == 2,
            f"{workload}/{fault}: fault schema mismatch",
        )
        _require(
            summary.get("target") == manifest.get("target"),
            f"{workload}/{fault}: target mismatch",
        )
        _require(
            summary.get("accepted") is True,
            f"{workload}/{fault}: final fault campaign rejected",
        )
        by_profile = {item.get("profile"): item for item in summary.get("profiles", [])}
        _require(
            set(by_profile) == set(PROFILE_ORDER),
            f"{workload}/{fault}: incomplete profile matrix",
        )
        for profile in PROFILE_ORDER:
            item = by_profile[profile]
            if item.get("disposition") == "not-applicable":
                rows.append(
                    {
                        "artifact": artifact,
                        "workload": workload,
                        "fault": fault,
                        "profile": profile,
                        "applicable": False,
                        "reason": item.get("reason"),
                    }
                )
                continue
            trials = item.get("trials")
            attempted = item.get("attempted_trials")
            admitted = item.get("admitted_trials")
            reached = item.get("reached_trials")
            _require(
                isinstance(attempted, int) and attempted >= 30,
                f"{workload}/{fault}/{profile}: too few trials",
            )
            _require(
                trials == attempted and admitted == attempted and reached == attempted,
                f"{workload}/{fault}/{profile}: not every final trial reached",
            )
            reach_outcomes = item.get("reach_outcomes")
            _require(
                isinstance(reach_outcomes, dict)
                and all(
                    isinstance(name, str) and isinstance(count, int) and count >= 0
                    for name, count in reach_outcomes.items()
                )
                and sum(reach_outcomes.values()) == reached,
                f"{workload}/{fault}/{profile}: invalid reach evidence summary",
            )
            _require(
                item.get("accepted") is True,
                f"{workload}/{fault}/{profile}: policy rejected",
            )
            rows.append(
                {
                    "artifact": artifact,
                    "workload": workload,
                    "fault": fault,
                    "profile": profile,
                    "applicable": True,
                    "trials": attempted,
                    "reached": reached,
                    "reach_outcomes": reach_outcomes,
                    "detections": item.get("detections"),
                    "detection_rate": item.get("detection_rate"),
                    "detection_interval": item.get("detection_wilson_95"),
                    "oracle_manifestations": item.get("oracle_manifestations"),
                    "oracle_rate": item.get("oracle_manifestation_rate"),
                }
            )
    return rows


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def collect_complexity(manifest: dict[str, Any], source_root: Path) -> dict[str, Any]:
    config = manifest.get("complexity", {})
    all_source_paths: set[Path] = set()
    for relative in config.get("source_directories", []):
        directory = source_root / relative
        _require(directory.is_dir(), f"missing complexity source directory {directory}")
        all_source_paths.update(path for path in directory.rglob("*") if path.is_file())
    all_source_paths = {
        path
        for path in all_source_paths
        if path.suffix in TEXT_SUFFIXES or path.name == "CMakeLists.txt"
    }
    shared = {
        "files": len(all_source_paths),
        "lines": sum(_count_lines(path) for path in all_source_paths),
    }
    source_digest = hashlib.sha256()
    for path in sorted(all_source_paths):
        source_digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(path.read_bytes())
        source_digest.update(b"\0")
    source_tree_sha256 = source_digest.hexdigest()
    _require(
        source_tree_sha256 == config.get("source_tree_sha256"),
        "complexity source tree differs from the frozen inventory",
    )
    test_paths: set[Path] = set()
    for relative in config.get("test_directories", []):
        directory = source_root / relative
        if directory.is_dir():
            test_paths.update(
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix in TEXT_SUFFIXES
            )
    engines: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        item = config.get("engines", {}).get(profile)
        _require(
            isinstance(item, dict), f"missing complexity declaration for {profile}"
        )
        dedicated = [
            source_root / relative for relative in item.get("dedicated_files", [])
        ]
        _require(
            all(path.is_file() for path in dedicated),
            f"missing dedicated source for {profile}",
        )
        pattern = re.compile(str(item.get("test_pattern")), re.IGNORECASE)
        matching_tests = sum(
            1
            for path in test_paths
            if pattern.search(path.read_text(encoding="utf-8", errors="replace"))
        )
        engines.append(
            {
                "profile": profile,
                "dedicated_files": len(dedicated),
                "dedicated_lines": sum(_count_lines(path) for path in dedicated),
                "matching_test_files": matching_tests,
                "runtime_model": item.get("runtime_model"),
                "failure_modes": item.get("failure_modes"),
                "risk": item.get("risk"),
            }
        )
    return {
        "shared": shared,
        "source_tree_sha256": source_tree_sha256,
        "engines": engines,
    }


def _fmt_number(value: Any, digits: int = 2) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_metric(cell: dict[str, Any] | None) -> str:
    if cell is None:
        return "—"
    if not cell.get("qualified"):
        return f"unqualified {cell.get('count')}/{cell.get('required')}"
    return (
        f"{_fmt_number(cell.get('slowdown'))}× / {_fmt_number(cell.get('timing_ms'), 3)} ms "
        f"(n={cell.get('count')}, CI {_fmt_number(cell.get('lower'))}–{_fmt_number(cell.get('upper'))}×)"
    )


def _mib(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value) / (1024 * 1024):.2f} MiB"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    _require(
        all(len(row) == len(headers) for row in rows),
        f"Markdown table has {len(headers)} headers and a ragged row",
    )
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend(
        "| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |"
        for row in rows
    )
    return output


def render_markdown(
    manifest: dict[str, Any],
    signature: dict[str, Any],
    performance: list[dict[str, Any]],
    detection: list[dict[str, Any]],
    clean_gates: dict[tuple[str, str], dict[str, int]],
    complexity: dict[str, Any],
) -> str:
    lines = [
        "# gfx1201 ConSan empirical results",
        "",
        "This file is generated by `consan_empirical_report.py` from the frozen artifact manifest.",
        "It contains measurements only; interpretation and recommendations are in",
        "[GFX1201_EMPIRICAL_STUDY.md](GFX1201_EMPIRICAL_STUDY.md).",
        "",
        "## Verified provenance",
        "",
        f"- Target: `{signature['target']}` on KFD node(s) `{','.join(signature['selected_topology'])}`.",
        f"- RocJITsu source: `{signature['source_head']}`.",
        f"- ConSan hook SHA-256: `{signature['hook']}`.",
        f"- Fault specification SHA-256: `{manifest['fault_spec_sha256']}`.",
        f"- Performance protocol: {manifest['required_rounds']} accepted paired rounds, "
        f"{manifest['baseline_drift_limit'] * 100:.0f}% maximum baseline drift, "
        f"{manifest['bootstrap_resamples']} bootstrap resamples, seed `{manifest['randomization_seed']}`.",
        "- All retained fault campaigns use at least 30 reached, independently contained processes per applicable pair.",
        "",
        "## Performance",
        "",
        "Ratios are paired medians against interpolated native baselines. Each qualified cell also gives",
        "absolute median latency, sample count, and the bootstrap 95% interval for the slowdown.",
        "",
    ]
    perf_rows: list[list[str]] = []
    for row in performance:
        if row["admitted"]:
            cells = row["metrics"]
            admission = "admitted"
        else:
            cells = {}
            admission = "rejected: " + str(row["admission_reason"])
        perf_rows.append(
            [
                row["workload"],
                PROFILE_LABELS[row["profile"]],
                admission,
                _fmt_metric(cells.get("cold_process")),
                _fmt_metric(cells.get("cold_workload")),
                _fmt_metric(cells.get("cold_device")),
                _fmt_metric(cells.get("warm_host")),
                _fmt_metric(cells.get("warm_device")),
                row["artifact"],
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "Workload",
                "Engine",
                "Admission",
                "Cold process",
                "Cold workload",
                "Cold device",
                "Warm host",
                "Warm device",
                "Artifact",
            ],
            perf_rows,
        )
    )
    lines.extend(
        [
            "",
            "An `unqualified` cell is retained but excluded from conclusions; all other shown timing cells meet the frozen sample requirement.",
            "",
        ]
    )

    structural_rows: list[list[str]] = []
    for row in performance:
        structural = row.get("structural")
        if not row["admitted"] or not structural:
            continue
        structural_rows.append(
            [
                row["workload"],
                PROFILE_LABELS[row["profile"]],
                f"{_fmt_number(structural.get('patch_ms'), 1)} ms",
                f"{_fmt_number(structural.get('waitcheck_ms'), 1)} ms",
                f"{_fmt_number(structural.get('inventory_ms'), 1)} ms",
                f"{_fmt_number(structural.get('growth_ratio'))}×",
                f"{int(structural.get('original_bytes', 0))} → {int(structural.get('patched_bytes', 0))}",
                _mib(structural.get("report_peak_live_bytes")),
                _mib(structural.get("transform_peak_reserved_bytes")) + " reserved VA",
                str(int(structural.get("descriptor_growth", 0))),
                str(int(structural.get("spill", 0))),
                str(int(structural.get("unsupported", 0))),
                f"{int(structural.get('metadata_nonzero_fields', 0))} / "
                f"{int(structural.get('metadata_compared_fields', 0))} compared",
                str(structural.get("samples")),
            ]
        )
    lines.extend(["## Structural cost", ""])
    lines.extend(
        _markdown_table(
            [
                "Workload",
                "Engine",
                "Patch",
                "Waitcheck",
                "Inventory",
                "Object ratio",
                "Modified bytes",
                "Report peak",
                "Transform peak",
                "Descriptor growth",
                "Spill",
                "Unsupported resource attempts",
                "Nonzero metadata deltas",
                "Samples",
            ],
            structural_rows,
        )
    )
    lines.extend(
        [
            "",
            "Object sizes sum only code objects actually modified in each process. `Transform peak` is the",
            "maximum reserved virtual-address budget reported by the transformer, not resident memory.",
            "",
            "## Detection",
            "",
        ]
    )
    detection_rows: list[list[str]] = []
    for row in detection:
        if not row["applicable"]:
            detection_rows.append(
                [
                    row["workload"],
                    row["fault"],
                    PROFILE_LABELS[row["profile"]],
                    "not applicable",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    row["artifact"],
                ]
            )
            continue
        interval = row["detection_interval"]
        gate = clean_gates.get((row["workload"], row["profile"]))
        reach_labels = {
            "detector-owned-runtime-diagnostic": "detector runtime",
            "independent-oracle-manifestation": "oracle runtime",
            "reviewed-unconditional-final-isa": "reviewed proof",
        }
        reach_basis = "; ".join(
            f"{reach_labels.get(name, name)}={count}"
            for name, count in sorted(row["reach_outcomes"].items())
        )
        detection_rows.append(
            [
                row["workload"],
                row["fault"],
                PROFILE_LABELS[row["profile"]],
                f"{row['reached']}/{row['trials']}",
                reach_basis,
                f"{row['oracle_manifestations']}/{row['trials']}",
                f"{row['detections']}/{row['trials']} ({row['detection_rate'] * 100:.1f}%)",
                f"{interval['lower'] * 100:.1f}–{interval['upper'] * 100:.1f}%",
                (
                    f"pass ({gate['runs']} runs; {gate['diagnostics']} diagnostics)"
                    if gate is not None
                    else "—"
                ),
                row["artifact"],
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "Workload",
                "Fault",
                "Engine",
                "Reached",
                "Reach basis",
                "Oracle failures",
                "Diagnoses",
                "Wilson 95%",
                "Unexpected clean diagnoses",
                "Artifact",
            ],
            detection_rows,
        )
    )
    lines.extend(["", "## Mechanical complexity inventory", ""])
    shared = complexity["shared"]
    lines.append(
        f"The two ConSan production directories contain {shared['files']} text/build files and "
        f"{shared['lines']} lines in total. Dedicated-file counts below are non-overlapping engine-owned "
        "surfaces; shared analysis, placement, ABI, configuration, and report code is deliberately not assigned to one engine."
    )
    lines.append("")
    complexity_rows = [
        [
            PROFILE_LABELS[item["profile"]],
            str(item["dedicated_files"]),
            str(item["dedicated_lines"]),
            str(item["matching_test_files"]),
            str(item["runtime_model"]),
            str(item["failure_modes"]),
            str(item["risk"]),
        ]
        for item in complexity["engines"]
    ]
    lines.extend(
        _markdown_table(
            [
                "Engine",
                "Dedicated files",
                "Dedicated lines",
                "Matching test files",
                "Runtime/ABI model",
                "Observed failures",
                "Maintenance risk",
            ],
            complexity_rows,
        )
    )
    lines.extend(["", "## Artifact inventory", ""])
    for name in manifest.get("performance_artifacts", []):
        lines.append(f"- Performance: `{name}`")
    for name in manifest.get("detection_artifacts", []):
        lines.append(f"- Detection: `{name}`")
    lines.append("")
    digest = hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
    lines.append(f"Generated-content SHA-256: `{digest}`")
    lines.append("")
    return "\n".join(lines)


def build_report(manifest_path: Path, artifact_base: Path, source_root: Path) -> str:
    manifest = _read_json(manifest_path)
    _require(manifest.get("schema_version") == 1, "study manifest schema mismatch")
    fault_spec = source_root / str(manifest.get("fault_spec_path", ""))
    _require(
        fault_spec.is_file(), f"missing checked-in fault specification {fault_spec}"
    )
    spec_hash = hashlib.sha256(fault_spec.read_bytes()).hexdigest()
    _require(
        spec_hash == manifest.get("fault_spec_sha256"),
        "checked-in fault-spec hash mismatch",
    )
    performance_documents, detection_documents = discover_documents(
        manifest, artifact_base
    )
    signature, _ = validate_common_provenance(
        manifest, performance_documents, detection_documents
    )
    performance, clean_gates = collect_performance(manifest, performance_documents)
    detection = collect_detection(manifest, detection_documents)
    complexity = collect_complexity(manifest, source_root)
    return render_markdown(
        manifest, signature, performance, detection, clean_gates, complexity
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--manifest",
        type=Path,
        default=here / "consan_empirical_gfx1201.json",
        help="frozen study artifact and complexity manifest",
    )
    parser.add_argument("--artifact-base", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=here.parents[4],
        help="rocm-systems repository root used for the mechanical source audit",
    )
    parser.add_argument("--output", type=Path, help="write Markdown to this path")
    parser.add_argument(
        "--check",
        type=Path,
        help="fail unless this checked-in Markdown file exactly matches generated output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = build_report(args.manifest, args.artifact_base, args.source_root)
        if args.check is not None:
            existing = args.check.read_text(encoding="utf-8")
            _require(existing == report, f"generated report differs from {args.check}")
        if args.output is not None:
            args.output.write_text(report, encoding="utf-8")
        elif args.check is None:
            print(report, end="")
    except (OSError, StudyError) as error:
        print(f"study validation error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
