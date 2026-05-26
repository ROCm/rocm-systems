#!/usr/bin/env python3

import argparse
import collections
import hashlib
import json
import os
import pathlib
import subprocess
import sys


BASE_MINIMIZER_ENV = {
    "ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE": "1",
    "ROCJITSU_AFL_DEBUG_EDGE_LIMIT": "0",
    "ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS": "1",
    "ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY": "hashed",
    "ROCJITSU_AFL_DEBUG_REQUIRE_LIVENESS_REGISTERS": "1",
}


DIAGNOSTIC_PROFILE_ENV = {
    "opaque-fresh": {
        "ROCJITSU_AFL_DEBUG_ALLOW_OPAQUE_FRESH_REGISTERS": "1",
    },
    "previous-bb-set": {},
}


EDGE_FIELDS = (
    "kernel",
    "kind",
    "patch_text_offset",
    "return_text_offset",
    "slot_policy",
    "state_sgpr",
    "saved_exec_sgpr",
    "tmp0_sgpr",
    "tmp1_sgpr",
    "workitem_vgpr",
    "tmp0_vgpr",
    "tmp1_vgpr",
    "tmp2_vgpr",
    "scratch_spill",
    "placement",
)


CANDIDATE_FIELDS = (
    "kernel",
    "kind",
    "patch_text_offset",
    "mnemonic",
    "required_sgprs",
    "required_vgprs",
    "allocated_sgprs",
    "allocated_vgprs",
    "sgpr_growth",
    "vgpr_growth",
    "state_sgpr",
    "saved_exec_sgpr",
    "tmp0_sgpr",
    "tmp1_sgpr",
    "workitem_vgpr",
    "tmp0_vgpr",
    "tmp1_vgpr",
    "tmp2_vgpr",
)


DESCRIPTOR_RESOURCE_FIELDS = (
    "kernel",
    "descriptor_file_offset",
    "wave32",
    "old_sgpr_count",
    "patched_sgpr_count",
    "old_vgpr_count",
    "patched_vgpr_count",
    "old_private_segment_fixed_size",
    "patched_private_segment_fixed_size",
    "sgpr_count_metadata_patch",
    "private_segment_metadata_patch",
    "resource_fields_changed",
)


DEVICE_EDGE_DELTA_FIELDS = (
    "trigger",
    "launches",
    "edge_slot_delta_count",
    "edge_counter_delta_total",
    "nonzero_edge_slots_total",
    "first_kernel",
    "last_kernel",
    "last_kind",
)


def load_jsonl(path):
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    except FileNotFoundError:
        return []
    return rows


def pick_fields(row, fields):
    return {field: row[field] for field in fields if field in row}


def minimizer_env_for_profile(profile):
    if profile not in DIAGNOSTIC_PROFILE_ENV:
        raise ValueError(f"unknown diagnostic profile: {profile}")
    env = dict(BASE_MINIMIZER_ENV)
    env.update(DIAGNOSTIC_PROFILE_ENV[profile])
    return env


def as_int(value):
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def sorted_counter(counter, limit=None):
    items = [
        {"reason": reason, "count": count}
        for reason, count in counter.most_common(limit)
    ]
    return items


def growth(resource, old_field, patched_field):
    return max(0, as_int(resource.get(patched_field)) - as_int(resource.get(old_field)))


def descriptor_resource_sample(resource):
    sampled = pick_fields(resource, DESCRIPTOR_RESOURCE_FIELDS)
    sampled["sgpr_growth"] = growth(resource, "old_sgpr_count", "patched_sgpr_count")
    sampled["vgpr_growth"] = growth(resource, "old_vgpr_count", "patched_vgpr_count")
    sampled["private_segment_growth"] = growth(
        resource,
        "old_private_segment_fixed_size",
        "patched_private_segment_fixed_size",
    )
    return sampled


def summarize_descriptor_resources(resources):
    sgpr_growth = [
        resource
        for resource in resources
        if growth(resource, "old_sgpr_count", "patched_sgpr_count")
    ]
    vgpr_growth = [
        resource
        for resource in resources
        if growth(resource, "old_vgpr_count", "patched_vgpr_count")
    ]
    private_segment_growth = [
        resource
        for resource in resources
        if growth(
            resource,
            "old_private_segment_fixed_size",
            "patched_private_segment_fixed_size",
        )
    ]
    sgpr_only_growth = [
        resource
        for resource in sgpr_growth
        if not growth(resource, "old_vgpr_count", "patched_vgpr_count")
        and not growth(
            resource,
            "old_private_segment_fixed_size",
            "patched_private_segment_fixed_size",
        )
    ]
    changed = [
        resource
        for resource in resources
        if resource.get("resource_fields_changed")
        or growth(resource, "old_sgpr_count", "patched_sgpr_count")
        or growth(resource, "old_vgpr_count", "patched_vgpr_count")
        or growth(
            resource,
            "old_private_segment_fixed_size",
            "patched_private_segment_fixed_size",
        )
    ]
    sampled_changes = sorted(
        (descriptor_resource_sample(resource) for resource in changed),
        key=lambda resource: (
            -as_int(resource.get("sgpr_growth")),
            -as_int(resource.get("vgpr_growth")),
            -as_int(resource.get("private_segment_growth")),
            str(resource.get("kernel", "")),
        ),
    )[:32]

    return {
        "updates": len(resources),
        "changed": len(changed),
        "sgpr_growth_updates": len(sgpr_growth),
        "vgpr_growth_updates": len(vgpr_growth),
        "private_segment_growth_updates": len(private_segment_growth),
        "sgpr_only_growth_updates": len(sgpr_only_growth),
        "sgpr_and_vgpr_growth_updates": sum(
            1
            for resource in resources
            if growth(resource, "old_sgpr_count", "patched_sgpr_count")
            and growth(resource, "old_vgpr_count", "patched_vgpr_count")
        ),
        "max_old_sgpr_count": max(
            [as_int(resource.get("old_sgpr_count")) for resource in resources] or [0]
        ),
        "max_patched_sgpr_count": max(
            [as_int(resource.get("patched_sgpr_count")) for resource in resources] or [0]
        ),
        "max_sgpr_growth": max(
            [growth(resource, "old_sgpr_count", "patched_sgpr_count") for resource in resources]
            or [0]
        ),
        "max_old_vgpr_count": max(
            [as_int(resource.get("old_vgpr_count")) for resource in resources] or [0]
        ),
        "max_patched_vgpr_count": max(
            [as_int(resource.get("patched_vgpr_count")) for resource in resources] or [0]
        ),
        "max_vgpr_growth": max(
            [growth(resource, "old_vgpr_count", "patched_vgpr_count") for resource in resources]
            or [0]
        ),
        "max_old_private_segment_fixed_size": max(
            [
                as_int(resource.get("old_private_segment_fixed_size"))
                for resource in resources
            ]
            or [0]
        ),
        "max_patched_private_segment_fixed_size": max(
            [
                as_int(resource.get("patched_private_segment_fixed_size"))
                for resource in resources
            ]
            or [0]
        ),
        "max_private_segment_growth": max(
            [
                growth(
                    resource,
                    "old_private_segment_fixed_size",
                    "patched_private_segment_fixed_size",
                )
                for resource in resources
            ]
            or [0]
        ),
        "changed_kernels": sorted(
            {resource.get("kernel", "") for resource in changed if resource.get("kernel")}
        )[:64],
        "changed_kernel_count": len(
            {resource.get("kernel", "") for resource in changed if resource.get("kernel")}
        ),
        "sampled_changes": sampled_changes,
        "sampled_changes_truncated": len(changed) > len(sampled_changes),
        "sampled_sgpr_only_changes": sorted(
            (descriptor_resource_sample(resource) for resource in sgpr_only_growth),
            key=lambda resource: str(resource.get("kernel", "")),
        )[:32],
        "sampled_sgpr_only_changes_truncated": len(sgpr_only_growth) > 32,
    }


def summarize_device_edge_deltas(rows):
    deltas = [row for row in rows if row.get("event") == "device_edge_delta"]
    return {
        "events": len(deltas),
        "edge_slot_delta_count": sum(
            as_int(row.get("edge_slot_delta_count")) for row in deltas
        ),
        "edge_counter_delta_total": sum(
            as_int(row.get("edge_counter_delta_total")) for row in deltas
        ),
        "max_nonzero_edge_slots_total": max(
            [as_int(row.get("nonzero_edge_slots_total")) for row in deltas] or [0]
        ),
        "first": pick_fields(deltas[0], DEVICE_EDGE_DELTA_FIELDS) if deltas else None,
        "last": pick_fields(deltas[-1], DEVICE_EDGE_DELTA_FIELDS) if deltas else None,
    }


def compact_descriptor_summary(summary):
    resources = summary.get("descriptor_resources", {})
    return {
        "updates": resources.get("updates", 0),
        "changed": resources.get("changed", 0),
        "sgpr_growth_updates": resources.get("sgpr_growth_updates", 0),
        "vgpr_growth_updates": resources.get("vgpr_growth_updates", 0),
        "private_segment_growth_updates": resources.get(
            "private_segment_growth_updates", 0
        ),
        "sgpr_only_growth_updates": resources.get("sgpr_only_growth_updates", 0),
        "sgpr_and_vgpr_growth_updates": resources.get(
            "sgpr_and_vgpr_growth_updates", 0
        ),
        "max_sgpr_growth": resources.get("max_sgpr_growth", 0),
        "max_vgpr_growth": resources.get("max_vgpr_growth", 0),
        "max_private_segment_growth": resources.get("max_private_segment_growth", 0),
        "changed_kernel_count": resources.get("changed_kernel_count", 0),
        "changed_kernels": resources.get("changed_kernels", [])[:16],
        "sampled_sgpr_only_changes": resources.get(
            "sampled_sgpr_only_changes", []
        )[:8],
    }


def compact_limit_summary(summary):
    compact = {
        "limit": summary.get("limit"),
        "returncode": summary.get("returncode"),
        "status": summary.get("status"),
        "diagnostic_profile": summary.get("diagnostic_profile"),
        "report": summary.get("report"),
        "patch_text_offset_filter": summary.get("patch_text_offset_filter"),
        "patch_events": summary.get("patch_events", 0),
        "successful_patch_events": summary.get("successful_patch_events", 0),
        "edge_sites_patched": summary.get("edge_sites_patched", 0),
        "hashed_edge_sites": summary.get("hashed_edge_sites", 0),
        "selected_edge_samples": summary.get("selected_edge_samples", 0),
        "selected_edge_samples_truncated": summary.get(
            "selected_edge_samples_truncated", False
        ),
        "probe_required_sgprs": summary.get("probe_required_sgprs", 0),
        "probe_required_vgprs": summary.get("probe_required_vgprs", 0),
        "first_selected_edge": summary.get("first_selected_edge"),
        "last_sampled_selected_edge": summary.get("last_sampled_selected_edge"),
        "descriptor_resources": compact_descriptor_summary(summary),
        "device_edge_deltas": summary.get("device_edge_deltas"),
        "top_skip_reasons": summary.get("top_skip_reasons", []),
        "no_site_kernel_count": summary.get("no_site_kernel_count", 0),
        "no_site_kernel_samples": summary.get("no_site_kernel_samples", []),
    }
    if "patch_text_offset_set" in summary:
        compact["patch_text_offset_set"] = summary["patch_text_offset_set"]
    if "patch_text_offset_base_set" in summary:
        compact["patch_text_offset_base_set"] = summary["patch_text_offset_base_set"]
    if "patch_text_offset_candidate" in summary:
        compact["patch_text_offset_candidate"] = summary["patch_text_offset_candidate"]
    return compact


def compact_offset_scan_summary(summary):
    compact = compact_limit_summary(summary)
    compact["patch_text_offset"] = summary.get("patch_text_offset_filter")
    return compact


def summarize_run(summaries, metadata=None):
    metadata = metadata or {}
    failures = [
        summary for summary in summaries if as_int(summary.get("returncode")) != 0
    ]
    successes = [
        summary for summary in summaries if as_int(summary.get("returncode")) == 0
    ]
    first_failure = failures[0] if failures else None
    first_failure_index = summaries.index(first_failure) if first_failure else -1
    successes_before_failure = (
        [
            summary
            for summary in summaries[:first_failure_index]
            if as_int(summary.get("returncode")) == 0
        ]
        if first_failure
        else []
    )
    return {
        "event": "opaque_fresh_minimization_summary",
        "status": "failed" if failures else "ok",
        "runs": len(summaries),
        "successful_runs": len(successes),
        "failed_runs": len(failures),
        "limits": [summary.get("limit") for summary in summaries],
        "failing_limits": [summary.get("limit") for summary in failures],
        "kernel_include": metadata.get("kernel_include"),
        "kernel_exclude": metadata.get("kernel_exclude"),
        "diagnostic_profile": metadata.get("diagnostic_profile"),
        "source_slot_policy": metadata.get("source_slot_policy"),
        "source_slot_policy_limits": metadata.get("source_slot_policy_limits"),
        "patch_text_offset": metadata.get("patch_text_offset"),
        "patch_text_offset_set": metadata.get("patch_text_offset_set"),
        "first_failure": (
            compact_limit_summary(first_failure) if first_failure else None
        ),
        "last_success_before_failure": (
            compact_limit_summary(successes_before_failure[-1])
            if successes_before_failure
            else None
        ),
        "last_success": compact_limit_summary(successes[-1]) if successes else None,
    }


def summarize_patch_offset_scan(summaries, metadata=None):
    metadata = metadata or {}
    failures = [
        summary for summary in summaries if as_int(summary.get("returncode")) != 0
    ]
    successes = [
        summary for summary in summaries if as_int(summary.get("returncode")) == 0
    ]
    failing_offsets = [
        summary.get("patch_text_offset_filter") for summary in failures
    ]
    passing_offsets = [
        summary.get("patch_text_offset_filter") for summary in successes
    ]
    return {
        "event": "opaque_fresh_patch_offset_scan_summary",
        "status": "failed" if failures else "ok",
        "runs": len(summaries),
        "successful_runs": len(successes),
        "failed_runs": len(failures),
        "limit": metadata.get("limit"),
        "kernel_include": metadata.get("kernel_include"),
        "kernel_exclude": metadata.get("kernel_exclude"),
        "diagnostic_profile": metadata.get("diagnostic_profile"),
        "source_slot_policy": metadata.get("source_slot_policy"),
        "source_slot_policy_limits": metadata.get("source_slot_policy_limits"),
        "source_base_slot_policy_limits": metadata.get(
            "source_base_slot_policy_limits"
        ),
        "base_patch_text_offset_set": metadata.get("base_patch_text_offset_set"),
        "source_report": metadata.get("source_report"),
        "source_report_edge_sites_patched": metadata.get(
            "source_report_edge_sites_patched", 0
        ),
        "source_report_selected_edge_samples": metadata.get(
            "source_report_selected_edge_samples", 0
        ),
        "source_report_selected_edge_samples_truncated": metadata.get(
            "source_report_selected_edge_samples_truncated", False
        ),
        "source_report_device_edge_delta_events": metadata.get(
            "source_report_device_edge_delta_events", 0
        ),
        "offsets": [summary.get("patch_text_offset_filter") for summary in summaries],
        "failing_offsets": failing_offsets,
        "passing_offsets": passing_offsets,
        "first_failing_offset": failing_offsets[0] if failing_offsets else None,
        "smallest_failing_offset": min(failing_offsets) if failing_offsets else None,
        "all_scanned_offsets_passed": len(summaries) != 0 and not failures,
        "single_offset_failure_found": bool(failures),
        "first_failure": (
            compact_offset_scan_summary(failures[0]) if failures else None
        ),
        "last_success": (
            compact_offset_scan_summary(successes[-1]) if successes else None
        ),
    }


def summarize_report(path):
    rows = load_jsonl(path)
    patch_events = [row for row in rows if row.get("event") == "patch_device_elf"]
    selected_edges = []
    opaque_fresh_candidates = []
    fresh_kernels = []
    descriptor_resources = []
    skip_reasons = collections.Counter()
    no_site_kernel_summaries = {}
    for event in patch_events:
        for resource in event.get("descriptor_resources", []):
            descriptor_resources.append(resource)
        for edge in event.get("sampled_selected_edges", []):
            selected_edges.append(pick_fields(edge, EDGE_FIELDS))
        for summary in event.get("kernel_summaries", []):
            kernel = summary.get("kernel", "")
            branch_candidates = as_int(summary.get("branch_candidates", 0))
            block_candidates = as_int(summary.get("block_candidates", 0))
            selected_sites = (
                as_int(summary.get("edge_sites_patched", 0))
                + as_int(summary.get("branch_edges_selected", 0))
                + as_int(summary.get("block_selected", 0))
            )
            kernel_skip_reasons = collections.Counter()
            for skip in summary.get("skip_reason_counts", []):
                if not isinstance(skip, dict):
                    continue
                reason = skip.get("reason")
                count = as_int(skip.get("count", 0))
                if reason and count:
                    skip_reasons[str(reason)] += count
                    kernel_skip_reasons[str(reason)] += count
            if kernel and selected_sites == 0 and (branch_candidates or block_candidates):
                no_site_summary = no_site_kernel_summaries.setdefault(
                    kernel,
                    {
                        "kernel": kernel,
                        "branch_candidates": 0,
                        "block_candidates": 0,
                        "events": 0,
                        "_skip_reasons": collections.Counter(),
                    },
                )
                no_site_summary["branch_candidates"] = max(
                    no_site_summary["branch_candidates"], branch_candidates
                )
                no_site_summary["block_candidates"] = max(
                    no_site_summary["block_candidates"], block_candidates
                )
                no_site_summary["events"] += 1
                no_site_summary["_skip_reasons"].update(kernel_skip_reasons)
            if summary.get("fresh_registers") or summary.get(
                "opaque_fresh_register_candidate_probe_points", 0
            ):
                fresh_kernels.append(
                    {
                        "kernel": kernel,
                        "fresh_registers": summary.get("fresh_registers", False),
                        "fresh_register_probe_points": summary.get(
                            "fresh_register_probe_points", 0
                        ),
                        "opaque_candidate_probe_points": summary.get(
                            "opaque_fresh_register_candidate_probe_points", 0
                        ),
                        "opaque_candidate_sgpr_growth_probe_points": summary.get(
                            "opaque_fresh_register_candidate_sgpr_growth_probe_points",
                            0,
                        ),
                        "opaque_candidate_vgpr_growth_probe_points": summary.get(
                            "opaque_fresh_register_candidate_vgpr_growth_probe_points",
                            0,
                        ),
                    }
                )
            for candidate in summary.get("sampled_opaque_fresh_register_candidates", []):
                sampled = pick_fields(candidate, CANDIDATE_FIELDS)
                if kernel and "kernel" not in sampled:
                    sampled["kernel"] = kernel
                opaque_fresh_candidates.append(sampled)

    edge_sites_patched = sum(
        row.get("edge_sites_patched", 0)
        for row in patch_events
        if isinstance(row.get("edge_sites_patched", 0), int)
    )
    no_site_kernels = []
    for summary in no_site_kernel_summaries.values():
        summary = dict(summary)
        summary["skip_reasons"] = sorted_counter(summary.pop("_skip_reasons"), 4)
        no_site_kernels.append(summary)

    return {
        "report": str(path),
        "patch_events": len(patch_events),
        "successful_patch_events": sum(1 for row in patch_events if row.get("success")),
        "edge_sites_patched": edge_sites_patched,
        "hashed_edge_sites": sum(
            row.get("hashed_edge_sites", 0)
            for row in patch_events
            if isinstance(row.get("hashed_edge_sites", 0), int)
        ),
        "probe_required_sgprs": max(
            [row.get("probe_required_sgprs", 0) for row in patch_events] or [0]
        ),
        "probe_required_vgprs": max(
            [row.get("probe_required_vgprs", 0) for row in patch_events] or [0]
        ),
        "first_selected_edge": selected_edges[0] if selected_edges else None,
        "last_sampled_selected_edge": selected_edges[-1] if selected_edges else None,
        "selected_edge_samples": len(selected_edges),
        "selected_edge_samples_truncated": edge_sites_patched > len(selected_edges),
        "first_opaque_fresh_candidate": (
            opaque_fresh_candidates[0] if opaque_fresh_candidates else None
        ),
        "fresh_kernels": fresh_kernels[:16],
        "descriptor_resources": summarize_descriptor_resources(descriptor_resources),
        "device_edge_deltas": summarize_device_edge_deltas(rows),
        "top_skip_reasons": sorted_counter(skip_reasons, 8),
        "no_site_kernel_count": len(no_site_kernels),
        "no_site_kernel_samples": sorted(
            no_site_kernels,
            key=lambda kernel: (
                as_int(kernel.get("branch_candidates", 0))
                + as_int(kernel.get("block_candidates", 0)),
                kernel.get("kernel", ""),
            ),
            reverse=True,
        )[:8],
    }


def patch_offsets_from_reports(paths, kernel_include=None, kernel_exclude=None,
                               slot_policy=None, slot_policy_limits=None):
    if slot_policy is not None and slot_policy_limits:
        raise ValueError("slot_policy and slot_policy_limits are mutually exclusive")
    if slot_policy_limits:
        offsets = []
        seen = set()
        for policy, limit in normalize_slot_policy_limits(slot_policy_limits):
            if limit <= 0:
                continue
            policy_offsets = patch_offsets_from_reports(
                paths,
                kernel_include=kernel_include,
                kernel_exclude=kernel_exclude,
                slot_policy=policy,
            )
            for offset in policy_offsets[:limit]:
                if offset in seen:
                    continue
                seen.add(offset)
                offsets.append(offset)
        return offsets

    offsets = []
    seen = set()
    for path in paths:
        for row in load_jsonl(path):
            if row.get("event") != "patch_device_elf":
                continue
            for edge in row.get("sampled_selected_edges", []):
                if not isinstance(edge, dict):
                    continue
                kernel = edge.get("kernel", "")
                if kernel_include and kernel_include not in kernel:
                    continue
                if kernel_exclude and kernel_exclude in kernel:
                    continue
                if slot_policy and edge.get("slot_policy") != slot_policy:
                    continue
                offset = edge.get("patch_text_offset")
                if not isinstance(offset, int):
                    continue
                if offset in seen:
                    continue
                seen.add(offset)
                offsets.append(offset)
    return offsets


def parse_limits(args):
    limits = list(args.limit or [])
    if args.limit_range:
        start, end = args.limit_range
        step = 1 if end >= start else -1
        limits.extend(range(start, end + step, step))
    if not limits:
        limits.append(1)
    return limits


def parse_int_auto(value):
    try:
        return int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc


def parse_slot_policy_limit(value):
    separator = "=" if "=" in value else ":"
    if separator not in value:
        raise argparse.ArgumentTypeError(
            "slot policy limits must use POLICY=COUNT"
        )
    policy, count_text = value.split(separator, 1)
    policy = policy.strip()
    if not policy:
        raise argparse.ArgumentTypeError("slot policy must not be empty")
    count = parse_int_auto(count_text.strip())
    if count < 0:
        raise argparse.ArgumentTypeError("slot policy limit must be non-negative")
    return policy, count


def normalize_slot_policy_limits(slot_policy_limits):
    normalized = []
    index_by_policy = {}
    for policy, count in slot_policy_limits:
        if policy in index_by_policy:
            index = index_by_policy[policy]
            normalized[index] = (policy, normalized[index][1] + count)
            continue
        index_by_policy[policy] = len(normalized)
        normalized.append((policy, count))
    return normalized


def compact_slot_policy_limits(slot_policy_limits):
    if not slot_policy_limits:
        return None
    return [
        {"slot_policy": policy, "limit": limit}
        for policy, limit in normalize_slot_policy_limits(slot_policy_limits)
    ]


def unique_offsets(offsets):
    unique = []
    seen = set()
    for offset in offsets:
        if offset in seen:
            continue
        seen.add(offset)
        unique.append(offset)
    return unique


def parse_patch_text_offset_set(values):
    offsets = []
    for value in values:
        for piece in value.replace(":", ",").replace(";", ",").split(","):
            piece = piece.strip()
            if piece:
                offsets.append(parse_int_auto(piece))
    return unique_offsets(offsets)


def append_candidate_to_base_offsets(base_offsets, candidate_offset):
    return unique_offsets(list(base_offsets) + [candidate_offset])


def patch_text_offset_set_digest(offsets):
    data = ",".join(str(offset) for offset in offsets).encode("ascii")
    return hashlib.sha1(data).hexdigest()[:12]


def compact_patch_text_offset_set(offsets):
    offsets = list(offsets)
    compact = {
        "count": len(offsets),
        "sha1": patch_text_offset_set_digest(offsets),
        "first": offsets[:8],
    }
    if len(offsets) > 8:
        compact["last"] = offsets[-8:]
    return compact


def command_from_args(args):
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    return command


def report_path_for_limit(report_dir, limit, patch_text_offset=None,
                          patch_text_offsets=None):
    if patch_text_offsets:
        compact = compact_patch_text_offset_set(patch_text_offsets)
        filename = (
            f"opaque-fresh-limit-{limit}-offset-set-"
            f"{compact['count']}-{compact['sha1']}.jsonl"
        )
        return report_dir / filename
    if patch_text_offset is None:
        return report_dir / f"opaque-fresh-limit-{limit}.jsonl"
    return report_dir / f"opaque-fresh-limit-{limit}-offset-0x{patch_text_offset:x}.jsonl"


def run_limit(args, limit, patch_text_offset=None, patch_text_offsets=None,
              candidate_patch_text_offset=None, base_patch_text_offsets=None):
    report_dir = pathlib.Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    active_patch_text_offsets = (
        patch_text_offsets
        if patch_text_offsets is not None
        else args.patch_text_offset_set
    )
    active_patch_text_offset = (
        None
        if active_patch_text_offsets
        else (args.patch_text_offset if patch_text_offset is None else patch_text_offset)
    )
    report_path = report_path_for_limit(
        report_dir,
        limit,
        active_patch_text_offset,
        active_patch_text_offsets,
    )
    try:
        report_path.unlink()
    except FileNotFoundError:
        pass

    env = os.environ.copy()
    env.update(minimizer_env_for_profile(args.diagnostic_profile))
    env["ROCJITSU_AFL_DEBUG_BRANCH_EDGE_LIMIT"] = str(limit)
    env["ROCJITSU_AFL_PATCH_REPORT"] = str(report_path)
    env.pop("ROCJITSU_AFL_DEBUG_EDGE_PATCH_TEXT_OFFSET", None)
    env.pop("ROCJITSU_AFL_DEBUG_EDGE_PATCH_TEXT_OFFSETS", None)
    if args.force_fresh_sgprs:
        env["ROCJITSU_AFL_DEBUG_FORCE_FRESH_SGPRS"] = "1"
    if args.force_fresh_vgprs:
        env["ROCJITSU_AFL_DEBUG_FORCE_FRESH_VGPRS"] = "1"
    if args.disable_vgpr_scratch_spills:
        env["ROCJITSU_AFL_DEBUG_DISABLE_VGPR_SCRATCH_SPILLS"] = "1"
    if args.require_device_edges:
        env["ROCJITSU_AFL_REQUIRE_DEVICE_EDGES"] = "1"
    if args.kernel_include:
        env["ROCJITSU_AFL_KERNEL_INCLUDE"] = args.kernel_include
    if args.kernel_exclude:
        env["ROCJITSU_AFL_KERNEL_EXCLUDE"] = args.kernel_exclude
    if active_patch_text_offsets:
        env["ROCJITSU_AFL_DEBUG_EDGE_PATCH_TEXT_OFFSETS"] = ",".join(
            str(offset) for offset in active_patch_text_offsets
        )
    elif active_patch_text_offset is not None:
        env["ROCJITSU_AFL_DEBUG_EDGE_PATCH_TEXT_OFFSET"] = str(active_patch_text_offset)
    if args.preload:
        old_preload = env.get("LD_PRELOAD")
        env["LD_PRELOAD"] = (
            args.preload if not old_preload else f"{args.preload}:{old_preload}"
        )

    completed = subprocess.run(command_from_args(args), env=env, check=False)
    summary = summarize_report(report_path)
    summary["limit"] = limit
    summary["returncode"] = completed.returncode
    summary["status"] = "ok" if completed.returncode == 0 else "failed"
    summary["diagnostic_profile"] = args.diagnostic_profile
    if active_patch_text_offset is not None:
        summary["patch_text_offset_filter"] = active_patch_text_offset
    if candidate_patch_text_offset is not None:
        summary["patch_text_offset_filter"] = candidate_patch_text_offset
        summary["patch_text_offset_candidate"] = candidate_patch_text_offset
    if base_patch_text_offsets:
        summary["patch_text_offset_base_set"] = compact_patch_text_offset_set(
            base_patch_text_offsets
        )
    if active_patch_text_offsets:
        summary["patch_text_offset_set"] = compact_patch_text_offset_set(
            active_patch_text_offsets
        )
    print(json.dumps(summary, sort_keys=True))
    return completed.returncode, summary


def patch_offset_chunks(offsets, n):
    chunk_size = max(1, (len(offsets) + n - 1) // n)
    return [offsets[i : i + chunk_size] for i in range(0, len(offsets), chunk_size)]


def minimize_patch_offset_set(args, limit, offsets):
    current = unique_offsets(offsets)
    summaries = []
    tested = set()

    def run_candidate(candidate):
        key = tuple(candidate)
        if key in tested:
            return None
        tested.add(key)
        rc, summary = run_limit(args, limit, patch_text_offsets=list(candidate))
        summaries.append(summary)
        return rc

    rc = run_candidate(current)
    if rc == 0:
        return summaries, current

    n = 2
    while len(current) >= 2:
        if (
            args.set_minimize_max_tests is not None
            and len(summaries) >= args.set_minimize_max_tests
        ):
            break

        reduced = False
        chunks = patch_offset_chunks(current, n)
        candidates = list(chunks)
        for chunk in chunks:
            chunk_set = set(chunk)
            complement = [offset for offset in current if offset not in chunk_set]
            if complement:
                candidates.append(complement)

        for candidate in candidates:
            if not candidate or len(candidate) == len(current):
                continue
            if (
                args.set_minimize_max_tests is not None
                and len(summaries) >= args.set_minimize_max_tests
            ):
                break
            candidate = unique_offsets(candidate)
            rc = run_candidate(candidate)
            if rc is None:
                continue
            if rc != 0:
                current = candidate
                n = max(2, n - 1)
                reduced = True
                break

        if reduced:
            continue
        if n >= len(current):
            break
        n = min(len(current), n * 2)

    return summaries, current


def summarize_patch_offset_set_minimization(summaries, minimized_offsets, metadata=None):
    metadata = metadata or {}
    failures = [
        summary for summary in summaries if as_int(summary.get("returncode")) != 0
    ]
    successes = [
        summary for summary in summaries if as_int(summary.get("returncode")) == 0
    ]
    return {
        "event": "opaque_fresh_patch_offset_set_minimization_summary",
        "status": "failed" if failures else "source-set-passed",
        "limit": metadata.get("limit"),
        "kernel_include": metadata.get("kernel_include"),
        "kernel_exclude": metadata.get("kernel_exclude"),
        "diagnostic_profile": metadata.get("diagnostic_profile"),
        "source_slot_policy": metadata.get("source_slot_policy"),
        "source_slot_policy_limits": metadata.get("source_slot_policy_limits"),
        "source_report": metadata.get("source_report"),
        "source_report_edge_sites_patched": metadata.get(
            "source_report_edge_sites_patched", 0
        ),
        "source_report_selected_edge_samples": metadata.get(
            "source_report_selected_edge_samples", 0
        ),
        "source_report_selected_edge_samples_truncated": metadata.get(
            "source_report_selected_edge_samples_truncated", False
        ),
        "runs": len(summaries),
        "successful_runs": len(successes),
        "failed_runs": len(failures),
        "minimized_offsets": list(minimized_offsets),
        "minimized_offset_set": compact_patch_text_offset_set(minimized_offsets),
        "last_failure": compact_limit_summary(failures[-1]) if failures else None,
        "last_success": compact_limit_summary(successes[-1]) if successes else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run or summarize a rocfuzz branch-edge minimization. The runner "
            "sets DEBUG-only coverage knobs; choose the diagnostic profile to "
            "decide whether opaque fresh-register guards are bypassed."
        )
    )
    parser.add_argument("--report", action="append", default=[],
                        help="summarize an existing ROCJITSU_AFL_PATCH_REPORT JSONL")
    parser.add_argument("--report-dir", help="directory for per-limit reports")
    parser.add_argument("--limit", type=int, action="append",
                        help="branch edge limit to run; may be repeated")
    parser.add_argument("--limit-range", type=int, nargs=2, metavar=("START", "END"),
                        help="inclusive branch edge limit range")
    parser.add_argument("--preload", help="librocjitsu_afl_preload.so path")
    parser.add_argument("--force-fresh-sgprs", action="store_true",
                        help="also force fresh SGPR allocation for selected probes")
    parser.add_argument("--force-fresh-vgprs", action="store_true",
                        help="also force fresh VGPR allocation for selected probes")
    parser.add_argument("--disable-vgpr-scratch-spills", action="store_true",
                        help=("disable VGPR scratch fallback so fresh-register "
                              "resource growth can be isolated"))
    parser.add_argument("--require-device-edges", action="store_true",
                        help="set ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1")
    parser.add_argument("--kernel-include",
                        help="set ROCJITSU_AFL_KERNEL_INCLUDE for this diagnostic run")
    parser.add_argument("--kernel-exclude",
                        help="set ROCJITSU_AFL_KERNEL_EXCLUDE for this diagnostic run")
    parser.add_argument("--diagnostic-profile",
                        choices=sorted(DIAGNOSTIC_PROFILE_ENV),
                        default="opaque-fresh",
                        help=("debug environment profile. opaque-fresh keeps "
                              "the legacy fresh-register override; "
                              "previous-bb-set keeps opaque fresh-register "
                              "growth disabled while minimizing branch patch "
                              "sets."))
    parser.add_argument("--source-slot-policy",
                        help="only extract sampled source offsets with this slot_policy")
    parser.add_argument("--source-slot-policy-limit",
                        type=parse_slot_policy_limit, action="append", default=[],
                        metavar="POLICY=COUNT",
                        help=("extract up to COUNT sampled source offsets for POLICY; "
                              "may be repeated to compose a mixed offset set"))
    parser.add_argument("--scan-base-source-slot-policy-limit",
                        type=parse_slot_policy_limit, action="append", default=[],
                        metavar="POLICY=COUNT",
                        help=("when scanning source offsets, prepend a fixed base set "
                              "extracted from sampled offsets with POLICY=COUNT"))
    parser.add_argument("--patch-text-offset", type=parse_int_auto,
                        help="select only a single edge patch_text_offset")
    parser.add_argument("--patch-text-offset-set", action="append", default=[],
                        metavar="OFFSETS",
                        help="select a comma-separated debug set of patch_text_offset values")
    parser.add_argument("--scan-patch-offsets-from", action="append", default=[],
                        metavar="REPORT",
                        help="scan sampled selected patch_text_offsets from a report")
    parser.add_argument("--minimize-patch-offset-set-from", action="append", default=[],
                        metavar="REPORT",
                        help="delta-reduce a failing sampled patch_text_offset set from a report")
    parser.add_argument("--set-minimize-max-tests", type=int,
                        help="maximum debug set-minimization target runs")
    parser.add_argument("--scan-offset-limit", type=int,
                        help="maximum number of extracted offsets to scan")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue after a failing command")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="target command after --")
    args = parser.parse_args()
    if args.source_slot_policy and args.source_slot_policy_limit:
        parser.error("--source-slot-policy and --source-slot-policy-limit are mutually exclusive")
    if args.scan_base_source_slot_policy_limit and not args.scan_patch_offsets_from:
        parser.error("--scan-base-source-slot-policy-limit requires --scan-patch-offsets-from")
    args.patch_text_offset_set = parse_patch_text_offset_set(
        args.patch_text_offset_set
    )
    source_slot_policy_limits = compact_slot_policy_limits(
        args.source_slot_policy_limit
    )
    source_base_slot_policy_limits = compact_slot_policy_limits(
        args.scan_base_source_slot_policy_limit
    )

    for report in args.report:
        print(json.dumps(summarize_report(pathlib.Path(report)), sort_keys=True))

    command = command_from_args(args)
    if not command:
        return 0
    if not args.report_dir:
        print("--report-dir is required when running a command", file=sys.stderr)
        return 2

    first_failure = 0
    summaries = []
    limits = parse_limits(args)
    scan_offsets = patch_offsets_from_reports(
        [pathlib.Path(path) for path in args.scan_patch_offsets_from],
        args.kernel_include,
        args.kernel_exclude,
        args.source_slot_policy,
        args.source_slot_policy_limit,
    )
    scan_base_offsets = patch_offsets_from_reports(
        [pathlib.Path(path) for path in args.scan_patch_offsets_from],
        args.kernel_include,
        args.kernel_exclude,
        slot_policy_limits=args.scan_base_source_slot_policy_limit,
    )
    minimize_offsets = patch_offsets_from_reports(
        [pathlib.Path(path) for path in args.minimize_patch_offset_set_from],
        args.kernel_include,
        args.kernel_exclude,
        args.source_slot_policy,
        args.source_slot_policy_limit,
    )
    if args.scan_offset_limit is not None:
        scan_offsets = scan_offsets[: args.scan_offset_limit]
        minimize_offsets = minimize_offsets[: args.scan_offset_limit]
    if args.scan_patch_offsets_from and not scan_offsets:
        print("no sampled selected patch offsets found to scan", file=sys.stderr)
        return 2
    if args.scan_base_source_slot_policy_limit and not scan_base_offsets:
        print("no sampled selected base patch offsets found to scan", file=sys.stderr)
        return 2
    if args.minimize_patch_offset_set_from and not minimize_offsets:
        print("no sampled selected patch offsets found to minimize", file=sys.stderr)
        return 2
    if minimize_offsets:
        source_summary = summarize_report(
            pathlib.Path(args.minimize_patch_offset_set_from[0])
        )
        for limit in limits:
            limit_summaries, minimized_offsets = minimize_patch_offset_set(
                args, limit, minimize_offsets
            )
            summaries.extend(limit_summaries)
            for summary in limit_summaries:
                if as_int(summary.get("returncode")) != 0 and first_failure == 0:
                    first_failure = as_int(summary.get("returncode"))
            print(
                json.dumps(
                    summarize_patch_offset_set_minimization(
                        limit_summaries,
                        minimized_offsets,
                        {
                            "limit": limit,
                            "kernel_include": args.kernel_include,
                            "kernel_exclude": args.kernel_exclude,
                            "diagnostic_profile": args.diagnostic_profile,
                            "source_slot_policy": args.source_slot_policy,
                            "source_slot_policy_limits": source_slot_policy_limits,
                            "source_report": args.minimize_patch_offset_set_from[0],
                            "source_report_edge_sites_patched": source_summary.get(
                                "edge_sites_patched", 0
                            ),
                            "source_report_selected_edge_samples": source_summary.get(
                                "selected_edge_samples", 0
                            ),
                            "source_report_selected_edge_samples_truncated": (
                                source_summary.get(
                                    "selected_edge_samples_truncated", False
                                )
                            ),
                        },
                    ),
                    sort_keys=True,
                )
            )
            if first_failure != 0 and not args.keep_going:
                break
    elif scan_offsets:
        source_summary = summarize_report(pathlib.Path(args.scan_patch_offsets_from[0]))
        for limit in limits:
            limit_summaries = []
            for offset in scan_offsets:
                if scan_base_offsets:
                    offset_set = append_candidate_to_base_offsets(
                        scan_base_offsets, offset
                    )
                    rc, summary = run_limit(
                        args,
                        limit,
                        patch_text_offsets=offset_set,
                        candidate_patch_text_offset=offset,
                        base_patch_text_offsets=scan_base_offsets,
                    )
                else:
                    rc, summary = run_limit(args, limit, offset)
                summaries.append(summary)
                limit_summaries.append(summary)
                if rc != 0 and first_failure == 0:
                    first_failure = rc
                if rc != 0 and not args.keep_going:
                    break
            print(
                json.dumps(
                    summarize_patch_offset_scan(
                        limit_summaries,
                        {
                            "limit": limit,
                            "kernel_include": args.kernel_include,
                            "kernel_exclude": args.kernel_exclude,
                            "diagnostic_profile": args.diagnostic_profile,
                            "source_slot_policy": args.source_slot_policy,
                            "source_slot_policy_limits": source_slot_policy_limits,
                            "source_base_slot_policy_limits": (
                                source_base_slot_policy_limits
                            ),
                            "base_patch_text_offset_set": (
                                compact_patch_text_offset_set(scan_base_offsets)
                                if scan_base_offsets
                                else None
                            ),
                            "source_report": args.scan_patch_offsets_from[0],
                            "source_report_edge_sites_patched": source_summary.get(
                                "edge_sites_patched", 0
                            ),
                            "source_report_selected_edge_samples": source_summary.get(
                                "selected_edge_samples", 0
                            ),
                            "source_report_selected_edge_samples_truncated": (
                                source_summary.get(
                                    "selected_edge_samples_truncated", False
                                )
                            ),
                            "source_report_device_edge_delta_events": source_summary.get(
                                "device_edge_deltas", {}
                            ).get("events", 0),
                        },
                    ),
                    sort_keys=True,
                )
            )
            if first_failure != 0 and not args.keep_going:
                break
    else:
        for limit in limits:
            rc, summary = run_limit(args, limit)
            summaries.append(summary)
            if rc != 0 and first_failure == 0:
                first_failure = rc
            if rc != 0 and not args.keep_going:
                break
    if summaries and not scan_offsets and not minimize_offsets:
        print(
            json.dumps(
                summarize_run(
                    summaries,
                    {
                        "kernel_include": args.kernel_include,
                        "kernel_exclude": args.kernel_exclude,
                        "diagnostic_profile": args.diagnostic_profile,
                        "source_slot_policy": args.source_slot_policy,
                        "source_slot_policy_limits": source_slot_policy_limits,
                        "patch_text_offset": args.patch_text_offset,
                        "patch_text_offset_set": (
                            compact_patch_text_offset_set(args.patch_text_offset_set)
                            if args.patch_text_offset_set
                            else None
                        ),
                    },
                ),
                sort_keys=True,
            )
        )
    return first_failure


if __name__ == "__main__":
    sys.exit(main())
