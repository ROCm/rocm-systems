#!/usr/bin/env python3

import argparse
import collections
import json
import sys


COUNTER_FIELDS = (
    "input_bytes",
    "entry_candidates",
    "entries_patched",
    "descriptor_updates",
    "edge_sites_selected",
    "edge_sites_patched",
    "edge_patch_failures",
    "local_text_caves",
    "appended_caves",
    "branch_range_failures",
    "hashed_edge_sites",
    "fixed_edge_sites",
    "fixed_slot_requests",
    "fixed_slots_reserved",
    "fixed_slot_exhaustions",
    "fixed_slot_collisions",
    "inline_slot_requests",
    "inline_slot_exhaustions",
    "branch_edges_degraded_to_fixed",
    "fixed_counter_branch_edge_aggregate_fallback_used",
    "fixed_counter_branch_edge_safety_fallback_used",
    "fixed_counter_branch_edge_liveness_fallback_used",
    "fixed_counter_branch_edge_placement_fallback_used",
    "exec_empty_fixed_counter_edges",
    "previous_bb_branch_edges_selected",
    "previous_bb_branch_sites_selected",
    "previous_bb_branch_sites_degraded_to_fixed",
    "edge_trampolines_planned",
    "previous_bb_branch_edge_trampolines_planned",
    "planned_appended_edge_trampolines",
    "planned_local_edge_trampolines",
    "previous_bb_branch_planned_appended_edge_trampolines",
    "previous_bb_branch_planned_local_edge_trampolines",
    "planned_edge_trampoline_bytes",
    "previous_bb_branch_edge_trampoline_bytes",
    "planned_appended_edge_trampoline_bytes",
    "planned_local_edge_trampoline_bytes",
    "previous_bb_branch_planned_appended_edge_trampoline_bytes",
    "previous_bb_branch_planned_local_edge_trampoline_bytes",
    "scratch_spill_probe_points",
    "vgpr_scratch_spill_probe_points",
    "sgpr_scratch_spill_probe_points",
    "probe_required_private_segment_bytes",
    "spill_bytes",
    "skipped_kernel_count",
)

KERNEL_COUNTER_FIELDS = (
    "reachable_blocks",
    "block_candidates",
    "block_selected",
    "skipped_unsafe",
    "skipped_liveness",
    "skipped_limit",
    "skipped_fixed_slot",
    "branch_candidates",
    "branch_edge_candidate_edges",
    "previous_bb_branch_edge_candidate_edges",
    "previous_bb_branch_site_candidate_sites",
    "branch_edge_budget",
    "previous_bb_branch_site_budget",
    "previous_bb_branch_edge_over_budget",
    "previous_bb_branch_site_over_budget",
    "fixed_counter_branch_edge_fallback_budget",
    "fixed_counter_branch_edge_aggregate_fallback_used",
    "fixed_counter_branch_edge_safety_fallback_used",
    "fixed_counter_branch_edge_liveness_fallback_used",
    "fixed_counter_branch_edge_placement_fallback_used",
    "fixed_counter_branch_edge_fallback_used",
    "exec_empty_fixed_counter_edges",
    "branch_edges_selected",
    "previous_bb_branch_edges_selected",
    "branch_edges_degraded_to_fixed",
    "previous_bb_branch_sites_selected",
    "previous_bb_branch_sites_degraded_to_fixed",
    "skipped_branch_unsafe",
    "skipped_branch_liveness",
    "skipped_branch_limit",
    "opaque_instruction_count",
    "unmodeled_opaque_instruction_count",
    "liveness_probe_points",
    "fresh_register_probe_points",
    "opaque_fresh_register_candidate_probe_points",
    "opaque_fresh_register_candidate_sgpr_growth_probe_points",
    "opaque_fresh_register_candidate_vgpr_growth_probe_points",
    "opaque_fresh_register_candidate_required_sgprs",
    "opaque_fresh_register_candidate_required_vgprs",
    "scratch_spill_probe_points",
    "vgpr_scratch_spill_probe_points",
    "sgpr_scratch_spill_probe_points",
    "fresh_register_growth_disabled_by_opaque_probe_points",
    "sgpr_scratch_spill_disabled_by_opaque_probe_points",
    "sgpr_scratch_spill_disabled_by_exec_condition_probe_points",
    "direct_exec_fixed_scratch_disabled_by_opaque_probe_points",
    "hashed_edge_sites",
    "fixed_edge_sites",
    "fixed_slot_requests",
    "fixed_slots_reserved",
    "fixed_slot_exhaustions",
    "fixed_slot_collisions",
    "inline_slot_requests",
    "inline_slot_exhaustions",
    "edge_trampolines_planned",
    "previous_bb_branch_edge_trampolines_planned",
    "planned_appended_edge_trampolines",
    "planned_local_edge_trampolines",
    "previous_bb_branch_planned_appended_edge_trampolines",
    "previous_bb_branch_planned_local_edge_trampolines",
    "planned_edge_trampoline_bytes",
    "previous_bb_branch_edge_trampoline_bytes",
    "planned_appended_edge_trampoline_bytes",
    "planned_local_edge_trampoline_bytes",
    "previous_bb_branch_planned_appended_edge_trampoline_bytes",
    "previous_bb_branch_planned_local_edge_trampoline_bytes",
)

MAX_COUNTER_FIELDS = (
    "largest_edge_trampoline_bytes",
    "largest_previous_bb_branch_edge_trampoline_bytes",
    "previous_bb_branch_afl_map_budget",
    "previous_bb_branch_afl_map_pressure_ppm",
    "previous_bb_branch_code_growth_pressure_ppm",
)
KERNEL_MAX_COUNTER_FIELDS = (
    "largest_edge_trampoline_bytes",
    "largest_previous_bb_branch_edge_trampoline_bytes",
    "previous_bb_branch_afl_map_budget",
    "previous_bb_branch_afl_map_pressure_ppm",
)

DELTA_COUNTER_FIELDS = (
    "launches",
    "hip_module_launches",
    "hip_runtime_launches",
    "runtime_shadow_launches",
    "lazy_ccob_launches",
    "entry_delta",
    "edge_slot_delta_count",
    "edge_counter_delta_total",
    "nonzero_edge_slots_total",
)

LOW_EDGE_REASON_RULES = (
    (
        "site_limit_cap",
        (
            "edge site limit",
            "branch edge site limit",
            "previous-BB branch site limit",
        ),
    ),
    (
        "loader_visible_scratch_growth",
        (
            "loader-visible private segment growth",
        ),
    ),
    (
        "liveness_or_register_pressure",
        (
            "liveness",
            "saved EXEC",
            "VGPR",
            "SGPR",
            "scratch",
            "register",
        ),
    ),
    (
        "cfg_shape_no_branch_edge",
        (
            "block has no instructions",
            "falls through without a branch terminator",
            "terminator exits kernel",
        ),
    ),
    (
        "unsupported_relocation_or_decode",
        (
            "not relocatable",
            "PC-relocatable",
            "not a direct patchable branch",
            "not branch control flow",
            "indirect control flow",
            "indirect call",
            "indirect branch",
            "no PC-relative target",
            "manipulates EXEC",
            "saveexec",
            "ALU wait",
            "wait-counter",
            "opaque",
            "unknown encoding",
            "SCC",
        ),
    ),
    (
        "known_unsafe_kernel_classifier",
        (
            "prior-invalid-dispatch-kernel",
            "runtime-internal-kernel",
        ),
    ),
    (
        "loader_scoped_kernel_filter",
        (
            "kernel-include-filter",
        ),
    ),
    (
        "cave_or_branch_range_pressure",
        (
            "text cave",
            "s_branch",
            "branch range",
        ),
    ),
)

LOW_EDGE_PATCH_REASON_CAUSES = {
    "unsupported_arch": "unsupported_gfxip_or_probe_model",
    "code_object_rejected": "loader_or_code_object_parse_failure",
    "temp_elf_write_failed": "loader_or_code_object_parse_failure",
    "no_kernel_descriptors": "loader_or_code_object_parse_failure",
}


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def as_int(value):
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def scaled_int(numerator, denominator, scale):
    if denominator <= 0 or scale <= 0:
        return 0
    return (numerator * scale) // denominator


def sorted_counter(counter):
    return {key: counter[key] for key in sorted(counter)}


def sorted_set(values):
    return sorted(value for value in values if value)


def format_counts(counter):
    return ", ".join(f"`{key}`={value}" for key, value in counter.items())


def format_edge_registers(edge):
    sgprs = []
    for key, label in (
        ("state_sgpr", "state"),
        ("saved_exec_sgpr", "exec"),
        ("tmp0_sgpr", "tmp0"),
        ("tmp1_sgpr", "tmp1"),
    ):
        if key in edge:
            sgprs.append(f"{label}=s{as_int(edge.get(key))}")
    vgprs = []
    for key, label in (
        ("workitem_vgpr", "wi"),
        ("tmp0_vgpr", "tmp0"),
        ("tmp1_vgpr", "tmp1"),
        ("tmp2_vgpr", "tmp2"),
    ):
        if key in edge:
            vgprs.append(f"{label}=v{as_int(edge.get(key))}")
    parts = []
    if sgprs:
        parts.append(" ".join(sgprs))
    if vgprs:
        parts.append(" ".join(vgprs))
    return "; ".join(parts)


def format_edge_scratch(edge):
    parts = []
    if edge.get("scratch_spill") and "scratch_address_vgpr" in edge:
        parts.append(f"addr=v{as_int(edge.get('scratch_address_vgpr'))}")
    spilled_vgprs = edge.get("scratch_spilled_vgprs") or []
    if spilled_vgprs:
        parts.append("spill_vgpr=" + ",".join(f"v{as_int(v)}" for v in spilled_vgprs))
    spilled_sgprs = edge.get("scratch_spilled_sgprs") or []
    if spilled_sgprs:
        parts.append("spill_sgpr=" + ",".join(f"s{as_int(s)}" for s in spilled_sgprs))
    return " ".join(parts)


def classify_low_edge_reason(reason):
    for cause, needles in LOW_EDGE_REASON_RULES:
        for needle in needles:
            if needle in reason:
                return cause
    return "other_skip_or_patch_failure"


def add_low_edge_reason(counter, reason, count):
    if not reason or count <= 0:
        return
    counter[classify_low_edge_reason(reason)] += count


def add_simple_kernel_causes(counter, summary):
    opaque_blocked_probe_points = as_int(
        summary.get("fresh_register_growth_disabled_by_opaque_probe_points")
    )
    if opaque_blocked_probe_points > 0:
        counter["opaque_decode_blocks_fresh_register_growth"] += opaque_blocked_probe_points
    opaque_fresh_sgpr_candidates = as_int(
        summary.get("opaque_fresh_register_candidate_sgpr_growth_probe_points")
    )
    if opaque_fresh_sgpr_candidates > 0:
        counter["opaque_fresh_candidate_requires_sgpr_growth"] += (
            opaque_fresh_sgpr_candidates
        )
    opaque_fresh_vgpr_candidates = as_int(
        summary.get("opaque_fresh_register_candidate_vgpr_growth_probe_points")
    )
    if opaque_fresh_vgpr_candidates > 0:
        counter["opaque_fresh_candidate_requires_vgpr_growth"] += (
            opaque_fresh_vgpr_candidates
        )
    sgpr_scratch_guarded = as_int(
        summary.get("sgpr_scratch_spill_disabled_by_opaque_probe_points")
    )
    if sgpr_scratch_guarded > 0:
        counter["opaque_decode_blocks_sgpr_scratch_spill"] += sgpr_scratch_guarded
    exec_condition_sgpr_guarded = as_int(
        summary.get("sgpr_scratch_spill_disabled_by_exec_condition_probe_points")
    )
    if exec_condition_sgpr_guarded > 0:
        counter["exec_condition_blocks_sgpr_scratch_spill"] += exec_condition_sgpr_guarded
    direct_exec_scratch_guarded = as_int(
        summary.get("direct_exec_fixed_scratch_disabled_by_opaque_probe_points")
    )
    if direct_exec_scratch_guarded > 0:
        counter["opaque_decode_blocks_direct_exec_fixed_scratch"] += (
            direct_exec_scratch_guarded
        )
    if (
        "reachable_blocks" not in summary
        and "branch_candidates" not in summary
        and "block_candidates" not in summary
    ):
        return
    reachable_blocks = as_int(summary.get("reachable_blocks"))
    branch_candidates = as_int(summary.get("branch_candidates"))
    block_candidates = as_int(summary.get("block_candidates"))
    block_selected = as_int(summary.get("block_selected"))
    branch_edges_selected = as_int(summary.get("branch_edges_selected"))
    if reachable_blocks <= 1:
        counter["straight_line_or_trivial_cfg"] += 1
    elif branch_candidates == 0 and block_candidates == 0:
        counter["straight_line_or_trivial_cfg"] += 1
    elif block_selected == 0 and branch_edges_selected == 0:
        counter["no_selected_device_sites"] += 1


def empty_kernel_summary(kernel):
    out = {
        "kernel": kernel,
        "patch_events": 0,
        "successful_patch_events": 0,
        "contexts": [],
        "device_images": [],
        "gfxips": [],
        "edge_instrumentation_reasons": [],
        "coverage_strategies": [],
        "coverage_signals": [],
        "branch_edge_slot_policy_reasons": [],
        "previous_bb_branch_site_budget_reasons": [],
        "fixed_counter_branch_edge_fallback_budget_reasons": [],
        "previous_bb_branch_aggregate_limit_kinds": [],
        "previous_bb_branch_aggregate_safety": [],
        "previous_bb_branch_aggregate_safety_reasons": [],
        "previous_bb_branch_overhead_statuses": [],
        "previous_bb_branch_overhead_reasons": [],
        "liveness_register_events": 0,
        "fresh_register_events": 0,
        "patchability_skip_events": 0,
        "sampled_opaque_instructions": [],
        "sampled_opaque_fresh_register_candidates": [],
        "sampled_skip_instructions": [],
        "sampled_selected_edges": [],
    }
    for field in KERNEL_COUNTER_FIELDS:
        out[field] = 0
    for field in KERNEL_MAX_COUNTER_FIELDS:
        out[field] = 0
    return out


def summarize(rows):
    event_counts = collections.Counter(row.get("event", "<missing>") for row in rows)
    contexts = collections.Counter()
    gfxips = collections.Counter()
    reasons = collections.Counter()
    edge_reasons = collections.Counter()
    coverage_signals = collections.Counter()
    failure_phases = collections.Counter()
    descriptor_resource_failure_reasons = collections.Counter()
    previous_bb_branch_overhead_statuses = collections.Counter()
    previous_bb_branch_overhead_reasons = collections.Counter()
    failures = collections.Counter()
    skip_reasons = collections.Counter()
    degradation_reasons = collections.Counter()
    sampled_selected_edges = []

    patch_totals = {
        "events": 0,
        "successful_events": 0,
        "failed_events": 0,
    }
    for field in COUNTER_FIELDS:
        patch_totals[field] = 0
    for field in MAX_COUNTER_FIELDS:
        patch_totals[field] = 0

    kernel_accumulators = {}
    kernel_contexts = collections.defaultdict(set)
    kernel_images = collections.defaultdict(set)
    kernel_gfxips = collections.defaultdict(set)
    kernel_edge_reasons = collections.defaultdict(set)
    kernel_coverage_strategies = collections.defaultdict(set)
    kernel_coverage_signals = collections.defaultdict(set)
    kernel_branch_slot_policy_reasons = collections.defaultdict(set)
    kernel_previous_bb_branch_site_budget_reasons = collections.defaultdict(set)
    kernel_fixed_counter_fallback_budget_reasons = collections.defaultdict(set)
    kernel_previous_bb_branch_aggregate_limit_kinds = collections.defaultdict(set)
    kernel_previous_bb_branch_aggregate_safety = collections.defaultdict(set)
    kernel_previous_bb_branch_aggregate_safety_reasons = collections.defaultdict(set)
    kernel_previous_bb_branch_overhead_statuses = collections.defaultdict(set)
    kernel_previous_bb_branch_overhead_reasons = collections.defaultdict(set)
    kernel_skip_reasons = collections.defaultdict(collections.Counter)
    kernel_degradation_reasons = collections.defaultdict(collections.Counter)
    low_edge_causes = collections.Counter()
    kernel_low_edge_causes = collections.defaultdict(collections.Counter)

    delta_totals = {"events": 0}
    for field in DELTA_COUNTER_FIELDS:
        delta_totals[field] = 0
    delta_triggers = collections.Counter()
    delta_last_kinds = collections.Counter()
    delta_last_kernels = collections.Counter()

    shadow_counts = collections.Counter()
    ccob_rebuild = {
        "events": 0,
        "successes": 0,
        "sibling_payloads_preserved": 0,
        "input_device_images": 0,
        "output_device_images": 0,
    }

    for row in rows:
        event = row.get("event")
        if event == "patch_device_elf":
            patch_totals["events"] += 1
            if row.get("success") is True:
                patch_totals["successful_events"] += 1
            else:
                patch_totals["failed_events"] += 1
            contexts[row.get("context", "")] += 1
            gfxips[row.get("gfxip", "")] += 1
            reasons[row.get("reason", "")] += 1
            edge_reasons[row.get("edge_instrumentation_reason", "")] += 1
            coverage_signal = row.get("coverage_signal", "")
            if coverage_signal:
                coverage_signals[coverage_signal] += 1
            failure_phase = row.get("failure_phase", "")
            if failure_phase:
                failure_phases[failure_phase] += 1
            descriptor_resource_failure_reason = row.get(
                "descriptor_resource_failure_reason", ""
            )
            if descriptor_resource_failure_reason:
                descriptor_resource_failure_reasons[
                    descriptor_resource_failure_reason
                ] += 1
            overhead_status = row.get("previous_bb_branch_overhead_status", "")
            if overhead_status:
                previous_bb_branch_overhead_statuses[overhead_status] += 1
            overhead_reason = row.get("previous_bb_branch_overhead_reason", "")
            if overhead_reason:
                previous_bb_branch_overhead_reasons[overhead_reason] += 1
            patch_reason = row.get("reason", "")
            patch_cause = LOW_EDGE_PATCH_REASON_CAUSES.get(patch_reason)
            if patch_cause:
                low_edge_causes[patch_cause] += 1
            for field in COUNTER_FIELDS:
                patch_totals[field] += as_int(row.get(field))
            for field in MAX_COUNTER_FIELDS:
                patch_totals[field] = max(patch_totals[field], as_int(row.get(field)))
            for failure in row.get("sampled_failures", []):
                if isinstance(failure, dict):
                    reason = failure.get("reason", "")
                    failures[reason] += 1
                    add_low_edge_reason(low_edge_causes, reason, 1)

            for edge in row.get("sampled_selected_edges", []):
                if not isinstance(edge, dict):
                    continue
                if len(sampled_selected_edges) < 32:
                    sampled_selected_edges.append(edge)
                kernel = edge.get("kernel", "")
                if kernel:
                    acc = kernel_accumulators.setdefault(kernel, empty_kernel_summary(kernel))
                    if len(acc["sampled_selected_edges"]) < 16:
                        acc["sampled_selected_edges"].append(edge)

            for summary in row.get("kernel_summaries", []):
                if not isinstance(summary, dict):
                    continue
                kernel = summary.get("kernel", "")
                if not kernel:
                    continue
                acc = kernel_accumulators.setdefault(kernel, empty_kernel_summary(kernel))
                add_simple_kernel_causes(kernel_low_edge_causes[kernel], summary)
                acc["patch_events"] += 1
                if row.get("success") is True:
                    acc["successful_patch_events"] += 1
                if summary.get("liveness_registers") is True:
                    acc["liveness_register_events"] += 1
                if summary.get("fresh_registers") is True:
                    acc["fresh_register_events"] += 1
                for field in KERNEL_COUNTER_FIELDS:
                    acc[field] += as_int(summary.get(field))
                for field in KERNEL_MAX_COUNTER_FIELDS:
                    acc[field] = max(acc[field], as_int(summary.get(field)))
                for degradation in summary.get("degradation_reason_counts", []):
                    if not isinstance(degradation, dict):
                        continue
                    reason = degradation.get("reason", "")
                    if not reason:
                        continue
                    count = as_int(degradation.get("count"))
                    degradation_reasons[reason] += count
                    kernel_degradation_reasons[kernel][reason] += count
                for sample in summary.get("sampled_opaque_instructions", []):
                    if len(acc["sampled_opaque_instructions"]) >= 16:
                        break
                    if isinstance(sample, dict):
                        acc["sampled_opaque_instructions"].append(sample)
                for sample in summary.get(
                    "sampled_opaque_fresh_register_candidates", []
                ):
                    if len(acc["sampled_opaque_fresh_register_candidates"]) >= 16:
                        break
                    if isinstance(sample, dict):
                        acc["sampled_opaque_fresh_register_candidates"].append(sample)
                counted_reasons = False
                for skip in summary.get("skip_reason_counts", []):
                    if not isinstance(skip, dict):
                        continue
                    reason = skip.get("reason", "")
                    if not reason:
                        continue
                    count = as_int(skip.get("count"))
                    skip_reasons[reason] += count
                    kernel_skip_reasons[kernel][reason] += count
                    add_low_edge_reason(low_edge_causes, reason, count)
                    add_low_edge_reason(kernel_low_edge_causes[kernel], reason, count)
                    counted_reasons = True
                if not counted_reasons:
                    for skip in summary.get("sampled_skips", []):
                        if not isinstance(skip, dict):
                            continue
                        reason = skip.get("reason", "")
                        if not reason:
                            continue
                        skip_reasons[reason] += 1
                        kernel_skip_reasons[kernel][reason] += 1
                        add_low_edge_reason(low_edge_causes, reason, 1)
                        add_low_edge_reason(kernel_low_edge_causes[kernel], reason, 1)
                for skip in summary.get("sampled_skips", []):
                    if len(acc["sampled_skip_instructions"]) >= 16:
                        break
                    if not isinstance(skip, dict):
                        continue
                    if "mnemonic" in skip or "words" in skip:
                        acc["sampled_skip_instructions"].append(skip)
                kernel_contexts[kernel].add(row.get("context", ""))
                kernel_images[kernel].add(row.get("device_image", ""))
                kernel_gfxips[kernel].add(row.get("gfxip", ""))
                kernel_edge_reasons[kernel].add(row.get("edge_instrumentation_reason", ""))
                kernel_coverage_strategies[kernel].add(summary.get("coverage_strategy", ""))
                coverage_signal = summary.get("coverage_signal", "")
                if coverage_signal:
                    kernel_coverage_signals[kernel].add(coverage_signal)
                branch_slot_reason = summary.get("branch_edge_slot_policy_reason", "")
                if branch_slot_reason:
                    kernel_branch_slot_policy_reasons[kernel].add(branch_slot_reason)
                previous_bb_site_budget_reason = summary.get(
                    "previous_bb_branch_site_budget_reason", ""
                )
                if previous_bb_site_budget_reason:
                    kernel_previous_bb_branch_site_budget_reasons[kernel].add(
                        previous_bb_site_budget_reason
                    )
                fixed_counter_fallback_reason = summary.get(
                    "fixed_counter_branch_edge_fallback_budget_reason", ""
                )
                if fixed_counter_fallback_reason:
                    kernel_fixed_counter_fallback_budget_reasons[kernel].add(
                        fixed_counter_fallback_reason
                    )
                aggregate_limit_kind = summary.get(
                    "previous_bb_branch_aggregate_limit_kind", ""
                )
                if aggregate_limit_kind:
                    kernel_previous_bb_branch_aggregate_limit_kinds[kernel].add(
                        aggregate_limit_kind
                    )
                aggregate_safety = summary.get(
                    "previous_bb_branch_aggregate_safety", ""
                )
                if aggregate_safety:
                    kernel_previous_bb_branch_aggregate_safety[kernel].add(
                        aggregate_safety
                    )
                aggregate_safety_reason = summary.get(
                    "previous_bb_branch_aggregate_safety_reason", ""
                )
                if aggregate_safety_reason:
                    kernel_previous_bb_branch_aggregate_safety_reasons[kernel].add(
                        aggregate_safety_reason
                    )
                overhead_status = summary.get(
                    "previous_bb_branch_overhead_status", ""
                )
                if overhead_status:
                    kernel_previous_bb_branch_overhead_statuses[kernel].add(
                        overhead_status
                    )
                overhead_reason = summary.get(
                    "previous_bb_branch_overhead_reason", ""
                )
                if overhead_reason:
                    kernel_previous_bb_branch_overhead_reasons[kernel].add(
                        overhead_reason
                    )

            for skipped in row.get("skipped_kernels", []):
                if not isinstance(skipped, dict):
                    continue
                kernel = skipped.get("kernel", "")
                reason = skipped.get("reason", "")
                if not kernel:
                    continue
                acc = kernel_accumulators.setdefault(kernel, empty_kernel_summary(kernel))
                acc["patchability_skip_events"] += 1
                if reason:
                    skip_reasons[reason] += 1
                    kernel_skip_reasons[kernel][reason] += 1
                    add_low_edge_reason(low_edge_causes, reason, 1)
                    add_low_edge_reason(kernel_low_edge_causes[kernel], reason, 1)
                kernel_contexts[kernel].add(row.get("context", ""))
                kernel_images[kernel].add(row.get("device_image", ""))
                kernel_gfxips[kernel].add(row.get("gfxip", ""))
                kernel_edge_reasons[kernel].add(row.get("edge_instrumentation_reason", ""))
                kernel_coverage_strategies[kernel].add("skipped")
                kernel_coverage_signals[kernel].add("skipped")

        elif event == "device_edge_delta":
            delta_totals["events"] += 1
            for field in DELTA_COUNTER_FIELDS:
                delta_totals[field] += as_int(row.get(field))
            delta_triggers[row.get("trigger", "")] += 1
            delta_last_kinds[row.get("last_kind", "")] += 1
            delta_last_kernels[row.get("last_kernel", "")] += 1

        elif isinstance(event, str) and (
            event.startswith("runtime_shadow_") or event.startswith("lazy_ccob_")
        ):
            shadow_counts[event] += 1

        elif event == "ccob_rebuild":
            ccob_rebuild["events"] += 1
            if row.get("success") is True:
                ccob_rebuild["successes"] += 1
            if row.get("sibling_payloads_preserved") is True:
                ccob_rebuild["sibling_payloads_preserved"] += 1
            ccob_rebuild["input_device_images"] += as_int(row.get("input_device_images"))
            ccob_rebuild["output_device_images"] += as_int(row.get("output_device_images"))

    patch_totals["previous_bb_branch_trampoline_avg_bytes_x100"] = scaled_int(
        patch_totals["previous_bb_branch_edge_trampoline_bytes"],
        patch_totals["previous_bb_branch_edge_trampolines_planned"],
        100,
    )
    patch_totals["previous_bb_branch_appended_trampoline_ratio_ppm"] = scaled_int(
        patch_totals["previous_bb_branch_planned_appended_edge_trampoline_bytes"],
        patch_totals["previous_bb_branch_edge_trampoline_bytes"],
        1000000,
    )
    patch_totals["previous_bb_branch_local_trampoline_ratio_ppm"] = scaled_int(
        patch_totals["previous_bb_branch_planned_local_edge_trampoline_bytes"],
        patch_totals["previous_bb_branch_edge_trampoline_bytes"],
        1000000,
    )
    patch_totals["previous_bb_branch_code_growth_pressure_ppm"] = max(
        patch_totals["previous_bb_branch_code_growth_pressure_ppm"],
        scaled_int(
            patch_totals["previous_bb_branch_planned_appended_edge_trampoline_bytes"],
            patch_totals["input_bytes"],
            1000000,
        ),
    )
    if patch_totals["previous_bb_branch_afl_map_budget"] > 0:
        patch_totals["previous_bb_branch_afl_map_pressure_ppm"] = max(
            patch_totals["previous_bb_branch_afl_map_pressure_ppm"],
            scaled_int(
                patch_totals["previous_bb_branch_edges_selected"],
                patch_totals["previous_bb_branch_afl_map_budget"],
                1000000,
            ),
        )

    kernels = []
    for kernel in sorted(kernel_accumulators):
        acc = kernel_accumulators[kernel]
        acc["contexts"] = sorted_set(kernel_contexts[kernel])
        acc["device_images"] = sorted_set(kernel_images[kernel])
        acc["gfxips"] = sorted_set(kernel_gfxips[kernel])
        acc["edge_instrumentation_reasons"] = sorted_set(kernel_edge_reasons[kernel])
        acc["coverage_strategies"] = sorted_set(kernel_coverage_strategies[kernel])
        acc["coverage_signals"] = sorted_set(kernel_coverage_signals[kernel])
        acc["branch_edge_slot_policy_reasons"] = sorted_set(
            kernel_branch_slot_policy_reasons[kernel]
        )
        acc["previous_bb_branch_site_budget_reasons"] = sorted_set(
            kernel_previous_bb_branch_site_budget_reasons[kernel]
        )
        acc["fixed_counter_branch_edge_fallback_budget_reasons"] = sorted_set(
            kernel_fixed_counter_fallback_budget_reasons[kernel]
        )
        acc["previous_bb_branch_aggregate_limit_kinds"] = sorted_set(
            kernel_previous_bb_branch_aggregate_limit_kinds[kernel]
        )
        acc["previous_bb_branch_aggregate_safety"] = sorted_set(
            kernel_previous_bb_branch_aggregate_safety[kernel]
        )
        acc["previous_bb_branch_aggregate_safety_reasons"] = sorted_set(
            kernel_previous_bb_branch_aggregate_safety_reasons[kernel]
        )
        acc["previous_bb_branch_overhead_statuses"] = sorted_set(
            kernel_previous_bb_branch_overhead_statuses[kernel]
        )
        acc["previous_bb_branch_overhead_reasons"] = sorted_set(
            kernel_previous_bb_branch_overhead_reasons[kernel]
        )
        acc["previous_bb_branch_trampoline_avg_bytes_x100"] = scaled_int(
            acc["previous_bb_branch_edge_trampoline_bytes"],
            acc["previous_bb_branch_edge_trampolines_planned"],
            100,
        )
        acc["previous_bb_branch_appended_trampoline_ratio_ppm"] = scaled_int(
            acc["previous_bb_branch_planned_appended_edge_trampoline_bytes"],
            acc["previous_bb_branch_edge_trampoline_bytes"],
            1000000,
        )
        acc["previous_bb_branch_local_trampoline_ratio_ppm"] = scaled_int(
            acc["previous_bb_branch_planned_local_edge_trampoline_bytes"],
            acc["previous_bb_branch_edge_trampoline_bytes"],
            1000000,
        )
        if acc["previous_bb_branch_afl_map_budget"] > 0:
            acc["previous_bb_branch_afl_map_pressure_ppm"] = max(
                acc["previous_bb_branch_afl_map_pressure_ppm"],
                scaled_int(
                    acc["previous_bb_branch_edges_selected"],
                    acc["previous_bb_branch_afl_map_budget"],
                    1000000,
                ),
            )
        acc["sampled_skip_reasons"] = sorted_counter(kernel_skip_reasons[kernel])
        acc["degradation_reasons"] = sorted_counter(kernel_degradation_reasons[kernel])
        acc["low_edge_causes"] = sorted_counter(kernel_low_edge_causes[kernel])
        kernels.append(acc)

    return {
        "event_counts": sorted_counter(event_counts),
        "patches": {
            **patch_totals,
            "contexts": sorted_counter(contexts),
            "gfxips": sorted_counter(gfxips),
            "reasons": sorted_counter(reasons),
            "edge_instrumentation_reasons": sorted_counter(edge_reasons),
            "coverage_signals": sorted_counter(coverage_signals),
            "failure_phases": sorted_counter(failure_phases),
            "descriptor_resource_failure_reasons": sorted_counter(
                descriptor_resource_failure_reasons
            ),
            "previous_bb_branch_overhead_statuses": sorted_counter(
                previous_bb_branch_overhead_statuses
            ),
            "previous_bb_branch_overhead_reasons": sorted_counter(
                previous_bb_branch_overhead_reasons
            ),
            "sampled_failure_reasons": sorted_counter(failures),
            "sampled_skip_reasons": sorted_counter(skip_reasons),
            "degradation_reasons": sorted_counter(degradation_reasons),
            "low_edge_causes": sorted_counter(low_edge_causes),
            "sampled_selected_edges": sampled_selected_edges,
        },
        "device_edge_delta": {
            **delta_totals,
            "triggers": sorted_counter(delta_triggers),
            "last_kinds": sorted_counter(delta_last_kinds),
            "last_kernels": sorted_counter(delta_last_kernels),
        },
        "shadow_loaders": sorted_counter(shadow_counts),
        "ccob_rebuild": ccob_rebuild,
        "kernels": kernels,
    }


def write_markdown(summary, label, out):
    print(f"# {label}", file=out)
    print("", file=out)
    patches = summary["patches"]
    deltas = summary["device_edge_delta"]
    print("| Metric | Value |", file=out)
    print("| --- | ---: |", file=out)
    print(f"| Patch events | {patches['events']} |", file=out)
    print(f"| Successful patch events | {patches['successful_events']} |", file=out)
    print(f"| Coverage signals | {format_counts(patches['coverage_signals'])} |", file=out)
    if patches["descriptor_resource_failure_reasons"]:
        print(
            f"| Descriptor resource failure reasons | "
            f"{format_counts(patches['descriptor_resource_failure_reasons'])} |",
            file=out,
        )
    print(f"| Edge sites selected | {patches['edge_sites_selected']} |", file=out)
    print(f"| Edge sites patched | {patches['edge_sites_patched']} |", file=out)
    print(f"| Previous-BB hashed edge sites | {patches['hashed_edge_sites']} |", file=out)
    print(f"| Fixed-counter edge sites | {patches['fixed_edge_sites']} |", file=out)
    print(
        f"| Branch edges degraded to fixed counters | "
        f"{patches['branch_edges_degraded_to_fixed']} |",
        file=out,
    )
    print(
        f"| Fixed-counter fallback causes | "
        f"agg={patches['fixed_counter_branch_edge_aggregate_fallback_used']}, "
        f"safety={patches['fixed_counter_branch_edge_safety_fallback_used']}, "
        f"live={patches['fixed_counter_branch_edge_liveness_fallback_used']}, "
        f"place={patches['fixed_counter_branch_edge_placement_fallback_used']} |",
        file=out,
    )
    print(
        f"| EXEC-empty fixed-counter edges | "
        f"{patches['exec_empty_fixed_counter_edges']} |",
        file=out,
    )
    print(
        f"| Previous-BB branch logical edges selected | "
        f"{patches['previous_bb_branch_edges_selected']} |",
        file=out,
    )
    print(
        f"| Previous-BB branch sites selected | "
        f"{patches['previous_bb_branch_sites_selected']} |",
        file=out,
    )
    print(
        f"| Previous-BB branch sites degraded to fixed counters | "
        f"{patches['previous_bb_branch_sites_degraded_to_fixed']} |",
        file=out,
    )
    print(f"| Planned edge trampolines | {patches['edge_trampolines_planned']} |", file=out)
    print(
        f"| Planned previous-BB branch trampolines | "
        f"{patches['previous_bb_branch_edge_trampolines_planned']} |",
        file=out,
    )
    print(
        f"| Planned edge trampoline bytes | "
        f"{patches['planned_edge_trampoline_bytes']} |",
        file=out,
    )
    print(
        f"| Planned previous-BB branch trampoline bytes | "
        f"{patches['previous_bb_branch_edge_trampoline_bytes']} |",
        file=out,
    )
    print(
        f"| Previous-BB branch overhead status | "
        f"{format_counts(patches['previous_bb_branch_overhead_statuses'])} |",
        file=out,
    )
    print(
        f"| Previous-BB branch AFL map pressure ppm | "
        f"{patches['previous_bb_branch_afl_map_pressure_ppm']} |",
        file=out,
    )
    print(
        f"| Previous-BB branch code growth pressure ppm | "
        f"{patches['previous_bb_branch_code_growth_pressure_ppm']} |",
        file=out,
    )
    print(
        f"| Previous-BB branch average trampoline bytes x100 | "
        f"{patches['previous_bb_branch_trampoline_avg_bytes_x100']} |",
        file=out,
    )
    print(
        f"| Previous-BB branch appended/local trampoline bytes | "
        f"{patches['previous_bb_branch_planned_appended_edge_trampoline_bytes']}/"
        f"{patches['previous_bb_branch_planned_local_edge_trampoline_bytes']} |",
        file=out,
    )
    print(
        f"| Planned appended/local trampoline bytes | "
        f"{patches['planned_appended_edge_trampoline_bytes']}/"
        f"{patches['planned_local_edge_trampoline_bytes']} |",
        file=out,
    )
    print(
        f"| Largest planned edge trampoline bytes | "
        f"{patches['largest_edge_trampoline_bytes']} |",
        file=out,
    )
    print(
        f"| Largest planned previous-BB branch trampoline bytes | "
        f"{patches['largest_previous_bb_branch_edge_trampoline_bytes']} |",
        file=out,
    )
    print(f"| Edge patch failures | {patches['edge_patch_failures']} |", file=out)
    print(f"| Device-edge delta events | {deltas['events']} |", file=out)
    print(f"| Device edge slots changed | {deltas['edge_slot_delta_count']} |", file=out)
    print(f"| Device edge counter delta | {deltas['edge_counter_delta_total']} |", file=out)
    print("", file=out)
    if patches["low_edge_causes"]:
        print("## Low-Edge Causes", file=out)
        print("", file=out)
        print("| Cause | Count |", file=out)
        print("| --- | ---: |", file=out)
        for cause, count in patches["low_edge_causes"].items():
            print(f"| `{cause}` | {count} |", file=out)
        print("", file=out)

    if patches["degradation_reasons"]:
        print("## Coverage Degradation Causes", file=out)
        print("", file=out)
        print("| Cause | Degraded Edges |", file=out)
        print("| --- | ---: |", file=out)
        for cause, count in patches["degradation_reasons"].items():
            print(f"| `{cause}` | {count} |", file=out)
        print("", file=out)

    if patches["sampled_selected_edges"]:
        print("## Selected Edge Samples", file=out)
        print("", file=out)
        print(
            "| Kernel | Kind | Patch | Policy | Fixed Slot | Exec0 | Scratch | Exec Source | Regs | Scratch Regs | Placement | Tramp Bytes |",
            file=out,
        )
        print(
            "| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- | ---: |",
            file=out,
        )
        for edge in patches["sampled_selected_edges"][:16]:
            scratch = "yes" if edge.get("scratch_spill") else "no"
            exec0 = "yes" if edge.get("force_lane0_exec_for_fixed_counter") else "no"
            print(
                f"| `{edge.get('kernel', '')}` | {edge.get('kind', '')} | "
                f"{edge.get('patch_text_offset', 0)} | {edge.get('slot_policy', '')} | "
                f"{edge.get('fixed_slot', 0)} | {exec0} | {scratch} | "
                f"{edge.get('scratch_address_exec_source', '')} | "
                f"{format_edge_registers(edge)} | {format_edge_scratch(edge)} | "
                f"{edge.get('placement', '')} | {edge.get('trampoline_bytes', 0)} |",
                file=out,
            )
        print("", file=out)

    fresh_candidates = []
    for kernel in summary["kernels"]:
        for candidate in kernel["sampled_opaque_fresh_register_candidates"]:
            if len(fresh_candidates) >= 16:
                break
            fresh_candidates.append((kernel["kernel"], candidate))
        if len(fresh_candidates) >= 16:
            break
    if fresh_candidates:
        print("## Opaque Fresh-Register Candidate Samples", file=out)
        print("", file=out)
        print(
            "| Kernel | Kind | Patch | Mnemonic | Policy | Required | Allocated | Growth | Regs |",
            file=out,
        )
        print("| --- | --- | ---: | --- | --- | --- | --- | --- | --- |", file=out)
        for kernel, candidate in fresh_candidates:
            growth = []
            if candidate.get("sgpr_growth"):
                growth.append("SGPR")
            if candidate.get("vgpr_growth"):
                growth.append("VGPR")
            print(
                f"| `{kernel}` | {candidate.get('kind', '')} | "
                f"{candidate.get('patch_text_offset', 0)} | "
                f"{candidate.get('mnemonic', '')} | "
                f"{candidate.get('slot_policy', '')} | "
                f"s{candidate.get('required_sgprs', 0)}/v{candidate.get('required_vgprs', 0)} | "
                f"s{candidate.get('allocated_sgprs', 0)}/v{candidate.get('allocated_vgprs', 0)} | "
                f"{','.join(growth)} | {format_edge_registers(candidate)} |",
                file=out,
            )
        print("", file=out)

    print("## Kernels", file=out)
    print("", file=out)
    kernel_columns = (
        "Kernel", "Strategies", "Signals", "Branch Policy Reasons",
        "PrevBB Site Budget Reasons", "PrevBB Limit", "PrevBB Aggregate",
        "Contexts", "Blocks", "Block Sites", "Branch Budget", "PrevBB Site Budget",
        "PrevBB Edge Over", "PrevBB Site Over", "Fixed Fallback Budget/Used",
        "Fixed Fallback Causes", "Branch Edges", "PrevBB Edges",
        "Fixed Fallback Edges", "PrevBB Sites", "PrevBB Site Fallbacks",
        "Trampolines", "PrevBB Trampolines",
        "Trampoline Bytes", "PrevBB Trampoline Bytes",
        "PrevBB App/Local Bytes", "PrevBB Map ppm", "PrevBB Avg Bytes x100",
        "PrevBB Overhead", "Appended/Local Bytes",
        "Max Trampoline", "Max PrevBB Trampoline",
        "Opaque Inst", "Unmodeled Opaque", "Opaque Fresh-Reg Gate",
        "Opaque Fresh Candidate", "Candidate SGPR Growth",
        "Candidate VGPR Growth", "SGPR Scratch Guard", "EXEC SGPR Guard",
        "Scratch Spill Probes", "VGPR Scratch", "SGPR Scratch",
        "Direct-EXEC Scratch Guard", "Skipped Unsafe", "Skipped Limit",
        "Patchability Skips",
    )
    kernel_alignments = (
        "---", "---", "---", "---", "---", "---", "---", "---",
        "---:", "---:", "---:", "---:", "---:", "---:", "---", "---",
        "---:", "---:", "---:", "---:", "---:", "---:", "---:",
        "---:", "---:", "---:", "---:", "---",
        "---:", "---:", "---:",
        "---:", "---:", "---:", "---:", "---:", "---:", "---:",
        "---:", "---:", "---:", "---:", "---:", "---:", "---:",
        "---:", "---:",
    )
    print("| " + " | ".join(kernel_columns) + " |", file=out)
    print("| " + " | ".join(kernel_alignments) + " |", file=out)
    for kernel in summary["kernels"]:
        contexts = ", ".join(kernel["contexts"])
        strategies = ", ".join(kernel["coverage_strategies"])
        signals = ", ".join(kernel["coverage_signals"])
        branch_policy_reasons = ", ".join(kernel["branch_edge_slot_policy_reasons"])
        previous_bb_site_budget_reasons = ", ".join(
            kernel["previous_bb_branch_site_budget_reasons"]
        )
        previous_bb_aggregate_limit_kinds = ", ".join(
            kernel["previous_bb_branch_aggregate_limit_kinds"]
        )
        previous_bb_aggregate_safety = ", ".join(
            kernel["previous_bb_branch_aggregate_safety"]
        )
        previous_bb_overhead = ", ".join(
            kernel["previous_bb_branch_overhead_statuses"]
        )
        fixed_fallback_causes = (
            f"agg={kernel['fixed_counter_branch_edge_aggregate_fallback_used']}, "
            f"safety={kernel['fixed_counter_branch_edge_safety_fallback_used']}, "
            f"live={kernel['fixed_counter_branch_edge_liveness_fallback_used']}, "
            f"place={kernel['fixed_counter_branch_edge_placement_fallback_used']}"
        )
        skipped_unsafe = (
            kernel["skipped_unsafe"]
            + kernel["skipped_liveness"]
            + kernel["skipped_branch_unsafe"]
            + kernel["skipped_branch_liveness"]
        )
        skipped_limit = kernel["skipped_limit"] + kernel["skipped_branch_limit"]
        print(
            f"| `{kernel['kernel']}` | {strategies} | {signals} | "
            f"{branch_policy_reasons} | {previous_bb_site_budget_reasons} | "
            f"{previous_bb_aggregate_limit_kinds} | "
            f"{previous_bb_aggregate_safety} | {contexts} | "
            f"{kernel['reachable_blocks']} | {kernel['block_selected']} | "
            f"{kernel['branch_edge_budget']} | "
            f"{kernel['previous_bb_branch_site_budget']} | "
            f"{kernel['previous_bb_branch_edge_over_budget']} | "
            f"{kernel['previous_bb_branch_site_over_budget']} | "
            f"{kernel['fixed_counter_branch_edge_fallback_budget']}/"
            f"{kernel['fixed_counter_branch_edge_fallback_used']} | "
            f"{fixed_fallback_causes} | "
            f"{kernel['branch_edges_selected']} | "
            f"{kernel['previous_bb_branch_edges_selected']} | "
            f"{kernel['branch_edges_degraded_to_fixed']} | "
            f"{kernel['previous_bb_branch_sites_selected']} | "
            f"{kernel['previous_bb_branch_sites_degraded_to_fixed']} | "
            f"{kernel['edge_trampolines_planned']} | "
            f"{kernel['previous_bb_branch_edge_trampolines_planned']} | "
            f"{kernel['planned_edge_trampoline_bytes']} | "
            f"{kernel['previous_bb_branch_edge_trampoline_bytes']} | "
            f"{kernel['previous_bb_branch_planned_appended_edge_trampoline_bytes']}/"
            f"{kernel['previous_bb_branch_planned_local_edge_trampoline_bytes']} | "
            f"{kernel['previous_bb_branch_afl_map_pressure_ppm']} | "
            f"{kernel['previous_bb_branch_trampoline_avg_bytes_x100']} | "
            f"{previous_bb_overhead} | "
            f"{kernel['planned_appended_edge_trampoline_bytes']}/"
            f"{kernel['planned_local_edge_trampoline_bytes']} | "
            f"{kernel['largest_edge_trampoline_bytes']} | "
            f"{kernel['largest_previous_bb_branch_edge_trampoline_bytes']} | "
            f"{kernel['opaque_instruction_count']} | "
            f"{kernel['unmodeled_opaque_instruction_count']} | "
            f"{kernel['fresh_register_growth_disabled_by_opaque_probe_points']} | "
            f"{kernel['opaque_fresh_register_candidate_probe_points']} | "
            f"{kernel['opaque_fresh_register_candidate_sgpr_growth_probe_points']} | "
            f"{kernel['opaque_fresh_register_candidate_vgpr_growth_probe_points']} | "
            f"{kernel['sgpr_scratch_spill_disabled_by_opaque_probe_points']} | "
            f"{kernel['sgpr_scratch_spill_disabled_by_exec_condition_probe_points']} | "
            f"{kernel['scratch_spill_probe_points']} | "
            f"{kernel['vgpr_scratch_spill_probe_points']} | "
            f"{kernel['sgpr_scratch_spill_probe_points']} | "
            f"{kernel['direct_exec_fixed_scratch_disabled_by_opaque_probe_points']} | "
            f"{skipped_unsafe} | {skipped_limit} | "
            f"{kernel['patchability_skip_events']} |",
            file=out,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize rocfuzz ROCJITSU_AFL_PATCH_REPORT JSONL diagnostics."
    )
    parser.add_argument("report")
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format.",
    )
    parser.add_argument("--label", default="rocfuzz patch report summary")
    args = parser.parse_args()

    summary = summarize(read_jsonl(args.report))
    if args.format == "json":
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        write_markdown(summary, args.label, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
