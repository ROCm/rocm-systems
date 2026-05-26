#!/usr/bin/env python3

import importlib.util
import io
import json
import subprocess
import sys


def load_module(path):
    spec = importlib.util.spec_from_file_location("summarize_patch_report", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    if len(sys.argv) != 3:
        print(
            "usage: patch_report_summary_unit.py "
            "<summarize_patch_report.py> <fixture.jsonl>",
            file=sys.stderr,
        )
        return 2

    tool_path = sys.argv[1]
    fixture_path = sys.argv[2]
    tool = load_module(tool_path)
    summary = tool.summarize(tool.read_jsonl(fixture_path))

    check(summary["event_counts"]["patch_device_elf"] == 2, "patch event count")
    check(summary["event_counts"]["device_edge_delta"] == 1, "delta event count")
    check(summary["patches"]["successful_events"] == 2, "successful patch count")
    check(summary["patches"]["edge_sites_selected"] == 7, "selected edge count")
    check(summary["patches"]["edge_sites_patched"] == 6, "patched edge count")
    check(summary["patches"]["failure_phases"] == {}, "successful fixture failure phases")
    check(summary["patches"]["branch_range_failures"] == 1, "branch range failures")
    check(summary["patches"]["contexts"]["runtime KPACK shadow"] == 1, "KPACK context")
    check(
        summary["patches"]["sampled_failure_reasons"]["branch range exceeds s_branch simm16"]
        == 1,
        "sampled failure reason",
    )
    check(
        summary["patches"]["sampled_skip_reasons"][
            "entry instruction is control flow and is not PC-relocatable yet"
        ]
        == 1,
        "sampled skip reason",
    )
    check(
        summary["patches"]["low_edge_causes"]
        == {
            "cave_or_branch_range_pressure": 1,
            "site_limit_cap": 1,
            "unsupported_relocation_or_decode": 2,
        },
        "low-edge causes",
    )
    check(summary["device_edge_delta"]["edge_slot_delta_count"] == 3, "edge delta slots")
    check(summary["device_edge_delta"]["last_kernels"]["kernel_b"] == 1, "last kernel")
    check(summary["shadow_loaders"]["runtime_shadow_function"] == 1, "runtime shadow count")
    check(summary["shadow_loaders"]["lazy_ccob_function"] == 1, "lazy CCOB count")
    check(summary["ccob_rebuild"]["sibling_payloads_preserved"] == 1, "CCOB preserve count")

    fixed_slot_summary = tool.summarize(
        [
            {
                "event": "patch_device_elf",
                "success": True,
                "context": "unit",
                "fixed_slot_collisions": 3,
                "probe_required_private_segment_bytes": 64,
                "spill_bytes": 48,
                "scratch_spill_probe_points": 2,
                "vgpr_scratch_spill_probe_points": 2,
                "sgpr_scratch_spill_probe_points": 1,
                "edge_trampolines_planned": 1,
                "planned_appended_edge_trampolines": 1,
                "planned_local_edge_trampolines": 0,
                "planned_edge_trampoline_bytes": 160,
                "planned_appended_edge_trampoline_bytes": 160,
                "planned_local_edge_trampoline_bytes": 0,
                "largest_edge_trampoline_bytes": 160,
                "sampled_selected_edges": [
                    {
                        "kernel": "fixed_kernel",
                        "kind": "cond-branch",
                        "patch_text_offset": 128,
                        "slot_policy": "fixed-counter",
                        "fixed_slot": 17,
                        "scratch_spill": True,
                        "scratch_address_exec_source": "exec",
                        "state_sgpr": 100,
                        "saved_exec_sgpr": 100,
                        "tmp0_sgpr": 104,
                        "tmp1_sgpr": 104,
                        "workitem_vgpr": 120,
                        "tmp0_vgpr": 121,
                        "tmp1_vgpr": 122,
                        "tmp2_vgpr": 123,
                        "scratch_address_vgpr": 4,
                        "scratch_spilled_vgprs": [5, 6],
                        "scratch_spilled_sgprs": [],
                        "placement": "appended-cave",
                        "trampoline_bytes": 160,
                    }
                ],
                "kernel_summaries": [
                    {
                        "kernel": "fixed_kernel",
                        "fixed_slot_collisions": 2,
                        "scratch_spill_probe_points": 2,
                        "vgpr_scratch_spill_probe_points": 2,
                        "sgpr_scratch_spill_probe_points": 1,
                        "edge_trampolines_planned": 1,
                        "planned_appended_edge_trampolines": 1,
                        "planned_local_edge_trampolines": 0,
                        "planned_edge_trampoline_bytes": 160,
                        "planned_appended_edge_trampoline_bytes": 160,
                        "planned_local_edge_trampoline_bytes": 0,
                        "largest_edge_trampoline_bytes": 160,
                    }
                ],
            }
        ]
    )
    check(
        fixed_slot_summary["patches"]["fixed_slot_collisions"] == 3,
        "fixed slot collisions should aggregate in patch counters",
    )
    check(
        fixed_slot_summary["kernels"][0]["fixed_slot_collisions"] == 2,
        "fixed slot collisions should aggregate in kernel counters",
    )
    check(
        fixed_slot_summary["patches"]["probe_required_private_segment_bytes"] == 64,
        "private segment requirements should aggregate in patch counters",
    )
    check(
        fixed_slot_summary["patches"]["spill_bytes"] == 48,
        "spill bytes should aggregate in patch counters",
    )
    check(
        fixed_slot_summary["patches"]["scratch_spill_probe_points"] == 2,
        "scratch spill probe points should aggregate in patch counters",
    )
    check(
        fixed_slot_summary["patches"]["vgpr_scratch_spill_probe_points"] == 2,
        "VGPR scratch spill probe points should aggregate in patch counters",
    )
    check(
        fixed_slot_summary["patches"]["sgpr_scratch_spill_probe_points"] == 1,
        "SGPR scratch spill probe points should aggregate in patch counters",
    )
    check(
        fixed_slot_summary["patches"]["edge_trampolines_planned"] == 1,
        "planned edge trampoline count should aggregate in patch counters",
    )
    check(
        fixed_slot_summary["patches"]["planned_edge_trampoline_bytes"] == 160,
        "planned edge trampoline bytes should aggregate in patch counters",
    )
    check(
        fixed_slot_summary["patches"]["largest_edge_trampoline_bytes"] == 160,
        "largest planned edge trampoline bytes should use max aggregation",
    )
    check(
        fixed_slot_summary["kernels"][0]["scratch_spill_probe_points"] == 2,
        "scratch spill probe points should aggregate in kernel counters",
    )
    check(
        fixed_slot_summary["kernels"][0]["vgpr_scratch_spill_probe_points"] == 2,
        "VGPR scratch spill probe points should aggregate in kernel counters",
    )
    check(
        fixed_slot_summary["kernels"][0]["sgpr_scratch_spill_probe_points"] == 1,
        "SGPR scratch spill probe points should aggregate in kernel counters",
    )
    check(
        fixed_slot_summary["kernels"][0]["planned_appended_edge_trampoline_bytes"] == 160,
        "planned appended trampoline bytes should aggregate in kernel counters",
    )

    trampoline_max_summary = tool.summarize(
        [
            {
                "event": "patch_device_elf",
                "success": True,
                "largest_edge_trampoline_bytes": 160,
                "kernel_summaries": [
                    {"kernel": "kernel", "largest_edge_trampoline_bytes": 160}
                ],
            },
            {
                "event": "patch_device_elf",
                "success": True,
                "largest_edge_trampoline_bytes": 96,
                "kernel_summaries": [
                    {"kernel": "kernel", "largest_edge_trampoline_bytes": 96}
                ],
            },
        ]
    )
    check(
        trampoline_max_summary["patches"]["largest_edge_trampoline_bytes"] == 160,
        "largest planned edge trampoline bytes should not be summed across reports",
    )
    check(
        trampoline_max_summary["kernels"][0]["largest_edge_trampoline_bytes"] == 160,
        "largest planned edge trampoline bytes should not be summed across kernel rows",
    )
    check(
        fixed_slot_summary["patches"]["sampled_selected_edges"][0]["slot_policy"]
        == "fixed-counter",
        "selected edge samples should be preserved in patch summaries",
    )
    check(
        fixed_slot_summary["kernels"][0]["sampled_selected_edges"][0][
            "scratch_address_exec_source"
        ]
        == "exec",
        "selected edge samples should be attached to kernel summaries",
    )
    check(
        fixed_slot_summary["patches"]["sampled_selected_edges"][0][
            "scratch_spilled_vgprs"
        ]
        == [5, 6],
        "selected edge register-allocation samples should be preserved",
    )
    check(
        tool.format_edge_registers(
            fixed_slot_summary["patches"]["sampled_selected_edges"][0]
        )
        == "state=s100 exec=s100 tmp0=s104 tmp1=s104; wi=v120 tmp0=v121 tmp1=v122 tmp2=v123",
        "selected edge register samples should format for markdown",
    )

    coverage_signal_summary = tool.summarize(
        [
            {
                "event": "patch_device_elf",
                "success": True,
                "coverage_signal": "hybrid-previous-bb-and-fixed",
                "previous_bb_branch_edges_selected": 8,
                "previous_bb_branch_afl_map_budget": 32768,
                "previous_bb_branch_edge_trampolines_planned": 5,
                "previous_bb_branch_edge_trampoline_bytes": 2000,
                "previous_bb_branch_planned_appended_edge_trampolines": 3,
                "previous_bb_branch_planned_local_edge_trampolines": 2,
                "previous_bb_branch_planned_appended_edge_trampoline_bytes": 1200,
                "previous_bb_branch_planned_local_edge_trampoline_bytes": 800,
                "largest_previous_bb_branch_edge_trampoline_bytes": 400,
                "previous_bb_branch_overhead_status": "partially-degraded-to-fixed",
                "kernel_summaries": [
                    {
                        "kernel": "hybrid_kernel",
                        "coverage_signal": "hybrid-previous-bb-and-fixed",
                        "previous_bb_branch_edges_selected": 8,
                        "previous_bb_branch_afl_map_budget": 32768,
                        "previous_bb_branch_edge_trampolines_planned": 5,
                        "previous_bb_branch_edge_trampoline_bytes": 2000,
                        "previous_bb_branch_planned_appended_edge_trampolines": 3,
                        "previous_bb_branch_planned_local_edge_trampolines": 2,
                        "previous_bb_branch_planned_appended_edge_trampoline_bytes": 1200,
                        "previous_bb_branch_planned_local_edge_trampoline_bytes": 800,
                        "largest_previous_bb_branch_edge_trampoline_bytes": 400,
                        "previous_bb_branch_overhead_status": (
                            "partially-degraded-to-fixed"
                        ),
                        "previous_bb_branch_aggregate_safety": (
                            "capped-previous-bb-with-fixed-fallback"
                        ),
                        "previous_bb_branch_aggregate_limit_kind": "edge-and-site-cap",
                        "previous_bb_branch_aggregate_safety_reason": (
                            "candidate previous-BB logical edges exceed the current "
                            "aggregate edge cap and writer sites exceed the current "
                            "aggregate state-writer cap; excess sites use fixed "
                            "counters when the smaller probe is safe"
                        ),
                        "previous_bb_branch_edge_over_budget": 12,
                        "previous_bb_branch_site_over_budget": 3,
                        "fixed_counter_branch_edge_fallback_budget": 128,
                        "fixed_counter_branch_edge_fallback_used": 12,
                        "fixed_counter_branch_edge_aggregate_fallback_used": 9,
                        "fixed_counter_branch_edge_safety_fallback_used": 2,
                        "fixed_counter_branch_edge_liveness_fallback_used": 1,
                        "fixed_counter_branch_edge_fallback_budget_reason": (
                            "auto-candidate-branch-edge-count"
                        ),
                    },
                    {
                        "kernel": "fixed_kernel",
                        "coverage_signal": "fixed-branch-counters",
                        "previous_bb_branch_aggregate_safety": (
                            "not-previous-bb-policy"
                        ),
                        "previous_bb_branch_aggregate_limit_kind": (
                            "not-previous-bb-policy"
                        ),
                        "fixed_counter_branch_edge_fallback_budget_reason": (
                            "not-previous-bb-policy"
                        ),
                    },
                ],
            },
            {
                "event": "patch_device_elf",
                "success": True,
                "coverage_signal": "previous-bb-edges",
                "previous_bb_branch_edges_selected": 4,
                "previous_bb_branch_afl_map_budget": 32768,
                "previous_bb_branch_edge_trampolines_planned": 2,
                "previous_bb_branch_edge_trampoline_bytes": 700,
                "previous_bb_branch_planned_appended_edge_trampolines": 0,
                "previous_bb_branch_planned_local_edge_trampolines": 2,
                "previous_bb_branch_planned_appended_edge_trampoline_bytes": 0,
                "previous_bb_branch_planned_local_edge_trampoline_bytes": 700,
                "largest_previous_bb_branch_edge_trampoline_bytes": 500,
                "previous_bb_branch_overhead_status": (
                    "within-current-resource-guards"
                ),
                "kernel_summaries": [
                    {
                        "kernel": "hybrid_kernel",
                        "coverage_signal": "previous-bb-edges",
                        "previous_bb_branch_edges_selected": 4,
                        "previous_bb_branch_afl_map_budget": 32768,
                        "previous_bb_branch_edge_trampolines_planned": 2,
                        "previous_bb_branch_edge_trampoline_bytes": 700,
                        "previous_bb_branch_planned_appended_edge_trampolines": 0,
                        "previous_bb_branch_planned_local_edge_trampolines": 2,
                        "previous_bb_branch_planned_appended_edge_trampoline_bytes": 0,
                        "previous_bb_branch_planned_local_edge_trampoline_bytes": 700,
                        "largest_previous_bb_branch_edge_trampoline_bytes": 500,
                        "previous_bb_branch_overhead_status": (
                            "within-current-resource-guards"
                        ),
                        "previous_bb_branch_aggregate_safety": (
                            "full-previous-bb-within-auto-budget"
                        ),
                        "previous_bb_branch_aggregate_limit_kind": "none",
                        "previous_bb_branch_edge_over_budget": 0,
                        "previous_bb_branch_site_over_budget": 0,
                        "fixed_counter_branch_edge_fallback_budget": 128,
                        "fixed_counter_branch_edge_fallback_used": 0,
                        "fixed_counter_branch_edge_aggregate_fallback_used": 0,
                        "fixed_counter_branch_edge_safety_fallback_used": 0,
                        "fixed_counter_branch_edge_liveness_fallback_used": 0,
                        "fixed_counter_branch_edge_fallback_budget_reason": (
                            "auto-candidate-branch-edge-count"
                        ),
                    }
                ],
            },
        ]
    )
    check(
        coverage_signal_summary["patches"]["coverage_signals"]
        == {
            "hybrid-previous-bb-and-fixed": 1,
            "previous-bb-edges": 1,
        },
        "patch coverage signals should aggregate by class",
    )
    signal_by_kernel = {
        kernel["kernel"]: kernel["coverage_signals"]
        for kernel in coverage_signal_summary["kernels"]
    }
    check(
        signal_by_kernel["hybrid_kernel"]
        == ["hybrid-previous-bb-and-fixed", "previous-bb-edges"],
        "kernel coverage signals should preserve every observed class",
    )
    check(
        signal_by_kernel["fixed_kernel"] == ["fixed-branch-counters"],
        "fixed counter kernel coverage signal should be preserved",
    )
    aggregate_safety_by_kernel = {
        kernel["kernel"]: kernel["previous_bb_branch_aggregate_safety"]
        for kernel in coverage_signal_summary["kernels"]
    }
    check(
        aggregate_safety_by_kernel["hybrid_kernel"]
        == [
            "capped-previous-bb-with-fixed-fallback",
            "full-previous-bb-within-auto-budget",
        ],
        "kernel aggregate previous-BB safety decisions should aggregate by class",
    )
    check(
        aggregate_safety_by_kernel["fixed_kernel"] == ["not-previous-bb-policy"],
        "fixed counter policy should report a non-previous-BB aggregate decision",
    )
    aggregate_limit_by_kernel = {
        kernel["kernel"]: kernel["previous_bb_branch_aggregate_limit_kinds"]
        for kernel in coverage_signal_summary["kernels"]
    }
    check(
        aggregate_limit_by_kernel["hybrid_kernel"] == ["edge-and-site-cap", "none"],
        "kernel aggregate previous-BB limit kinds should aggregate by class",
    )
    check(
        aggregate_limit_by_kernel["fixed_kernel"] == ["not-previous-bb-policy"],
        "fixed counter policy should report a non-previous-BB aggregate limit",
    )
    hybrid_kernel = next(
        kernel
        for kernel in coverage_signal_summary["kernels"]
        if kernel["kernel"] == "hybrid_kernel"
    )
    check(
        hybrid_kernel["fixed_counter_branch_edge_fallback_budget"] == 256
        and hybrid_kernel["fixed_counter_branch_edge_fallback_used"] == 12,
        "fixed fallback budget and use should aggregate per kernel",
    )
    check(
        hybrid_kernel["fixed_counter_branch_edge_aggregate_fallback_used"] == 9
        and hybrid_kernel["fixed_counter_branch_edge_safety_fallback_used"] == 2
        and hybrid_kernel["fixed_counter_branch_edge_liveness_fallback_used"] == 1
        and hybrid_kernel["fixed_counter_branch_edge_placement_fallback_used"] == 0,
        "fixed fallback cause counters should aggregate per kernel",
    )
    check(
        hybrid_kernel["previous_bb_branch_edge_over_budget"] == 12
        and hybrid_kernel["previous_bb_branch_site_over_budget"] == 3,
        "previous-BB aggregate budget overage should aggregate per kernel",
    )
    check(
        coverage_signal_summary["patches"]["previous_bb_branch_edges_selected"] == 12
        and hybrid_kernel["previous_bb_branch_edges_selected"] == 12,
        "selected previous-BB branch edge counts should aggregate",
    )
    check(
        coverage_signal_summary["patches"][
            "previous_bb_branch_edge_trampolines_planned"
        ]
        == 7
        and coverage_signal_summary["patches"][
            "previous_bb_branch_edge_trampoline_bytes"
        ]
        == 2700
        and coverage_signal_summary["patches"][
            "largest_previous_bb_branch_edge_trampoline_bytes"
        ]
        == 500
        and hybrid_kernel["previous_bb_branch_edge_trampolines_planned"] == 7
        and hybrid_kernel["previous_bb_branch_edge_trampoline_bytes"] == 2700
        and hybrid_kernel["largest_previous_bb_branch_edge_trampoline_bytes"] == 500,
        "previous-BB branch trampoline footprint should aggregate",
    )
    check(
        hybrid_kernel["previous_bb_branch_planned_appended_edge_trampoline_bytes"]
        == 1200
        and hybrid_kernel["previous_bb_branch_planned_local_edge_trampoline_bytes"]
        == 1500
        and hybrid_kernel["previous_bb_branch_trampoline_avg_bytes_x100"] == 38571
        and hybrid_kernel["previous_bb_branch_appended_trampoline_ratio_ppm"] == 444444
        and hybrid_kernel["previous_bb_branch_local_trampoline_ratio_ppm"] == 555555
        and hybrid_kernel["previous_bb_branch_afl_map_pressure_ppm"] == 366,
        "previous-BB branch overhead footprint should aggregate per kernel",
    )
    check(
        coverage_signal_summary["patches"][
            "previous_bb_branch_planned_appended_edge_trampoline_bytes"
        ]
        == 1200
        and coverage_signal_summary["patches"][
            "previous_bb_branch_planned_local_edge_trampoline_bytes"
        ]
        == 1500
        and coverage_signal_summary["patches"][
            "previous_bb_branch_trampoline_avg_bytes_x100"
        ]
        == 38571
        and coverage_signal_summary["patches"][
            "previous_bb_branch_appended_trampoline_ratio_ppm"
        ]
        == 444444
        and coverage_signal_summary["patches"][
            "previous_bb_branch_local_trampoline_ratio_ppm"
        ]
        == 555555
        and coverage_signal_summary["patches"][
            "previous_bb_branch_afl_map_pressure_ppm"
        ]
        == 366,
        "previous-BB branch overhead footprint should aggregate across patches",
    )
    check(
        coverage_signal_summary["patches"]["previous_bb_branch_overhead_statuses"]
        == {
            "partially-degraded-to-fixed": 1,
            "within-current-resource-guards": 1,
        },
        "previous-BB branch overhead status should aggregate by class",
    )
    check(
        hybrid_kernel["previous_bb_branch_overhead_statuses"]
        == ["partially-degraded-to-fixed", "within-current-resource-guards"],
        "previous-BB branch overhead status should aggregate per kernel",
    )
    check(
        hybrid_kernel["fixed_counter_branch_edge_fallback_budget_reasons"]
        == ["auto-candidate-branch-edge-count"],
        "fixed fallback budget reasons should aggregate",
    )
    check(
        hybrid_kernel["previous_bb_branch_aggregate_safety_reasons"]
        == [
            "candidate previous-BB logical edges exceed the current aggregate edge cap and writer sites exceed the current aggregate state-writer cap; excess sites use fixed counters when the smaller probe is safe"
        ],
        "aggregate safety reasons should aggregate",
    )

    aggregate_skip_summary = tool.summarize(
        [
            {
                "event": "patch_device_elf",
                "success": True,
                "context": "unit",
                "kernel_summaries": [
                    {
                        "kernel": "diagnostic_kernel",
                        "skip_reason_counts": [
                            {
                                "kind": "branch",
                                "reason": "no liveness-safe state SGPR pair",
                                "count": 7,
                            }
                        ],
                        "sampled_skips": [
                            {
                                "kind": "branch",
                                "text_offset": 16,
                                "reason": "terminator sample should not override counts",
                                "mnemonic": "s_cbranch_scc1",
                                "instruction_size": 4,
                                "instruction_flags": 2,
                                "words": [3213164546],
                            }
                        ],
                    }
                ],
            }
        ]
    )
    diagnostic_kernel = aggregate_skip_summary["kernels"][0]
    check(
        diagnostic_kernel["sampled_skip_reasons"]["no liveness-safe state SGPR pair"] == 7,
        "aggregate skip reason counts",
    )
    check(
        diagnostic_kernel["low_edge_causes"]["liveness_or_register_pressure"] == 7,
        "aggregate low-edge causes",
    )
    check(
        "terminator sample should not override counts"
        not in diagnostic_kernel["sampled_skip_reasons"],
        "aggregate skip reason counts should supersede samples",
    )
    check(
        diagnostic_kernel["sampled_skip_instructions"][0]["mnemonic"] == "s_cbranch_scc1"
        and diagnostic_kernel["sampled_skip_instructions"][0]["words"] == [3213164546],
        "aggregate should preserve sampled skipped instruction details",
    )

    degradation_summary = tool.summarize(
        [
            {
                "event": "patch_device_elf",
                "success": True,
                "context": "unit",
                "branch_edges_degraded_to_fixed": 3,
                "kernel_summaries": [
                    {
                        "kernel": "hybrid_kernel",
                        "branch_edges_degraded_to_fixed": 3,
                        "degradation_reason_counts": [
                            {
                                "kind": "branch",
                                "reason": "no liveness-safe saved EXEC SGPR pair",
                                "count": 2,
                            },
                            {
                                "kind": "branch",
                                "reason": "no liveness-safe VGPR run",
                                "count": 1,
                            },
                        ],
                    }
                ],
            }
        ]
    )
    hybrid_kernel = degradation_summary["kernels"][0]
    check(
        degradation_summary["patches"]["degradation_reasons"][
            "no liveness-safe saved EXEC SGPR pair"
        ]
        == 2,
        "aggregate degradation reasons",
    )
    check(
        hybrid_kernel["degradation_reasons"]["no liveness-safe VGPR run"] == 1,
        "kernel degradation reasons",
    )

    cfg_shape_summary = tool.summarize(
        [
            {
                "event": "patch_device_elf",
                "success": True,
                "context": "unit",
                "kernel_summaries": [
                    {
                        "kernel": "cfg_shape_kernel",
                        "skip_reason_counts": [
                            {
                                "kind": "branch",
                                "reason": "block falls through without a branch terminator",
                                "count": 5,
                            },
                            {
                                "kind": "branch",
                                "reason": "terminator exits kernel",
                                "count": 1,
                            },
                            {
                                "kind": "branch",
                                "reason": (
                                    "terminator is indirect call and needs "
                                    "return-address-preserving coverage policy"
                                ),
                                "count": 1,
                            },
                            {
                                "kind": "branch",
                                "reason": (
                                    "terminator is indirect branch and needs "
                                    "dynamic-target coverage policy"
                                ),
                                "count": 1,
                            },
                        ],
                    }
                ],
            }
        ]
    )
    cfg_shape_kernel = cfg_shape_summary["kernels"][0]
    check(
        cfg_shape_kernel["low_edge_causes"]["cfg_shape_no_branch_edge"] == 6,
        "fallthrough and exits should classify as CFG shape",
    )
    check(
        cfg_shape_kernel["low_edge_causes"]["unsupported_relocation_or_decode"] == 2,
        "indirect control flow should stay an unsupported relocation/decode cause",
    )

    skipped_kernel_summary = tool.summarize(
        [
            {
                "event": "patch_device_elf",
                "success": False,
                "context": "unit",
                "device_image": "raw:gfx1201",
                "gfxip": "gfx1201",
                "skipped_kernel_count": 1,
                "skipped_kernels": [
                    {
                        "kernel": "twiddle_gen_radices_sp",
                        "reason": "prior-invalid-dispatch-kernel",
                    }
                ],
            }
        ]
    )
    skipped_kernel = skipped_kernel_summary["kernels"][0]
    check(
        skipped_kernel_summary["patches"]["skipped_kernel_count"] == 1,
        "skipped kernel count should aggregate in patch counters",
    )
    check(
        skipped_kernel["patchability_skip_events"] == 1,
        "skipped kernel should appear in per-kernel summary",
    )
    check(
        skipped_kernel["sampled_skip_reasons"]["prior-invalid-dispatch-kernel"] == 1,
        "skipped kernel reason should aggregate",
    )
    check(
        skipped_kernel["low_edge_causes"]["known_unsafe_kernel_classifier"] == 1,
        "skipped kernel reason should classify as a low-edge cause",
    )

    loader_filter_summary = tool.summarize(
        [
            {
                "event": "patch_device_elf",
                "success": True,
                "context": "unit",
                "skipped_kernels": [
                    {
                        "kernel": "unlaunched_template_kernel",
                        "reason": "kernel-include-filter",
                    }
                ],
                "kernel_summaries": [
                    {
                        "kernel": "launched_kernel",
                        "skip_reason_counts": [
                            {
                                "kind": "branch",
                                "reason": "kernel-include-filter",
                                "count": 3,
                            }
                        ],
                    }
                ],
            }
        ]
    )
    loader_kernels = {
        kernel["kernel"]: kernel for kernel in loader_filter_summary["kernels"]
    }
    check(
        loader_filter_summary["patches"]["low_edge_causes"][
            "loader_scoped_kernel_filter"
        ]
        == 4,
        "loader-scoped kernel filters should classify separately",
    )
    check(
        loader_kernels["unlaunched_template_kernel"]["low_edge_causes"][
            "loader_scoped_kernel_filter"
        ]
        == 1,
        "skipped loader-scoped kernels should preserve per-kernel causes",
    )
    check(
        loader_kernels["launched_kernel"]["low_edge_causes"][
            "loader_scoped_kernel_filter"
        ]
        == 3,
        "kernel summary filters should preserve per-kernel causes",
    )

    opaque_summary = tool.summarize(
        [
            {
                "event": "patch_device_elf",
                "success": True,
                "context": "unit",
                "kernel_summaries": [
                    {
                        "kernel": "opaque_kernel",
                        "opaque_instruction_count": 3,
                        "unmodeled_opaque_instruction_count": 1,
                        "fresh_register_growth_disabled_by_opaque_probe_points": 5,
                        "opaque_fresh_register_candidate_probe_points": 5,
                        "opaque_fresh_register_candidate_sgpr_growth_probe_points": 5,
                        "opaque_fresh_register_candidate_vgpr_growth_probe_points": 2,
                        "opaque_fresh_register_candidate_required_sgprs": 16,
                        "opaque_fresh_register_candidate_required_vgprs": 104,
                        "sgpr_scratch_spill_disabled_by_opaque_probe_points": 4,
                        "sgpr_scratch_spill_disabled_by_exec_condition_probe_points": 6,
                        "direct_exec_fixed_scratch_disabled_by_opaque_probe_points": 2,
                        "sampled_opaque_instructions": [
                            {
                                "mnemonic": "vopd_opaque",
                                "text_offset": 32,
                                "liveness_modeled": True,
                                "words": [3355443200, 3735928559],
                            }
                        ],
                        "sampled_opaque_fresh_register_candidates": [
                            {
                                "kind": "branch",
                                "patch_text_offset": 64,
                                "mnemonic": "s_cbranch_scc1",
                                "required_sgprs": 16,
                                "required_vgprs": 80,
                                "allocated_sgprs": 8,
                                "allocated_vgprs": 80,
                                "sgpr_growth": True,
                                "vgpr_growth": False,
                                "previous_bb_probe_registers": True,
                                "stable_state_sgpr": False,
                                "slot_policy": "previous-bb-hash",
                                "state_sgpr": 8,
                                "saved_exec_sgpr": 10,
                                "tmp0_sgpr": 12,
                                "tmp1_sgpr": 14,
                                "workitem_vgpr": 73,
                                "tmp0_vgpr": 74,
                                "tmp1_vgpr": 75,
                                "tmp2_vgpr": 76,
                                "words": [3221225472],
                            }
                        ],
                    }
                ],
            }
        ]
    )
    opaque_kernel = opaque_summary["kernels"][0]
    check(opaque_kernel["opaque_instruction_count"] == 3, "opaque instruction count")
    check(
        opaque_kernel["unmodeled_opaque_instruction_count"] == 1,
        "unmodeled opaque instruction count",
    )
    check(
        opaque_kernel["fresh_register_growth_disabled_by_opaque_probe_points"] == 5,
        "opaque fresh-register gate probe points",
    )
    check(
        opaque_kernel["opaque_fresh_register_candidate_probe_points"] == 5,
        "opaque fresh-register candidate probe points",
    )
    check(
        opaque_kernel[
            "opaque_fresh_register_candidate_sgpr_growth_probe_points"
        ]
        == 5,
        "opaque fresh-register candidate SGPR growth probe points",
    )
    check(
        opaque_kernel[
            "opaque_fresh_register_candidate_vgpr_growth_probe_points"
        ]
        == 2,
        "opaque fresh-register candidate VGPR growth probe points",
    )
    check(
        opaque_kernel["opaque_fresh_register_candidate_required_sgprs"] == 16
        and opaque_kernel["opaque_fresh_register_candidate_required_vgprs"] == 104,
        "opaque fresh-register candidate register requirements",
    )
    check(
        opaque_kernel["sgpr_scratch_spill_disabled_by_opaque_probe_points"] == 4,
        "opaque SGPR scratch-spill guard probe points",
    )
    check(
        opaque_kernel[
            "sgpr_scratch_spill_disabled_by_exec_condition_probe_points"
        ]
        == 6,
        "EXEC-conditioned SGPR scratch-spill guard probe points",
    )
    check(
        opaque_kernel[
            "direct_exec_fixed_scratch_disabled_by_opaque_probe_points"
        ]
        == 2,
        "opaque direct-EXEC scratch guard probe points",
    )
    check(
        opaque_kernel["low_edge_causes"]["opaque_decode_blocks_fresh_register_growth"] == 5,
        "opaque fresh-register gate low-edge cause",
    )
    check(
        opaque_kernel["low_edge_causes"][
            "opaque_fresh_candidate_requires_sgpr_growth"
        ]
        == 5,
        "opaque fresh-register candidate SGPR low-edge cause",
    )
    check(
        opaque_kernel["low_edge_causes"][
            "opaque_fresh_candidate_requires_vgpr_growth"
        ]
        == 2,
        "opaque fresh-register candidate VGPR low-edge cause",
    )
    check(
        opaque_kernel["low_edge_causes"]["opaque_decode_blocks_sgpr_scratch_spill"] == 4,
        "opaque SGPR scratch-spill guard low-edge cause",
    )
    check(
        opaque_kernel["low_edge_causes"]["exec_condition_blocks_sgpr_scratch_spill"] == 6,
        "EXEC-conditioned SGPR scratch-spill guard low-edge cause",
    )
    check(
        opaque_kernel[
            "low_edge_causes"
        ]["opaque_decode_blocks_direct_exec_fixed_scratch"]
        == 2,
        "opaque direct-EXEC scratch guard low-edge cause",
    )
    check(
        opaque_kernel["sampled_opaque_instructions"]
        == [
            {
                "mnemonic": "vopd_opaque",
                "text_offset": 32,
                "liveness_modeled": True,
                "words": [3355443200, 3735928559],
            }
        ],
        "opaque instruction samples",
    )
    check(
        opaque_kernel["sampled_opaque_fresh_register_candidates"][0][
            "patch_text_offset"
        ]
        == 64,
        "opaque fresh-register candidate sample offset",
    )
    check(
        tool.format_edge_registers(
            opaque_kernel["sampled_opaque_fresh_register_candidates"][0]
        )
        == "state=s8 exec=s10 tmp0=s12 tmp1=s14; wi=v73 tmp0=v74 tmp1=v75 tmp2=v76",
        "opaque fresh-register candidate register sample formatting",
    )

    failure_phase_summary = tool.summarize(
        [
            {
                "event": "patch_device_elf",
                "success": False,
                "reason": "no_patchable_sites",
                "failure_phase": "plan",
                "descriptor_resource_failure_reason": (
                    "SGPR growth unsupported because SGPR count comes from AMDGPU metadata"
                ),
            },
            {
                "event": "patch_device_elf",
                "success": False,
                "reason": "entry_redirect_failed",
                "failure_phase": "install",
            },
        ]
    )
    check(
        failure_phase_summary["patches"]["failure_phases"] == {"install": 1, "plan": 1},
        "failure phases should aggregate by phase",
    )
    check(
        failure_phase_summary["patches"]["descriptor_resource_failure_reasons"]
        == {
            "SGPR growth unsupported because SGPR count comes from AMDGPU metadata": 1
        },
        "descriptor resource failure reasons should aggregate",
    )

    kernels = {kernel["kernel"]: kernel for kernel in summary["kernels"]}
    check(set(kernels) == {"kernel_a", "kernel_b"}, "kernel set")
    check(kernels["kernel_a"]["patch_events"] == 2, "kernel_a patch events")
    check(kernels["kernel_a"]["block_selected"] == 2, "kernel_a block selected")
    check(kernels["kernel_a"]["branch_edges_selected"] == 4, "kernel_a branch selected")
    check(kernels["kernel_a"]["liveness_register_events"] == 2, "kernel_a liveness")
    check(kernels["kernel_a"]["fresh_register_events"] == 1, "kernel_a fresh registers")
    check(kernels["kernel_a"]["fresh_register_probe_points"] == 1,
          "kernel_a fresh register probes")
    check(
        kernels["kernel_a"]["coverage_strategies"]
        == [
            "entry-previous-bb-block-and-previous-bb-branch",
            "self-contained-previous-bb-branch",
        ],
        "kernel_a coverage strategies",
    )
    check(
        kernels["kernel_a"]["contexts"] == ["HSA memory reader", "runtime KPACK shadow"],
        "kernel_a contexts",
    )
    check(
        kernels["kernel_a"]["sampled_skip_reasons"][
            "entry instruction is control flow and is not PC-relocatable yet"
        ]
        == 1,
        "kernel_a skip reason",
    )
    check(kernels["kernel_b"]["skipped_limit"] == 1, "kernel_b skipped limit")
    check(
        kernels["kernel_b"]["coverage_strategies"] == ["entry-previous-bb-block"],
        "kernel_b coverage strategies",
    )
    check(kernels["kernel_b"]["skipped_branch_unsafe"] == 1, "kernel_b branch unsafe")
    check(kernels["kernel_b"]["sampled_skip_reasons"]["edge site limit"] == 1, "kernel_b limit")

    json_output = subprocess.check_output(
        [sys.executable, tool_path, "--format", "json", fixture_path],
        text=True,
    )
    parsed_cli_summary = json.loads(json_output)
    check(parsed_cli_summary["patches"]["edge_sites_patched"] == 6, "CLI JSON output")

    markdown_output = subprocess.check_output(
        [sys.executable, tool_path, "--format", "markdown", "--label", "fixture", fixture_path],
        text=True,
    )
    check("# fixture" in markdown_output, "markdown heading")
    check("`kernel_a`" in markdown_output, "markdown kernel row")
    check("Device edge slots changed" in markdown_output, "markdown metrics")
    degradation_markdown = io.StringIO()
    tool.write_markdown(degradation_summary, "degradation", degradation_markdown)
    check(
        "Coverage Degradation Causes" in degradation_markdown.getvalue(),
        "markdown degradation causes",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
