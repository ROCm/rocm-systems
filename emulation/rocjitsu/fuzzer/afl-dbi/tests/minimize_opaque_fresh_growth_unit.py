#!/usr/bin/env python3

import importlib.util
import json
import pathlib
import sys
import tempfile


def load_module(path):
    spec = importlib.util.spec_from_file_location("minimize_opaque_fresh_growth", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def main():
    if len(sys.argv) != 2:
        print(
            "usage: minimize_opaque_fresh_growth_unit.py "
            "<minimize_opaque_fresh_growth.py>",
            file=sys.stderr,
        )
        return 2

    tool = load_module(sys.argv[1])
    opaque_env = tool.minimizer_env_for_profile("opaque-fresh")
    previous_bb_env = tool.minimizer_env_for_profile("previous-bb-set")
    check(
        opaque_env["ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY"] == "hashed",
        "opaque profile should force hashed branch slots",
    )
    check(
        previous_bb_env["ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY"] == "hashed",
        "previous-BB profile should force hashed branch slots",
    )
    check(
        opaque_env.get("ROCJITSU_AFL_DEBUG_ALLOW_OPAQUE_FRESH_REGISTERS") == "1",
        "opaque profile should enable opaque fresh-register diagnostics",
    )
    check(
        "ROCJITSU_AFL_DEBUG_ALLOW_OPAQUE_FRESH_REGISTERS" not in previous_bb_env,
        "previous-BB profile should keep opaque fresh-register growth disabled",
    )
    try:
        tool.minimizer_env_for_profile("unknown")
        check(False, "unknown diagnostic profile should fail")
    except ValueError:
        pass

    rows = [
        {
            "event": "patch_device_elf",
            "success": True,
            "edge_sites_patched": 3,
            "hashed_edge_sites": 2,
            "probe_required_sgprs": 12,
            "probe_required_vgprs": 59,
            "descriptor_resources": [
                {
                    "kernel": "Cijk_Ailk_Bljk_MT128x64",
                    "descriptor_file_offset": 256,
                    "wave32": True,
                    "old_sgpr_count": 8,
                    "patched_sgpr_count": 16,
                    "old_vgpr_count": 80,
                    "patched_vgpr_count": 80,
                    "old_private_segment_fixed_size": 0,
                    "patched_private_segment_fixed_size": 0,
                    "sgpr_count_metadata_patch": "in-place-bytes",
                    "resource_fields_changed": True,
                },
                {
                    "kernel": "other_kernel",
                    "descriptor_file_offset": 512,
                    "wave32": False,
                    "old_sgpr_count": 32,
                    "patched_sgpr_count": 32,
                    "old_vgpr_count": 96,
                    "patched_vgpr_count": 104,
                    "old_private_segment_fixed_size": 16,
                    "patched_private_segment_fixed_size": 64,
                    "private_segment_metadata_patch": "rebuilt-note-section",
                    "resource_fields_changed": True,
                },
            ],
            "sampled_selected_edges": [
                {
                    "kernel": "Cijk_Ailk_Bljk_MT128x64",
                    "kind": "cond-branch",
                    "patch_text_offset": 128,
                    "slot_policy": "previous-bb-hash",
                },
                {
                    "kernel": "Cijk_Ailk_Bljk_MT128x64",
                    "kind": "branch",
                    "patch_text_offset": 256,
                    "slot_policy": "previous-bb-hash",
                },
                {
                    "kernel": "other_kernel",
                    "kind": "branch",
                    "patch_text_offset": 512,
                    "slot_policy": "previous-bb-hash",
                },
                {
                    "kernel": "Cijk_Ailk_Bljk_MT128x64",
                    "kind": "branch",
                    "patch_text_offset": 768,
                    "slot_policy": "fixed-counter",
                }
            ],
            "kernel_summaries": [
                {
                    "kernel": "Cijk_Ailk_Bljk_MT128x64",
                    "fresh_registers": True,
                    "fresh_register_probe_points": 1,
                    "branch_edges_selected": 3,
                    "opaque_fresh_register_candidate_probe_points": 1,
                    "opaque_fresh_register_candidate_sgpr_growth_probe_points": 1,
                    "opaque_fresh_register_candidate_vgpr_growth_probe_points": 0,
                    "sampled_opaque_fresh_register_candidates": [
                        {
                            "kind": "cond-branch",
                            "patch_text_offset": 128,
                            "required_sgprs": 12,
                            "required_vgprs": 80,
                        }
                    ],
                    "skip_reason_counts": [
                        {
                            "kind": "branch",
                            "reason": "fresh SGPR growth disabled by target resource model",
                            "count": 4,
                        },
                        {
                            "kind": "branch",
                            "reason": "branch edge site limit",
                            "count": 2,
                        },
                    ],
                },
                {
                    "kernel": "no_site_kernel",
                    "coverage_signal": "none",
                    "branch_candidates": 5,
                    "branch_edges_selected": 0,
                    "skip_reason_counts": [
                        {
                            "kind": "branch",
                            "reason": "no liveness-safe allocated SGPR pair",
                            "count": 1,
                        }
                    ],
                },
            ],
        },
        {
            "event": "device_edge_delta",
            "trigger": "hipDeviceSynchronize",
            "launches": 1,
            "edge_slot_delta_count": 3,
            "edge_counter_delta_total": 7,
            "nonzero_edge_slots_total": 3,
            "first_kernel": "Cijk_Ailk_Bljk_MT128x64",
            "last_kernel": "Cijk_Ailk_Bljk_MT128x64",
            "last_kind": "runtime_shadow",
        },
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        report = pathlib.Path(temp_dir) / "report.jsonl"
        write_jsonl(report, rows)
        summary = tool.summarize_report(report)
        mt_offsets = tool.patch_offsets_from_reports(
            [report], kernel_include="MT128x64"
        )
        mt_previous_bb_offsets = tool.patch_offsets_from_reports(
            [report], kernel_include="MT128x64", slot_policy="previous-bb-hash"
        )
        mt_fixed_offsets = tool.patch_offsets_from_reports(
            [report], kernel_include="MT128x64", slot_policy="fixed-counter"
        )
        mt_mixed_policy_offsets = tool.patch_offsets_from_reports(
            [report],
            kernel_include="MT128x64",
            slot_policy_limits=[
                ("previous-bb-hash", 1),
                ("fixed-counter", 1),
            ],
        )
        mt_merged_policy_offsets = tool.patch_offsets_from_reports(
            [report],
            kernel_include="MT128x64",
            slot_policy_limits=[
                ("previous-bb-hash", 1),
                ("previous-bb-hash", 1),
                ("fixed-counter", 1),
            ],
        )

    resources = summary["descriptor_resources"]
    check(resources["updates"] == 2, "descriptor resource update count")
    check(resources["changed"] == 2, "descriptor resource changed count")
    check(resources["sgpr_growth_updates"] == 1, "SGPR growth update count")
    check(resources["vgpr_growth_updates"] == 1, "VGPR growth update count")
    check(resources["sgpr_only_growth_updates"] == 1, "SGPR-only growth update count")
    check(
        resources["sgpr_and_vgpr_growth_updates"] == 0,
        "SGPR+VGPR growth update count",
    )
    check(
        resources["private_segment_growth_updates"] == 1,
        "private segment growth update count",
    )
    check(resources["max_sgpr_growth"] == 8, "max SGPR growth")
    check(resources["max_vgpr_growth"] == 8, "max VGPR growth")
    check(resources["max_private_segment_growth"] == 48, "max private segment growth")
    check(
        resources["sampled_changes"][0]["kernel"] == "Cijk_Ailk_Bljk_MT128x64",
        "sampled descriptor changes should prioritize SGPR growth",
    )
    check(
        resources["sampled_changes"][0]["old_sgpr_count"] == 8,
        "descriptor sample should preserve old SGPR count",
    )
    check(
        resources["sampled_changes"][0]["patched_sgpr_count"] == 16,
        "descriptor sample should preserve patched SGPR count",
    )
    check(
        resources["sampled_sgpr_only_changes"][0]["kernel"]
        == "Cijk_Ailk_Bljk_MT128x64",
        "SGPR-only descriptor samples should identify the MT kernel",
    )
    check(
        "Cijk_Ailk_Bljk_MT128x64" in resources["changed_kernels"],
        "changed kernel list should include MT kernel",
    )
    check(not resources["sampled_changes_truncated"], "sample should not be truncated")

    deltas = summary["device_edge_deltas"]
    check(deltas["events"] == 1, "device-edge delta event count")
    check(deltas["edge_slot_delta_count"] == 3, "device-edge delta slots")
    check(deltas["edge_counter_delta_total"] == 7, "device-edge counter delta")
    check(deltas["max_nonzero_edge_slots_total"] == 3, "nonzero edge slots")
    check(
        deltas["last"]["trigger"] == "hipDeviceSynchronize",
        "last delta trigger should be preserved",
    )
    check(
        summary["first_opaque_fresh_candidate"]["kernel"] == "Cijk_Ailk_Bljk_MT128x64",
        "opaque fresh candidate should still be reported",
    )
    check(mt_offsets == [128, 256, 768],
          "MT patch-offset extraction should preserve order")
    check(mt_previous_bb_offsets == [128, 256],
          "MT previous-BB patch-offset extraction should filter by policy")
    check(mt_fixed_offsets == [768],
          "MT fixed-counter patch-offset extraction should filter by policy")
    check(mt_mixed_policy_offsets == [128, 768],
          "MT mixed policy patch-offset extraction should preserve policy order")
    check(mt_merged_policy_offsets == [128, 256, 768],
          "MT mixed policy patch-offset extraction should merge duplicate policies")
    compact_policy_limits = tool.compact_slot_policy_limits([
        ("previous-bb-hash", 1),
        ("previous-bb-hash", 2),
        ("fixed-counter", 1),
    ])
    check(
        compact_policy_limits == [
            {"slot_policy": "previous-bb-hash", "limit": 3},
            {"slot_policy": "fixed-counter", "limit": 1},
        ],
        "slot policy limits should be compact and ordered",
    )
    base_plus_candidate = tool.append_candidate_to_base_offsets(
        mt_previous_bb_offsets, 768
    )
    check(base_plus_candidate == [128, 256, 768],
          "base plus candidate offset set should append candidates")
    duplicate_base_plus_candidate = tool.append_candidate_to_base_offsets(
        mt_previous_bb_offsets, 128
    )
    check(duplicate_base_plus_candidate == [128, 256],
          "base plus candidate offset set should deduplicate candidates")
    try:
        tool.patch_offsets_from_reports(
            [report],
            kernel_include="MT128x64",
            slot_policy="previous-bb-hash",
            slot_policy_limits=[("fixed-counter", 1)],
        )
        check(False, "conflicting source policy filters should fail")
    except ValueError:
        pass
    parsed_offset_set = tool.parse_patch_text_offset_set(["0x80, 256", "0x80;512"])
    check(parsed_offset_set == [128, 256, 512],
          "patch-offset set parsing should preserve unique order")
    compact_offset_set = tool.compact_patch_text_offset_set(range(10))
    check(compact_offset_set["count"] == 10,
          "compact offset set should preserve count")
    check(compact_offset_set["first"] == list(range(8)),
          "compact offset set should preserve first offsets")
    check(compact_offset_set["last"] == [2, 3, 4, 5, 6, 7, 8, 9],
          "compact offset set should preserve tail offsets")
    check(summary["selected_edge_samples"] == 4, "selected-edge sample count")
    check(
        not summary["selected_edge_samples_truncated"],
        "selected-edge sample completeness",
    )
    check(
        summary["top_skip_reasons"][0]["reason"]
        == "fresh SGPR growth disabled by target resource model",
        "minimizer summary should surface no-site liveness reason",
    )
    check(summary["top_skip_reasons"][0]["count"] == 4,
          "minimizer summary should retain no-site liveness count")
    check(summary["no_site_kernel_count"] == 1,
          "no-site kernels should be counted")
    check(
        summary["no_site_kernel_samples"][0]["skip_reasons"][0]["reason"]
        == "no liveness-safe allocated SGPR pair",
        "no-site kernel samples should retain local skip reasons",
    )

    ok_summary = dict(summary)
    ok_summary["limit"] = 44
    ok_summary["returncode"] = 0
    ok_summary["status"] = "ok"
    ok_summary["diagnostic_profile"] = "previous-bb-set"
    ok_summary["patch_text_offset_filter"] = 0x20
    failed_summary = dict(summary)
    failed_summary["limit"] = 45
    failed_summary["returncode"] = -6
    failed_summary["status"] = "failed"
    failed_summary["diagnostic_profile"] = "previous-bb-set"
    failed_summary["patch_text_offset_filter"] = 0x40
    failed_summary["patch_text_offset_candidate"] = 0x40
    failed_summary["patch_text_offset_base_set"] = tool.compact_patch_text_offset_set(
        [128, 256]
    )
    failed_summary["patch_text_offset_set"] = tool.compact_patch_text_offset_set(
        [128, 256, 0x40]
    )
    failed_summary["device_edge_deltas"] = {
        "events": 0,
        "edge_slot_delta_count": 0,
        "edge_counter_delta_total": 0,
        "max_nonzero_edge_slots_total": 0,
        "first": None,
        "last": None,
    }
    run_summary = tool.summarize_run(
        [ok_summary, failed_summary],
        {
            "kernel_include": "MT128x64",
            "kernel_exclude": None,
            "diagnostic_profile": "previous-bb-set",
            "source_slot_policy": "previous-bb-hash",
            "source_slot_policy_limits": compact_policy_limits,
            "patch_text_offset": 0x40,
        },
    )
    check(
        run_summary["event"] == "opaque_fresh_minimization_summary",
        "run summary event name",
    )
    check(run_summary["status"] == "failed", "run summary status")
    check(run_summary["failing_limits"] == [45], "run summary failing limits")
    check(run_summary["kernel_include"] == "MT128x64", "run summary kernel include")
    check(run_summary["diagnostic_profile"] == "previous-bb-set",
          "run summary diagnostic profile")
    check(run_summary["source_slot_policy"] == "previous-bb-hash",
          "run summary source slot policy")
    check(run_summary["source_slot_policy_limits"] == compact_policy_limits,
          "run summary source slot policy limits")
    check(run_summary["patch_text_offset"] == 0x40, "run summary patch offset")
    check(
        run_summary["first_failure"]["limit"] == 45,
        "run summary first failure limit",
    )
    check(
        run_summary["first_failure"]["diagnostic_profile"] == "previous-bb-set",
        "compact failure should retain diagnostic profile",
    )
    check(
        run_summary["first_failure"]["patch_text_offset_filter"] == 0x40,
        "run summary first failure patch-offset filter",
    )
    check(
        run_summary["last_success_before_failure"]["limit"] == 44,
        "run summary last success before failure",
    )
    check(
        run_summary["first_failure"]["descriptor_resources"][
            "sgpr_only_growth_updates"
        ]
        == 1,
        "run summary should retain compact descriptor growth class",
    )
    check(
        run_summary["first_failure"]["first_selected_edge"]["kernel"]
        == "Cijk_Ailk_Bljk_MT128x64",
        "run summary should retain sampled patch/register tuple",
    )
    check(
        run_summary["first_failure"]["selected_edge_samples"] == 4,
        "run summary should retain selected-edge sample count",
    )
    check(
        run_summary["first_failure"]["top_skip_reasons"][0]["reason"]
        == "fresh SGPR growth disabled by target resource model",
        "compact failure should retain top skip reasons",
    )
    check(
        not run_summary["first_failure"]["selected_edge_samples_truncated"],
        "run summary should retain selected-edge sample completeness",
    )
    offset_scan_summary = tool.summarize_patch_offset_scan(
        [ok_summary, failed_summary],
        {
            "limit": 45,
            "kernel_include": "MT128x64",
            "diagnostic_profile": "previous-bb-set",
            "source_slot_policy": "fixed-counter",
            "source_slot_policy_limits": compact_policy_limits,
            "source_base_slot_policy_limits": [
                {"slot_policy": "previous-bb-hash", "limit": 2},
            ],
            "base_patch_text_offset_set": tool.compact_patch_text_offset_set(
                [128, 256]
            ),
            "source_report": "/tmp/report.jsonl",
            "source_report_edge_sites_patched": 3,
            "source_report_selected_edge_samples": 4,
            "source_report_selected_edge_samples_truncated": False,
            "source_report_device_edge_delta_events": 0,
        },
    )
    check(
        offset_scan_summary["event"] == "opaque_fresh_patch_offset_scan_summary",
        "offset scan summary event name",
    )
    check(offset_scan_summary["first_failing_offset"] == 0x40,
          "offset scan first failing offset")
    check(offset_scan_summary["smallest_failing_offset"] == 0x40,
          "offset scan smallest failing offset")
    check(not offset_scan_summary["all_scanned_offsets_passed"],
          "offset scan should record failures")
    check(offset_scan_summary["source_report_edge_sites_patched"] == 3,
          "offset scan should retain source report edge count")
    check(offset_scan_summary["source_report_selected_edge_samples"] == 4,
          "offset scan should retain source selected-edge sample count")
    check(offset_scan_summary["source_slot_policy"] == "fixed-counter",
          "offset scan should retain source slot policy")
    check(offset_scan_summary["diagnostic_profile"] == "previous-bb-set",
          "offset scan should retain diagnostic profile")
    check(offset_scan_summary["source_slot_policy_limits"] == compact_policy_limits,
          "offset scan should retain source slot policy limits")
    check(
        offset_scan_summary["source_base_slot_policy_limits"]
        == [{"slot_policy": "previous-bb-hash", "limit": 2}],
        "offset scan should retain source base slot policy limits",
    )
    check(
        offset_scan_summary["base_patch_text_offset_set"]["count"] == 2,
        "offset scan should retain compact base offset set",
    )
    check(
        offset_scan_summary["first_failure"]["patch_text_offset_candidate"] == 0x40,
        "offset scan compact failure should retain candidate offset",
    )
    check(
        offset_scan_summary["first_failure"]["patch_text_offset_base_set"]["count"] == 2,
        "offset scan compact failure should retain base set",
    )
    check(not offset_scan_summary["source_report_selected_edge_samples_truncated"],
          "offset scan should retain source selected-edge sample completeness")
    offset_set_summary = tool.summarize_patch_offset_set_minimization(
        [ok_summary, failed_summary],
        [128, 256],
        {
            "limit": 45,
            "kernel_include": "MT128x64",
            "diagnostic_profile": "previous-bb-set",
            "source_slot_policy": "previous-bb-hash",
            "source_slot_policy_limits": compact_policy_limits,
            "source_report": "/tmp/report.jsonl",
            "source_report_edge_sites_patched": 3,
            "source_report_selected_edge_samples": 4,
            "source_report_selected_edge_samples_truncated": False,
        },
    )
    check(
        offset_set_summary["event"]
        == "opaque_fresh_patch_offset_set_minimization_summary",
        "offset-set minimization summary event name",
    )
    check(offset_set_summary["minimized_offsets"] == [128, 256],
          "offset-set minimization should retain minimized set")
    check(offset_set_summary["minimized_offset_set"]["count"] == 2,
          "offset-set minimization should summarize minimized count")
    check(offset_set_summary["source_slot_policy"] == "previous-bb-hash",
          "offset-set minimization should retain source slot policy")
    check(offset_set_summary["diagnostic_profile"] == "previous-bb-set",
          "offset-set minimization should retain diagnostic profile")
    check(offset_set_summary["source_slot_policy_limits"] == compact_policy_limits,
          "offset-set minimization should retain source slot policy limits")
    check(offset_set_summary["last_failure"]["patch_text_offset_filter"] == 0x40,
          "offset-set minimization should retain last failing run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
