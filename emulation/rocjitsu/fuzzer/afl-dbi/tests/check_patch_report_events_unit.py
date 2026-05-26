#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_check(tool_path, report_path, *args):
    return subprocess.run(
        [sys.executable, str(tool_path), str(report_path), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    if len(sys.argv) != 2:
        print("usage: check_patch_report_events_unit.py <check_patch_report_events.py>",
              file=sys.stderr)
        return 2

    tool_path = Path(sys.argv[1])
    fixture = {
        "event": "patch_device_elf",
        "probe_required_private_segment_bytes": 32,
        "spill_bytes": 16,
        "vgpr_scratch_spill_probe_points": 2,
        "sgpr_scratch_spill_probe_points": 1,
        "kernel_summaries": [
            {
                "kernel": "kernel_a",
                "self_contained_probe": False,
                "coverage_strategy": "entry-previous-bb-block",
                "fixed_edge_sites": 0,
                "vgpr_scratch_spill_probe_points": 0,
                "sgpr_scratch_spill_probe_points": 0,
            },
            {
                "kernel": "kernel_b",
                "self_contained_probe": True,
                "coverage_strategy": "self-contained-fixed-branch",
                "fixed_edge_sites": 1,
                "vgpr_scratch_spill_probe_points": 2,
                "sgpr_scratch_spill_probe_points": 1,
                "sampled_skips": [
                    {
                        "reason": "guarded",
                        "mnemonic": "s_cbranch_scc1",
                        "text_offset": 64,
                    },
                    {
                        "reason": "guarded",
                        "mnemonic": "s_cbranch_scc1",
                        "text_offset": 128,
                    },
                ],
            },
        ],
        "descriptor_resources": [
            {
                "kernel": "kernel_b",
                "wave32": True,
                "old_private_segment_enabled": False,
                "patched_private_segment_enabled": True,
                "old_private_segment_fixed_size": 16,
                "patched_private_segment_fixed_size": 32,
                "spill_bytes": 16,
                "sgpr_count_metadata_patch": "in-place-bytes",
                "private_segment_metadata_patch": "in-place-bytes",
                "resource_fields_changed": True,
            }
        ],
        "sampled_selected_edges": [
            {
                "kernel": "kernel_b",
                "kind": "cond-branch",
                "patch_text_offset": 128,
                "slot_policy": "fixed-counter",
                "fixed_slot": 17,
                "self_contained_probe": True,
                "scratch_spill": True,
                "vgpr_scratch_spill": True,
                "sgpr_scratch_spill": False,
                "scratch_address_exec_source": "exec",
                "state_sgpr": 100,
                "saved_exec_sgpr": 100,
                "workitem_vgpr": 120,
                "scratch_address_vgpr": 4,
                "scratch_spilled_vgprs": [5, 6],
                "scratch_spilled_sgprs": [],
                "placement": "appended-cave",
            }
        ],
        "skipped_kernels": [
            {
                "kernel": "kernel_c",
                "reason": "prior-invalid-dispatch-kernel",
                "entry_probe_safe": False,
                "branch_probe_safe": False,
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "report.jsonl"
        report_path.write_text(json.dumps(fixture) + "\n", encoding="utf-8")

        positive = run_check(
            tool_path,
            report_path,
            "--require-list-item",
            "patch_device_elf",
            "kernel_summaries",
            "kernel=kernel_b",
            "self_contained_probe=true",
            "coverage_strategy=self-contained-fixed-branch",
            "fixed_edge_sites=1",
            "vgpr_scratch_spill_probe_points=2",
            "sgpr_scratch_spill_probe_points=1",
            "--require-field",
            "patch_device_elf",
            "vgpr_scratch_spill_probe_points",
            "2",
            "--require-field",
            "patch_device_elf",
            "sgpr_scratch_spill_probe_points",
            "1",
            "--require-list-item",
            "patch_device_elf",
            "descriptor_resources",
            "kernel=kernel_b",
            "wave32=true",
            "patched_private_segment_enabled=true",
            "patched_private_segment_fixed_size=32",
            "spill_bytes=16",
            "sgpr_count_metadata_patch=in-place-bytes",
            "private_segment_metadata_patch=in-place-bytes",
            "resource_fields_changed=true",
            "--require-list-item",
            "patch_device_elf",
            "sampled_selected_edges",
            "kernel=kernel_b",
            "kind=cond-branch",
            "slot_policy=fixed-counter",
            "scratch_spill=true",
            "vgpr_scratch_spill=true",
            "scratch_address_exec_source=exec",
            "state_sgpr=100",
            "saved_exec_sgpr=100",
            "scratch_address_vgpr=4",
            "placement=appended-cave",
            "--require-list-item",
            "patch_device_elf",
            "skipped_kernels",
            "kernel=kernel_c",
            "reason=prior-invalid-dispatch-kernel",
            "entry_probe_safe=false",
            "--require-list-item-count",
            "patch_device_elf",
            "kernel_summaries.sampled_skips",
            "2",
            "reason=guarded",
            "mnemonic=s_cbranch_scc1",
        )
        check(positive.returncode == 0, positive.stderr)

        negative = run_check(
            tool_path,
            report_path,
            "--require-list-item",
            "patch_device_elf",
            "kernel_summaries",
            "kernel=kernel_b",
            "self_contained_probe=false",
        )
        check(negative.returncode == 1, "negative same-item check should fail")
        check(
            "missing list item match" in negative.stderr,
            "negative check should explain the missing same-item match",
        )

        invalid = run_check(
            tool_path,
            report_path,
            "--require-list-item",
            "patch_device_elf",
            "kernel_summaries",
            "kernel_b",
        )
        check(invalid.returncode == 2, "invalid same-item syntax should be rejected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
