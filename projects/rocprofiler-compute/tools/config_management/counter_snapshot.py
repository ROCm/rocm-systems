#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Check the snapshot of the counters profile mode collects.

For each analysis config arch, the snapshot lists what detect_counters()
returns for each panel file (-b <panel>), and under "always" the
normalization-unit counters every profile collects. The check also confirms
that the panels add up to the default profile and to --membw-analysis.
Profiling does not read the snapshot; it makes changes to the profiled
counters visible in review.

Usage:
    python counter_snapshot.py           # report changes, exit 1 if any
    python counter_snapshot.py --update  # rewrite the snapshot
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # rocprofiler-compute/
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config
from rocprof_compute_soc.soc_base import offline_soc
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_common import MEMBW_ANALYSIS_PANEL_ID, canonical_config_arch
from utils.utils_counter_defs import extract_counters_and_variables

SNAPSHOT_PATH = Path(__file__).resolve().parent / ".profiled_counters.json"


def detected_counters(
    gpu_arch: str, filter_blocks: list[str], membw_analysis: bool
) -> set[str]:
    """detect_counters() on the shipped configs, with TCC channel counters in
    their template form (e.g. "TCC_HIT["), because the channel count is only
    known on the GPU."""
    soc = offline_soc(
        config.analysis_configs_dir, gpu_arch, filter_blocks, membw_analysis
    )
    counters, _ = soc.detect_counters()
    return {re.sub(r"\[0\]$", "[", counter) for counter in counters}


def arch_snapshot(
    config_arch_dir: Path, gpu_arch: str
) -> tuple[dict[str, list[str]], list[str]]:
    """Return the snapshot entry of one config arch, and where its panels do
    not add up to the default or --membw-analysis profile."""
    always, _ = extract_counters_and_variables(
        "", mi_gpu_specs.get_gpu_series(gpu_arch)
    )
    entry = {"always": sorted(always)}
    for config_file in sorted(config_arch_dir.glob("*.yaml")):
        panel = config_file.name.split("_")[0]
        counters = detected_counters(gpu_arch, [str(int(panel) // 100)], True)
        if counters - always:
            entry[panel] = sorted(counters - always)

    mismatches = []
    membw_panel = str(MEMBW_ANALYSIS_PANEL_ID)
    for membw_analysis in (False, True):
        expected = set().union(
            *(c for panel, c in entry.items() if membw_analysis or panel != membw_panel)
        )
        detected = detected_counters(gpu_arch, [], membw_analysis)
        if detected != expected:
            mismatches.append(
                f"{config_arch_dir.name} membw_analysis={membw_analysis}: "
                f"profile only {sorted(detected - expected)}, "
                f"panels only {sorted(expected - detected)}"
            )
    return entry, mismatches


def diff_snapshots(
    recorded: dict[str, dict[str, list[str]]],
    current: dict[str, dict[str, list[str]]],
) -> list[str]:
    """Describe every counter added or removed, per arch and panel."""
    changes: list[str] = []
    for arch in sorted(recorded.keys() | current.keys()):
        old_panels = recorded.get(arch, {})
        new_panels = current.get(arch, {})
        for panel in sorted(old_panels.keys() | new_panels.keys()):
            old = set(old_panels.get(panel, []))
            new = set(new_panels.get(panel, []))
            if new - old:
                changes.append(f"{arch} {panel}: added {', '.join(sorted(new - old))}")
            if old - new:
                changes.append(
                    f"{arch} {panel}: removed {', '.join(sorted(old - new))}"
                )
    return changes


def compute_snapshot() -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    """Return the snapshot of every config arch, and all mismatches."""
    arch_by_config_arch = {
        canonical_config_arch(arch): arch for arch in mi_gpu_specs.get_gpu_series_dict()
    }
    snapshot: dict[str, dict[str, list[str]]] = {}
    mismatches: list[str] = []
    for config_arch_dir in sorted(config.analysis_configs_dir.iterdir()):
        if not config_arch_dir.is_dir():
            continue
        if config_arch_dir.name not in arch_by_config_arch:
            raise KeyError(f"{config_arch_dir.name} is missing from mi_gpu_spec.yaml")
        entry, arch_mismatches = arch_snapshot(
            config_arch_dir, arch_by_config_arch[config_arch_dir.name]
        )
        snapshot[config_arch_dir.name] = entry
        mismatches.extend(arch_mismatches)
    return snapshot, mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the profiled counters.")
    parser.add_argument("--update", action="store_true", help="rewrite the snapshot")
    args = parser.parse_args()

    current, mismatches = compute_snapshot()
    if mismatches:
        print("\nPANELS DO NOT ADD UP TO THE PROFILE:")
        for mismatch in mismatches:
            print("  - " + mismatch)
        return 1
    if args.update:
        SNAPSHOT_PATH.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote {SNAPSHOT_PATH.relative_to(PROJECT_ROOT)}.")
        return 0

    recorded = (
        json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        if SNAPSHOT_PATH.is_file()
        else {}
    )
    changes = diff_snapshots(recorded, current)
    if not changes:
        print("Profiled counter snapshot check passed.")
        return 0
    print("\nPROFILED COUNTERS CHANGED:")
    for change in changes:
        print("  - " + change)
    print("Run tools/config_management/counter_snapshot.py --update and commit it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
