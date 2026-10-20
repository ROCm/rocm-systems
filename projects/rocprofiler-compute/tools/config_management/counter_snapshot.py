#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Check the snapshot of the counters profile mode collects.

The snapshot lists, per analysis config arch, the counters of each panel file
and, under "always", the normalization-unit counters every profile collects.
Profiling does not read it; it makes changes to the profiled counters visible
in review.

Usage:
    python counter_snapshot.py           # report changes, exit 1 if any
    python counter_snapshot.py --update  # rewrite the snapshot
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # rocprofiler-compute/
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config
from rocprof_compute_soc.soc_base import iter_analysis_config_metrics
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_common import canonical_config_arch
from utils.utils_counter_defs import extract_counters_and_variables

SNAPSHOT_PATH = Path(__file__).resolve().parent / ".profiled_counters.json"


def panel_counters(
    expressions_by_panel: dict[str, list[str]], gpu_series: str
) -> dict[str, list[str]]:
    """Return the sorted counters of each panel that uses any, plus "always".

    TCC channel counters keep their template form (e.g. "TCC_HIT["), because
    the channel count is only known on the GPU.
    """
    always, _ = extract_counters_and_variables("", gpu_series)
    counters_by_panel = {"always": sorted(always)}
    for panel, expressions in sorted(expressions_by_panel.items()):
        counters, _ = extract_counters_and_variables(
            "\n".join(expressions), gpu_series, include_supported_denom=False
        )
        if counters:
            counters_by_panel[panel] = sorted(counters)
    return counters_by_panel


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


def compute_snapshot() -> dict[str, dict[str, list[str]]]:
    """Read every arch's analysis configs and return its panel_counters."""
    series_by_arch = {
        canonical_config_arch(arch): mi_gpu_specs.get_gpu_series(arch)
        for arch in mi_gpu_specs.get_gpu_series_dict()
    }
    snapshot: dict[str, dict[str, list[str]]] = {}
    for arch_dir in sorted(config.analysis_configs_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        if arch_dir.name not in series_by_arch:
            raise KeyError(f"{arch_dir.name} is missing from mi_gpu_spec.yaml")
        expressions_by_panel: dict[str, list[str]] = defaultdict(list)
        for panel, *_, expressions in iter_analysis_config_metrics(arch_dir, set()):
            expressions_by_panel[panel].extend(expressions)
        snapshot[arch_dir.name] = panel_counters(
            expressions_by_panel, series_by_arch[arch_dir.name]
        )
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the profiled counters.")
    parser.add_argument("--update", action="store_true", help="rewrite the snapshot")
    args = parser.parse_args()

    current = compute_snapshot()
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
