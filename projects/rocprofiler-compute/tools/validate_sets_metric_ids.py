#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook: verify gfx*_sets.yaml files.

Checks:
  1. Every metric ID maps to the correct metric name in the analysis config.
  2. Each set's counters fit within a single profiling pass (per-block limits
     from perfmon_config in mi_gpu_spec.yaml).

Metric ID format X.Y.Z:
  X = panel_config_id // 100
  Y = metric_table_id % 100  (metric_table_id = X*100 + Y)
  Z = 0-indexed position of the metric within that table's ordered metric dict
"""

import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETS_DIR = PROJECT_ROOT / "src" / "rocprof_compute_soc" / "profile_configs" / "sets"
ANALYSIS_DIR = PROJECT_ROOT / "src" / "rocprof_compute_soc" / "analysis_configs"

# Make src/ importable so we can reuse the canonical counter definitions.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_common import canonical_config_arch
from utils.utils_counter_defs import (
    counters_over_block_limits,
    extract_counters_and_variables,
    metric_expressions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_metric_id(metric_id: str) -> tuple[int | None, int | None]:
    """Parse 'X.Y.Z' into (metric_table_id, metric_index)."""
    tokens = metric_id.split(".")
    if len(tokens) < 3:
        return None, None
    try:
        x, y, z = int(tokens[0]), int(tokens[1]), int(tokens[2])
    except ValueError:
        return None, None
    return x * 100 + y, z


def load_analysis_configs(arch: str) -> dict[int, dict[str, Any]]:
    """Return {metric_table_id: metric_table} for *arch*.

    Metric names keep their YAML order, so list(result[table_id]["metric"])
    gives the names for index-based lookups.
    """
    arch_dir = ANALYSIS_DIR / (canonical_config_arch(arch) or arch)
    if not arch_dir.is_dir():
        return {}
    result: dict[int, dict[str, Any]] = {}
    for config_path in sorted(arch_dir.glob("*.yaml")):
        data = yaml.safe_load(config_path.read_text())
        panel = (data or {}).get("Panel Config", {})
        for source in panel.get("data source", []):
            metric_table = source.get("metric_table", {})
            table_id = metric_table.get("id")
            if table_id is None:
                continue
            if metric_table.get("metric"):
                result[table_id] = metric_table
    return result


# ---------------------------------------------------------------------------
# Validation (single pass over all sets files)
# ---------------------------------------------------------------------------


def validate() -> list[str]:
    """Run all checks, return error messages."""
    errors: list[str] = []
    shared_arch_expansions = {"gfx115x": ["gfx1150", "gfx1151", "gfx1152", "gfx1153"]}

    for sets_path in sorted(SETS_DIR.glob("gfx*_sets.yaml")):
        arch = sets_path.stem.replace("_sets", "")
        sets_data = yaml.safe_load(sets_path.read_text())
        analysis = load_analysis_configs(arch)
        target_arches = shared_arch_expansions.get(arch, [arch])

        for s in sets_data.get("sets", []):
            set_option = s.get("set_option", "<unknown>")
            expressions: list[str] = []

            for entry in s.get("metric", []):
                for metric_id_raw, expected_name in entry.items():
                    metric_id = str(metric_id_raw)
                    expected_name = str(expected_name)
                    table_id, idx = resolve_metric_id(metric_id)

                    # --- Check 1: metric ID maps to correct name ---
                    if table_id is None or idx is None:
                        errors.append(
                            f"[{arch}] set '{set_option}': metric ID "
                            f"'{metric_id}' is not fully qualified (need X.Y.Z)"
                        )
                        continue

                    if table_id not in analysis:
                        errors.append(
                            f"[{arch}] set '{set_option}': metric_table "
                            f"{table_id} not found in analysis configs "
                            f"(metric ID {metric_id})"
                        )
                        continue

                    table = analysis[table_id]
                    metric_names = list(table["metric"])
                    if idx >= len(metric_names):
                        errors.append(
                            f"[{arch}] set '{set_option}': index {idx} out of "
                            f"range for table {table_id} which has "
                            f"{len(metric_names)} metrics (metric ID {metric_id}). "
                            f"Metrics: {metric_names}"
                        )
                        continue

                    actual = metric_names[idx]
                    # Allow the sets file to use a qualified name that
                    # includes context from the panel/table title
                    # (e.g. "vL1D Cache Utilization" matches "Utilization"
                    # when the table is under the vL1D Cache panel).
                    if actual != expected_name and not expected_name.endswith(
                        " " + actual
                    ):
                        errors.append(
                            f"[{arch}] set '{set_option}': metric ID "
                            f"{metric_id} is labeled '{expected_name}' but "
                            f"analysis config (table {table_id}) has "
                            f"'{actual}' at index {idx}. "
                            f"Full table: {metric_names}"
                        )

                    # Expressions for the single-pass check
                    expressions.extend(
                        metric_expressions(table, table["metric"][actual])
                    )

            if not expressions:
                continue

            for target_arch in target_arches:
                gpu_series = mi_gpu_specs.get_gpu_series(target_arch)
                if not gpu_series:
                    errors.append(
                        f"[{target_arch}] arch missing from mi_gpu_spec.yaml; "
                        f"add it under mi_gpu_spec.<series>.gpu_archs"
                    )
                    continue

                limits = mi_gpu_specs.get_perfmon_config(target_arch)
                counters, _ = extract_counters_and_variables(
                    "\n".join(expressions), gpu_series
                )
                over_limit = counters_over_block_limits(counters, limits)
                for block, block_ctrs in over_limit.items():
                    errors.append(
                        f"[{target_arch}] set '{set_option}': block {block} needs "
                        f"{len(block_ctrs)} counters but limit is "
                        f"{limits[block]}. "
                        f"Counters: {sorted(block_ctrs)}"
                    )

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    errors = validate()
    if errors:
        print("Sets metric ID validation failed:\n")
        for e in errors:
            print(f"  ERROR: {e}")
        print(f"\n{len(errors)} error(s) found.")
        return 1
    print("Sets metric ID validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
