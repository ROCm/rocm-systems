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
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETS_DIR = PROJECT_ROOT / "src" / "rocprof_compute_soc" / "profile_configs" / "sets"
ANALYSIS_DIR = PROJECT_ROOT / "src" / "rocprof_compute_soc" / "analysis_configs"
GPU_SPEC_PATH = PROJECT_ROOT / "src" / "utils" / "mi_gpu_spec.yaml"

# Make src/ importable so we can reuse the canonical counter definitions.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils.utils_counter_defs import (  # noqa: E402
    counter_to_block,
    extract_counters,
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


def load_analysis_configs(arch: str) -> dict[int, dict[str, dict]]:
    """Return {metric_table_id: {metric_name: formula_dict}} for *arch*.

    Metric name ordering is preserved (dict insertion order), so
    ``list(result[table_id])`` gives the ordered name list needed for
    index-based lookups.
    """
    arch_dir = ANALYSIS_DIR / arch
    if not arch_dir.is_dir():
        return {}
    result: dict[int, dict[str, dict]] = {}
    for config_path in sorted(arch_dir.glob("*.yaml")):
        data = yaml.safe_load(config_path.read_text())
        panel = (data or {}).get("Panel Config", {})
        for source in panel.get("data source", []):
            mt = source.get("metric_table", {})
            table_id = mt.get("id")
            if table_id is None:
                continue
            metrics = mt.get("metric", {})
            if metrics:
                result[table_id] = dict(metrics)
    return result


def load_perfmon_configs() -> dict[str, dict[str, int]]:
    """Return {gpu_arch: {block: max_counters}} from mi_gpu_spec.yaml."""
    data = yaml.safe_load(GPU_SPEC_PATH.read_text())
    result: dict[str, dict[str, int]] = {}
    for series in data.get("mi_gpu_spec", []):
        for arch_entry in series.get("gpu_archs", []):
            arch = arch_entry.get("gpu_arch")
            perfmon = arch_entry.get("perfmon_config", {})
            if arch and perfmon:
                # Filter out non-block entries like TCC_channels
                result[arch] = {
                    k: v
                    for k, v in perfmon.items()
                    if isinstance(v, int) and "_" not in k
                }
    return result


def _flatten_formula_values(d: dict) -> str:
    """Recursively collect all string values from a metric formula dict."""
    parts: list[str] = []
    for v in d.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            parts.append(_flatten_formula_values(v))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Validation (single pass over all sets files)
# ---------------------------------------------------------------------------


def validate() -> list[str]:
    """Run all checks, return error messages."""
    errors: list[str] = []
    perfmon_configs = load_perfmon_configs()

    for sets_path in sorted(SETS_DIR.glob("gfx*_sets.yaml")):
        arch = sets_path.stem.replace("_sets", "")
        sets_data = yaml.safe_load(sets_path.read_text())

        # Load analysis configs once per arch
        analysis = load_analysis_configs(arch)
        limits = perfmon_configs.get(arch)

        for s in sets_data.get("sets", []):
            set_option = s.get("set_option", "<unknown>")
            formula_texts: list[str] = []

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

                    table_metrics = analysis[table_id]
                    metric_names = list(table_metrics)
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

                    # Collect formula text for single-pass check
                    formula = table_metrics.get(actual)
                    if formula is not None:
                        formula_texts.append(
                            _flatten_formula_values(formula)
                            if isinstance(formula, dict)
                            else str(formula)
                        )

            # --- Check 2: counters fit in single pass ---
            if not formula_texts or limits is None:
                continue

            counters = extract_counters("\n".join(formula_texts))
            # *_ACCUM is the per-bucket alias for SQ_ACCUM_PREV_HIRES, which is
            # injected automatically by the profiler for level counters
            counters = {c for c in counters if not c.endswith("_ACCUM")}
            block_counters: dict[str, set[str]] = defaultdict(set)
            for c in counters:
                block_counters[counter_to_block(c)].add(c)

            for block, block_ctrs in sorted(block_counters.items()):
                if block not in limits:
                    continue
                if len(block_ctrs) > limits[block]:
                    errors.append(
                        f"[{arch}] set '{set_option}': block {block} needs "
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
