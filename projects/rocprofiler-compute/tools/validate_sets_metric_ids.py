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

``print`` is the CLI-output channel for a pre-commit hook; ``T201`` is
disabled at the module level rather than per-call.
"""

# ruff: noqa: T201

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

import yaml

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SETS_DIR: Final[Path] = (
    PROJECT_ROOT / "src" / "rocprof_compute_soc" / "profile_configs" / "sets"
)
ANALYSIS_DIR: Final[Path] = (
    PROJECT_ROOT / "src" / "rocprof_compute_soc" / "analysis_configs"
)
GPU_SPEC_PATH: Final[Path] = PROJECT_ROOT / "src" / "utils" / "mi_gpu_spec.yaml"

# Make src/ importable so we can reuse the canonical counter definitions.
# src/ has no __init__.py (and making it a proper package is out of scope
# for this validator), so we extend sys.path the way tests/conftest.py does.
_SRC: Final[str] = str(PROJECT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# pylint: disable=wrong-import-position  # sys.path.insert must run first.
from utils.utils_counter_defs import (  # noqa: E402
    counter_to_block,
    extract_counters,
    load_perfmon_configs,
)

# Metadata keys that appear in perfmon_config alongside real block caps
# (e.g. TCC_channels) and must not be treated as per-block limits. Explicit
# allow-list — if mi_gpu_spec.yaml grows new metadata it must be added here.
_METADATA_KEYS: Final[frozenset[str]] = frozenset({"TCC_channels"})

# A metric formula in analysis_configs/*.yaml is a recursive tree of
# strings (leaf formulas) nested under dicts. We don't expect lists in
# practice, but typing them here keeps the boundary honest.
FormulaTree: TypeAlias = "str | dict[str, FormulaTree] | list[FormulaTree]"
TableMetrics: TypeAlias = "dict[str, FormulaTree]"
ArchAnalysis: TypeAlias = "dict[int, TableMetrics]"

# Metric IDs must have exactly three dot-separated segments (``X.Y.Z``).
_METRIC_ID_SEGMENTS: Final[int] = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_metric_id(metric_id: str) -> tuple[int | None, int | None]:
    """Parse 'X.Y.Z' into (metric_table_id, metric_index).

    Returns ``(None, None)`` if the id is not fully qualified (fewer than
    three segments) *or* if any segment is not numeric — the caller reports
    a single "not fully qualified" error in either case.
    """
    tokens = metric_id.split(".")
    if len(tokens) < _METRIC_ID_SEGMENTS:
        return None, None
    try:
        x, y, z = int(tokens[0]), int(tokens[1]), int(tokens[2])
    except ValueError:
        return None, None
    return x * 100 + y, z


def load_analysis_configs(arch: str) -> ArchAnalysis:  # noqa: C901
    """Return {metric_table_id: {metric_name: formula_tree}} for *arch*.

    Metric name ordering is preserved (dict insertion order), so
    ``list(result[table_id])`` gives the ordered name list needed for
    index-based lookups. Each YAML layer is guarded with ``isinstance`` so
    that a malformed file cannot inject unexpected types past this boundary
    — hence the complexity ruff flags is load-bearing (one ``isinstance``
    per YAML level) and is accepted.
    """
    arch_dir = ANALYSIS_DIR / arch
    if not arch_dir.is_dir():
        return {}
    result: ArchAnalysis = {}
    for config_path in sorted(arch_dir.glob("*.yaml")):
        raw: Any = yaml.safe_load(config_path.read_text())
        if not isinstance(raw, dict):
            continue
        panel_raw = raw.get("Panel Config")
        if not isinstance(panel_raw, dict):
            continue
        sources = panel_raw.get("data source") or []
        if not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, dict):
                continue
            mt = source.get("metric_table")
            if not isinstance(mt, dict):
                continue
            table_id = mt.get("id")
            if not isinstance(table_id, int):
                continue
            metrics = mt.get("metric")
            if not isinstance(metrics, dict):
                continue
            # Every key in metric-tables is a metric-name string in practice;
            # drop any entries that are not so we preserve the TableMetrics
            # contract downstream.
            table: TableMetrics = {
                str(name): cast("FormulaTree", body)
                for name, body in metrics.items()
                if isinstance(name, str)
            }
            if table:
                result[table_id] = table
    return result


def _perfmon_caps() -> dict[str, dict[str, int]]:
    """Return ``{gpu_arch: {block: max_counters}}`` via the shared loader.

    Thin wrapper around :func:`utils.utils_counter_defs.load_perfmon_configs`
    so the validator cannot drift from the runtime spec-reader.
    """
    return load_perfmon_configs(GPU_SPEC_PATH, metadata_keys=_METADATA_KEYS)


def _flatten_formula_values(tree: FormulaTree) -> str:
    """Recursively collect all string leaves from a metric formula tree."""
    if isinstance(tree, str):
        return tree
    if isinstance(tree, dict):
        return "\n".join(_flatten_formula_values(v) for v in tree.values())
    if isinstance(tree, list):
        return "\n".join(_flatten_formula_values(v) for v in tree)
    return ""


# ---------------------------------------------------------------------------
# Validation (single pass over all sets files)
# ---------------------------------------------------------------------------


def validate() -> list[str]:  # noqa: C901, PLR0912  # pylint: disable=too-many-locals,too-many-branches
    """Run all checks, return error messages.

    The two checks (metric-ID -> name mapping and single-pass counter
    budgets) are intentionally fused into one nested loop so each sets
    file is read and parsed once. Splitting the function would require
    re-materializing the per-arch analysis config and formula list, so
    the branch/local counts are tracked-but-accepted.
    """
    errors: list[str] = []
    perfmon_configs = _perfmon_caps()

    for sets_path in sorted(SETS_DIR.glob("gfx*_sets.yaml")):
        arch = sets_path.stem.replace("_sets", "")
        sets_raw: Any = yaml.safe_load(sets_path.read_text())
        if not isinstance(sets_raw, dict):
            continue

        # Load analysis configs once per arch
        analysis = load_analysis_configs(arch)
        limits = perfmon_configs.get(arch)

        sets_list = sets_raw.get("sets") or []
        if not isinstance(sets_list, list):
            continue
        for s in sets_list:
            if not isinstance(s, dict):
                continue
            set_option = str(s.get("set_option", "<unknown>"))
            formula_texts: list[str] = []

            metric_entries = s.get("metric") or []
            if not isinstance(metric_entries, list):
                continue
            for entry in metric_entries:
                if not isinstance(entry, dict):
                    continue
                for metric_id_raw, expected_name_raw in entry.items():
                    metric_id = str(metric_id_raw)
                    expected_name = str(expected_name_raw)
                    table_id, idx = resolve_metric_id(metric_id)

                    # --- Check 1: metric ID maps to correct name ---
                    if table_id is None or idx is None:
                        errors.append(
                            f"[{arch}] set '{set_option}': metric ID "
                            f"'{metric_id}' is not fully qualified (need "
                            f"three numeric segments X.Y.Z)"
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
                        formula_texts.append(_flatten_formula_values(formula))

            # --- Check 2: counters fit in single pass ---
            if not formula_texts or limits is None:
                continue

            # extract_counters already filters SYNTHETIC_COUNTERS
            # (SQ_ACCUM_PREV_HIRES); no manual .discard() needed.
            counters = extract_counters("\n".join(formula_texts))
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
    """CLI entry point; prints each error and returns a non-zero exit code."""
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
