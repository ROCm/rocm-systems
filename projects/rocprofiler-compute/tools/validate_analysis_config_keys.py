#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook: verify analysis config header/metric key consistency.

For every metric_table in each analysis config YAML, checks that metric
entry keys match the header keys.  A mismatch means the runtime parser
(src/utils/parser.py  _build_metric_table_df) will either silently drop
values or crash with a column-count error.

Checks:
  1. (ERROR) A metric entry is missing a key declared in the header
     → will crash at runtime (row shorter than column list).
  2. (WARNING) A metric entry has an extra key not in the header
     → silently dropped at runtime, likely a typo.
"""

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "src" / "rocprof_compute_soc" / "analysis_configs"

METRIC_METADATA_KEYS = {"coll_level", "alias"}
# Memory chart (panel 03xx) metrics carry a "unit" key consumed by the
# chart renderer, not by the DataFrame parser.  Only exempt it there so
# the same typo in any other panel is still flagged.
MEMORY_CHART_PANEL_PREFIX = 3
MEMORY_CHART_EXTRA_KEYS = {"unit"}


def validate() -> tuple[list[str], list[str]]:
    """Run all checks, return (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    for arch_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not arch_dir.is_dir() or not arch_dir.name.startswith("gfx"):
            continue
        arch = arch_dir.name

        for config_path in sorted(arch_dir.glob("*.yaml")):
            data = yaml.safe_load(config_path.read_text())
            panel = (data or {}).get("Panel Config", {})

            for source in panel.get("data source", []):
                mt = source.get("metric_table", {})
                table_id = mt.get("id")
                header = mt.get("header")
                metrics = mt.get("metric")
                if table_id is None or header is None or not metrics:
                    continue

                title = mt.get("title", "")
                is_simple_box = mt.get("cli_style") == "simple_box"

                header_formula_keys = set(header) - {"metric", "expr"}

                for metric_name, entries in metrics.items():
                    if metric_name == "placeholder_range":
                        continue
                    if not isinstance(entries, dict):
                        continue

                    metric_keys = set(entries) - METRIC_METADATA_KEYS
                    if is_simple_box:
                        metric_keys -= {"expr"}

                    missing = header_formula_keys - metric_keys
                    extra = metric_keys - header_formula_keys
                    if table_id // 100 == MEMORY_CHART_PANEL_PREFIX:
                        extra -= MEMORY_CHART_EXTRA_KEYS

                    loc = f'[{arch}] {config_path.name} table {table_id} "{title}"'

                    if missing:
                        errors.append(
                            f'{loc}:\n  metric "{metric_name}" is missing '
                            f"header key(s): {sorted(missing)} "
                            f"(header keys: {sorted(header_formula_keys)})"
                        )

                    if extra:
                        warnings.append(
                            f'{loc}:\n  metric "{metric_name}" has extra '
                            f"key(s) not in header: {sorted(extra)} "
                            f"(header keys: {sorted(header_formula_keys)})"
                        )

    return errors, warnings


def main() -> int:
    errors, warnings = validate()

    if warnings:
        print("Analysis config key warnings:\n")
        for w in warnings:
            print(f"  WARNING: {w}")
        print()

    if errors:
        print("Analysis config key validation FAILED:\n")
        for e in errors:
            print(f"  ERROR: {e}")
        print(f"\n{len(errors)} error(s), {len(warnings)} warning(s) found.")
        return 1

    if warnings:
        print(f"{len(warnings)} warning(s), 0 errors. Passed.")
    else:
        print("Analysis config key validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
