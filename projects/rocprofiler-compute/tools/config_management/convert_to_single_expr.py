#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Convert panel YAML files from per-stat expressions to single expr: format.

For multi-stat metrics (avg/min/max): extract the core expression from
``avg:`` by stripping the outer ``AVG(...)`` wrapper, write as ``expr:``,
and remove ``avg:``, ``min:``, ``max:`` keys.

For single-stat metrics (value): extract from ``value:`` by stripping
``AVG(...)`` if it is the outermost call, write as ``expr:``, and
remove ``value:`` key.

Removes ``pop:`` entries entirely (computed in Python at eval time).
Preserves ``peak:``, ``unit:``, ``type:``, ``transaction:``,
``coll_level:``, ``alias:``, and all other non-stat keys.

Usage:
    python convert_to_single_expr.py <arch_dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_management import utils_ruamel as cm_utils  # noqa: E402

STAT_WRAPPERS = {"AVG", "MIN", "MAX", "MEDIAN", "STD"}

STAT_KEYS = {"avg", "min", "max", "value", "pop"}

PRESERVE_KEYS = {
    "unit",
    "peak",
    "type",
    "transaction",
    "coll_level",
    "alias",
    "style",
}


def strip_outer_stat_call(expression: str) -> str:
    """Strip the outermost stat wrapper if present.

    Returns the inner expression when the outermost function call
    is one of AVG/MIN/MAX/MEDIAN/STD.  If the expression starts
    with a different construct, returns it unchanged.
    """
    stripped = str(expression).strip()
    for func_name in STAT_WRAPPERS:
        prefix = f"{func_name}("
        if stripped.startswith(prefix):
            inner = _extract_balanced_arg(stripped, len(func_name))
            if inner is not None:
                return inner.strip()
    return stripped


def _extract_balanced_arg(
    text: str, func_name_len: int
) -> Optional[str]:
    """Return the content between balanced parens after *func_name*.

    Given ``AVG((a + b))``, with *func_name_len* = 3, this returns
    ``(a + b)``.  Returns None if the closing paren does not match
    the outermost open paren at position *func_name_len*.
    """
    open_pos = func_name_len
    if open_pos >= len(text) or text[open_pos] != "(":
        return None

    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                trailing = text[i + 1 :].strip()
                if trailing:
                    return None
                return text[open_pos + 1 : i]
    return None


def convert_metric(
    metric_name: str,
    fields: Any,
    header_keys: set[str],
) -> tuple[bool, dict[str, Any]]:
    """Convert a single metric entry to expr: format.

    Returns (changed, new_fields).
    """
    if not isinstance(fields, dict):
        return False, fields

    if "expr" in fields:
        return False, fields

    new_fields: dict[str, Any] = {}
    changed = False

    has_avg_key = "avg" in fields
    has_value_key = "value" in fields

    if has_avg_key:
        raw_expr = strip_outer_stat_call(str(fields["avg"]))
        new_fields["expr"] = raw_expr
        changed = True
    elif has_value_key:
        value_str = str(fields["value"])
        raw_expr = strip_outer_stat_call(value_str)
        new_fields["expr"] = raw_expr
        changed = True

    for key, value in fields.items():
        if key in STAT_KEYS:
            continue
        if key in PRESERVE_KEYS:
            new_fields[key] = value

    return changed, new_fields


def convert_metric_table(
    table: dict[str, Any],
) -> int:
    """Convert all metrics in a metric_table in-place.

    Returns the number of metrics converted.
    """
    metrics = table.get("metric")
    if not metrics or not isinstance(metrics, dict):
        return 0

    header = table.get("header", {})
    header_keys = set(header.keys())

    converted_count = 0
    replacements: list[tuple[str, dict]] = []

    for metric_name, fields in metrics.items():
        changed, new_fields = convert_metric(
            metric_name, fields, header_keys
        )
        if changed:
            replacements.append((metric_name, new_fields))
            converted_count += 1

    for metric_name, new_fields in replacements:
        metrics[metric_name] = new_fields

    return converted_count


def convert_file(
    filepath: Path, *, dry_run: bool = False
) -> int:
    """Convert a single YAML file.  Returns number of metrics converted."""
    data = cm_utils.load_yaml(filepath, round_trip=True)

    panel_config = data.get("Panel Config")
    if not panel_config:
        return 0

    data_sources = panel_config.get("data source", [])
    if not isinstance(data_sources, list):
        return 0

    total_converted = 0
    for ds_item in data_sources:
        if not isinstance(ds_item, dict):
            continue
        table = ds_item.get("metric_table")
        if table and isinstance(table, dict):
            total_converted += convert_metric_table(table)

    if total_converted > 0 and not dry_run:
        cm_utils.strip_existing_header(data)
        cm_utils.save_yaml(data, filepath)

    return total_converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert panel YAMLs from per-stat expressions "
            "to single expr: format."
        )
    )
    parser.add_argument(
        "arch_dir",
        help="Architecture config directory to convert",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report conversions without writing files",
    )
    args = parser.parse_args()

    arch_path = Path(args.arch_dir)
    if not arch_path.is_dir():
        print(f"Error: {arch_path} is not a directory")
        sys.exit(1)

    total_files = 0
    total_metrics = 0

    for yaml_file in sorted(arch_path.glob("*.yaml")):
        converted = convert_file(
            yaml_file, dry_run=args.dry_run
        )
        if converted > 0:
            action = "Would convert" if args.dry_run else "Converted"
            print(
                f"{action} {converted} metric(s) in "
                f"{yaml_file.name}"
            )
            total_files += 1
            total_metrics += converted

    print(
        f"\nTotal: {total_metrics} metric(s) in "
        f"{total_files} file(s)"
    )


if __name__ == "__main__":
    main()
