#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook: validate the analysis views.

For every arch of the metric library, check that:
  1. its view files have known fields, render types and CLI styles, and the views tree
     holds only metric library arch directories;
  2. every metric a view shows is in the arch's metric library, with a measured peak
     where the row asks for one;
  3. the views keep the analysis_configs panels' structure: the same panels, tables,
     headers and table settings, in order, and the same rows in order, leaving out only
     rows that show no equation (and tables left empty by them); each panel's -b alias
     is the config template's; every row's legacy position is its row in the panel
     file; and every metric the arch shows has one legacy_id, that position. Values,
     units, peaks and descriptions come from the metric library and are not compared.
     An arch without a roofline benchmark (gfx908) has no roofline panel to compare.
Check 3 is temporary: it goes with the panel files (AIPROFCOMP-821 PR 12).
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config
from utils.analysis_views import (
    METRIC_TABLE,
    NO_VALUE,
    default_views_dir,
    iter_view_rows,
    legacy_position,
    panel_aliases,
    read_view_files,
    row_label,
    row_metric_ids,
    view_metric_errors,
    view_schema_errors,
    views_to_panel_configs,
)
from utils.metric_library import (
    METRIC_LIBRARY_PARENT_ARCH,
    default_metric_library_dir,
    load_metric_library,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_common import (
    ROOFLINE_PANEL_ID,
    canonical_config_arch,
    get_arch_panel_id_to_alias,
    load_panel_configs,
)

# Table settings that must match the panel's; rows are compared by name.
TABLE_KEYS = ("id", "title", "header", "cli_style", "source", "columnwise")
# Row columns that show no equation.
NON_EQUATION_COLUMNS = ("unit", "peak", "pct_of_peak")


def validate(
    views_dir: Path, metric_dir: Path, analysis_dir: Optional[Path] = None
) -> List[str]:
    """Validate every arch's views. Return error messages.

    With analysis_dir, also check each arch's views against its analysis configs.
    """
    errors = [
        f"[{path.relative_to(views_dir)}] is not a metric library arch directory"
        for path in sorted(Path(views_dir).iterdir())
        if path.name not in METRIC_LIBRARY_PARENT_ARCH
    ]
    for arch in METRIC_LIBRARY_PARENT_ARCH:
        documents = read_view_files(arch, views_dir)
        schema_errors = view_schema_errors({
            str(path.relative_to(views_dir)): document
            for path, document in documents.items()
        })
        if schema_errors:
            errors.extend(schema_errors)
            continue
        views = list(documents.values())
        metrics = load_metric_library(metric_dir, arch).metrics
        arch_errors = view_metric_errors(views, metrics)
        if not arch_errors and analysis_dir is not None:
            arch_errors = structure_errors(views, metrics, analysis_dir, arch)
        errors.extend(f"[{arch}] {error}" for error in arch_errors)
    return errors


def structure_errors(
    views: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
    analysis_dir: Path,
    arch: str,
) -> List[str]:
    """Check one arch's views against its analysis configs' structure. Temporary."""
    expected = load_panel_configs([str(analysis_dir / arch)])
    positions = _panel_positions(expected)
    actual = views_to_panel_configs(views, metrics)
    if not _has_roofline_benchmark(arch):
        expected.pop(ROOFLINE_PANEL_ID, None)
        actual.pop(ROOFLINE_PANEL_ID, None)
    if list(expected) != list(actual):
        return [f"panels: analysis configs have {list(expected)}, views {list(actual)}"]
    errors = []
    for panel_id, panel in expected.items():
        errors.extend(
            f"panel {panel_id}: {error}"
            for error in _panel_errors(panel, actual[panel_id])
        )
    errors.extend(_alias_errors(views, arch))
    errors.extend(_legacy_id_errors(views, metrics, positions))
    return errors


def main() -> int:
    """Entry point. Returns 0 on success, 1 on validation failure."""
    errors = validate(
        default_views_dir(), default_metric_library_dir(), config.analysis_configs_dir
    )
    if errors:
        print("Analysis view validation failed:\n")
        for error in errors:
            print(f"  ERROR: {error}")
        print(f"\n{len(errors)} error(s) found.")
        return 1
    print("Analysis view validation passed.")
    return 0


def _has_roofline_benchmark(arch: str) -> bool:
    return any(
        canonical_config_arch(device) == arch
        for device in mi_gpu_specs.get_roofline_archs()
    )


def _panel_errors(expected: Dict[str, Any], actual: Dict[str, Any]) -> List[str]:
    if expected["title"] != actual["title"]:
        return [f"title {actual['title']!r}, analysis configs {expected['title']!r}"]
    kept = [
        source
        for source in expected["data source"]
        if not _left_out_table(*_table(source))
    ]
    kinds = [list(source) for source in kept]
    if kinds != [list(source) for source in actual["data source"]] or [
        _table(source)[1]["id"] for source in kept
    ] != [_table(source)[1]["id"] for source in actual["data source"]]:
        return ["tables differ from the analysis configs' tables with rows"]
    errors = []
    for want, got in zip(kept, actual["data source"]):
        errors.extend(_table_errors(_table(want)[1], _table(got)[1]))
    return errors


def _table(source: Dict[str, Any]) -> Any:
    ((render, table),) = source.items()
    return render, table


def _left_out_table(render: str, table: Dict[str, Any]) -> bool:
    """Return whether a panel table is left out of the views: rows, none showing one."""
    rows = _table_rows(table) if render == METRIC_TABLE else {}
    return bool(rows) and not any(map(_shows_equation, rows.values()))


def _table_errors(want: Dict[str, Any], got: Dict[str, Any]) -> List[str]:
    where = f"table {want['id']}"
    errors = [
        f"{where}: {key} {got.get(key)!r}, analysis configs {want.get(key)!r}"
        for key in TABLE_KEYS
        if json.dumps(want.get(key)) != json.dumps(got.get(key))
    ]
    want_rows = _table_rows(want)
    shown = [name for name, row in want_rows.items() if _shows_equation(row)]
    got_names = list(_table_rows(got))
    if got_names != shown:
        errors.append(f"{where}: rows {got_names}, analysis configs {shown}")
    placeholder = (want.get("metric") or {}).get("placeholder_range")
    if placeholder != (got.get("metric") or {}).get("placeholder_range"):
        errors.append(f"{where}: placeholder_range differs")
    return errors


def _table_rows(table: Dict[str, Any]) -> Dict[str, Any]:
    rows = table.get("metric") or {}
    return {name: row for name, row in rows.items() if name != "placeholder_range"}


def _shows_equation(row: Dict[str, Any]) -> bool:
    return any(
        str(value) != NO_VALUE
        for column, value in row.items()
        if column not in NON_EQUATION_COLUMNS
    )


def _alias_errors(views: List[Dict[str, Any]], arch: str) -> List[str]:
    template_aliases = get_arch_panel_id_to_alias(arch)
    return [
        f"panel {panel_number}: -b alias {alias!r}, config template has "
        f"{template_aliases.get(panel_number)!r}"
        for panel_number, alias in panel_aliases(views).items()
        if template_aliases.get(panel_number) != alias
    ]


def _legacy_id_errors(
    views: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
    positions: Dict[str, Any],
) -> List[str]:
    """Check that each row's legacy position is its row in the panel files
    (positions: _panel_positions()), that a stored legacy_id is that position, and
    that every metric has one legacy_id."""
    errors = []
    stored: Dict[str, List[str]] = {}
    for position, row in iter_view_rows(views):
        old = legacy_position(position, row)
        name = row_label(row, metrics)
        if positions.get(old) != (position.rsplit(".", 1)[0], name):
            errors.append(f"{position}: {old} is not {name!r}'s panel position")
        if row.get("legacy_id", old) != old:
            errors.append(f"{position}: legacy_id {row['legacy_id']} is not {old}")
        for metric_id in row_metric_ids(row):
            stored.setdefault(metric_id, [])
            if "legacy_id" in row:
                stored[metric_id].append(row["legacy_id"])
    errors.extend(
        f"{metric_id}: has legacy ids {legacy_ids}, not one"
        for metric_id, legacy_ids in stored.items()
        if len(legacy_ids) != 1
    )
    return errors


def _panel_positions(panels: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Return numeric row id -> (table number, row name) of the panel files' rows."""
    positions = {}
    for panel in panels.values():
        for source in panel["data source"]:
            render, table = _table(source)
            if render != METRIC_TABLE:
                continue
            number = f"{table['id'] // 100}.{table['id'] % 100}"
            for index, name in enumerate(_table_rows(table)):
                positions[f"{number}.{index}"] = (number, name)
    return positions


if __name__ == "__main__":
    sys.exit(main())
