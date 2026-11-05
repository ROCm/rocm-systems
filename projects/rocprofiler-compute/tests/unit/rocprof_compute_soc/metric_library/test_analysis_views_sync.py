# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Check that the analysis views keep the analysis configs' structure, arch by arch.

Until the cutover, each view is one analysis config panel:

- the same panels, tables, headers and table settings, in order;
- the same rows in order, leaving out only rows that show no equation (and tables left
  empty by them);
- each panel's -b alias is the config template's;
- every row is a row of the same panel table, and every metric the arch shows has one
  legacy_id, its row's numeric id in the panel file.

Values, units, peaks and descriptions come from the metric library and are not compared.
The check skips the panels the metric library sync test skips (the roofline panel of an
arch without a roofline benchmark).
"""

import json
import re
from typing import Any, Dict, List

import common
import pytest

import config
from rocprof_compute_soc.metric_library import loader
from utils.analysis_views import (
    METRIC_TABLE,
    NO_VALUE,
    iter_view_rows,
    load_views,
    panel_aliases,
    row_label,
    row_metric_ids,
    views_to_panel_configs,
)
from utils.utils_common import load_panel_configs, load_yaml

# Table settings that must match the panel's; rows are compared by name.
TABLE_KEYS = ("id", "title", "header", "cli_style", "source", "columnwise")
# Row columns that show no equation.
NON_EQUATION_COLUMNS = ("unit", "peak", "pct_of_peak")


def structure_errors(
    views: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
    arch: str,
) -> List[str]:
    """Return how one arch's views differ from its analysis configs' structure."""
    expected = load_panel_configs([str(config.analysis_configs_dir / arch)])
    positions = _panel_positions(expected)
    actual = views_to_panel_configs(views, metrics)
    for file_id in common.unshown_file_ids(arch):
        expected.pop(int(file_id), None)
        actual.pop(int(file_id), None)
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
    template_aliases = _template_aliases(arch)
    return [
        f"panel {panel_number}: -b alias {alias!r}, config template has "
        f"{template_aliases.get(panel_number)!r}"
        for panel_number, alias in panel_aliases(views).items()
        if template_aliases.get(panel_number) != alias
    ]


def _template_aliases(arch: str) -> Dict[str, str]:
    """Return panel number -> -b alias from the config template whose name prefix
    matches arch, "" for none; {} when no template matches."""
    for path in sorted(config.analysis_configs_dir.glob("*_config_template.yaml")):
        match = re.match(r"(gfx\d+)_config_template\.yaml$", path.name)
        if match and arch.startswith(match.group(1)):
            panels = (load_yaml(path) or {}).get("panels") or []
            return {
                str(panel["panel_id"]): (panel.get("panel_alias") or "")
                for panel in panels
            }
    return {}


def _legacy_id_errors(
    views: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
    positions: Dict[Any, str],
) -> List[str]:
    """Check that each row is a row of the same panel table (positions:
    _panel_positions()), that a stored legacy_id is that row's numeric id, and that
    every metric has one legacy_id."""
    errors = []
    stored: Dict[str, List[str]] = {}
    for position, row in iter_view_rows(views):
        table = position.rsplit(".", 1)[0]
        name = row_label(row, metrics)
        old = positions.get((table, name))
        if old is None:
            errors.append(f"{position}: {name!r} is not a row of panel table {table}")
            continue
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


def _panel_positions(panels: Dict[int, Dict[str, Any]]) -> Dict[Any, str]:
    """Return (table number, row name) -> numeric row id of the panel files' rows."""
    positions = {}
    for panel in panels.values():
        for source in panel["data source"]:
            render, table = _table(source)
            if render != METRIC_TABLE:
                continue
            number = f"{table['id'] // 100}.{table['id'] % 100}"
            for index, name in enumerate(_table_rows(table)):
                positions[(number, name)] = f"{number}.{index}"
    return positions


def swap_rows(views: List[Dict[str, Any]]) -> None:
    table = next(t for v in views for t in v["tables"] if t["id"] == 1102)
    table["metrics"][0], table["metrics"][1] = table["metrics"][1], table["metrics"][0]


def drop_row(views: List[Dict[str, Any]]) -> None:
    table = next(t for v in views for t in v["tables"] if t["id"] == 1102)
    del table["metrics"][2]


def drop_column(views: List[Dict[str, Any]]) -> None:
    table = next(t for v in views for t in v["tables"] if t["id"] == 1102)
    del table["header"]["max"]


def change_alias(views: List[Dict[str, Any]]) -> None:
    next(v for v in views if v["legacy_panel_id"] == 1100)["alias"] = "pipeline"


def move_legacy_id(views: List[Dict[str, Any]]) -> None:
    dict(iter_view_rows(views))["11.2.1"]["legacy_id"] = "11.2.2"


def arch_errors(arch: str, views: List[Dict[str, Any]]) -> List[str]:
    metrics = loader.MetricLibrary.load(
        loader.default_metric_library_dir(), arch
    ).metrics
    return structure_errors(views, metrics, arch)


@pytest.mark.parametrize("arch", list(loader.METRIC_LIBRARY_PARENT_ARCH))
def test_views_keep_the_panel_structure(arch):
    assert arch_errors(arch, load_views(arch)) == []


@pytest.mark.parametrize(
    "change, error",
    [
        (swap_rows, "panel 1100: table 1102: rows"),
        (drop_row, "panel 1100: table 1102: rows"),
        (drop_column, "panel 1100: table 1102: header"),
        (change_alias, "panel 11: -b alias 'pipeline', config template has 'cu_pipe'"),
        (move_legacy_id, "11.2.1: legacy_id 11.2.2 is not 11.2.1"),
    ],
)
def test_a_changed_structure_is_reported(change, error):
    """gfx942's views with one change no longer keep its panels' structure."""
    views = load_views("gfx942")
    change(views)
    errors = arch_errors("gfx942", views)
    assert any(e.startswith(error) for e in errors), errors
