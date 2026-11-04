# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Analysis views: the panels analyze shows, built from the metric library.

A view is one panel of one arch. Its table rows name metric library ids: the library
holds formulas, units, peaks and descriptions, and a row only sets how the panel shows
its metric (label, rounding, measured peak, literal column values).

Each metric library arch has one view file per panel in views/<arch>/, which
utils_common.load_views() reads, with BUILT_IN_VIEWS for the panels every arch shows
from the workload's CSV files. The panel id comes from the file name
(0300_memory_chart.yaml is panel 300), table and row ids from their position (301 is
the first table, 3.1.0 its first row).

views_to_panel_configs() builds the panel configs that load_panel_configs() reads from
analysis_configs/. Analyze reads its panel configs through arch_panel_configs().
Standard library and vendored YAML only, because profile imports this module.
"""

import ast
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import config
from utils.metrics.expression import (
    AGGREGATION_FUNCTIONS,
    format_equation,
    parse_equation,
)
from utils.utils_common import (
    BUILT_IN_VIEWS,
    METRIC_TABLE,
    PLACEHOLDER_RANGE_KEY,
    RAW_CSV_TABLE,
    VIEW_FILE_RE,
    canonical_config_arch,
    iter_view_rows,
    load_panel_configs,
    mapping_field_errors,
    numbered_view_tables,
    row_metric_ids,
    view_table_render,
)
from utils.utils_counter_defs import per_dispatch_partner, summed_collectables

# Table kinds, which are also the panel configs' data source keys.
RENDERS = (METRIC_TABLE, RAW_CSV_TABLE, "pc_sampling_table")
# Columns that show a row's own formula.
MAIN_COLUMNS = ("avg", "value", "expr")
# Columns that aggregate a row's formula over kernels.
AGGREGATE_COLUMNS = {"min": "MIN", "max": "MAX", "std dev": "STD"}
# The aggregates a per-kernel formula drops. MIN and MAX stay: in a formula they
# compare values, as in MIN(a, b).
KERNEL_AGGREGATES = AGGREGATION_FUNCTIONS - {"MIN", "MAX"}
VIEW_FIELDS = ("view", "alias", "tables")
TABLE_FIELDS = (
    "render",
    "title",
    "header",
    PLACEHOLDER_RANGE_KEY,
    "metrics",
    "source",
    "columnwise",
)
# Table fields the panel config copies unchanged.
TABLE_CONFIG_KEYS = ("title", "header", "source", "columnwise")
ROW_FIELDS = (
    "metric",
    "columns",
    "label",
    "round",
    "peak",
    "pct_of_peak",
    "values",
)
# How panel configs write a missing column value.
NO_VALUE = "None"
# The row "peak" that shows the measured peak instead of the theoretical one.
EMPIRICAL_PEAK = "empirical"


def views_to_panel_configs(
    views: Dict[int, Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
    collectables: Dict[str, str],
) -> "OrderedDict[int, Dict[str, Any]]":
    """Return the panel configs of an arch's views, as load_panel_configs() does.

    Args:
        views: The arch's views (load_views()).
        metrics: The arch's MetricLibrary.metrics.
        collectables: The arch's MetricLibrary.collectable_formulas().
    """
    partners = _per_dispatch_partners(collectables)
    return OrderedDict(
        (panel_id, _panel_config(panel_id, views[panel_id], metrics, partners))
        for panel_id in sorted(views)
    )


def arch_panel_configs(arch: str) -> "OrderedDict[int, Dict[str, Any]]":
    """Return the panel configs analyze shows for arch, from analysis_configs/."""
    config_arch = canonical_config_arch(arch) or arch
    return load_panel_configs([str(config.analysis_configs_dir / config_arch)])


def row_label(row: Dict[str, Any], metrics: Dict[str, Dict[str, Any]]) -> str:
    """Return the name a row shows: its label, or its metric's name.

    A row with per-column metrics has no single metric name, so it needs a label.
    """
    if "label" in row:
        return row["label"]
    return metrics[row["metric"]]["name"]


def view_schema_errors(documents: Dict[str, Any]) -> List[str]:
    """Return the shape errors of one arch's view documents, keyed by file path;
    view_metric_errors() checks the views against the metric library."""
    errors = _file_name_errors(list(documents))
    for source, document in documents.items():
        errors.extend(f"[{source}] {error}" for error in _view_shape_errors(document))
    return errors


def view_metric_errors(
    views: Dict[int, Dict[str, Any]], metrics: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Return what in an arch's views does not match its metric library, including
    the library's metrics and measured peaks that no row shows."""
    errors = [
        f"{position} {error}"
        for position, row in iter_view_rows(views)
        for error in _row_metric_errors(row, metrics)
    ]
    return errors + _unshown_metric_errors(views, metrics)


# ---------------------------------------------------------------------------
# Panel configs
# ---------------------------------------------------------------------------


def _per_dispatch_partners(collectables: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Return each summed collectable's per-dispatch partner, or None if it has none.

    A summed collectable is one value for all dispatches, so a Min, Max or Std Dev
    over kernels reads its partner instead.
    """
    return {
        name: per_dispatch_partner(name, collectables)
        for name in summed_collectables(collectables)
    }


def _panel_config(
    panel_id: int,
    view: Dict[str, Any],
    metrics: Dict[str, Dict[str, Any]],
    partners: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    return {
        "id": panel_id,
        "title": view["view"],
        "data source": [
            {
                view_table_render(table): _table_config(
                    table_id, table, metrics, partners
                )
            }
            for table_id, table in numbered_view_tables(panel_id, view)
        ],
        "metrics_description": _panel_descriptions(view, metrics),
    }


def _table_config(
    table_id: int,
    table: Dict[str, Any],
    metrics: Dict[str, Dict[str, Any]],
    partners: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    table_config = {"id": table_id}
    table_config.update({key: table[key] for key in TABLE_CONFIG_KEYS if key in table})
    if "metrics" not in table:
        return table_config
    columns = [column for column in table["header"] if column != "metric"]
    rows = {
        row_label(row, metrics): _row_values(row, columns, metrics, partners)
        for row in table["metrics"]
    }
    if PLACEHOLDER_RANGE_KEY in table:
        rows[PLACEHOLDER_RANGE_KEY] = table[PLACEHOLDER_RANGE_KEY]
    table_config["metric"] = rows
    return table_config


def _row_values(
    row: Dict[str, Any],
    columns: List[str],
    metrics: Dict[str, Dict[str, Any]],
    partners: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """Return a row's value in each column, in column order."""
    metric = metrics.get(row.get("metric"))
    values = {}
    for column in columns:
        if column in row.get("values", {}):
            values[column] = row["values"][column]
        elif column in row.get("columns", {}):
            metric_id = row["columns"][column]
            values[column] = metrics[metric_id]["formula"] if metric_id else NO_VALUE
        elif column in MAIN_COLUMNS:
            values[column] = _main_formula(row, metric)
        elif column in AGGREGATE_COLUMNS:
            values[column] = _aggregate(column, metric, partners)
        elif column == "unit":
            values[column] = metric["unit"] if metric else NO_VALUE
        elif column == "peak":
            values[column] = _peak(row, metric)
        elif column == "pct_of_peak":
            values[column] = row.get("pct_of_peak", True)
        else:
            raise ValueError(f"row {row} has no value for column {column!r}")
    return values


def _main_formula(row: Dict[str, Any], metric: Optional[Dict[str, Any]]) -> str:
    """Return the formula a row shows in its main column, rounded if it asks."""
    if metric is None:
        return NO_VALUE
    if "round" in row:
        return f"ROUND({metric['formula']}, {row['round']})"
    return metric["formula"]


def _aggregate(
    column: str,
    metric: Optional[Dict[str, Any]],
    partners: Dict[str, Optional[str]],
) -> str:
    """Return the column's aggregate over kernels of the metric's per-kernel formula."""
    if metric is None:
        return NO_VALUE
    per_kernel = _PerKernelFormula(partners).visit(parse_equation(metric["formula"]))
    function = ast.Name(AGGREGATE_COLUMNS[column], ast.Load())
    return format_equation(ast.Call(function, [per_kernel], []))


def _peak(row: Dict[str, Any], metric: Optional[Dict[str, Any]]) -> Union[int, str]:
    """Return the peak a row shows: its metric's theoretical or measured peak."""
    field = "peak_empirical" if row.get("peak") == EMPIRICAL_PEAK else "peak"
    if metric is None or field not in metric:
        return NO_VALUE
    return metric[field]


def _panel_descriptions(
    view: Dict[str, Any], metrics: Dict[str, Dict[str, Any]]
) -> Dict[str, str]:
    """Return label -> description; the first row with a label describes it."""
    descriptions: Dict[str, str] = {}
    for table in view["tables"]:
        for row in table.get("metrics", []):
            label = row_label(row, metrics)
            metric = metrics.get(row.get("metric"))
            if label not in descriptions and metric is not None:
                descriptions[label] = metric["description"]
    return descriptions


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _file_name_errors(paths: List[str]) -> List[str]:
    """Report bad view file names, built-in panel ids, and panel ids with two files."""
    errors = []
    paths_by_panel: Dict[int, List[str]] = {}
    for path in paths:
        match = VIEW_FILE_RE.match(Path(path).name)
        if match is None:
            errors.append(
                f"[{path}] the file name must be the panel id and the view name, "
                "e.g. 0300_memory_chart.yaml"
            )
            continue
        panel_id = int(match.group(1))
        if panel_id in BUILT_IN_VIEWS:
            errors.append(f"[{path}] panel {panel_id} is a built-in view")
        paths_by_panel.setdefault(panel_id, []).append(path)
    errors.extend(
        f"[{', '.join(panel_paths)}] panel {panel_id} has more than one view file"
        for panel_id, panel_paths in sorted(paths_by_panel.items())
        if len(panel_paths) > 1
    )
    return errors


def _view_shape_errors(document: object) -> List[str]:
    if not isinstance(document, dict):
        return ["a view must be a mapping"]
    errors = mapping_field_errors(document, ("view", "tables"), VIEW_FIELDS)
    tables = document.get("tables")
    if not isinstance(tables, list):
        return [*errors, "'tables' must be a list"]
    for position, table in enumerate(tables, start=1):
        errors.extend(
            f"table {position}: {error}" for error in _table_shape_errors(table)
        )
    return errors


def _table_shape_errors(table: object) -> List[str]:
    if not isinstance(table, dict):
        return ["a table must be a mapping"]
    errors = mapping_field_errors(table, ("title",), TABLE_FIELDS)
    render = view_table_render(table)
    if render not in RENDERS:
        errors.append(f"render {render!r} is not one of {RENDERS}")
    if render == METRIC_TABLE:
        errors.extend(_metric_table_shape_errors(table))
    elif "metrics" in table or "header" in table:
        errors.append(f"a {render} has no header or metrics")
    return errors


def _metric_table_shape_errors(table: Dict[str, Any]) -> List[str]:
    header = table.get("header")
    if not isinstance(header, dict) or next(iter(header), None) != "metric":
        return ["'header' must be a mapping whose first column is 'metric'"]
    if not isinstance(table.get("metrics"), list):
        return ["'metrics' must be a list"]
    columns = [column for column in header if column != "metric"]
    errors = []
    for index, row in enumerate(table["metrics"]):
        errors.extend(
            f"row {index}: {error}" for error in _row_shape_errors(row, columns)
        )
    return errors


def _row_shape_errors(row: object, columns: List[str]) -> List[str]:
    if not isinstance(row, dict):
        return ["a row must be a mapping"]
    errors = mapping_field_errors(row, (), ROW_FIELDS)
    if ("metric" in row) == ("columns" in row):
        errors.append("needs one of 'metric' and 'columns'")
    elif "columns" in row and "label" not in row:
        errors.append("needs a 'label' when it has 'columns'")
    if row.get("peak", EMPIRICAL_PEAK) != EMPIRICAL_PEAK:
        errors.append(f"'peak' may only be {EMPIRICAL_PEAK!r}")
    errors.extend(
        f"'{field}' names {column!r}, which is not a column"
        for field in ("columns", "values")
        for column in row.get(field, {})
        if column not in columns
    )
    return errors


def _row_metric_errors(
    row: Dict[str, Any], metrics: Dict[str, Dict[str, Any]]
) -> List[str]:
    errors = [
        f"{metric_id}: not a metric of the arch's library"
        for metric_id in row_metric_ids(row)
        if metric_id not in metrics
    ]
    metric = metrics.get(row.get("metric"))
    if not errors and row.get("peak") == EMPIRICAL_PEAK:
        if metric is None or "peak_empirical" not in metric:
            errors.append(f"{row.get('metric')} has no 'peak_empirical'")
    return errors


def _unshown_metric_errors(
    views: Dict[int, Dict[str, Any]], metrics: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Report each metric no view row shows, and each measured peak no row asks for."""
    shown: Set[str] = set()
    shown_empirical: Set[str] = set()
    for _, row in iter_view_rows(views):
        shown.update(row_metric_ids(row))
        if row.get("peak") == EMPIRICAL_PEAK:
            shown_empirical.add(row["metric"])
    errors = [
        f"{metric_id}: no view shows it"
        for metric_id in metrics
        if metric_id not in shown
    ]
    errors.extend(
        f"{metric_id}: no row shows its 'peak_empirical'"
        for metric_id, metric in metrics.items()
        if "peak_empirical" in metric and metric_id not in shown_empirical
    )
    return errors


class _PerKernelFormula(ast.NodeTransformer):
    """Turn the syntax tree of a metric formula into its value per kernel: each
    SUM(x) and AVG(x) becomes x, and each summed collectable (one value for all
    dispatches) becomes its per-dispatch partner."""

    def __init__(self, partners: Dict[str, Optional[str]]) -> None:
        self.partners = partners

    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        if node.func.id in KERNEL_AGGREGATES and len(node.args) == 1:
            return node.args[0]
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if not node.id.startswith("$"):
            return node
        partner = self.partners.get(node.id[1:])
        return ast.Name(f"${partner}", ast.Load()) if partner else node
