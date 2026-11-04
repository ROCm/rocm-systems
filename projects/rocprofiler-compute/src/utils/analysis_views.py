# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Analysis views: the panels analyze shows, built from the metric library.

A view is one panel of one arch: its tables, and for each table row the metric it
shows, by metric library id. Formulas, units, peaks and descriptions come from the
library; a view row stores only how the panel shows the metric: a label other than
the metric's name, rounding, the measured peak instead of the theoretical one, and the
literal values of typed columns. views_to_panel_configs() turns an arch's views into
the panel configs that load_panel_configs() reads from analysis_configs/.

views/<arch>/ holds the views of one metric library arch. "legacy_id" marks the one
numeric id (11.2.3) each metric keeps from the panel files until numeric ids are
retired.

arch_panel_configs() returns the panel configs analyze shows; tools/validate_views.py
checks the view files.
Standard library and the vendored YAML only: profile mode imports it.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import config
from utils.equation_format import normalize_equation
from utils.utils_common import canonical_config_arch, load_panel_configs
from vendored import yaml

# Table kinds, as the panel configs' data source keys; a table without "render" is a
# metric table.
METRIC_TABLE = "metric_table"
RENDERS = (METRIC_TABLE, "raw_csv_table", "pc_sampling_table")
# How the CLI draws a metric table (tty.py, memory_chart).
CLI_STYLES = (
    "simple_box",
    "simple_bar",
    "simple_multiple_bar",
    "mem_chart",
    "Roofline",
)
# A row shows its formula in the first of these columns its table has.
MAIN_COLUMNS = ("avg", "value", "expr")
# Columns that aggregate the main formula over kernels, with the function they apply.
AGGREGATE_COLUMNS = {"min": "MIN", "max": "MAX", "std dev": "STD"}
VIEW_FIELDS = ("view", "legacy_panel_id", "alias", "tables")
TABLE_FIELDS = (
    "render",
    "id",
    "title",
    "header",
    "cli_style",
    "placeholder_range",
    "metrics",
    "source",
    "columnwise",
)
ROW_FIELDS = (
    "metric",
    "metrics",
    "label",
    "round",
    "peak",
    "pct_of_peak",
    "values",
    "legacy_id",
)
# The value of a column a row has no value for, as the panel configs write it.
NO_VALUE = "None"
# A row's "peak" that shows its metric's measured peak instead of the theoretical one.
EMPIRICAL_PEAK = "empirical"


def default_views_dir() -> Path:
    """Return the views tree shipped with rocprof-compute."""
    return Path(config.rocprof_compute_home) / "rocprof_compute_soc" / "views"


def read_view_files(arch: str, views_dir: Optional[Path] = None) -> Dict[Path, Any]:
    """Read arch's view files, keyed by path, in file name order.

    Args:
        arch: A GPU arch, e.g. "gfx942" or "gfx1151".
        views_dir: The views tree; the shipped one by default.

    Returns:
        Each file's document; none if the arch has no views.
    """
    arch_dir = Path(views_dir or default_views_dir()) / canonical_config_arch(arch)
    documents = {}
    for path in sorted(arch_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as file:
            documents[path] = yaml.safe_load(file)
    return documents


def load_views(arch: str, views_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return arch's views, in file name order."""
    return list(read_view_files(arch, views_dir).values())


def views_to_panel_configs(
    views: List[Dict[str, Any]], metrics: Dict[str, Dict[str, Any]]
) -> "OrderedDict[int, Dict[str, Any]]":
    """Return the panel configs of an arch's views, as load_panel_configs() does.

    Args:
        views: One arch's views (load_views()).
        metrics: The arch's metric library metrics (MetricLibrary.metrics).

    Returns:
        Each view's panel config, keyed by panel id, in panel id order.
    """
    configs = {view["legacy_panel_id"]: _panel_config(view, metrics) for view in views}
    return OrderedDict(sorted(configs.items()))


def arch_panel_configs(arch: str) -> "OrderedDict[int, Dict[str, Any]]":
    """Return the panel configs analyze shows for an arch."""
    config_arch = canonical_config_arch(arch) or arch
    return load_panel_configs([str(config.analysis_configs_dir / config_arch)])


def panel_aliases(views: List[Dict[str, Any]]) -> Dict[str, str]:
    """Return panel number -> -b alias, "" for none (get_arch_panel_id_to_alias())."""
    return {
        str(view["legacy_panel_id"] // 100): view.get("alias") or "" for view in views
    }


def iter_view_rows(views: List[Dict[str, Any]]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield each metric table row of an arch's views with its numeric id."""
    for view in views:
        for table in view["tables"]:
            if _render(table) != METRIC_TABLE:
                continue
            for index, row in enumerate(table["metrics"]):
                yield f"{table['id'] // 100}.{table['id'] % 100}.{index}", row


def row_metric_ids(row: Dict[str, Any]) -> List[str]:
    """Return the metric ids a row shows: its metric, or each column's."""
    if "metric" in row:
        return [row["metric"]]
    return [metric_id for metric_id in row.get("metrics", {}).values() if metric_id]


def row_label(row: Dict[str, Any], metrics: Dict[str, Dict[str, Any]]) -> str:
    """Return the name a row shows: its label, or its metric's name."""
    if "label" in row:
        return row["label"]
    return metrics[row["metric"]]["name"]


def view_schema_errors(documents: Dict[str, Any]) -> List[str]:
    """Return what is wrong with the shape of view documents, keyed by source name.

    Checks the fields each level may have, render types and CLI styles, that a row
    names its metric or its columns' metrics, and that a row with columns' metrics
    has a label.
    """
    errors = []
    for source, document in documents.items():
        errors.extend(f"[{source}] {error}" for error in _view_shape_errors(document))
    return errors


def view_metric_errors(
    views: List[Dict[str, Any]], metrics: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Return what in an arch's views the arch's metric library cannot show.

    Checks that panel ids are unique, every metric id exists, and a measured peak
    exists.
    """
    errors = []
    panel_ids = [view["legacy_panel_id"] for view in views]
    errors.extend(
        f"panel {panel_id} has more than one view"
        for panel_id in sorted({pid for pid in panel_ids if panel_ids.count(pid) > 1})
    )
    for position, row in iter_view_rows(views):
        errors.extend(
            f"{position} {error}" for error in _row_metric_errors(row, metrics)
        )
    return errors


# ---------------------------------------------------------------------------
# Panel configs
# ---------------------------------------------------------------------------


def _render(table: Dict[str, Any]) -> str:
    return table.get("render", METRIC_TABLE)


def _panel_config(
    view: Dict[str, Any], metrics: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    return {
        "id": view["legacy_panel_id"],
        "title": view["view"],
        "data source": [
            {_render(table): _table_config(table, metrics)} for table in view["tables"]
        ],
        "metrics_description": _panel_descriptions(view, metrics),
    }


def _table_config(
    table: Dict[str, Any], metrics: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    config_keys = ("id", "title", "header", "source", "columnwise", "cli_style")
    table_config = {key: table[key] for key in config_keys if key in table}
    if "metrics" not in table:
        return table_config
    columns = [column for column in table["header"] if column != "metric"]
    rows = {
        row_label(row, metrics): _row_values(row, columns, metrics)
        for row in table["metrics"]
    }
    if "placeholder_range" in table:
        rows["placeholder_range"] = table["placeholder_range"]
    table_config["metric"] = rows
    return table_config


def _row_values(
    row: Dict[str, Any], columns: List[str], metrics: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Return a row's value in each column, in column order."""
    metric = metrics.get(row.get("metric"))
    values = {}
    for column in columns:
        if column in row.get("values", {}):
            values[column] = row["values"][column]
        elif column in row.get("metrics", {}):
            metric_id = row["metrics"][column]
            values[column] = metrics[metric_id]["formula"] if metric_id else NO_VALUE
        elif column in MAIN_COLUMNS:
            values[column] = _main_formula(row, metric)
        elif column in AGGREGATE_COLUMNS:
            values[column] = _aggregate(column, metric)
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


def _aggregate(column: str, metric: Optional[Dict[str, Any]]) -> str:
    """Return a column's aggregate over kernels of the metric's formula, per kernel."""
    if metric is None:
        return NO_VALUE
    per_kernel = _per_kernel(metric["formula"])
    return normalize_equation(f"{AGGREGATE_COLUMNS[column]}({per_kernel})")


def _per_kernel(formula: str) -> str:
    """Return a formula per kernel: each SUM(x) and AVG(x) over kernels becomes (x)."""
    text = formula
    for function in ("SUM(", "AVG("):
        start = text.find(function)
        while start != -1:
            open_paren = start + len(function) - 1
            close_paren = _closing_paren(text, open_paren)
            inner = text[open_paren + 1 : close_paren]
            text = f"{text[:start]}({inner}){text[close_paren + 1 :]}"
            start = text.find(function)
    return text


def _closing_paren(text: str, open_paren: int) -> int:
    depth = 0
    for index in range(open_paren, len(text)):
        depth += {"(": 1, ")": -1}.get(text[index], 0)
        if depth == 0:
            return index
    raise ValueError(f"unbalanced parentheses in {text!r}")


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


def _view_shape_errors(document: object) -> List[str]:
    if not isinstance(document, dict):
        return ["a view must be a mapping"]
    errors = _field_errors(document, VIEW_FIELDS, ("view", "legacy_panel_id", "tables"))
    if not isinstance(document.get("legacy_panel_id", 0), int):
        errors.append("'legacy_panel_id' must be an integer")
    tables = document.get("tables")
    if not isinstance(tables, list):
        return [*errors, "'tables' must be a list"]
    for table in tables:
        errors.extend(_table_shape_errors(table))
    return errors


def _table_shape_errors(table: object) -> List[str]:
    if not isinstance(table, dict):
        return ["a table must be a mapping"]
    where = f"table {table.get('id')}"
    errors = _field_errors(table, TABLE_FIELDS, ("id", "title"))
    render = _render(table)
    if render not in RENDERS:
        errors.append(f"render {render!r} is not one of {RENDERS}")
    if "cli_style" in table and table["cli_style"] not in CLI_STYLES:
        errors.append(f"cli_style {table['cli_style']!r} is not one of {CLI_STYLES}")
    if render == METRIC_TABLE:
        errors.extend(_metric_table_shape_errors(table))
    elif "metrics" in table or "header" in table:
        errors.append(f"a {render} has no header or metrics")
    return [f"{where}: {error}" for error in errors]


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
    errors = _field_errors(row, ROW_FIELDS, ())
    if ("metric" in row) == ("metrics" in row):
        errors.append("needs one of 'metric' and 'metrics'")
    elif "metrics" in row and "label" not in row:
        errors.append("needs a 'label' when it has 'metrics'")
    if row.get("peak", EMPIRICAL_PEAK) != EMPIRICAL_PEAK:
        errors.append(f"'peak' may only be {EMPIRICAL_PEAK!r}")
    errors.extend(
        f"'{field}' names {column!r}, which is not a column"
        for field in ("metrics", "values")
        for column in row.get(field, {})
        if column not in columns
    )
    if not isinstance(row.get("legacy_id", ""), str):
        errors.append("'legacy_id' must be a string")
    return errors


def _field_errors(
    item: Dict[str, Any], known: Tuple[str, ...], required: Tuple[str, ...]
) -> List[str]:
    errors = [f"'{field}' is required" for field in required if field not in item]
    errors.extend(
        f"'{field}' is not a known field" for field in item if field not in known
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
