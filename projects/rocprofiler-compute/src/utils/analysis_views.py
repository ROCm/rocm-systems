# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Analysis views: the panels analyze shows, built from the metric library.

A view is one panel of one arch. Its table rows name metric library ids: the library
holds formulas, units, peaks and descriptions, and a row only sets how the panel shows
its metric (label, rounding, measured peak, literal column values).

Each metric library arch has one view file per panel in views/<arch>/; BUILT_IN_VIEWS
adds the panels every arch shows from the workload's CSV files. The panel id comes
from the file name (0300_memory_chart.yaml is panel 300), table and row ids from their
position (301 is the first table, 3.1.0 its first row).

views_to_panel_configs() builds the panel configs that load_panel_configs() reads from
analysis_configs/. Analyze reads its panel configs through arch_panel_configs(). Profile
still reads analysis_configs/, which test_metric_library_sync.py keeps equal in
structure to the views' panel configs.
Standard library and vendored YAML only, because profile imports this module.
"""

import copy
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import config
from rocprof_compute_soc.metric_library import loader
from utils.equation_format import (
    AGGREGATION_FUNCTIONS,
    ASTNode,
    BinaryOpNode,
    FunctionCallNode,
    UnaryOpNode,
    VariableNode,
    format_equation,
    parse_equation,
)
from utils.logger import console_error
from utils.utils_common import (
    INVALID_BLOCK_HINT,
    METRIC_ID_RE,
    PLACEHOLDER_RANGE_KEY,
    canonical_config_arch,
    convert_metric_id_to_panel_info,
    format_row_id,
    load_yaml,
    mapping_field_errors,
    resolve_block_token,
)
from utils.utils_counter_defs import (
    PER_DISPATCH_SUFFIX,
    parse_counters_text,
    per_dispatch_partner,
    summed_collectables,
)

# Table kinds, which are also the panel configs' data source keys.
METRIC_TABLE = "metric_table"
RAW_CSV_TABLE = "raw_csv_table"
RENDERS = (METRIC_TABLE, RAW_CSV_TABLE, "pc_sampling_table")
# Columns that show a row's own formula.
MAIN_COLUMNS = ("avg", "value", "expr")
# Columns that aggregate a row's formula over kernels.
AGGREGATE_COLUMNS = {"min": "MIN", "max": "MAX", "std dev": "STD"}
# A table with this column is a simple box: Min, quartiles and Max over dispatches.
SIMPLE_BOX_COLUMN = "expr"
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
VIEW_FILE_RE = re.compile(r"^(\d\d00)_[a-z0-9_]+\.yaml$")
# The panel --list-stats shows.
TOP_STATS_PANEL_ID = 0
# The views every arch has, by panel id. They show the workload's CSV files,
# not metrics.
BUILT_IN_VIEWS: Dict[int, Dict[str, Any]] = {
    TOP_STATS_PANEL_ID: {
        "view": "Top Stats",
        "alias": "topstats",
        "tables": [
            {
                "render": RAW_CSV_TABLE,
                "title": "Top Kernels",
                "source": "pmc_kernel_top.csv",
            },
            {
                "render": RAW_CSV_TABLE,
                "title": "Dispatch List",
                "source": "pmc_dispatch_info.csv",
            },
        ],
    },
    100: {
        "view": "System Info",
        "alias": "sysinfo",
        "tables": [
            {
                "render": RAW_CSV_TABLE,
                "title": "System Info",
                "source": "sysinfo.csv",
                "columnwise": True,
            }
        ],
    },
}


def read_view_files(arch: str) -> Dict[Path, Any]:
    """Return arch's view file documents by path, in file name order."""
    arch_dir = config.analysis_views_dir / canonical_config_arch(arch)
    return {path: load_yaml(path) for path in sorted(arch_dir.glob("*.yaml"))}


def load_views(arch: str) -> "OrderedDict[int, Dict[str, Any]]":
    """Return arch's views, built-in ones included, sorted by panel id."""
    return views_from_files(read_view_files(arch))


def views_from_files(documents: Dict[Path, Any]) -> "OrderedDict[int, Dict[str, Any]]":
    """Return the built-in views and those of read_view_files(), sorted by panel id."""
    views = copy.deepcopy(BUILT_IN_VIEWS)
    views.update((view_panel_id(path.name), view) for path, view in documents.items())
    return OrderedDict(sorted(views.items()))


def view_panel_id(file_name: str) -> int:
    """Return the panel id of a view file name, e.g. 300 for 0300_memory_chart.yaml."""
    match = VIEW_FILE_RE.match(file_name)
    if match is None:
        raise ValueError(f"{file_name}: not a view file name")
    return int(match.group(1))


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
    partners = per_dispatch_partners(collectables)
    return OrderedDict(
        (panel_id, _panel_config(panel_id, views[panel_id], metrics, partners))
        for panel_id in sorted(views)
    )


def per_dispatch_partners(collectables: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Return each summed collectable's per-dispatch partner, or None if it has none.

    A summed collectable is one value for all dispatches, so a Min, Max or Std Dev
    over kernels reads its partner instead.
    """
    return {
        name: per_dispatch_partner(name, collectables)
        for name in summed_collectables(collectables)
    }


def arch_panel_configs(arch: str) -> "OrderedDict[int, Dict[str, Any]]":
    """Return arch's panel configs, built from its views and metric library.

    An arch without a metric library has no views, so it gets {} and shows no panels.
    """
    if not loader.has_metric_library(arch):
        return OrderedDict()
    library = loader.MetricLibrary.load(loader.default_metric_library_dir(), arch)
    return views_to_panel_configs(
        load_views(arch), library.metrics, library.collectable_formulas()
    )


def panel_aliases(views: Dict[int, Dict[str, Any]]) -> Dict[str, str]:
    """Return panel number -> -b alias, "" for none, of an arch's views."""
    return {
        str(panel_id // 100): view.get("alias") or ""
        for panel_id, view in views.items()
    }


def get_arch_panel_id_to_alias(arch: str) -> Dict[str, str]:
    """Return panel_aliases() of arch's views; {} for an arch without views."""
    if not loader.has_metric_library(arch):
        return {}
    return panel_aliases(load_views(arch))


def get_arch_alias_to_panel_id(arch: str) -> Dict[str, str]:
    """Return -b alias -> panel number, for the panels that have an alias."""
    return {
        alias: pid for pid, alias in get_arch_panel_id_to_alias(arch).items() if alias
    }


def convert_filter_blocks_to_panel_ids(
    filter_blocks: List[str], arch: Optional[str] = None
) -> Set[int]:
    """Return the panel ids (e.g. {200, 1100}) that -b tokens select.

    Numeric ids ("2", "11.1") select their panel; other tokens are arch's panel aliases.
    """
    alias_map: Dict[str, str] = (
        get_arch_alias_to_panel_id(arch)
        if arch and any(not METRIC_ID_RE.match(str(bid)) for bid in filter_blocks)
        else {}
    )
    resolved: Set[int] = set()
    for bid in filter_blocks:
        metric_id = resolve_block_token(str(bid), alias_map)
        if metric_id is None:
            console_error(f"Invalid --block value {str(bid)!r}.{INVALID_BLOCK_HINT}")
        resolved.add(int(convert_metric_id_to_panel_info(metric_id)[0]))
    return resolved


def iter_view_rows(
    views: Dict[int, Dict[str, Any]],
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield each metric table row of an arch's views with its numeric id."""
    for position, row, _ in _iter_rows_with_columns(views):
        yield position, row


def row_metric_ids(row: Dict[str, Any]) -> List[str]:
    """Return the metric ids a row shows: its metric, or each column's."""
    if "metric" in row:
        return [row["metric"]]
    return [metric_id for metric_id in row.get("columns", {}).values() if metric_id]


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
    views: Dict[int, Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
    collectables: Dict[str, str],
) -> List[str]:
    """Return what in an arch's views does not match its metric library, including
    the library's metrics and measured peaks that no row shows."""
    partners = per_dispatch_partners(collectables)
    errors = []
    for position, row, columns in _iter_rows_with_columns(views):
        row_errors = _row_metric_errors(row, metrics)
        simple_box = SIMPLE_BOX_COLUMN in columns
        if not row_errors and (
            simple_box or any(column in AGGREGATE_COLUMNS for column in columns)
        ):
            row_errors = _summed_collectable_errors(row, metrics, partners, simple_box)
        errors.extend(f"{position} {error}" for error in row_errors)
    return errors + _unshown_metric_errors(views, metrics)


# ---------------------------------------------------------------------------
# Panel configs
# ---------------------------------------------------------------------------


def _render(table: Dict[str, Any]) -> str:
    return table.get("render", METRIC_TABLE)


def _iter_rows_with_columns(
    views: Dict[int, Dict[str, Any]],
) -> Iterator[Tuple[str, Dict[str, Any], List[str]]]:
    """Yield what iter_view_rows() does, plus the row's table header columns."""
    for panel_id, view in views.items():
        for table_id, table in _numbered_tables(panel_id, view):
            if _render(table) != METRIC_TABLE:
                continue
            for index, row in enumerate(table["metrics"]):
                yield format_row_id(table_id, index), row, list(table["header"])


def _numbered_tables(
    panel_id: int, view: Dict[str, Any]
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield each table of a view with its table id: the panel id plus its position."""
    for index, table in enumerate(view["tables"]):
        yield panel_id + index + 1, table


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
            {_render(table): _table_config(table_id, table, metrics, partners)}
            for table_id, table in _numbered_tables(panel_id, view)
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
    per_kernel = _per_kernel(parse_equation(metric["formula"]), partners)
    return format_equation(FunctionCallNode(AGGREGATE_COLUMNS[column], [per_kernel]))


def _per_kernel(node: ASTNode, partners: Dict[str, Optional[str]]) -> ASTNode:
    """Return a formula tree per kernel: each SUM(x) and AVG(x) becomes x, and each
    summed collectable (one value for all dispatches) becomes its per-dispatch partner.
    """
    if isinstance(node, FunctionCallNode):
        args = [_per_kernel(arg, partners) for arg in node.args]
        if node.name in KERNEL_AGGREGATES and len(args) == 1:
            return args[0]
        return FunctionCallNode(node.name, args)
    if isinstance(node, BinaryOpNode):
        return BinaryOpNode(
            node.op, _per_kernel(node.left, partners), _per_kernel(node.right, partners)
        )
    if isinstance(node, UnaryOpNode):
        return UnaryOpNode(node.op, _per_kernel(node.operand, partners))
    if isinstance(node, VariableNode):
        partner = partners.get(node.name.lstrip("$"))
        return VariableNode(f"${partner}") if partner else node
    return node


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
    render = _render(table)
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


def _summed_collectable_errors(
    row: Dict[str, Any],
    metrics: Dict[str, Dict[str, Any]],
    partners: Dict[str, Optional[str]],
    simple_box: bool,
) -> List[str]:
    """Report the summed collectables a row reads where it needs per-dispatch values.

    A simple-box formula shows as it is, so it may read none. Min, Max and Std Dev read
    a summed collectable's per-dispatch partner instead, so each needs one.
    """
    if "metric" not in row:
        return []
    _, names = parse_counters_text(metrics[row["metric"]]["formula"])
    reader = "the simple-box formula reads" if simple_box else "Min and Max read"
    return [
        f"{row['metric']}: {reader} ${name}, a sum over all dispatches; "
        f"{'use' if partners[name] else 'add'} ${name}{PER_DISPATCH_SUFFIX}"
        for name in sorted(names & partners.keys())
        if simple_box or partners[name] is None
    ]


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
