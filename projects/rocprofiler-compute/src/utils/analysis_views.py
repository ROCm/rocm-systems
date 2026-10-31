# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Analysis views: the panels analyze shows, built from the metric library.

A view is one panel: its tables, and for each table row the metric it shows. Formulas,
units, peaks and descriptions come from the metric library; a view row stores only
what the panel shows differently: a label other than the metric's name, a display unit,
rounding, which peak, literal columns, a description that differs, or a display form
(DISPLAY_FORMS). views_to_panel_configs() turns an arch's views into the panel configs
that load_panel_configs() reads from analysis_configs/.

views/ holds the views of every arch, views/<family>/ (FAMILY_DIRS) and views/<arch>/
those of one family or arch, and an "archs" key narrows a view, a table or a row
further. A row's numeric id (11.2.3) is its position; "legacy_id" records the one
numeric id each metric keeps until numeric ids are retired.

Only tools use this module until the cutover; tools/validate_views.py checks the files.
Standard library and the vendored YAML only: profile mode imports it.
"""

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import config
from utils.equation_format import normalize_equation
from utils.metric_library import METRIC_LIBRARY_PARENT_ARCH, arch_chain
from utils.utils_common import canonical_config_arch
from vendored import yaml

# The directory of the views a metric library family shares, by the family's base arch.
FAMILY_DIRS = {"gfx908": "cdna"}
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
ROUND_RE = re.compile(r"^\s*ROUND\((.*),\s*\d+\s*\)\s*$", re.DOTALL)
# Metric unit -> the unit of a row that shows the metric per second, not per nanosecond.
SECONDS_SCALED_UNITS = {"GB/s": "Bytes/s"}
NANOSECONDS_DENOMINATOR = "/ SUM(End_Timestamp - Start_Timestamp)"
SECONDS_DENOMINATOR = "/ (SUM(End_Timestamp - Start_Timestamp) / 1e9)"
# Alias -> canonical unit. A display scale such as Bytes/s vs GB/s is not an alias.
UNIT_ALIASES = {
    "Gbps": "GB/s",
    "GFLOPs": "GFLOP/s",
    "GIOPs": "GIOP/s",
    "(Req + $normUnit)": "(Requests + $normUnit)",
    "Instr/cycle": "Instructions per Cycle",
    "Waves/cycle": "Wavefronts per Cycle",
    "Cycles/wave": "Cycles per Wavefront",
    "KB per Wave": "KB per Wavefront",
    "Wave": "Wavefronts",
    "Waves": "Wavefronts",
}
# (metric unit, row unit) pairs a row may show without converting the value: the
# aliases, and rows that name another unit until the metric owners settle it.
DISPLAY_UNITS = {
    *((canonical, alias) for alias, canonical in UNIT_ALIASES.items()),
    ("(Count + $normUnit)", "(Requests + $normUnit)"),
    ("(IOPS + $normUnit)", "(FLOPs + $normUnit)"),
    ("(Instructions + $normUnit)", "(Wavefronts + $normUnit)"),
    ("(Requests + $normUnit)", "(Count + $normUnit)"),
    ("(Requests + $normUnit)", "(Cycles + $normUnit)"),
    ("(Wavefronts + $normUnit)", "(Instructions + $normUnit)"),
    ("Bytes per Workgroup", "Bytes"),
    ("Count", "WGPs per Cycle"),
    ("Cycles", "Cycles per Request"),
    ("Wavefronts", "Count"),
    ("Workgroups", "Count"),
}
_L2_BW_EQUATION = (
    "{} * SUM(TCP_TCC_WRITE_REQ_sum + TCP_TCC_ATOMIC_WITH_RET_REQ_sum "
    "+ TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum + TCP_TCC_READ_REQ_sum) "
    "/ SUM(End_Timestamp - Start_Timestamp)"
)
# Forms a row may show instead of its metric's (a row's "display"), by name: the main
# column's formula ("formula"), from which the aggregate columns follow, or an
# aggregate column's own equation. Each is an equation a panel shows today, kept until
# the metric owners choose one form (AIPROFCOMP-820 FOLLOWUPS).
DISPLAY_FORMS: Dict[str, Dict[str, str]] = {
    # The roofline's L2 bandwidth at its bytes per request (64 up to gfx941, then 128).
    "roofline_l2_bw_64b": {"formula": _L2_BW_EQUATION.format(64)},
    "roofline_l2_bw_128b": {"formula": _L2_BW_EQUATION.format(128)},
    "roofline_lds_bw": {
        "formula": "SUM((SQ_LDS_IDX_ACTIVE - SQ_LDS_BANK_CONFLICT) * 4 "
        "* $lds_banks_per_cu) / SUM(End_Timestamp - Start_Timestamp)"
    },
    "pipeline_mfma_util": {
        "formula": "100 * SUM(SQ_VALU_MFMA_BUSY_CYCLES) "
        "/ (4 * SUM($cu_per_gpu * $GRBM_GUI_ACTIVE_PER_XCD))"
    },
    "noise_clamped_vl1d_hit_rate": {
        "formula": "100 * SUM(NOISE_CLAMP(TCP_TOTAL_CACHE_ACCESSES_sum "
        "- (TCP_TCC_READ_REQ_sum + TCP_TCC_WRITE_REQ_sum "
        "+ TCP_TCC_ATOMIC_WITH_RET_REQ_sum + TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum), "
        "TCP_TOTAL_CACHE_ACCESSES_sum)) / SUM(TCP_TOTAL_CACHE_ACCESSES_sum)"
    },
    "fabric_rd_bw_by_request_size": {
        "formula": "SUM(128 * TCC_EA0_RDREQ_128B_sum + 64 * TCC_EA0_RDREQ_64B_sum "
        "+ 32 * TCC_EA0_RDREQ_32B_sum) / SUM(End_Timestamp - Start_Timestamp)"
    },
    # gfx950 panel 3000 shows each latency as AVG/AVG as well as SUM/SUM: the counters
    # are summed across CUs (channels), so both give the same value.
    "avg_vl1d_average_lat": {
        "formula": "AVG(TCP_TCP_LATENCY_sum) / AVG(TCP_TOTAL_ACCESSES_sum)"
    },
    "avg_vl1d_l2_rd_average_lat": {
        "formula": "AVG(TCP_TCC_READ_REQ_LATENCY_sum) / AVG(TCP_TCC_READ_REQ_sum)"
    },
    "avg_vl1d_l2_wr_average_lat": {
        "formula": "AVG(TCP_TCC_WRITE_REQ_LATENCY_sum) / AVG(TCP_TCC_WRITE_REQ_sum)"
    },
    "avg_vl1d_utcl1_average_lat": {
        "formula": "AVG(TCP_CLIENT_UTCL1_INFLIGHT_sum) / AVG(TCP_UTCL1_REQUEST_sum)"
    },
    "avg_vl1d_utcl1_to_utcl2_average_lat": {
        "formula": "AVG(TCP_UTCL1_UTCL2_INFLIGHT_sum) "
        "/ AVG(TCP_UTCL1_TRANSLATION_MISS_sum)"
    },
    "avg_fabric_rd_lat": {
        "formula": "AVG(TCC_EA0_RDREQ_LEVEL_sum) / AVG(TCC_EA0_RDREQ_sum)"
    },
    "avg_fabric_wr_lat": {
        "formula": "AVG(TCC_EA0_WRREQ_LEVEL_sum) / AVG(TCC_EA0_WRREQ_sum)"
    },
    "avg_fabric_atomic_lat": {
        "formula": "AVG(TCC_EA0_ATOMIC_LEVEL_sum) / AVG(TCC_EA0_ATOMIC_sum)"
    },
    "l2_util_whole_channels": {
        "formula": "100 * SUM(TCC_BUSY_sum) "
        "/ (SUM($GRBM_GUI_ACTIVE_PER_XCD) * TO_INT($total_l2_chan))"
    },
    "valu_flops_per_valu_inst": {
        "formula": "SUM($wave_size * SQ_INSTS_VALU_sum) "
        "/ SUM(End_Timestamp - Start_Timestamp)"
    },
    "valu_flops_f64_per_dp_inst": {
        "formula": "SUM($wave_size * SQ_INSTS_VALU_DP_sum) "
        "/ SUM(End_Timestamp - Start_Timestamp)"
    },
    "gl0_hit_rate_times_100": {
        "formula": "SUM(TX_VMW_REQ_sum - TX_VMW_REQ_MISS_sum) / SUM(TX_VMW_REQ_sum) "
        "* 100"
    },
    # Aggregates that do not follow the row's formula.
    "spill_atomic_insts_max_as_min": {
        "max": "MIN(TA_BUFFER_ATOMIC_WAVEFRONTS_sum / $denom)"
    },
    "gl0_lookup_hit_rate_per_kernel_ratio": {
        "min": "100 * MIN(1 - TX_VMW_REQ_MISS_sum / TX_VMW_REQ_sum)",
        "max": "100 * MAX(1 - TX_VMW_REQ_MISS_sum / TX_VMW_REQ_sum)",
    },
    "gl0_sector_lookup_hit_rate_per_kernel_ratio": {
        "min": "100 * MIN(1 - TX_VMW_SECTOR_MISS_sum / TX_VMW_SECTORS_sum)",
        "max": "100 * MAX(1 - TX_VMW_SECTOR_MISS_sum / TX_VMW_SECTORS_sum)",
    },
}
VIEW_FIELDS = ("view", "legacy_panel_id", "alias", "archs", "tables")
TABLE_FIELDS = (
    "render",
    "id",
    "title",
    "archs",
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
    "archs",
    "unit",
    "round",
    "display",
    "unavailable",
    "peak",
    "pct_of_peak",
    "values",
    "description",
    "description_note",
    "legacy_id",
)
# The value of a column a row has no value for, as the panel configs write it.
NO_VALUE = "None"
# A row's "peak" that selects its metric's measured peak.
EMPIRICAL_PEAK = "empirical"


def default_views_dir() -> Path:
    """Return the views tree shipped with rocprof-compute."""
    return Path(config.rocprof_compute_home) / "rocprof_compute_soc" / "views"


def view_dirs(arch: str, views_dir: Path) -> List[Path]:
    """Return the directories that hold arch's views: shared, family, then arch.

    Args:
        arch: A GPU arch, e.g. "gfx942" or "gfx1151".
        views_dir: The views tree.

    Returns:
        The existing directories, each once.
    """
    config_arch = canonical_config_arch(arch)
    family_base = arch_chain(config_arch)[0]
    names = ["", FAMILY_DIRS.get(family_base, family_base), config_arch]
    dirs = list(OrderedDict.fromkeys(Path(views_dir) / name for name in names))
    return [path for path in dirs if path.is_dir()]


def read_view_files(arch: str, views_dir: Optional[Path] = None) -> Dict[Path, Any]:
    """Read every view file in arch's view directories, keyed by path, by file name."""
    paths = [
        path
        for directory in view_dirs(arch, views_dir or default_views_dir())
        for path in directory.glob("*.yaml")
    ]
    documents = {}
    for path in sorted(paths, key=lambda path: path.name):
        with open(path, encoding="utf-8") as file:
            documents[path] = yaml.safe_load(file)
    return documents


def load_views(arch: str, views_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return arch's views, without the tables and rows of other archs.

    Args:
        arch: A GPU arch.
        views_dir: The views tree; the shipped one by default.

    Returns:
        The views as select_arch_views() returns them.
    """
    return select_arch_views(list(read_view_files(arch, views_dir).values()), arch)


def select_arch_views(
    documents: List[Dict[str, Any]], arch: str
) -> List[Dict[str, Any]]:
    """Return the views of documents that serve arch, narrowed to arch.

    Tables and rows of other archs are left out, "archs" keys are dropped, and a
    legacy_id map becomes arch's id, or is dropped where arch has none.
    """
    config_arch = canonical_config_arch(arch)
    return [
        _select_view(document, config_arch)
        for document in documents
        if _serves(document, config_arch)
    ]


def views_to_panel_configs(
    views: List[Dict[str, Any]], metrics: Dict[str, Dict[str, Any]]
) -> "OrderedDict[int, Dict[str, Any]]":
    """Return the panel configs of an arch's views, as load_panel_configs() does.

    Args:
        views: One arch's views (select_arch_views()).
        metrics: The arch's metric library metrics (MetricLibrary.metrics).

    Returns:
        Each view's panel config, keyed by panel id, in panel id order.
    """
    configs = {view["legacy_panel_id"]: _panel_config(view, metrics) for view in views}
    return OrderedDict(sorted(configs.items()))


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


def view_schema_errors(documents: Dict[str, Any]) -> List[str]:
    """Return what is wrong with the shape of view documents, keyed by source name.

    Checks the fields each level may have, the known archs, render types, CLI styles
    and display forms, and that a row names its metric or its label.
    """
    errors = []
    for source, document in documents.items():
        errors.extend(f"[{source}] {error}" for error in _view_shape_errors(document))
    return errors


def view_metric_errors(
    views: List[Dict[str, Any]], metrics: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Return what in an arch's views the arch's metric library cannot show.

    Checks that panel ids are unique, every metric id exists, a row's unit is its
    metric's or a known display unit of it, and a measured peak exists.
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
# Selecting an arch
# ---------------------------------------------------------------------------


def _render(table: Dict[str, Any]) -> str:
    return table.get("render", METRIC_TABLE)


def _serves(item: Dict[str, Any], arch: str) -> bool:
    return "archs" not in item or arch in item["archs"]


def _select_view(document: Dict[str, Any], arch: str) -> Dict[str, Any]:
    view = {key: value for key, value in document.items() if key != "archs"}
    view["tables"] = [
        _select_table(table, arch)
        for table in document["tables"]
        if _serves(table, arch)
    ]
    return view


def _select_table(table: Dict[str, Any], arch: str) -> Dict[str, Any]:
    selected = {key: value for key, value in table.items() if key != "archs"}
    if "metrics" in table:
        selected["metrics"] = [
            _select_row(row, arch) for row in table["metrics"] if _serves(row, arch)
        ]
    return selected


def _select_row(row: Dict[str, Any], arch: str) -> Dict[str, Any]:
    selected = {key: value for key, value in row.items() if key != "archs"}
    legacy_id = row.get("legacy_id")
    if isinstance(legacy_id, dict):
        selected.pop("legacy_id")
        if arch in legacy_id:
            selected["legacy_id"] = legacy_id[arch]
    return selected


# ---------------------------------------------------------------------------
# Panel configs
# ---------------------------------------------------------------------------


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
        _row_label(row, metrics): _row_values(row, columns, metrics)
        for row in table["metrics"]
    }
    if "placeholder_range" in table:
        rows["placeholder_range"] = table["placeholder_range"]
    table_config["metric"] = rows
    return table_config


def _row_label(row: Dict[str, Any], metrics: Dict[str, Dict[str, Any]]) -> str:
    if "label" in row:
        return row["label"]
    return metrics[row["metric"]]["name"]


def _row_values(
    row: Dict[str, Any], columns: List[str], metrics: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Return a row's value in each column, in column order."""
    metric = metrics.get(row.get("metric"))
    unit = row.get("unit", metric["unit"] if metric else None)
    in_seconds = metric is not None and SECONDS_SCALED_UNITS.get(metric["unit"]) == unit
    main_column = next((column for column in MAIN_COLUMNS if column in columns), None)
    main = _main_formula(row, metric, in_seconds) if main_column else None
    form = DISPLAY_FORMS.get(row.get("display"), {})
    values = {}
    for column in columns:
        if column in row.get("values", {}):
            values[column] = row["values"][column]
        elif column in row.get("metrics", {}):
            values[column] = _column_formula(row["metrics"][column], metrics)
        elif column == main_column:
            values[column] = main
        elif column in AGGREGATE_COLUMNS:
            values[column] = form.get(column) or _aggregate(column, main)
        elif column == "unit":
            values[column] = unit
        elif column == "peak":
            values[column] = _peak(row, metric, in_seconds)
        elif column == "pct_of_peak":
            values[column] = row.get("pct_of_peak", True)
        else:
            raise ValueError(f"row {row} has no value for column {column!r}")
    return values


def _main_formula(
    row: Dict[str, Any], metric: Optional[Dict[str, Any]], in_seconds: bool
) -> str:
    """Return the formula a row shows in its main column."""
    if row.get("unavailable") or metric is None:
        return NO_VALUE
    formula = DISPLAY_FORMS.get(row.get("display"), {}).get("formula")
    formula = formula or metric["formula"]
    if in_seconds:
        formula = _per_second_formula(formula)
    if "round" in row:
        formula = f"ROUND({formula}, {row['round']})"
    return formula


def _column_formula(
    metric_id: Optional[str], metrics: Dict[str, Dict[str, Any]]
) -> str:
    return metrics[metric_id]["formula"] if metric_id else NO_VALUE


def _aggregate(column: str, main: str) -> str:
    """Return a column's aggregate over kernels of the main formula, per kernel."""
    if main == NO_VALUE:
        return NO_VALUE
    match = ROUND_RE.match(main)
    formula = match.group(1) if match else main
    return normalize_equation(f"{AGGREGATE_COLUMNS[column]}({_per_kernel(formula)})")


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


def _peak(
    row: Dict[str, Any], metric: Optional[Dict[str, Any]], in_seconds: bool
) -> Union[int, str]:
    """Return the peak a row shows: its metric's, or its own if it has no metric."""
    if metric is None:
        return row.get("peak", NO_VALUE)
    choice = row.get("peak", "peak")
    if choice is None:
        return NO_VALUE
    field = "peak_empirical" if choice == EMPIRICAL_PEAK else "peak"
    if field not in metric:
        return NO_VALUE
    return _per_second_peak(metric[field]) if in_seconds else metric[field]


def _per_second_formula(formula: str) -> str:
    if not formula.endswith(NANOSECONDS_DENOMINATOR):
        raise ValueError(f"{formula!r} does not divide by the kernel time")
    return formula[: -len(NANOSECONDS_DENOMINATOR)] + SECONDS_DENOMINATOR


def _per_second_peak(peak: Union[int, str]) -> str:
    """Return a GB/s peak in Bytes/s: "<MHz term> / 1000" as "<MHz term> * 1e6"."""
    text = str(peak)
    if text.endswith(" / 1000"):
        return text[: -len(" / 1000")] + " * 1e6"
    return f"{text} * 1e9"


def _panel_descriptions(
    view: Dict[str, Any], metrics: Dict[str, Dict[str, Any]]
) -> Dict[str, str]:
    """Return label -> description; the first row with a label describes it."""
    descriptions: Dict[str, str] = {}
    described = set()
    for table in view["tables"]:
        for row in table.get("metrics", []):
            label = _row_label(row, metrics)
            if label in described:
                continue
            described.add(label)
            description = _row_description(row, metrics.get(row.get("metric")))
            if description is not None:
                descriptions[label] = description
    return descriptions


def _row_description(
    row: Dict[str, Any], metric: Optional[Dict[str, Any]]
) -> Optional[str]:
    if "description" in row:
        return row["description"]
    if metric is None:
        return None
    if "description_note" in row:
        return f"{metric['description']} {row['description_note']}"
    return metric["description"]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _view_shape_errors(document: object) -> List[str]:
    if not isinstance(document, dict):
        return ["a view must be a mapping"]
    errors = _field_errors(document, VIEW_FIELDS, ("view", "legacy_panel_id", "tables"))
    errors.extend(_archs_errors(document))
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
    errors.extend(_archs_errors(table))
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
    errors.extend(_archs_errors(row))
    if "metric" in row and "metrics" in row:
        errors.append("has both 'metric' and 'metrics'")
    if "metric" not in row and "label" not in row:
        errors.append("needs a 'label' when it has no 'metric'")
    if "display" in row and row["display"] not in DISPLAY_FORMS:
        errors.append(f"display {row['display']!r} is not in DISPLAY_FORMS")
    errors.extend(
        f"'{field}' names {column!r}, which is not a column"
        for field in ("metrics", "values")
        for column in row.get(field, {})
        if column not in columns
    )
    errors.extend(_legacy_id_errors(row.get("legacy_id", "")))
    return errors


def _legacy_id_errors(legacy_id: object) -> List[str]:
    if isinstance(legacy_id, str):
        return []
    if not isinstance(legacy_id, dict):
        return ["'legacy_id' must be a string or a mapping of arch to string"]
    return _unknown_arch_errors(list(legacy_id), "legacy_id")


def _archs_errors(item: Dict[str, Any]) -> List[str]:
    if "archs" not in item:
        return []
    if not isinstance(item["archs"], list):
        return ["'archs' must be a list"]
    return _unknown_arch_errors(item["archs"], "archs")


def _unknown_arch_errors(archs: List[Any], field: str) -> List[str]:
    return [
        f"'{field}' names {arch!r}, which is not a metric library arch"
        for arch in archs
        if arch not in METRIC_LIBRARY_PARENT_ARCH
    ]


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
    if errors or metric is None:
        return errors
    if "unit" in row and not _is_display_unit(metric["unit"], row["unit"]):
        errors.append(
            f"unit {row['unit']!r} is not a display unit of {metric['unit']!r}"
        )
    if row.get("peak") == EMPIRICAL_PEAK and "peak_empirical" not in metric:
        errors.append(f"{row['metric']} has no 'peak_empirical'")
    return errors


def _is_display_unit(metric_unit: str, row_unit: str) -> bool:
    return (
        SECONDS_SCALED_UNITS.get(metric_unit) == row_unit
        or (metric_unit, row_unit) in DISPLAY_UNITS
    )
