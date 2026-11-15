# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Check that the metric library and the analysis views match the analysis configs,
arch by arch.

Temporary: this test is deleted with the analysis configs. Until then, the library
copies the analysis configs:

- every equation of a panel row is a library formula, and every formula has a row;
- every row has the unit of its metric;
- every row peak is a peak of its metric, and every library peak has a row;
- the library collectables equal the analysis config collectables;
- each library set lists the metrics of its old set in profile_configs/sets.

And each view is one analysis config panel:

- the same panels, tables, headers and table settings, in order (table ids are
  positions in the views, so they are not compared);
- the same rows in order, compared by the metric ids they show (row_metric_id() finds
  a panel row's ids), leaving out only rows that show no equation (and tables left
  empty by them);
- each panel's -b alias is the config template's.

Row texts are not compared: a view row shows its label or its metric's library name.
Values, units, peaks and descriptions come from the metric library and are not compared.
The views leave out the panels in LEFT_OUT_PANELS.

The checks skip the roofline panel of an arch without a roofline benchmark (gfx908).
"""

import json
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import pytest
from validate_metric_library import (
    EMPIRICAL_PEAK_RE,
    SHARED_FORMULA_IDS,
    UNIT_ALIASES,
    arch_ceilings,
    canonical_formula,
)

import config
from rocprof_compute_soc.metric_library import loader
from rocprof_compute_soc.soc_base import iter_analysis_config_metrics
from utils.analysis_views import (
    MAIN_COLUMNS,
    NO_VALUE,
    TABLE_CONFIG_KEYS,
    views_to_panel_configs,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_common import (
    METRIC_TABLE,
    PLACEHOLDER_RANGE_KEY,
    ROOFLINE_PANEL_ID,
    canonical_config_arch,
    config_template_path_for_arch,
    format_row_id,
    iter_view_rows,
    load_panel_configs,
    load_views,
    load_yaml,
    panel_aliases,
    parse_sets_yaml,
    row_metric_ids,
)
from utils.utils_counter_defs import Collectables, metric_value_keys


class Row(NamedTuple):
    """One equation of an analysis config row."""

    legacy_id: str
    label: str
    equation: Optional[str]
    peak: Optional[str]
    unit: Optional[str]


# Row value columns read as fields, not as equations.
ROW_FIELDS = ("unit", "peak")
# Kernel time divisors in seconds (Bytes/s rows) and in nanoseconds (GB/s rows).
SECONDS_SCALE = "/(SUM(End_Timestamp-Start_Timestamp)/1e9)"
# A Bytes/s row can also scale a GB/s equation: 1e9*(<GB/s equation>)
BYTES_SCALE_PREFIX = "1e9*("
NANOSECONDS = "/SUM(End_Timestamp-Start_Timestamp)"
# Metric unit -> row unit when the row divides by seconds.
SECONDS_SCALE_UNITS = {"GB/s": "Bytes/s"}
# Roofline L2 bandwidth, with the bytes per request as a format field.
ROOFLINE_L2_BW_EQUATION = (
    "{} * SUM(TCP_TCC_WRITE_REQ_sum + TCP_TCC_ATOMIC_WITH_RET_REQ_sum "
    "+ TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum + TCP_TCC_READ_REQ_sum) "
    "/ SUM(End_Timestamp - Start_Timestamp)"
)
ROOFLINE_L2_BW = (
    "L2 Cache Bandwidth",
    "vl1d.l2_bw",
    ROOFLINE_L2_BW_EQUATION.format(64),
)
ROOFLINE_L2_BW_128B = (
    "L2 Cache Bandwidth",
    "vl1d.l2_bw",
    ROOFLINE_L2_BW_EQUATION.format(128),
)
ROOFLINE_LDS_BW = (
    "LDS Bandwidth",
    "lds.theoretical_bw",
    "SUM((SQ_LDS_IDX_ACTIVE - SQ_LDS_BANK_CONFLICT) * 4 * $lds_banks_per_cu) "
    "/ SUM(End_Timestamp - Start_Timestamp)",
)
PIPELINE_MFMA_UTIL = (
    "MFMA Utilization",
    "compute.mfma_util",
    "100 * SUM(SQ_VALU_MFMA_BUSY_CYCLES) "
    "/ (4 * SUM($cu_per_gpu * $GRBM_GUI_ACTIVE_PER_XCD))",
)
# Rows with another form of a metric formula (positional id -> label, metric id,
# row equation).
DISPLAY_VARIANTS = {
    "gfx90a": {
        "4.1.9": ROOFLINE_L2_BW,
        "4.1.11": ROOFLINE_LDS_BW,
        "11.2.7": PIPELINE_MFMA_UTIL,
    },
    "gfx942": {
        "4.1.10": ROOFLINE_L2_BW_128B,
        "4.1.12": ROOFLINE_LDS_BW,
        "11.2.7": PIPELINE_MFMA_UTIL,
    },
    "gfx950": {
        "2.1.23": (
            "L2-Fabric Read BW",
            "fabric.rd_bw",
            "SUM(128 * TCC_EA0_RDREQ_128B_sum + 64 * TCC_EA0_RDREQ_64B_sum "
            "+ 32 * TCC_EA0_RDREQ_32B_sum) / SUM(End_Timestamp - Start_Timestamp)",
        ),
        "4.1.12": ROOFLINE_L2_BW_128B,
        "4.1.14": ROOFLINE_LDS_BW,
        "11.2.9": PIPELINE_MFMA_UTIL,
    },
    "gfx115x": {
        "3.4.3": (
            "LDS Estimated Bandwidth",
            "lds.theoretical_bw",
            "32 * 4 * SUM(SQC_LDS_IDX_ACTIVE_sum - SQC_LDS_BANK_CONFLICT_sum)"
            " / (SUM(End_Timestamp - Start_Timestamp) / 1e9)",
        ),
        "4.1.0": (
            "VALU FLOPs",
            "compute.valu_flops",
            "SUM($wave_size * SQ_INSTS_VALU_sum)"
            " / SUM(End_Timestamp - Start_Timestamp)",
        ),
        "4.1.1": (
            "VALU FLOPs (F64)",
            "compute.valu_flops_f64",
            "SUM($wave_size * SQ_INSTS_VALU_DP_sum)"
            " / SUM(End_Timestamp - Start_Timestamp)",
        ),
    },
    "gfx1250": {
        "3.3.4": (
            "GL0 Hit Rate",
            "gl0.hit_rate",
            "SUM(TX_VMW_REQ_sum - TX_VMW_REQ_MISS_sum) / SUM(TX_VMW_REQ_sum) * 100",
        ),
    },
}
# Rows with a different unit from their metric (positional id -> label, unit).
CDNA_UNIT_VARIANTS = {"7.1.8": ("LDS Allocation", "Bytes")}
UNIT_VARIANTS = {
    arch: CDNA_UNIT_VARIANTS for arch in ("gfx908", "gfx90a", "gfx942", "gfx950")
}
UNIT_VARIANTS["gfx115x"] = {
    "3.4.0": ("LDS Req", "(Requests + $normUnit)"),
    "7.2.4": ("LDS Allocation", "Bytes"),
    "9.2.0": ("Total Requests", "(Count + $normUnit)"),
    "9.2.1": ("Read Requests", "(Count + $normUnit)"),
    "9.2.2": ("Write Requests", "(Count + $normUnit)"),
    "9.4.0": ("GL1 Read Requests", "(Count + $normUnit)"),
    "9.4.2": ("GL1 Write Requests", "(Count + $normUnit)"),
    "11.2.0": ("Total Requests", "(Count + $normUnit)"),
    "11.2.1": ("Read Requests", "(Count + $normUnit)"),
    "11.2.2": ("Write Requests", "(Count + $normUnit)"),
    "11.4.0": ("GL2 Read Requests", "(Count + $normUnit)"),
    "11.4.1": ("GL2 Read 128B Requests", "(Count + $normUnit)"),
    "11.4.2": ("GL2 Write Requests", "(Count + $normUnit)"),
}
UNIT_VARIANTS["gfx1201"] = {
    "2.1.6": ("LDS Bank Conflicts", "Cycles"),
    "3.4.0": ("LDS Req", "(Requests + $normUnit)"),
    "3.9.0": ("Memory Read Requests", "(Requests + $normUnit)"),
    "3.9.1": ("Memory Write Requests", "(Requests + $normUnit)"),
    "7.2.4": ("LDS Allocation", "Bytes"),
    "9.2.0": ("Total Requests", "(Count + $normUnit)"),
    "11.3.0": ("Total Requests", "(Count + $normUnit)"),
}
UNIT_VARIANTS["gfx1250"] = {
    "2.1.17": ("Active WGPs", "WGPs per Cycle"),
    "3.4.0": ("LDS Load Requests", "(Requests + $normUnit)"),
    "3.4.1": ("LDS Store Requests", "(Requests + $normUnit)"),
    "3.4.2": ("LDS Atomic Requests", "(Requests + $normUnit)"),
    "3.4.3": ("Async Load Requests", "(Requests + $normUnit)"),
    "3.4.4": ("Async Store Requests", "(Requests + $normUnit)"),
    "3.4.14": ("TDM Load Requests", "(Requests + $normUnit)"),
    "3.4.15": ("TDM Store Requests", "(Requests + $normUnit)"),
    "6.2.0": ("Dispatched Workgroups", "Count"),
    "6.2.1": ("Dispatched Wavefronts", "Count"),
    "7.2.5": ("LDS Allocation", "Bytes"),
}
# Row columns that show no equation.
NON_EQUATION_COLUMNS = ("unit", "peak", "pct_of_peak")
# Analysis config panels with no view: the SDK has no PC sampling on gfx908.
LEFT_OUT_PANELS = {"gfx908": (2100,)}


def member_arch(arch: str) -> str:
    """Return a GPU arch with the analysis configs of arch (gfx1150 for gfx115x)."""
    return next(
        gpu_arch
        for gpu_arch in sorted(mi_gpu_specs.get_gpu_series_dict())
        if canonical_config_arch(gpu_arch) == arch
    )


def unshown_file_ids(arch: str) -> Set[str]:
    """Return the ids of analysis config files hidden on arch."""
    if member_arch(arch) in mi_gpu_specs.get_roofline_archs():
        return set()
    return {f"{ROOFLINE_PANEL_ID:04d}"}


def row_forms(table: Dict[str, Any], row: Dict[str, Any]) -> List[str]:
    """Return the columns of row with an equation."""
    main_forms = [form for form in MAIN_COLUMNS if form in row]
    if main_forms:
        return main_forms[:1]
    return [
        key for key in row if key in metric_value_keys(table) and key not in ROW_FIELDS
    ]


def read_rows(arch: str) -> List[Row]:
    """Return every row equation in the analysis configs of arch.

    The placeholder_range key of a per-channel table adds no row.
    """
    rows = []
    for item in iter_analysis_config_metrics(
        config.analysis_configs_dir / arch, unshown_file_ids(arch)
    ):
        legacy_id = format_row_id(item.table_id, item.metric_idx)
        rows.extend(
            Row(
                legacy_id,
                item.metric_name,
                canonical_formula(item.metric[form]),
                canonical_formula(item.metric.get("peak")),
                item.metric.get("unit"),
            )
            for form in row_forms(item.table, item.metric)
        )
    return rows


def in_seconds(row: Row) -> bool:
    """Return True if row is a Bytes/s row: it divides by the kernel time in
    seconds, or multiplies a GB/s equation by 1e9."""
    return row.equation is not None and (
        row.equation.endswith(SECONDS_SCALE)
        or row.equation.startswith(BYTES_SCALE_PREFIX)
    )


def gb_per_second(row: Row) -> Optional[str]:
    """Return the equation of row in GB/s, with the kernel time in nanoseconds."""
    if row.equation is not None and row.equation.endswith(SECONDS_SCALE):
        return row.equation[: -len(SECONDS_SCALE)] + NANOSECONDS
    if row.equation is not None and row.equation.startswith(BYTES_SCALE_PREFIX):
        return row.equation[len(BYTES_SCALE_PREFIX) : -1]
    return row.equation


def row_peak(row: Row, peak: str) -> str:
    """Return a GB/s library peak in the units of row.

    A Bytes/s row writes "<MHz term> / 1000" as "<MHz term> * 1e6", and a measured peak
    "$<ceiling>_empirical_peak" as "$<ceiling>_empirical_peak * 1e9".
    """
    if not in_seconds(row):
        return peak
    if peak.endswith("/1000"):
        return peak[: -len("/1000")] + "*1e6"
    return f"{peak}*1e9"


def row_metric_id(row: Row, library: loader.MetricLibrary, arch: str) -> Optional[str]:
    """Return the metric id of row, or None if no metric matches.

    A row with the name of a SHARED_FORMULA_IDS id as its label gets that id.
    """
    variants = DISPLAY_VARIANTS.get(arch, {})
    if row.legacy_id in variants:
        metric_id = variants[row.legacy_id][1]
        return metric_id if metric_id in library.metrics else None
    formulas = {
        canonical_formula(metric["formula"]): metric_id
        for metric_id, metric in library.metrics.items()
        if metric_id not in SHARED_FORMULA_IDS
    }
    names = {
        metric["name"].casefold(): metric_id
        for metric_id, metric in library.metrics.items()
    }
    metric_id = formulas.get(gb_per_second(row))
    named_id = names.get(row.label.casefold())
    if metric_id is not None and SHARED_FORMULA_IDS.get(named_id) == metric_id:
        return named_id
    return metric_id


def old_sets_by_metric_id(arch_view: SimpleNamespace) -> Dict[str, Dict[str, Any]]:
    """Return each old set of the arch, with the metric ids of the view rows its
    numeric ids select."""
    rows = dict(iter_view_rows(load_views(arch_view.arch)))
    old_sets = {}
    for set_option, entry in parse_sets_yaml(member_arch(arch_view.arch)).items():
        metric_ids = [
            metric_id
            for item in entry["metric"]
            for position in item
            for metric_id in row_metric_ids(rows[str(position)])
        ]
        old_sets[set_option] = {"title": entry["title"], "metrics": metric_ids}
    return old_sets


def load_arch_view(arch: str) -> SimpleNamespace:
    """Return one arch's library, its analysis config rows and the metric id of each
    row."""
    library = loader.MetricLibrary.load(loader.default_metric_library_dir(), arch)
    rows = read_rows(arch)
    return SimpleNamespace(
        arch=arch,
        library=library,
        rows=rows,
        metric_ids=[row_metric_id(row, library, arch) for row in rows],
    )


def structure_errors(
    views: Dict[int, Dict[str, Any]], arch_view: SimpleNamespace
) -> List[str]:
    """Return how one arch's views differ from its analysis configs' structure."""
    arch = arch_view.arch
    expected = load_panel_configs([str(config.analysis_configs_dir / arch)])
    for panel_id in LEFT_OUT_PANELS.get(arch, ()):
        del expected[panel_id]
    actual = views_to_panel_configs(views, arch_view.library.metrics, {})
    for file_id in unshown_file_ids(arch):
        expected.pop(int(file_id), None)
        actual.pop(int(file_id), None)
    if list(expected) != list(actual):
        return [f"panels: analysis configs have {list(expected)}, views {list(actual)}"]
    row_ids = panel_row_metric_ids(arch_view)
    errors = []
    for panel_id, panel in expected.items():
        errors.extend(
            f"panel {panel_id}: {error}"
            for error in panel_errors(panel, actual[panel_id], views[panel_id], row_ids)
        )
    errors.extend(alias_errors(views, arch))
    return errors


def panel_row_metric_ids(arch_view: SimpleNamespace) -> Dict[str, List[str]]:
    """Return the metric ids each shown analysis config row shows, by row id."""
    row_ids: Dict[str, List[str]] = {}
    for row, metric_id in zip(arch_view.rows, arch_view.metric_ids):
        row_ids.setdefault(row.legacy_id, [])
        if metric_id is not None:
            row_ids[row.legacy_id].append(metric_id)
    return row_ids


def panel_errors(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    view: Dict[str, Any],
    row_ids: Dict[str, List[str]],
) -> List[str]:
    """Return how a view's panel config differs from its analysis config panel."""
    if expected["title"] != actual["title"]:
        return [f"title {actual['title']!r}, analysis configs {expected['title']!r}"]
    kept = [
        source
        for source in expected["data source"]
        if not left_out_table(*table_of_source(source))
    ]
    kinds = [list(source) for source in kept]
    if kinds != [list(source) for source in actual["data source"]]:
        return ["tables differ from the analysis configs' tables with rows"]
    errors = []
    for want, got, table in zip(kept, actual["data source"], view["tables"]):
        errors.extend(
            table_errors(
                table_of_source(want)[1], table_of_source(got)[1], table, row_ids
            )
        )
    return errors


def table_of_source(source: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return the render type and the table of a panel's data source entry."""
    ((render, table),) = source.items()
    return render, table


def left_out_table(render: str, table: Dict[str, Any]) -> bool:
    """Return whether a panel table is left out of the views: rows, none showing one."""
    rows = table_rows(table) if render == METRIC_TABLE else {}
    return bool(rows) and not any(map(shows_equation, rows.values()))


def table_errors(
    want: Dict[str, Any],
    got: Dict[str, Any],
    view_table: Dict[str, Any],
    row_ids: Dict[str, List[str]],
) -> List[str]:
    """Return how a view's table differs from its analysis config table."""
    where = f"table {want['id']}"
    errors = [
        f"{where}: {key} {got.get(key)!r}, analysis configs {want.get(key)!r}"
        for key in TABLE_CONFIG_KEYS
        if json.dumps(want.get(key)) != json.dumps(got.get(key))
    ]
    shown = [
        ids
        for name, row, ids in rows_with_metric_ids(want, row_ids)
        if shows_equation(row)
    ]
    got_ids = [row_metric_ids(row) for row in view_table.get("metrics", [])]
    if got_ids != shown:
        errors.append(f"{where}: rows {got_ids}, analysis configs {shown}")
    placeholder = (want.get("metric") or {}).get(PLACEHOLDER_RANGE_KEY)
    if placeholder != (got.get("metric") or {}).get(PLACEHOLDER_RANGE_KEY):
        errors.append(f"{where}: {PLACEHOLDER_RANGE_KEY} differs")
    return errors


def table_rows(table: Dict[str, Any]) -> Dict[str, Any]:
    """Return a panel table's rows without its placeholder range."""
    rows = table.get("metric") or {}
    return {name: row for name, row in rows.items() if name != PLACEHOLDER_RANGE_KEY}


def rows_with_metric_ids(
    table: Dict[str, Any], row_ids: Dict[str, List[str]]
) -> List[Tuple[str, Dict[str, Any], List[str]]]:
    """Return each row of a panel table with its metric ids; a row's id counts the
    placeholder range key, as read_rows() does."""
    rows = table.get("metric") or {}
    return [
        (name, row, row_ids.get(format_row_id(table["id"], index), []))
        for index, (name, row) in enumerate(rows.items())
        if name != PLACEHOLDER_RANGE_KEY
    ]


def shows_equation(row: Dict[str, Any]) -> bool:
    """Return whether a panel row shows an equation in any column."""
    return any(
        str(value) != NO_VALUE
        for column, value in row.items()
        if column not in NON_EQUATION_COLUMNS
    )


def alias_errors(views: Dict[int, Dict[str, Any]], arch: str) -> List[str]:
    """Return each panel whose -b alias differs from the config template's."""
    template_path = config_template_path_for_arch(arch)
    template = load_yaml(template_path) if template_path else None
    template_aliases = {
        str(panel["panel_id"]): (panel.get("panel_alias") or "")
        for panel in (template or {}).get("panels") or []
    }
    return [
        f"panel {panel_number}: -b alias {alias!r}, config template has "
        f"{template_aliases.get(panel_number)!r}"
        for panel_number, alias in panel_aliases(views).items()
        if template_aliases.get(panel_number) != alias
    ]


def swap_rows(views: Dict[int, Dict[str, Any]]) -> None:
    table = views[1100]["tables"][1]
    table["metrics"][0], table["metrics"][1] = table["metrics"][1], table["metrics"][0]


def drop_row(views: Dict[int, Dict[str, Any]]) -> None:
    del views[1100]["tables"][1]["metrics"][2]


def drop_column(views: Dict[int, Dict[str, Any]]) -> None:
    del views[1100]["tables"][1]["header"]["max"]


def change_alias(views: Dict[int, Dict[str, Any]]) -> None:
    views[1100]["alias"] = "pipeline"


@pytest.fixture(scope="module", params=list(loader.METRIC_LIBRARY_PARENT_ARCH))
def arch_view(request) -> SimpleNamespace:
    """One arch's library, its analysis config rows and the metric id of each row."""
    return load_arch_view(request.param)


def test_every_row_shows_a_library_formula(arch_view):
    variants = DISPLAY_VARIANTS.get(arch_view.arch, {})
    errors = []
    for row, metric_id in zip(arch_view.rows, arch_view.metric_ids):
        if row.legacy_id in variants:
            label, variant_id, equation = variants[row.legacy_id]
            if row.label != label or variant_id not in arch_view.library.metrics:
                errors.append(f"{row.legacy_id}: update DISPLAY_VARIANTS")
            elif row.equation != canonical_formula(equation):
                errors.append(f"{row.legacy_id}: not the listed form of {variant_id}")
        elif row.equation is not None and metric_id is None:
            errors.append(f"{row.legacy_id} {row.label}: not a library formula")
    assert errors == []


def test_every_library_formula_is_shown(arch_view):
    shown = {
        metric_id
        for row, metric_id in zip(arch_view.rows, arch_view.metric_ids)
        if row.equation is not None
    }
    assert sorted(set(arch_view.library.metrics) - shown) == []


def test_row_units_are_metric_units(arch_view):
    unit_variants = UNIT_VARIANTS.get(arch_view.arch, {})
    errors = []
    for row, metric_id in zip(arch_view.rows, arch_view.metric_ids):
        if row.unit is None or metric_id is None:
            continue
        metric_unit = arch_view.library.metrics[metric_id]["unit"]
        label, unit = unit_variants.get(row.legacy_id, (row.label, metric_unit))
        if row.legacy_id in unit_variants and unit == metric_unit:
            errors.append(f"{row.legacy_id}: remove it from UNIT_VARIANTS")
        elif row.label != label:
            errors.append(f"{row.legacy_id} is now {row.label!r}; update UNIT_VARIANTS")
        else:
            if in_seconds(row):
                unit = SECONDS_SCALE_UNITS.get(unit, unit)
            if UNIT_ALIASES.get(row.unit, row.unit) != unit:
                errors.append(f"{row.legacy_id}: {row.unit!r} is not {unit!r}")
    assert errors == []


def test_row_peaks_are_metric_peaks(arch_view):
    """Skip measured row peaks with an unmeasured ceiling on the arch; an unknown
    ceiling still fails."""
    unmeasured = mi_gpu_specs.get_all_roofline_columns() - arch_ceilings(arch_view.arch)
    errors = []
    shown_fields = defaultdict(set)
    for row, metric_id in zip(arch_view.rows, arch_view.metric_ids):
        if row.peak is None or metric_id is None:
            continue
        ceiling = EMPIRICAL_PEAK_RE.search(row.peak)
        if ceiling and ceiling.group(1) in unmeasured:
            continue
        metric = arch_view.library.metrics[metric_id]
        fields = {
            field
            for field in ("peak", "peak_empirical")
            if field in metric
            and row.peak == row_peak(row, canonical_formula(metric[field]))
        }
        shown_fields[metric_id].update(fields)
        if not fields:
            errors.append(f"{row.legacy_id}: {row.peak} is no peak of {metric_id}")
    errors.extend(
        f"{metric_id}: no row shows its '{field}'"
        for metric_id, metric in arch_view.library.metrics.items()
        for field in ("peak", "peak_empirical")
        if field in metric and field not in shown_fields[metric_id]
    )
    assert errors == []


def test_listed_variants_have_rows(arch_view):
    positions = {row.legacy_id for row in arch_view.rows}
    listed = [
        *DISPLAY_VARIANTS.get(arch_view.arch, {}),
        *UNIT_VARIANTS.get(arch_view.arch, {}),
    ]
    assert sorted(set(listed) - positions) == []


def test_collectables_match_analysis_configs(arch_view):
    library_collectables = {
        name: canonical_formula(collectable["formula"])
        for name, collectable in arch_view.library.collectables.items()
    }
    expected = {
        name: canonical_formula(formula)
        for name, formula in Collectables.of_arch(arch_view.arch).items()
    }
    assert library_collectables == expected


def test_sets_match_profile_configs(arch_view):
    """Each library set has the title of its old set and the metrics of its rows."""
    assert arch_view.library.sets == old_sets_by_metric_id(arch_view)


def test_views_keep_the_panel_structure(arch_view):
    assert structure_errors(load_views(arch_view.arch), arch_view) == []


@pytest.mark.parametrize(
    "change, error",
    [
        (swap_rows, "panel 1100: table 1102: rows"),
        (drop_row, "panel 1100: table 1102: rows"),
        (drop_column, "panel 1100: table 1102: header"),
        (change_alias, "panel 11: -b alias 'pipeline', config template has 'cu_pipe'"),
    ],
)
def test_a_changed_structure_is_reported(change, error):
    """gfx942's views with one change no longer keep its panels' structure."""
    views = load_views("gfx942")
    change(views)
    errors = structure_errors(views, load_arch_view("gfx942"))
    assert any(e.startswith(error) for e in errors), errors
