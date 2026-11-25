# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/analysis_views.py: the panel config a view becomes (its
structure and the values each row gives), the errors for broken views, and the profile
mode imports.
"""

import copy
import importlib
import sys
from typing import Any, Dict, Tuple

from conftest import ProfileModeImportGuard

from rocprof_compute_soc.metric_library import loader
from utils.analysis_views import (
    view_metric_errors,
    view_schema_errors,
    views_to_panel_configs,
)
from utils.utils_common import (
    get_arch_alias_to_panel_id,
    iter_view_rows,
    load_views,
    metric_ids_by_name,
)

ARCH = "gfx942"
# gfx942's compute.ipc_issued per kernel: the per-dispatch partners of its summed
# collectables.
IPC_ISSUED_PER_KERNEL = (
    "$ipc_valu_vmem_salu_smem_per_dispatch"
    " + $ipc_branch_sendmsg_vskipped_lds_per_dispatch"
)
# A view with one mistake of each kind the shape check reports.
BROKEN_VIEW = {
    "view": "Compute Units - Compute Pipeline",
    "archs": ["gfx942"],
    "tables": [
        {
            "title": "Compute Speed-of-Light",
            "render": "chart",
        },
        {
            "title": "Pipeline Statistics",
            "header": {"metric": "Metric", "avg": "Avg", "unit": "Unit"},
            "metrics": [
                {"metric": "compute.ipc", "columns": {"avg": "compute.ipc"}},
                {"columns": {"avg": "compute.ipc"}},
                {"metric": "compute.salu_util", "values": {"xfer": "HBM"}},
                {"metric": "compute.vmem_util", "peak": "theoretical"},
                {"metric": "compute.mfma_util", "unit": "Percent"},
            ],
        },
    ],
}
# A view with a CSV table and a per-channel metric table, and the library metrics
# it shows.
TOY_VIEW = {
    "view": "L2 Cache (per Channel)",
    "alias": "l2_per_channel",
    "tables": [
        {
            "render": "raw_csv_table",
            "title": "System Info",
            "source": "sysinfo.csv",
            "columnwise": True,
        },
        {
            "title": "L2 Cache Hit Rate (Percent)",
            "header": {"metric": "Metric", "expr": "Expression"},
            "placeholder_range": {"::_1": "$total_l2_chan"},
            "metrics": [
                {"metric": "l2.chan_hit_rate", "label": "Channel ::_1"},
                {"metric": "l2.chan_util"},
            ],
        },
    ],
}
TOY_METRICS = {
    "l2.chan_hit_rate": {
        "name": "L2 Hit Rate",
        "formula": "TCC_HIT[::_1]",
        "unit": "Percent",
        "description": "Hit rate of one channel.",
    },
    "l2.chan_util": {
        "name": "L2 Utilization",
        "formula": "TCC_BUSY[::_1]",
        "unit": "Percent",
        "description": "Busy cycles of one channel.",
    },
}


def first_row_with(
    views: Dict[int, Dict[str, Any]], field: str
) -> Tuple[str, Dict[str, Any]]:
    """Return the numeric id and the view row of the first row that has field."""
    return next(
        (position, row) for position, row in iter_view_rows(views) if field in row
    )


def metric_rows(arch):
    """Return the arch's views, metrics, and panel config rows by numeric id."""
    views = load_views(arch)
    library = loader.MetricLibrary.load(loader.default_metric_library_dir(), arch)
    metrics = library.metrics
    rows = {}
    panels = views_to_panel_configs(views, metrics, library.collectable_formulas())
    for panel in panels.values():
        for source in panel["data source"]:
            table = source.get("metric_table")
            if not table:
                continue
            for name, metric_id in metric_ids_by_name(table).items():
                rows[metric_id] = (name, table["metric"][name])
    return views, metrics, rows


def test_rows_show_their_metric_as_the_view_asks():
    """Each display field of a view row changes what its panel config row shows."""
    views, metrics, rows = metric_rows(ARCH)
    ipc_position = next(
        position
        for position, row in iter_view_rows(views)
        if row.get("metric") == "compute.ipc_issued"
    )
    ipc = metrics["compute.ipc_issued"]
    assert rows[ipc_position] == (
        ipc["name"],
        {
            "avg": ipc["formula"],
            "min": f"MIN({IPC_ISSUED_PER_KERNEL})",
            "max": f"MAX({IPC_ISSUED_PER_KERNEL})",
            "unit": ipc["unit"],
        },
    )
    position, row = first_row_with(views, "round")
    formula = metrics[row["metric"]]["formula"]
    assert rows[position][1]["value"] == f"ROUND({formula}, {row['round']})"
    position, row = first_row_with(views, "peak")
    assert rows[position][1]["peak"] == metrics[row["metric"]]["peak_empirical"]
    position, row = first_row_with(views, "pct_of_peak")
    assert rows[position][1]["pct_of_peak"] is row["pct_of_peak"] is False
    position, row = first_row_with(views, "values")
    assert rows[position][0] == row["label"]
    assert rows[position][1].items() >= row["values"].items()
    position, row = first_row_with(views, "columns")
    assert rows[position][1] == {
        column: metrics[metric_id]["formula"]
        for column, metric_id in row["columns"].items()
    }


def test_panel_configs_keep_the_view_structure():
    """A view becomes a panel config: panel id from the file prefix, table ids from
    their position, table settings as the view gives them, one row per view row
    named by its label or its metric's name, and each row's library description."""
    panels = views_to_panel_configs({1800: TOY_VIEW}, TOY_METRICS, {})

    assert panels == {
        1800: {
            "id": 1800,
            "title": "L2 Cache (per Channel)",
            "data source": [
                {
                    "raw_csv_table": {
                        "id": 1801,
                        "title": "System Info",
                        "source": "sysinfo.csv",
                        "columnwise": True,
                    }
                },
                {
                    "metric_table": {
                        "id": 1802,
                        "title": "L2 Cache Hit Rate (Percent)",
                        "header": {"metric": "Metric", "expr": "Expression"},
                        "metric": {
                            "Channel ::_1": {"expr": "TCC_HIT[::_1]"},
                            "L2 Utilization": {"expr": "TCC_BUSY[::_1]"},
                            "placeholder_range": {"::_1": "$total_l2_chan"},
                        },
                    }
                },
            ],
            "metrics_description": {
                "Channel ::_1": "Hit rate of one channel.",
                "L2 Utilization": "Busy cycles of one channel.",
            },
        }
    }


def test_view_schema_errors():
    """A broken view, a second view of its panel, a file name without a panel id, and
    a file for a built-in view."""
    source = "gfx942/1100_compute_pipeline.yaml"
    errors = view_schema_errors({
        source: BROKEN_VIEW,
        "gfx942/1100_copy.yaml": {"view": "Copy", "tables": []},
        "gfx942/compute.yaml": {"view": "Compute", "tables": []},
        "gfx942/0000_top_stats.yaml": {"view": "Top Stats", "tables": []},
    })
    assert errors == [
        "[gfx942/compute.yaml] the file name must be the panel id and the view name, "
        "e.g. 0300_memory_chart.yaml",
        "[gfx942/0000_top_stats.yaml] panel 0 is a built-in view",
        f"[{source}, gfx942/1100_copy.yaml] panel 1100 has more than one view file",
        f"[{source}] 'archs' is not a known field",
        f"[{source}] table 1: render 'chart' is not one of "
        "('metric_table', 'raw_csv_table', 'pc_sampling_table')",
        f"[{source}] table 2: row 0: needs one of 'metric' and 'columns'",
        f"[{source}] table 2: row 1: needs a 'label' when it has 'columns'",
        f"[{source}] table 2: row 2: 'values' names 'xfer', which is not a column",
        f"[{source}] table 2: row 3: 'peak' may only be 'empirical'",
        f"[{source}] table 2: row 4: 'unit' is not a known field",
    ]


def test_view_metric_errors():
    """The shipped gfx942 views, with a metric and a measured peak the library lacks,
    and a metric and a measured peak no row shows."""
    views = load_views(ARCH)
    library = loader.MetricLibrary.load(loader.default_metric_library_dir(), ARCH)
    assert view_metric_errors(views, library.metrics) == []
    broken = copy.deepcopy(views)
    rows = dict(iter_view_rows(broken))
    rows["11.2.0"]["metric"] = "compute.made_up"
    rows["2.1.11"]["peak"] = "empirical"
    del rows["4.1.0"]["peak"]
    rows["16.2.0"]["metric"] = "compute.ipc_issued"
    assert view_metric_errors(broken, library.metrics) == [
        "2.1.11 compute.branch_util has no 'peak_empirical'",
        "11.2.0 compute.made_up: not a metric of the arch's library",
        "vl1d.stalled_on_l2_data: no view shows it",
        "compute.valu_flops_f16: no row shows its 'peak_empirical'",
    ]


def test_an_arch_reads_only_the_aliases_of_its_own_views():
    """The alias of a view one arch has and another lacks is unknown on the other."""
    archs = list(loader.METRIC_LIBRARY_PARENT_ARCH)
    aliases = {
        arch: {
            view.get("alias"): panel_id for panel_id, view in load_views(arch).items()
        }
        for arch in archs
    }
    alias, panel_id, arch, other_arch = next(
        (alias, panel_id, arch, other_arch)
        for arch in archs
        for alias, panel_id in aliases[arch].items()
        if alias
        for other_arch in archs
        if alias not in aliases[other_arch]
    )
    assert get_arch_alias_to_panel_id(arch)[alias] == str(panel_id // 100)
    assert alias not in get_arch_alias_to_panel_id(other_arch)


def test_imports_only_profile_mode_dependencies(monkeypatch):
    """The views and the equation reader use only stdlib, project and vendored code.
    The re-import rebinds the package attributes, so they are restored too: later
    tests patch these modules by name."""
    for module in ("utils.analysis_views", "utils.metrics.expression"):
        package, name = module.rsplit(".", 1)
        monkeypatch.setattr(sys.modules[package], name, sys.modules[module])
        monkeypatch.delitem(sys.modules, module, raising=False)
    with ProfileModeImportGuard():
        importlib.import_module("utils.analysis_views")
