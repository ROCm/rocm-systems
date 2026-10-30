# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Build the self-contained memory chart page from evaluated chart views."""

import re
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional

from memory_chart.html.db_rows import read_workload_chart_rows
from memory_chart.html.views import (
    ChartScope,
    ChartView,
    ChartViewSet,
    MembwDetail,
    SlotValue,
    build_chart_views_from_db,
    chart_metric_ids,
    stored_chart_tables,
)
from utils import analysis_orm as orm
from utils import schema
from utils.html_report import report_shell as html_report
from utils.logger import console_log, console_warning
from utils.utils_common import normalize_filter_to_str_list

MODEL_ID = "mem-chart-model"
_ASSETS_DIR = Path(__file__).parent / "assets"

# The file is always device 0 whatever the filters, as roofline-gpu-0.html is
# device 0 (ROOFLINE_DEVICE_ID in utils/roofline_calc.py); this constant is separate
MEMORY_CHART_GPU_ID = 0


def memory_chart_html_name() -> str:
    """The file name of one workload's memory chart, memory-chart-gpu-<N>.html."""
    return f"memory-chart-gpu-{MEMORY_CHART_GPU_ID}.html"


def write_mem_chart_report_from_db(
    workload_row: orm.Workload,
    arch_config: schema.ArchConfig,
    args: Namespace,
    scope: ChartScope,
    output_path: Path,
) -> Optional[Path]:
    """Write the report from one workload's stored metric rows.

    The rows come from the analysis database the workload belongs to, and the
    scope is the run's filters. Returns None when the rows cannot reproduce the
    counters' page: a -k run, whose rows keep only the selected kernels; a run
    with no stored panel 3 table, as the counters path skips it; or no stored
    memory chart layout.
    """
    if scope.kernel_ids:
        console_log(
            "memory chart",
            "Skipping HTML report: -k runs store only the selected kernels",
        )
        return None

    rows = read_workload_chart_rows(workload_row, chart_metric_ids(arch_config))
    if not any(
        300 < table_id < 400 for table_id in stored_chart_tables(rows, arch_config)
    ):
        return None

    if rows.layout_json is None:
        console_log("memory chart", f"Skipping HTML report: no layout for {rows.arch}")
        return None

    views = build_chart_views_from_db(rows, arch_config, args, scope)
    return _write_report(views, scope, args.normal_unit, output_path)


def _write_report(
    views: ChartViewSet,
    scope: ChartScope,
    normalization: str,
    output_path: Path,
) -> Optional[Path]:
    """Write the offline page to output_path, creating its folder if needed."""
    document = build_mem_chart_document(
        views,
        heading="3. Memory Chart",
        scope_note=_scope_note(scope),
        normalization=normalization,
    )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document, encoding="utf-8")
    except OSError as error:
        console_warning(
            "memory chart", f"Failed to write HTML report {output_path}: {error}"
        )
        return None

    console_log("memory chart", f"Saved HTML report: {output_path}")
    return output_path


def build_mem_chart_document(
    views: ChartViewSet,
    *,
    heading: str,
    scope_note: str,
    normalization: str = "",
) -> str:
    """Embed the diagram and all evaluated views in one offline HTML document."""
    model = {
        "title": "Memory Chart",
        "heading": heading,
        "arch": views.arch,
        "scopeNote": scope_note,
        "normalization": normalization,
        "layout": views.diagram,
        "aggregate": _view_model(views.aggregate),
        "kernels": [
            {
                "index": entry.index,
                "name": entry.name,
                "view": _view_model(view),
            }
            for entry, view in zip(views.kernels, views.kernel_views)
        ],
        "initialKernel": views.initial_kernel,
    }

    return html_report.build_document(
        title=f"{heading} — {views.arch}",
        body_html=html_report.read_asset(_ASSETS_DIR, "mem_chart_body.html"),
        page_css=html_report.read_asset(_ASSETS_DIR, "mem_chart.css"),
        page_js=html_report.read_asset(_ASSETS_DIR, "mem_chart.js"),
        model_id=MODEL_ID,
        model=model,
    )


def _filter_ids(value: object) -> List[str]:
    """Return unique filter tokens in stable numeric order."""
    if value is None or (isinstance(value, (list, tuple, str)) and not value):
        return []

    tokens = {
        re.sub(r"^>\s*(\d+)$", r"gt\1", token.strip())
        for token in normalize_filter_to_str_list(value)
    }

    return sorted(
        tokens,
        key=lambda token: (
            (0, int(token), token) if token.isdecimal() else (1, token, token)
        ),
    )


def _scope_note(chart_scope: ChartScope) -> str:
    """Describe the filters that scope the aggregate view."""
    scope: List[str] = []
    gpu_ids = _filter_ids(chart_scope.gpu_ids)
    dispatch_ids = _filter_ids(chart_scope.dispatch_ids)

    if gpu_ids:
        scope.append(f"GPU {', '.join(gpu_ids)}")

    if dispatch_ids:
        dispatch_labels = [
            f"> {token[2:]}" if re.fullmatch(r"gt\d+", token) else token
            for token in dispatch_ids
        ]
        scope.append(f"dispatches {', '.join(dispatch_labels)}")

    return " · ".join(scope)


def _view_model(view: ChartView) -> Dict[str, object]:
    """Convert one frozen Python view to the page's JSON field names."""
    return {
        "key": view.key,
        "slots": {slot.slot_id: _slot_model(slot) for slot in view.slots},
        "tables": [
            {
                "id": table.table_id,
                "title": table.title,
                "columns": table.columns,
                "rows": table.rows,
            }
            for table in view.tables
        ],
        "membw": _membw_model(view.membw),
    }


def _slot_model(slot: SlotValue) -> Dict[str, object]:
    """Keep terminal text and the numeric value used by percentage bars."""
    percent = None
    if slot.bar is not None and slot.numeric is not None:
        percent = min(100, max(0, slot.numeric))

    return {
        "metric": slot.metric,
        "text": slot.text,
        "numeric": slot.numeric,
        "percent": percent,
        "bar": slot.bar,
        "unitLabel": slot.unit_label,
    }


def _membw_model(detail: Optional[MembwDetail]) -> Optional[Dict[str, object]]:
    """Keep selection-specific availability, annotations, and guidance."""
    if detail is None:
        return None

    return {
        "availability": detail.availability,
        "status": detail.status,
        "annotations": [
            {
                "blockId": annotation.block_id,
                "label": annotation.label,
                "value": annotation.value,
            }
            for annotation in detail.annotations
        ],
        "guidanceBlocks": detail.guidance_blocks,
    }
