# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Build the self-contained memory chart page from evaluated chart views."""

from pathlib import Path
from typing import Dict, Optional

from memory_chart.html.views import ChartView, ChartViewSet, MembwDetail, SlotValue
from utils.html_report import document as html_report

MODEL_ID = "mem-chart-model"
_ASSETS_DIR = Path(__file__).parent / "assets"


def build_mem_chart_document(
    views: ChartViewSet,
    *,
    heading: str,
    scope_note: str,
) -> str:
    """Embed the diagram and all evaluated views in one offline HTML document."""
    model = {
        "title": "Memory Chart",
        "heading": heading,
        "arch": views.arch,
        "scopeNote": scope_note,
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
