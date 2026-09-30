# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Build the self-contained memory chart page from evaluated chart views."""

import hashlib
import re
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional

from memory_chart import loader
from memory_chart.html.views import (
    ChartView,
    ChartViewSet,
    MembwDetail,
    SlotValue,
    build_chart_views,
)
from memory_chart.mem_chart import format_mem_chart_heading
from utils import schema
from utils.html_report import document as html_report
from utils.logger import console_log, console_warning
from utils.utils_common import normalize_filter_to_str_list

MODEL_ID = "mem-chart-model"
_ASSETS_DIR = Path(__file__).parent / "assets"


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


def mem_chart_html_path(
    workload_dir: Path, gpu_ids: object, dispatch_ids: object
) -> Path:
    """Name a report for its effective GPU and dispatch filters."""
    gpu_tokens = _filter_ids(gpu_ids)
    dispatch_tokens = _filter_ids(dispatch_ids)
    gpu_group = f"_gpu-{'_'.join(gpu_tokens)}" if gpu_tokens else ""
    joined_dispatch = "_".join(dispatch_tokens)
    dispatch_group = f"_d-{joined_dispatch}" if dispatch_tokens else ""
    stem = f"mem_chart{gpu_group}{dispatch_group}"
    if len(stem) > 200 and dispatch_tokens:
        digest = hashlib.sha1(joined_dispatch.encode("utf-8")).hexdigest()[:8]
        dispatch_group = f"_d-{len(dispatch_tokens)}ids-{digest}"
        stem = f"mem_chart{gpu_group}{dispatch_group}"
    return workload_dir / f"{stem}.html"


def _scope_note(workload: schema.Workload) -> str:
    """Describe the filters that scope the aggregate view."""
    scope = ["All kernels"]
    gpu_ids = _filter_ids(workload.filter_gpu_ids)
    dispatch_ids = _filter_ids(workload.filter_dispatch_ids)
    if gpu_ids:
        scope.append(f"GPU {', '.join(gpu_ids)}")
    if dispatch_ids:
        dispatch_labels = [
            f"> {token[2:]}" if re.fullmatch(r"gt\d+", token) else token
            for token in dispatch_ids
        ]
        scope.append(f"dispatches {', '.join(dispatch_labels)}")
    return " · ".join(scope)


def write_mem_chart_report(
    workload: schema.Workload,
    arch_config: schema.ArchConfig,
    args: Namespace,
    workload_dir: Path,
) -> Optional[Path]:
    """Write one offline report when an analyzed memory chart is available."""
    if not any(300 < table_id < 400 for table_id in workload.dfs):
        return None
    arch = str(workload.sys_info.iloc[0]["gpu_arch"])
    if not loader.has_layout(arch):
        console_log("memory chart", f"Skipping HTML report: no layout for {arch}")
        return None

    views = build_chart_views(workload, arch_config, args)
    document = build_mem_chart_document(
        views,
        heading=format_mem_chart_heading(args.normal_unit, panel_id=300),
        scope_note=_scope_note(workload),
    )
    path = mem_chart_html_path(
        workload_dir, workload.filter_gpu_ids, workload.filter_dispatch_ids
    )
    try:
        path.write_text(document, encoding="utf-8")
    except OSError as error:
        console_warning("memory chart", f"Failed to write HTML report {path}: {error}")
        return None
    console_log("memory chart", f"Saved HTML report: {path}")
    return path


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
