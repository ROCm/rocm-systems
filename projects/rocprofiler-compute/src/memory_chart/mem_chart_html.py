# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Write the memory chart of an analyzed workload as a self-contained HTML page.

The page draws the layout file the terminal chart is drawn from, with the values
analyze shows in the terminal.
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from membw_analysis.models import MemBwAnalysisResult
from membw_analysis.summary import stalls_at, status_line
from memory_chart.loader import Layout, Layouts, layout_data
from memory_chart.render import LEGEND_ENTRIES, edge_segments
from memory_chart.units import (
    PERCENT,
    format_value,
    numeric,
    unit_of,
)
from utils.html_report.report_shell import build_page, truncate_kernel_name
from utils.utils_common import build_metric_details

MODEL_ID = "mem-chart-model"
_ASSETS_DIR = Path(__file__).parent / "assets"


def chart_view(
    gpu_arch: str,
    values: Mapping[str, Any],
    units: Mapping[str, str],
    membw: Optional[MemBwAnalysisResult],
) -> Dict[str, Any]:
    """What the page shows for one set of chart values.

    Each metric of the layout has its text as the terminal chart shows it, and
    its number for a bar. Stalls are listed by block, and the membw status line
    or guidance follows the chart, as in the terminal.
    """
    layout = _layout(gpu_arch)
    status = status_line(membw) if membw is not None else None
    return {
        "values": {
            metric_id: {
                "text": format_value(values.get(metric_id), unit_of(units, metric_id)),
                "number": numeric(values.get(metric_id)),
            }
            for metric_id in sorted(layout.metrics())
        },
        "stalls": {
            block.id: [
                {
                    "label": node.label,
                    "text": format_value(value, PERCENT),
                    "number": numeric(value),
                    "guidance": _guidance_index(membw, node.guidance_id),
                }
                for node, value in stalls_at(membw, block.stall_level)
            ]
            for block in layout.blocks()
            if block.stall_level
        },
        "status": status,
        "guidance": list(membw.guidance_blocks)
        if membw is not None and status is None
        else [],
    }


def mem_chart_document(
    gpu_arch: str,
    heading: str,
    panel: Dict[str, Any],
    aggregate: Dict[str, Any],
    kernels: Sequence[Tuple[str, Dict[str, Any]]] = (),
) -> str:
    """The page for the chart of an arch: the aggregate view, and a view per kernel
    as (kernel name, view) in kernel list order.

    Each metric on the page carries its panel config name and description.
    """
    details = build_metric_details({panel["id"]: panel}, None)
    model: Dict[str, Any] = {
        "heading": heading,
        "metrics": {
            metric_id: {
                "name": details[metric_id][0],
                "description": details[metric_id][2],
            }
            for metric_id in sorted(_layout(gpu_arch).metrics())
            if metric_id in details
        },
        "layout": layout_data(gpu_arch),
        "edges": _edges(gpu_arch),
        "legend": [
            {"category": category, "label": label}
            for _, label, category in LEGEND_ENTRIES
        ],
        "views": {
            "aggregate": aggregate,
            "kernels": [
                {"name": name, "label": truncate_kernel_name(name), "view": view}
                for name, view in kernels
            ],
        },
    }
    return build_page(
        _ASSETS_DIR, "mem_chart", title=heading, model_id=MODEL_ID, model=model
    )


def _layout(gpu_arch: str) -> Layout:
    layout = Layouts.for_arch(gpu_arch)
    if layout is None:
        raise ValueError(f"{gpu_arch} has no memory chart layout")
    return layout


def _guidance_index(
    membw: Optional[MemBwAnalysisResult], guidance_id: Optional[str]
) -> Optional[int]:
    """Position of the guidance block for a guidance id, or None without one."""
    block_ids = membw.guidance_block_ids if membw is not None else ()
    return block_ids.index(guidance_id) if guidance_id in block_ids else None


def _edges(gpu_arch: str) -> List[List[Dict[str, Any]]]:
    """The arrows of each edge as the terminal chart groups them: by the block they
    line up with (None: centred), as indexes into the layout file's arrows."""
    layout = _layout(gpu_arch)
    index = {id(arrow): i for i, arrow in enumerate(layout.arrows)}
    return [
        [
            {"anchor": anchor, "arrows": [index[id(arrow)] for arrow in arrows]}
            for anchor, arrows in segments.items()
        ]
        for segments in edge_segments(layout)
    ]
