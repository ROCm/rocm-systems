# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Assemble the interactive standalone roofline HTML document."""

import functools
import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

from roofline.roofline_frame import (
    FRAME_MIN_DECADES,
    FRAME_PAD,
    FRAME_SLOPE_SKEW,
)
from roofline.roofline_hover import KERNEL_NAME_FONT_FAMILY

ALL_PEAKS_VALUE = "all"

ROOF_EXTRAP_MIN_AI = 1e-150
ROOF_EXTRAP_MAX_AI = 1e150

_PLOT_DIV_ID = "roofline-plot"


@functools.lru_cache(maxsize=None)
def _read_asset(name: str) -> str:
    """Read a bundled asset file, cached for repeated calls in one run."""
    return (Path(__file__).parent / "assets" / name).read_text(encoding="utf-8")


def _json_safe(value: object) -> object:
    """Replace non-finite floats with None for browser-safe JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class RooflineViewModel:
    """JSON model embedded in the page for the client-side controller."""

    peaks: List[str] = field(default_factory=list)
    peak_colors: Dict[str, str] = field(default_factory=dict)
    default_peak: Optional[str] = None
    kernels: List[Dict[str, Any]] = field(default_factory=list)
    kernel_trace_indices: List[int] = field(default_factory=list)
    roofline_traces: List[Dict[str, Any]] = field(default_factory=list)
    compute_traces: List[Dict[str, Any]] = field(default_factory=list)
    compute_overlay_traces: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize the model for embedding in a <script> tag."""
        payload = {
            "divId": _PLOT_DIV_ID,
            "peaks": self.peaks,
            "peakColors": self.peak_colors,
            "defaultPeak": self.default_peak,
            "kernels": self.kernels,
            "kernelTraceIndices": self.kernel_trace_indices,
            "rooflineTraces": self.roofline_traces,
            "computeTraces": self.compute_traces,
            "computeOverlayTraces": self.compute_overlay_traces,
            "roofExtremeMaxAi": ROOF_EXTRAP_MAX_AI,
            "allPeaksValue": ALL_PEAKS_VALUE,
            "framePad": FRAME_PAD,
            "frameMinDecades": FRAME_MIN_DECADES,
            "frameSlopeSkew": FRAME_SLOPE_SKEW,
            "kernelNameFontFamily": KERNEL_NAME_FONT_FAMILY,
        }
        return json.dumps(_json_safe(payload), allow_nan=False).replace("</", "<\\/")


def build_interactive_document(
    figure: go.Figure,
    view_model: RooflineViewModel,
    title: str = "Empirical Roofline Analysis",
) -> str:
    """Build a fully self-contained interactive roofline HTML document."""
    figure.update_layout(showlegend=False)
    fragment = figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=_PLOT_DIV_ID,
        config={
            "displayModeBar": False,
            "responsive": True,
            "scrollZoom": True,
            "doubleClick": False,
        },
    )

    page_template = Template(_read_asset("roofline_plot.html"))
    return page_template.substitute(
        TITLE=html.escape(title),
        PEAK_TITLE=html.escape(
            "Plot each kernel at its arithmetic intensity for this memory level, "
            "matching the (AI axis) marker in the Bandwidth rooflines panel. "
            "All peaks plots every level at once."
        ),
        RUNTIME_TITLE=html.escape(
            "Show only the heaviest kernels whose combined percent of GPU "
            "resident time reaches this cutoff. The rightmost stop shows every "
            "plotted kernel."
        ),
        CSS=_read_asset("roofline_plot.css"),
        PLOT_FRAGMENT=fragment,
        MODEL_JSON=view_model.to_json(),
        JS=_read_asset("roofline_plot.js"),
    )
