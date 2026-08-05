# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Assemble the interactive standalone roofline HTML document.

Wraps the roofline plotly figure in a self-contained page whose controls are
all driven by a single embedded JSON model:
* a memory-peak dropdown,
* a click-to-isolate kernel panel with a cumulative-runtime filter, and
* a bandwidth-roofline panel with click-to-isolate roofs.
"""

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

KERNEL_NAME_FONT_FAMILY = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

ALL_PEAKS_VALUE = "all"

ROOF_EXTRAP_MAX_AI = 1e150

# Id Plotly renders the graph div under and the controller finds it by.
_PLOT_DIV_ID = "roofline-plot"


@functools.lru_cache(maxsize=None)
def _read_asset(name: str) -> str:
    """Read a bundled asset (HTML/CSS/JS) inlined into the document.

    Cached because both the Ops and Flops documents are written in one run and
    the assets never change at runtime.
    """
    return (Path(__file__).parent / "assets" / name).read_text(encoding="utf-8")


def _json_safe(value: object) -> object:
    """Recursively replace non-finite floats with None.
    This is to ensure that the browser can parse the embedded model.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class RooflineViewModel:
    """Client-facing description of the interactive roofline.

    Attributes:
        peaks: Ordered memory levels that have at least one point (e.g.
            ["L1", "L2", "HBM", "LDS"]). Empty on a roofs-only figure.
        peak_colors: Map from memory level to its roof color, used to color an
            isolated kernel's dots and every roofline panel row by memory level.
            Covers every level drawn, whether or not a kernel reaches it.
        default_peak: Memory level shown on load, or ALL_PEAKS_VALUE for every
            level at once.
        kernels: One entry per plotted kernel, in the same order as
            kernel_trace_indices:
            {"name", "color", "pctRuntime",
            "points": [{"peak", "ai", "perf", "hoverCells"}]}.
            This carries only what the client itself needs: pctRuntime (percent
            of total runtime) orders the kernel panel and drives the runtime
            filter, and hoverCells holds the two per-point tooltip values
            (peak throughput and percent of roofline, preformatted). Everything
            else a tooltip shows is constant across a kernel's points and so is
            baked into that trace's hover template instead of restated here.
            pctRuntime is None when the underlying data is missing.
        kernel_trace_indices: Indices into figure.data of the per-kernel
            scatter traces, in the same order as kernels.
        roofline_traces: Bandwidth-roof (memory-level) line traces, each
            {"level", "traceIndex", "bandwidth", "kneeAi", "kneePerf"}. Clicking
            a roofline panel row isolates the matching trace. The knee is where
            the diagonal turns over into the compute ceiling capping it, carried
            here because the client frames on it; it is None when the figure has
            no compute roof and the diagonal is drawn open-ended.
        compute_traces: Horizontal compute-ceiling traces (VALU/matrix), each
            {"traceIndex", "label", "peakPerf"}.
        compute_overlay_traces: One hidden highlight trace per compute ceiling,
            each {"traceIndex", "peakPerf"}. While roofs are isolated the base
            ceiling dims and its overlay carries the bright cap from the
            isolated slope rightward.
    """

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
        # The stops are the kernels' own cumulative percentages, so the last one
        # is whatever they add up to rather than a flat 100%.
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
