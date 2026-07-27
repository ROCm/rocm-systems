# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Assemble the interactive standalone roofline HTML document.

Wraps the roofline plotly figure in a self-contained page whose controls are
all driven by a single embedded JSON model:
* a memory-peak dropdown,
* a click-to-isolate kernel panel with a cumulative-runtime filter, and
* a bandwidth-roofline panel with click-to-isolate roofs.
"""

from __future__ import annotations

import functools
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import plotly.graph_objects as go

from roofline.roofline_shared import (
    ALL_PEAKS_LABEL,
    ALL_PEAKS_VALUE,
    FALLBACK_COLOR,
    FRAME_MIN_DECADES,
    FRAME_PAD,
    FRAME_ROOF_SEGMENT_DECADES,
    KERNEL_NAME_FONT_FAMILY,
    PLOT_DIM_OPACITY,
    ROOF_EXTRAP_MAX_AI,
    ROOF_SAMPLES,
)

PLOT_DIV_ID = "roofline-plot"

_ASSETS_DIR = Path(__file__).parent / "assets"

_PLOT_CONFIG: dict[str, Any] = {
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": True,
    "doubleClick": False,
}

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
__CSS__
</style>
</head>
<body>
<div class="roofline-app">
  <div class="roofline-toolbar">
    <label class="roofline-control" for="roofline-peak-select">Memory peak
      <select id="roofline-peak-select"
              aria-label="Memory peak for kernel points"></select>
    </label>
    <span class="roofline-hint">Scroll to zoom &middot; drag to pan &middot;
      double-click to reset</span>
    <button type="button" id="roofline-reset-view"
            class="roofline-btn roofline-btn-sm"
            title="Frame the kernels currently shown">Reset zoom</button>
    <button type="button" id="roofline-export-png"
            class="roofline-btn roofline-btn-sm"
            title="Download the current chart as a PNG image">Export PNG</button>
  </div>
  <div class="roofline-body">
    <div class="roofline-plot-col">
__PLOT_FRAGMENT__
    </div>
    <div class="roofline-panel-wrap">
    <aside class="roofline-panel roofline-panel--kernels">
      <div class="roofline-panel-title">
        <span class="roofline-panel-title-label">Kernels
          <span id="roofline-kernel-count" class="roofline-panel-count"></span>
        </span>
        <button type="button" id="roofline-show-all"
                class="roofline-btn roofline-btn-sm">Show all kernels</button>
      </div>
      <p class="roofline-panel-help">Click a row to show only that kernel; click
        again to show all. Ctrl+click (&#8984;+click on Mac) to add or remove
        kernels.</p>
      <div id="roofline-runtime-filter" class="roofline-runtime-filter">
        <label for="roofline-runtime-threshold"
               title="Show only the heaviest kernels whose combined GPU resident
time reaches this percentage. 100% shows every kernel.">
          Runtime shown
          <span id="roofline-runtime-value" class="roofline-runtime-value">100%</span>
        </label>
        <input type="range" id="roofline-runtime-threshold" min="0" max="0"
               step="1" value="0"
               aria-label="Cumulative percent of GPU resident time to display">
      </div>
      <ul id="roofline-kernel-list" class="roofline-panel-list"></ul>
    </aside>
    <aside class="roofline-panel roofline-panel--roofs">
      <div class="roofline-panel-title">
        <span class="roofline-panel-title-label">Bandwidth rooflines
          <span id="roofline-roof-count" class="roofline-panel-count"></span>
        </span>
        <button type="button" id="roofline-show-all-roofs"
                class="roofline-btn roofline-btn-sm">Show all rooflines</button>
      </div>
      <p class="roofline-panel-help">Click a row to show only that roofline; click
        again to show all. Ctrl+click (&#8984;+click on Mac) to add or remove
        rooflines.</p>
      <ul id="roofline-roof-list" class="roofline-panel-list roofline-roof-list">
      </ul>
    </aside>
    </div>
  </div>
</div>
<script id="roofline-model" type="application/json">__MODEL_JSON__</script>
<script>
__JS__
</script>
</body>
</html>
"""


def build_interactive_document(
    figure: go.Figure,
    view_model: RooflineViewModel,
    title: str = "Empirical Roofline Analysis",
) -> str:
    """Build a fully self-contained interactive roofline HTML document."""
    fragment = figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=view_model.div_id,
        config=_PLOT_CONFIG,
    )

    substitutions = {
        "TITLE": _escape_html(title),
        "CSS": _read_asset("roofline_plot.css"),
        "PLOT_FRAGMENT": fragment,
        "MODEL_JSON": view_model.to_json(),
        "JS": _read_asset("roofline_plot.js"),
    }
    return re.sub(
        r"__(TITLE|CSS|PLOT_FRAGMENT|MODEL_JSON|JS)__",
        lambda match: substitutions[match.group(1)],
        _PAGE_TEMPLATE,
    )


@functools.cache
def _read_asset(name: str) -> str:
    """Read a bundled asset (CSS/JS) inlined into the document.

    Cached because both the Ops and Flops documents are written in one run and
    the assets never change at runtime.
    """
    return (_ASSETS_DIR / name).read_text(encoding="utf-8")


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


def _escape_html(text: str) -> str:
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


@dataclass
class RooflineViewModel:
    """Client-facing description of the interactive roofline.

    Attributes:
        peaks: Ordered memory levels that have at least one point (e.g.
            ["L1", "L2", "HBM", "LDS"]).
        peak_colors: Map from memory level to its roof color, used to color an
            isolated kernel's dots by memory level
        default_peak: Memory region shown on load
        kernels: One entry per plotted kernel, in the same order as
            kernel_trace_indices:
            {"name", "color", "count", "totalTime", "pctRuntime", "limiter",
            "points": [{"peak", "ai", "perf", "status", "pctRoof", "peakPerf",
            "hover"}]}. count/totalTime/pctRuntime are the dispatch count,
            aggregate time (in time_unit), and percent of total runtime;
            limiter is the specific binding roof; pctRoof is the percent of the
            roofline achieved at each point; hover is the prebuilt tooltip body.
            The client reads pctRuntime, color, name, and points[*].{ai, perf,
            peak, hover}; the rest support the server-built hover. Any may be
            None when the underlying data is missing.
        kernel_trace_indices: Indices into figure.data of the per-kernel
            scatter traces, in the same order as kernels.
        roofline_traces: Bandwidth-roof (memory-level) line traces; clicking one
            in the legend isolates it, each {"level", "traceIndex", "bandwidth"}.
        compute_traces: Horizontal compute-ceiling traces (VALU/matrix), each
            {"traceIndex", "peakPerf"}. Kept off the legend. While roofs are
            isolated the base ceiling dims and its compute_overlay_traces
            counterpart carries the bright cap from the isolated slope rightward.
        div_id: Id of the Plotly graph div.
    """

    peaks: list[str] = field(default_factory=list)
    peak_colors: dict[str, str] = field(default_factory=dict)
    default_peak: Optional[str] = None
    kernels: list[dict[str, Any]] = field(default_factory=list)
    kernel_trace_indices: list[int] = field(default_factory=list)
    roofline_traces: list[dict[str, Any]] = field(default_factory=list)
    compute_traces: list[dict[str, Any]] = field(default_factory=list)
    compute_overlay_traces: list[dict[str, Any]] = field(default_factory=list)
    ceiling_dense_hi: float = 0.0
    roof_samples: int = ROOF_SAMPLES
    div_id: str = PLOT_DIV_ID

    def to_json(self) -> str:
        """Serialize the model for embedding in a <script> tag."""
        payload = {
            "divId": self.div_id,
            "peaks": self.peaks,
            "peakColors": self.peak_colors,
            "defaultPeak": self.default_peak,
            "kernels": self.kernels,
            "kernelTraceIndices": self.kernel_trace_indices,
            "rooflineTraces": self.roofline_traces,
            "computeTraces": self.compute_traces,
            "computeOverlayTraces": self.compute_overlay_traces,
            "ceilingDenseHi": self.ceiling_dense_hi,
            "roofSamples": self.roof_samples,
            "roofExtremeMaxAi": ROOF_EXTRAP_MAX_AI,
            "allPeaksValue": ALL_PEAKS_VALUE,
            "allPeaksLabel": ALL_PEAKS_LABEL,
            "fallbackColor": FALLBACK_COLOR,
            "plotDimOpacity": PLOT_DIM_OPACITY,
            "framePad": FRAME_PAD,
            "frameMinDecades": FRAME_MIN_DECADES,
            "frameRoofSegmentDecades": FRAME_ROOF_SEGMENT_DECADES,
            "kernelNameFontFamily": KERNEL_NAME_FONT_FAMILY,
        }
        return json.dumps(_json_safe(payload), allow_nan=False).replace("</", "<\\/")
