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
import html
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import plotly.graph_objects as go

KERNEL_NAME_FONT_FAMILY = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

ALL_PEAKS_VALUE = "all"

FRAME_PAD = 1.6
FRAME_MIN_DECADES = 2.5
FRAME_SLOPE_SKEW = 2.0

ROOF_EXTRAP_MAX_AI = 1e150

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
    <label class="roofline-control" id="roofline-peak-control"
           for="roofline-peak-select"
           title="__PEAK_TITLE__">AI axis
      <select id="roofline-peak-select"
              aria-label="Memory level for the arithmetic intensity axis">
      </select>
    </label>
    <span class="roofline-hint">Scroll to zoom &middot; drag to pan &middot;
      double-click to reset</span>
    <button type="button" id="roofline-reset-view"
            class="roofline-btn roofline-btn-sm"
            title="Frame the kernels currently shown">Reset zoom</button>
    <button type="button" id="roofline-export-png"
            class="roofline-btn roofline-btn-sm"
            title="Download the current chart as a PNG image">Export PNG</button>
    <button type="button" id="roofline-theme-toggle"
            class="roofline-btn roofline-btn-sm" aria-pressed="false">Dark mode</button>
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
        <label for="roofline-runtime-threshold" title="__RUNTIME_TITLE__">
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
    figure.update_layout(showlegend=False)
    fragment = figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=view_model.div_id,
        config={
            "displayModeBar": False,
            "responsive": True,
            "scrollZoom": True,
            "doubleClick": False,
        },
    )

    substitutions = {
        "TITLE": html.escape(title),
        "PEAK_TITLE": html.escape(
            "Plot each kernel at its arithmetic intensity for this memory level, "
            "matching the (AI axis) marker in the Bandwidth rooflines panel. "
            "All peaks plots every level at once."
        ),
        # The stops are the kernels' own cumulative percentages, so the last one
        # is whatever they add up to rather than a flat 100%.
        "RUNTIME_TITLE": html.escape(
            "Show only the heaviest kernels whose combined percent of GPU "
            "resident time reaches this cutoff. The rightmost stop shows every "
            "plotted kernel."
        ),
        "CSS": _read_asset("roofline_plot.css"),
        "PLOT_FRAGMENT": fragment,
        "MODEL_JSON": view_model.to_json(),
        "JS": _read_asset("roofline_plot.js"),
    }
    return re.sub(
        r"__(TITLE|PEAK_TITLE|RUNTIME_TITLE|CSS|PLOT_FRAGMENT|MODEL_JSON|JS)__",
        lambda match: substitutions[match.group(1)],
        _PAGE_TEMPLATE,
    )


@functools.cache
def _read_asset(name: str) -> str:
    """Read a bundled asset (CSS/JS) inlined into the document.

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
            {"level", "traceIndex", "bandwidth"}. Clicking a roofline panel row
            isolates the matching trace.
        compute_traces: Horizontal compute-ceiling traces (VALU/matrix), each
            {"traceIndex", "label", "peakPerf"}.
        compute_overlay_traces: One hidden highlight trace per compute ceiling,
            each {"traceIndex", "peakPerf"}. While roofs are isolated the base
            ceiling dims and its overlay carries the bright cap from the
            isolated slope rightward.
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
    div_id: str = "roofline-plot"

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
            "roofExtremeMaxAi": ROOF_EXTRAP_MAX_AI,
            "allPeaksValue": ALL_PEAKS_VALUE,
            "allPeaksLabel": "All peaks",
            # Marker color for a level or kernel with no assigned color.
            "fallbackColor": "#888888",
            # Opacity of the non-isolated roofs and ceilings while isolating.
            "plotDimOpacity": 0.15,
            "framePad": FRAME_PAD,
            "frameMinDecades": FRAME_MIN_DECADES,
            "frameSlopeSkew": FRAME_SLOPE_SKEW,
            "kernelNameFontFamily": KERNEL_NAME_FONT_FAMILY,
        }
        return json.dumps(_json_safe(payload), allow_nan=False).replace("</", "<\\/")
