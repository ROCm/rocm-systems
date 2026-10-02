# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Assemble the interactive standalone roofline HTML document."""

import html
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Dict, Final, List, Optional

import plotly.graph_objects as go

from roofline.roofline_hover import KERNEL_NAME_FONT_FAMILY
from utils.html_report.document import build_document, read_asset

ALL_PEAKS_VALUE = "all"

ROOF_EXTRAP_MIN_AI = 1e-150
ROOF_EXTRAP_MAX_AI = 1e150

_PLOT_DIV_ID = "roofline-plot"
ASSETS_DIR: Final[Path] = Path(__file__).parent / "assets"


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
    precisions: List[str] = field(default_factory=list)
    default_precisions: List[str] = field(default_factory=list)
    frame: Optional[Dict[str, List[float]]] = None

    def to_payload(self) -> Dict[str, Any]:
        """Return the model values for the shared document builder."""
        return {
            "divId": _PLOT_DIV_ID,
            "peaks": self.peaks,
            "peakColors": self.peak_colors,
            "defaultPeak": self.default_peak,
            "kernels": self.kernels,
            "kernelTraceIndices": self.kernel_trace_indices,
            "rooflineTraces": self.roofline_traces,
            "computeTraces": self.compute_traces,
            "computeOverlayTraces": self.compute_overlay_traces,
            "precisions": self.precisions,
            "defaultPrecisions": self.default_precisions,
            "frame": self.frame,
            "roofExtremeMaxAi": ROOF_EXTRAP_MAX_AI,
            "allPeaksValue": ALL_PEAKS_VALUE,
            "kernelNameFontFamily": KERNEL_NAME_FONT_FAMILY,
        }


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

    body_template = Template(read_asset(ASSETS_DIR, "roofline_plot.html"))
    body_html = body_template.substitute(
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
        PLOT_FRAGMENT=fragment,
    )
    return build_document(
        title=title,
        body_html=body_html,
        page_css=read_asset(ASSETS_DIR, "roofline_plot.css"),
        page_js=read_asset(ASSETS_DIR, "roofline_plot.js"),
        model_id="roofline-model",
        model=view_model.to_payload(),
    )
