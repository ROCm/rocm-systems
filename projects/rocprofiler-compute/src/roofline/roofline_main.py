# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
import plotext as plt
import plotly.colors as pcolors
import plotly.graph_objects as go
from dash import dcc, html

from roofline.roofline_hover import (
    build_compute_peak_hover,
    build_point_hover,
    build_roof_hover,
    wrap_hover_name,
)
from roofline.roofline_html import (
    RooflineViewModel,
    build_interactive_document,
)
from roofline.roofline_shared import (
    ALL_PEAKS_VALUE,
    DEFAULT_AXIS_BOUNDS,
    DEFAULT_PEAK,
    ROOF_DENSE_PAD_FACTOR,
    ROOF_EXTRAP_MAX_AI,
    ROOF_EXTRAP_MIN_AI,
    ROOF_SAMPLES,
    TRACE_COLORS,
)
from utils.kernel_name_shortener import format_kernel_signature
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.roofline_calc import (
    CACHE_LEVELS,
    SUPPORTED_DATATYPES,
    OpsSupport,
    construct_roof,
    sanitize_mem_level,
)
from utils.specs import MachineSpecs
from utils.utils_analysis import get_matrix_ops_type

# ROOFLINE_SUPPORTED lists the supported gfx architectures.
ROOFLINE_SUPPORTED = list(SUPPORTED_DATATYPES.keys())

# One color per kernel from a high-contrast qualitative palette.
_KERNEL_PALETTE: list[str] = pcolors.qualitative.Dark24 + pcolors.qualitative.Light24

# Plotly layout styling for the interactive figure.
_PLOT_GRID_COLOR = "rgba(0, 0, 0, 0.08)"
_PLOT_MARGIN = dict(l=82, r=40, b=62, t=62, pad=4, autoexpand=False)
_HOVERLABEL_BORDER_COLOR = "rgba(0, 0, 0, 0.15)"
_HOVERLABEL_FONT_COLOR = "#1b1f24"

# Per-kernel scatter marker styling.
_KERNEL_MARKER_SIZE = 10
_KERNEL_MARKER_LINE_WIDTH = 0.5
_KERNEL_MARKER_LINE_COLOR = "black"

# Compute-ceiling highlight overlay line width (thicker than the base ceiling).
_COMPUTE_OVERLAY_LINE_WIDTH = 3


def get_color(category: str, backend: str = "html") -> str:
    key = category.removeprefix("ai_").lower()

    if key not in TRACE_COLORS:
        raise RuntimeError(f"Invalid category passed to get_color(): {category}")
    if backend not in TRACE_COLORS[key]:
        raise RuntimeError(f"Invalid backend passed to get_color(): {backend}")

    return TRACE_COLORS[key][backend]


def build_kernel_colors(num_kernels: int) -> list[str]:
    """Assign one color per kernel, cycling only past the palette size."""
    if num_kernels <= 0:
        return []
    palette = _KERNEL_PALETTE
    return [palette[index % len(palette)] for index in range(num_kernels)]


def roofline_axis_bounds(
    ceiling_data: dict[str, Any],
    ai_data: dict[str, Any],
    sanitized_cache_hierarchy: list[str],
    pad: float = 1.6,
) -> tuple[float, float, float, float]:
    """Compute explicit log-axis bounds."""
    roof_keys = [level.lower() for level in sanitized_cache_hierarchy]
    roof_keys += ["valu", "matrix_ops"]

    xs: list[float] = []
    ys: list[float] = []

    for key in roof_keys:
        line = ceiling_data.get(key)
        if line and len(line) >= 2 and line[0] and line[1]:
            xs.extend(v for v in line[0] if v is not None and v > 0)
            ys.extend(v for v in line[1] if v is not None and v > 0)

    for cache_level in CACHE_LEVELS:
        points = ai_data.get(cache_level)
        if not points:
            continue
        xs.extend(v for v in points[0] if v is not None and v > 0)
        ys.extend(v for v in points[1] if v is not None and v > 0)

    if not xs or not ys:
        return DEFAULT_AXIS_BOUNDS

    return min(xs) / pad, max(xs) * pad, min(ys) / pad, max(ys) * pad


class Roofline:
    def __init__(
        self,
        args: argparse.Namespace,
        mspec: MachineSpecs,
        run_parameters: dict[str, Any],
    ) -> None:
        self.__args = args
        self.__mspec = mspec
        self.__run_parameters = run_parameters
        self.__ai_data: Optional[dict[str, Any]] = None
        self.__ceiling_data: Optional[dict[str, Any]] = None
        self.__view_models: dict[str, RooflineViewModel] = {}

    def roof_setup(self) -> None:
        workload_dir_val = self.__run_parameters.get("workload_dir")

        if not workload_dir_val:
            console_error(
                "Workload directory is not set. Cannot perform setup.", exit=False
            )
            return

        base_dir = str(workload_dir_val)

        base_path = Path(base_dir)

        if base_path.name == "workloads" and base_path.parent == Path.cwd():
            app_name = getattr(self.__args, "name", "default_app_name")
            gpu_model_name = getattr(self.__mspec, "gpu_model", "default_gpu_model")

            # Create the new path
            new_path = base_path / app_name / gpu_model_name

            # Update workload_dir with the new path, maintaining original data structure
            if isinstance(workload_dir_val, list):
                # Update the nested list structure
                if isinstance(workload_dir_val[0], (list, tuple)):
                    self.__run_parameters["workload_dir"][0][0] = str(new_path)
                else:
                    self.__run_parameters["workload_dir"][0] = str(new_path)
            else:
                # Update string value
                self.__run_parameters["workload_dir"] = str(new_path)

            final_dir = str(new_path)
        else:
            final_dir = base_dir

        # Create the directory
        Path(final_dir).mkdir(parents=True, exist_ok=True)

    def _determine_kernel_bound_status(
        self,
        ai_value: float,
        cache_level: str,
        ceiling_data: dict[str, Any],
    ) -> str:
        """
        Calculate if a kernel point is memory-bound or compute-bound
        based on its own cache level's roofline
        """
        bandwidth = self._peak_value(ceiling_data, cache_level.removeprefix("ai_"))
        if not bandwidth:
            return "Unknown"

        cap, _ = self._active_compute_cap(ceiling_data)
        if cap == float("inf"):
            return "Unknown"

        x_intersect = cap / bandwidth
        if ai_value < x_intersect:
            return "Memory"
        return "Compute"

    @staticmethod
    def _peak_value(ceiling_data: dict[str, Any], key: str) -> Optional[float]:
        """Scalar peak of a ceiling entry, or None when the entry is missing/empty."""
        data = ceiling_data.get(key)
        if (
            isinstance(data, (list, tuple))
            and len(data) >= 3
            and isinstance(data[2], (int, float))
        ):
            return float(data[2])
        return None

    @staticmethod
    def _sample_ceiling(
        left_x: float, peak_perf: float, dense_hi: float
    ) -> tuple[list[float], list[float]]:
        """Dense points for a flat compute ceiling from its left endpoint across
        the visible window, plus one extreme-right anchor, so the whole line is
        hoverable yet still extends far past any zoom."""
        hi = max(dense_hi, left_x)
        xs = np.logspace(np.log10(left_x), np.log10(hi), ROOF_SAMPLES).tolist()
        xs.append(ROOF_EXTRAP_MAX_AI)
        ys = [peak_perf] * len(xs)
        return xs, ys

    def _add_compute_ceiling(
        self,
        fig: go.Figure,
        label: str,
        color_key: str,
        ceiling: list,
        ops_flops: str,
        max_bw: float,
        roof_dense_hi: float,
        dtype: str,
    ) -> None:
        """Draw a flat compute-peak line plus a hidden highlight overlay."""
        peak_perf = ceiling[1][0]
        left_x = peak_perf / max_bw if max_bw > 0 else ceiling[0][0]
        xs, ys = self._sample_ceiling(left_x, peak_perf, roof_dense_hi)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                name=f"Peak {label}",
                mode="lines",
                showlegend=False,
                line=dict(color=get_color(color_key)),
                hovertemplate=build_compute_peak_hover(
                    label, ceiling[2], ops_flops, dtype
                ),
            )
        )
        vm = self.__view_models.get(ops_flops)
        if vm is None:
            return
        vm.compute_traces.append({
            "traceIndex": len(fig.data) - 1,
            "peakPerf": peak_perf,
        })
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"Peak {label} (isolated)",
                mode="lines",
                showlegend=False,
                visible=False,
                line=dict(
                    color=get_color(color_key), width=_COMPUTE_OVERLAY_LINE_WIDTH
                ),
                hoverinfo="skip",
            )
        )
        vm.compute_overlay_traces.append({
            "traceIndex": len(fig.data) - 1,
            "peakPerf": peak_perf,
        })

    def _compute_peaks(self, ceiling_data: dict[str, Any]) -> list[tuple[str, float]]:
        """Present compute ceilings for this figure's datatype."""
        peaks: list[tuple[str, float]] = []
        valu_peak = self._peak_value(ceiling_data, "valu")
        if valu_peak and valu_peak > 0:
            peaks.append(("VALU", valu_peak))
        matrix_peak = self._peak_value(ceiling_data, "matrix_ops")
        if matrix_peak and matrix_peak > 0:
            peaks.append((self._matrix_label(), matrix_peak))
        return peaks

    def _active_compute_cap(self, ceiling_data: dict[str, Any]) -> tuple[float, str]:
        """The single compute ceiling the roofline envelope is capped at."""
        peaks = self._compute_peaks(ceiling_data)
        if not peaks:
            return float("inf"), ""
        label, value = max(peaks, key=lambda item: item[1])
        return value, label

    def _global_compute_peaks(self, ops_flops: str) -> list[tuple[str, float]]:
        """Every compute roof drawn on this figure, each labeled with its
        datatype.
        """
        want_int = ops_flops == "OP"
        gpu_arch = getattr(self.__mspec, "gpu_arch", "unknown_arch")
        peaks: list[tuple[str, float]] = []
        for dt in self.__run_parameters.get("roofline_data_type", []):
            dt = str(dt)
            if dt.startswith("I") != want_int:
                continue
            if (
                gpu_arch not in SUPPORTED_DATATYPES
                or dt not in SUPPORTED_DATATYPES[gpu_arch]
            ):
                continue
            ceiling = construct_roof(
                roofline_parameters=self.__run_parameters,
                dtype=dt,
                mspec=self.__mspec,
                ai_data=self.__ai_data,
            )
            if self._supports(dt, OpsSupport.VALU):
                valu_peak = self._peak_value(ceiling, "valu")
                if valu_peak and valu_peak > 0:
                    peaks.append((f"{dt} VALU", valu_peak))
            if self._supports(dt, OpsSupport.MATRIX):
                matrix_peak = self._peak_value(ceiling, "matrix_ops")
                if matrix_peak and matrix_peak > 0:
                    peaks.append((f"{dt} {self._matrix_label()}", matrix_peak))
        return peaks

    def _roof_value_at(
        self,
        ai_value: float,
        cache_key: str,
        ceiling_data: dict[str, Any],
        cap: float,
    ) -> Optional[float]:
        """Roofline throughput (peak) at this AI for the point's memory level:
        min(bandwidth * AI, active compute cap); None when unavailable."""
        bandwidth = self._peak_value(ceiling_data, cache_key)
        if not bandwidth or ai_value <= 0:
            return None
        roof = bandwidth * ai_value
        if cap != float("inf"):
            roof = min(roof, cap)
        return roof if roof > 0 else None

    def _determine_kernel_limiter(
        self,
        level_ai: dict[str, float],
        ceiling_data: dict[str, Any],
    ) -> str:
        """Name the specific binding roof for a kernel: the roof with the lowest
        achievable performance at the kernel's operating point. The compute
        candidate is the single active cap (the highest peak the envelope is
        drawn to), so the limiter agrees with the drawn roof and pctRoof."""
        candidates: list[tuple[float, str]] = []
        for level_name, ai_value in level_ai.items():
            bandwidth = self._peak_value(ceiling_data, level_name.lower())
            if bandwidth and ai_value > 0:
                candidates.append((bandwidth * ai_value, level_name))

        cap, cap_label = self._active_compute_cap(ceiling_data)
        if cap != float("inf"):
            candidates.append((cap, cap_label))

        if not candidates:
            return "Unknown"
        return min(candidates, key=lambda candidate: candidate[0])[1]

    def _build_kernel_traces(
        self,
        kernel_names: list[str],
        kernel_colors: list[str],
        sanitized_cache_hierarchy: list[str],
        ceiling_data: dict[str, Any],
        ops_flops: str,
    ) -> tuple[list[go.Scatter], list[dict[str, Any]]]:
        """Build one marker trace per kernel plus the matching view-model data."""
        hovertemplate = "%{customdata[0]}<extra></extra>"

        traces: list[go.Scatter] = []
        kernels_model: list[dict[str, Any]] = []

        # Per-kernel stats joined, aligned index-for-index with kernel_names.
        counts = self.__ai_data.get("counts", []) or []
        total_time = self.__ai_data.get("totalTime", []) or []
        pct_runtime = self.__ai_data.get("pctRuntime", []) or []
        time_unit = self.__ai_data.get("timeUnit", "") or ""
        # The compute cap is the same for every kernel; compute it once.
        cap, _ = self._active_compute_cap(ceiling_data)

        for kernel_index, kernel_name in enumerate(kernel_names):
            xs: list[float] = []
            ys: list[float] = []
            points: list[dict[str, Any]] = []
            level_ai: dict[str, float] = {}

            for cache_level in CACHE_LEVELS:
                level_name = cache_level.removeprefix("ai_").upper()
                if level_name not in sanitized_cache_hierarchy:
                    continue
                level_points = self.__ai_data.get(cache_level)
                if not level_points or kernel_index >= len(level_points[0]):
                    continue
                ai_value = level_points[0][kernel_index]
                performance = level_points[1][kernel_index]
                if not (ai_value > 0 and performance > 0):
                    continue

                status = self._determine_kernel_bound_status(
                    ai_value=ai_value,
                    cache_level=cache_level,
                    ceiling_data=ceiling_data,
                )
                roof_val = self._roof_value_at(
                    ai_value=ai_value,
                    cache_key=cache_level.removeprefix("ai_"),
                    ceiling_data=ceiling_data,
                    cap=cap,
                )
                pct_roof = (
                    100.0 * performance / roof_val
                    if roof_val and performance > 0
                    else None
                )
                xs.append(ai_value)
                ys.append(performance)
                points.append({
                    "peak": level_name,
                    "ai": ai_value,
                    "perf": performance,
                    "status": status,
                    "pctRoof": pct_roof,
                    "peakPerf": roof_val,
                })
                level_ai[level_name] = ai_value

            if not xs:
                continue

            color = (
                kernel_colors[kernel_index]
                if kernel_index < len(kernel_colors)
                else None
            )
            count_val = counts[kernel_index] if kernel_index < len(counts) else None
            time_val = (
                total_time[kernel_index] if kernel_index < len(total_time) else None
            )
            pct_val = (
                pct_runtime[kernel_index] if kernel_index < len(pct_runtime) else None
            )
            limiter = self._determine_kernel_limiter(level_ai, ceiling_data)

            # Trim the demangled name for display so kernels
            # differing only by template argument types stay distinct.
            display_name = format_kernel_signature(kernel_name)
            # Build each point's tooltip now that the kernel-level fields exist.
            wrapped_name = wrap_hover_name(display_name)
            for point in points:
                point["hover"] = build_point_hover(
                    name_html=wrapped_name,
                    point=point,
                    limiter=limiter,
                    count=count_val,
                    total_time=time_val,
                    time_unit=time_unit,
                    pct_runtime=pct_val,
                    ops_flops=ops_flops,
                )
            customdata = [[point["hover"]] for point in points]

            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    name=display_name,
                    mode="markers",
                    legendgroup=display_name,
                    showlegend=False,
                    marker=dict(
                        color=color,
                        size=_KERNEL_MARKER_SIZE,
                        line=dict(
                            width=_KERNEL_MARKER_LINE_WIDTH,
                            color=_KERNEL_MARKER_LINE_COLOR,
                        ),
                    ),
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                )
            )
            kernels_model.append({
                "name": display_name,
                "color": color,
                "points": points,
                "count": count_val,
                "totalTime": time_val,
                "pctRuntime": pct_val,
                "limiter": limiter,
            })

        return traces, kernels_model

    @demarcate
    def construct_plotly_figures(
        self, ai_data: dict[str, Any]
    ) -> tuple[Optional[go.Figure], Optional[go.Figure], str, str]:
        """
        Build raw Plotly figure objects from pre-computed AI data.

        Returns (ops_figure, flops_figure, ops_dt_list, flops_dt_list).
        No I/O or HTML wrapping.
        """
        self.roof_setup()
        self.__view_models = {}

        console_debug("roofline", f"Path: {self.__run_parameters.get('workload_dir')}")

        self.__ai_data = ai_data

        msg = "AI at each mem level:"
        for key, value in self.__ai_data.items():
            msg += f"\n\t{key} -> {value}"
        console_debug(msg)

        kernel_names_data = None
        if self.__ai_data and self.__ai_data.get("kernelNames"):
            kernel_names_data = {"kernel_names": self.__ai_data["kernelNames"]}

        ops_figure = flops_figure = None
        ops_dt_list = flops_dt_list = ""

        for dt in self.__run_parameters.get("roofline_data_type", []):
            gpu_arch = getattr(self.__mspec, "gpu_arch", "unknown_arch")
            if (
                gpu_arch not in SUPPORTED_DATATYPES
                or str(dt) not in SUPPORTED_DATATYPES[gpu_arch]
            ):
                console_error(
                    f"{dt} is not a supported datatype for roofline profiling on "
                    f"{getattr(self.__mspec, 'gpu_model', 'N/A')} (arch: {gpu_arch})- "
                    f"cannot construct HTML plot",
                    exit=False,
                )
                continue

            ops_flops = "Ops" if str(dt).startswith("I") else "Flops"

            if ops_flops == "Ops":
                if ops_figure:
                    ops_figure = self.generate_plot(
                        dtype=str(dt),
                        fig=ops_figure,
                    )
                else:
                    ops_figure = self.generate_plot(
                        dtype=str(dt),
                        kernel_names_data=kernel_names_data,
                    )
                ops_dt_list += "_" + str(dt)

            if ops_flops == "Flops":
                if flops_figure:
                    flops_figure = self.generate_plot(
                        dtype=str(dt),
                        fig=flops_figure,
                    )
                else:
                    flops_figure = self.generate_plot(
                        dtype=str(dt),
                        kernel_names_data=kernel_names_data,
                    )
                flops_dt_list += "_" + str(dt)

        return ops_figure, flops_figure, ops_dt_list, flops_dt_list

    def save_html_files(
        self,
        ops_figure: Optional[go.Figure],
        flops_figure: Optional[go.Figure],
        ops_dt_list: str,
        flops_dt_list: str,
    ) -> None:
        """Write Plotly figures to standalone HTML files on disk."""
        dev_id = str(self.__run_parameters["device_id"])
        kernel_list = ""
        if self.__run_parameters.get("kernel_filter", False):
            kernels = getattr(self.__args, "gpu_kernel", None)
            if kernels:
                flat = [
                    str(k)
                    for group in kernels
                    for k in (group if isinstance(group, list) else [group])
                ]
                for name in sorted(flat):
                    kernel_list += "_" + name

        workload_dir = self.__run_parameters["workload_dir"]
        prefix = f"{workload_dir}/empirRoof_gpu-{dev_id}"

        wrote = False
        if ops_figure:
            document = build_interactive_document(
                ops_figure,
                self.__view_models.get("OP", RooflineViewModel()),
                title="Empirical Roofline Analysis (Ops)",
            )
            path = f"{prefix}{ops_dt_list}{kernel_list}.html"
            Path(path).write_text(document, encoding="utf-8")
            wrote = True

        if flops_figure:
            document = build_interactive_document(
                flops_figure,
                self.__view_models.get("FLOP", RooflineViewModel()),
                title="Empirical Roofline Analysis (Flops)",
            )
            path = f"{prefix}{flops_dt_list}{kernel_list}.html"
            Path(path).write_text(document, encoding="utf-8")
            wrote = True

        if wrote:
            console_log("roofline", "Roofline HTML files saved.")

    @staticmethod
    def generate_html_section(
        ops_figure: Optional[go.Figure],
        flops_figure: Optional[go.Figure],
    ) -> Optional[html.Section]:
        """Wrap Plotly figures in Dash HTML components for WebUI embedding."""
        if ops_figure is None and flops_figure is None:
            return None

        ops_graph = (
            html.Div(
                className="float-child",
                children=[
                    html.H3(children="Empirical Roofline Analysis (Ops)"),
                    dcc.Graph(figure=ops_figure),
                ],
            )
            if ops_figure
            else None
        )

        flops_graph = (
            html.Div(
                className="float-child",
                children=[
                    html.H3(children="Empirical Roofline Analysis (Flops)"),
                    dcc.Graph(figure=flops_figure),
                ],
            )
            if flops_figure
            else None
        )

        return html.Section(
            id="roofline",
            children=[
                html.Div(
                    className="float-container",
                    children=[
                        ops_graph,
                        flops_graph,
                    ],
                )
            ],
        )

    @demarcate
    def generate_plot(
        self,
        dtype: str,
        fig: Optional[go.Figure] = None,
        kernel_names_data: Optional[dict] = None,
    ) -> go.Figure:
        """
        Create graph object from ai_data (coordinate points) and ceiling_data
        (peak FLOP and BW) data.
        """
        is_new_figure = fig is None
        has_kernel_names = kernel_names_data is not None and is_new_figure
        skip_kernels = not is_new_figure

        sanitized_cache_hierarchy = sanitize_mem_level(
            self.__run_parameters["mem_level"], self.__mspec.gpu_model
        )

        if fig is None:
            fig = go.Figure()

        self.__ceiling_data = construct_roof(
            roofline_parameters=self.__run_parameters,
            dtype=dtype,
            mspec=self.__mspec,
            ai_data=self.__ai_data,
        )
        console_debug("roofline", f"Ceiling data:\n{self.__ceiling_data}")

        if all(
            v is None or all(x is None for x in v) for v in self.__ceiling_data.values()
        ):
            console_warning(
                "Unable to generate roofline plot due to missing or corrupted "
                "benchmark data. Returning empty figure."
            )
            return fig

        ops_flops = "OP" if dtype.startswith("I") else "FLOP"
        include_kernels = ops_flops == "FLOP" and not skip_kernels and has_kernel_names

        x_lo, x_hi, y_lo, y_hi = roofline_axis_bounds(
            self.__ceiling_data, self.__ai_data or {}, sanitized_cache_hierarchy
        )
        # Roofs are densely sampled across so they stay hoverable
        # throughout the visible range.
        roof_dense_lo = x_lo / ROOF_DENSE_PAD_FACTOR
        roof_dense_hi = x_hi * ROOF_DENSE_PAD_FACTOR

        # Draw roofs across the full extent.
        view_bounds = (x_lo, x_hi, y_lo, y_hi)
        if include_kernels:
            self._build_kernel_view_model(
                fig,
                sanitized_cache_hierarchy,
                ops_flops,
                roof_dense_hi,
            )

        roofline_trace_indices, max_bw = self._build_bandwidth_roofs(
            fig,
            sanitized_cache_hierarchy,
            ops_flops,
            roof_dense_lo,
            roof_dense_hi,
        )

        # Attach the memory-roof trace indices to the view model built this call
        # so the client controller can isolate roofs by the selected peak.
        if include_kernels:
            view_model = self.__view_models.get(ops_flops)
            if view_model is not None:
                view_model.roofline_traces = [
                    {"level": roof_level, "traceIndex": idx, "bandwidth": bw}
                    for roof_level, (idx, bw) in roofline_trace_indices.items()
                ]

        self._draw_compute_ceilings(fig, dtype, ops_flops, max_bw, roof_dense_hi)

        if is_new_figure:
            self._apply_plotly_layout(fig, dtype, ops_flops, view_bounds)
        else:
            self._extend_stacked_title(fig, dtype)

        return fig

    def _build_kernel_view_model(
        self,
        fig: go.Figure,
        sanitized_cache_hierarchy: list[str],
        ops_flops: str,
        roof_dense_hi: float,
    ) -> None:
        """Add per-kernel scatter traces and build the client view model."""
        kernel_names = self.__ai_data.get("kernelNames", [])
        kernel_colors = build_kernel_colors(len(kernel_names))
        kernel_traces, kernels_model = self._build_kernel_traces(
            kernel_names=kernel_names,
            kernel_colors=kernel_colors,
            sanitized_cache_hierarchy=sanitized_cache_hierarchy,
            ceiling_data=self.__ceiling_data,
            ops_flops=ops_flops,
        )

        first_index = len(fig.data)
        for kernel_trace in kernel_traces:
            fig.add_trace(kernel_trace)
        trace_indices = list(range(first_index, first_index + len(kernel_traces)))

        present_peaks = self._present_peaks(kernels_model, sanitized_cache_hierarchy)
        default_peak = (
            DEFAULT_PEAK
            if DEFAULT_PEAK in present_peaks
            else (present_peaks[0] if present_peaks else ALL_PEAKS_VALUE)
        )
        self.__view_models[ops_flops] = RooflineViewModel(
            peaks=present_peaks,
            peak_colors={peak: get_color(peak.lower()) for peak in present_peaks},
            default_peak=default_peak,
            kernels=kernels_model,
            kernel_trace_indices=trace_indices,
            ceiling_dense_hi=roof_dense_hi,
            roof_samples=ROOF_SAMPLES,
        )

    def _present_peaks(
        self,
        kernels_model: list[dict[str, Any]],
        sanitized_cache_hierarchy: list[str],
    ) -> list[str]:
        """Memory levels (in cache order) that at least one kernel point uses."""
        present: list[str] = []
        for cache_level in CACHE_LEVELS:
            level_name = cache_level.removeprefix("ai_").upper()
            if level_name not in sanitized_cache_hierarchy:
                continue
            if any(
                point["peak"] == level_name
                for kernel in kernels_model
                for point in kernel["points"]
            ):
                present.append(level_name)
        return present

    def _supports(self, dtype: str, flag: OpsSupport) -> bool:
        """Whether this datatype supports the given op class on this arch."""
        return flag in SUPPORTED_DATATYPES[self.__mspec.gpu_arch][dtype]

    def _matrix_label(self) -> str:
        """Matrix-op label (MFMA/WMMA) for this GPU series."""
        return get_matrix_ops_type(
            getattr(self.__mspec, "gpu_series", "unknown_series")
        )

    def _build_bandwidth_roofs(
        self,
        fig: go.Figure,
        sanitized_cache_hierarchy: list[str],
        ops_flops: str,
        roof_dense_lo: float,
        roof_dense_hi: float,
    ) -> tuple[dict[str, tuple[int, float]], float]:
        """Draw one diagonal bandwidth roof per memory level.

        Returns the {level: (trace_index, bandwidth)} map and the peak bandwidth.
        """
        roofline_trace_indices: dict[str, tuple[int, float]] = {}
        max_bw = 0.0
        compute_peaks = self._global_compute_peaks(ops_flops)
        cap = max((value for _, value in compute_peaks), default=float("inf"))
        for level in sanitized_cache_hierarchy:
            line_data = self.__ceiling_data.get(level.lower())
            if not (
                line_data
                and isinstance(line_data, (list, tuple))
                and len(line_data) >= 3
            ):
                continue
            peak_bw_val = line_data[2]
            max_bw = max(max_bw, peak_bw_val)
            level_key = level.upper()
            # Bandwidth is datatype-independent, so a roof for this memory level
            # may already be drawn from a prior datatype's pass; keep one entry.
            if any(trace.name == level_key for trace in fig.data):
                continue
            ridge_x = (
                cap / peak_bw_val
                if cap != float("inf") and peak_bw_val > 0
                else roof_dense_hi
            )
            self._add_bandwidth_roof(
                fig,
                level,
                peak_bw_val,
                ridge_x,
                roof_dense_lo,
                compute_peaks,
                ops_flops,
            )
            roofline_trace_indices[level_key] = (len(fig.data) - 1, peak_bw_val)
        return roofline_trace_indices, max_bw

    def _add_bandwidth_roof(
        self,
        fig: go.Figure,
        level: str,
        peak_bw_val: float,
        ridge_x: float,
        roof_dense_lo: float,
        compute_peaks: list[tuple[str, float]],
        ops_flops: str,
    ) -> None:
        """Add the diagonal y = BW * AI roof up to the tallest compute roof."""
        level_key = level.upper()
        dense_lo = min(roof_dense_lo, ridge_x)
        diag_x = [ROOF_EXTRAP_MIN_AI] + np.logspace(
            np.log10(dense_lo), np.log10(ridge_x), ROOF_SAMPLES
        ).tolist()
        diag_y = [peak_bw_val * x for x in diag_x]
        fig.add_trace(
            go.Scatter(
                x=diag_x,
                y=diag_y,
                name=level_key,
                mode="lines",
                line=dict(color=get_color(level.lower())),
                hovertemplate=build_roof_hover(
                    level_key,
                    peak_bw_val,
                    compute_peaks,
                    ops_flops,
                ),
            )
        )

    def _draw_compute_ceilings(
        self,
        fig: go.Figure,
        dtype: str,
        ops_flops: str,
        max_bw: float,
        roof_dense_hi: float,
    ) -> None:
        """Draw the flat VALU/matrix compute peaks that cap every roofline."""
        if self._supports(dtype, OpsSupport.VALU):
            valu_data = self.__ceiling_data.get("valu")
            if valu_data:
                self._add_compute_ceiling(
                    fig,
                    "VALU",
                    "valu",
                    valu_data,
                    ops_flops,
                    max_bw,
                    roof_dense_hi,
                    dtype,
                )
        if self._supports(dtype, OpsSupport.MATRIX):
            matrix_data = self.__ceiling_data.get("matrix_ops")
            if matrix_data:
                self._add_compute_ceiling(
                    fig,
                    self._matrix_label(),
                    "matrix_ops",
                    matrix_data,
                    ops_flops,
                    max_bw,
                    roof_dense_hi,
                    dtype,
                )

    def _apply_plotly_layout(
        self,
        fig: go.Figure,
        dtype: str,
        ops_flops: str,
        view_bounds: tuple[float, float, float, float],
    ) -> None:
        """Apply log axes, initial framing, and shared styling to a new figure."""
        view_x_lo, view_x_hi, view_y_lo, view_y_hi = view_bounds
        fig.update_xaxes(
            type="log",
            range=[float(np.log10(view_x_lo)), float(np.log10(view_x_hi))],
            title_text=f"Arithmetic Intensity ({ops_flops}s/Byte)",
            gridcolor=_PLOT_GRID_COLOR,
        )
        fig.update_yaxes(
            type="log",
            range=[float(np.log10(view_y_lo)), float(np.log10(view_y_hi))],
            title_text=f"Performance (G{ops_flops}/sec)",
            gridcolor=_PLOT_GRID_COLOR,
        )
        fig.update_layout(
            template="plotly_white",
            title=dict(
                text=f"Empirical Roofline Analysis ({dtype})",
                x=0.5,
                xanchor="center",
                font=dict(size=15),
            ),
            # Fill the container.
            autosize=True,
            # Pan on drag / zoom on wheel.
            dragmode="pan",
            hovermode="closest",
            margin=dict(**_PLOT_MARGIN),
            # Roofs are listed/isolated in the side panel, so the in-plot
            # Plotly legend is turned off.
            showlegend=False,
            hoverlabel=dict(
                bgcolor="white",
                bordercolor=_HOVERLABEL_BORDER_COLOR,
                align="left",
                font=dict(size=13, color=_HOVERLABEL_FONT_COLOR),
            ),
        )

    def _extend_stacked_title(self, fig: go.Figure, dtype: str) -> None:
        """Extend an existing figure's title to list every stacked datatype."""
        if not fig.layout.title.text:
            return
        title_text = fig.layout.title.text
        if "(" in title_text and ")" in title_text:
            prefix = title_text.split("(")[0]
            existing_types = title_text.split("(")[1].split(")")[0]
            if dtype not in existing_types.split(", "):
                fig.layout.title.text = f"{prefix}({existing_types}, {dtype})"

    def cli_generate_plot(
        self,
        dtype: str,
        ai_data: dict[str, Any],
    ) -> Optional[str]:
        """
        Plot CLI mode roofline analysis in terminal using plotext

        :param dtype: The datatype to be profiled
        :param ai_data: Pre-computed arithmetic intensity data from calc_ai_analyze
        :return: Build the current figure using plot.build(),
        or None if datatype is not valid for the architecture
        :rtype: str or None
        """
        console_debug("roofline", "Generating roofline plot for CLI")

        if not (str(dtype) in SUPPORTED_DATATYPES[str(self.__mspec.gpu_arch)].keys()):
            console_error(
                f"{dtype} is not a supported datatype for roofline profiling on "
                f"{getattr(self.__mspec, 'gpu_model', 'N/A')} (arch: "
                f"{self.__mspec.gpu_arch})- cannot construct CLI plot",
                exit=False,
            )
            return

        if not ai_data:
            console_warning(
                "roofline",
                "Skipping roofline charting due to invalid arithmetic intensity data",
            )
            return

        self.__ai_data = ai_data

        workload_dir = self.__run_parameters.get("workload_dir", "")
        if not (Path(workload_dir) / "roofline.csv").is_file():
            console_log(
                "roofline",
                f"{workload_dir}/roofline.csv does not exist",
            )
            return None

        self.__ceiling_data = construct_roof(
            roofline_parameters=self.__run_parameters,
            dtype=dtype,
            mspec=self.__mspec,
        )

        self.roof_setup()

        sanitized_cache_hierarchy = sanitize_mem_level(
            self.__run_parameters["mem_level"], self.__mspec.gpu_model
        )

        kernel_markers = {
            0: "star",
            1: "cross",
            2: "sd",
            3: "shamrock",
            4: "at",
            5: "atom",
        }

        plt.clf()
        plt.plotsize(plt.tw(), plt.th())

        ops_flops = "OP" if dtype.startswith("I") else "FLOP"

        for cache_level in sanitized_cache_hierarchy:
            cache_key = cache_level.lower()

            # cache_data layout:
            #   [0] list[float] — x-axis coords for AI: [start_AI, ridge_point_AI]
            #   [1] list[float] — y-axis coords for performance: [start_perf, peak_perf]
            #   [2] float       — scalar peak bandwidth (GB/s)
            cache_data = self.__ceiling_data.get(cache_key)

            if not cache_data or cache_data[0] is None:
                continue
            plt.plot(
                cache_data[0],
                cache_data[1],
                label=f"{cache_level}-{dtype}",
                marker="braille",
                color=get_color(cache_level, backend="cli"),
            )
            plt.text(
                f"{round(cache_data[2])} GB/s",
                x=cache_data[0][0],
                y=cache_data[1][0],
                background="black",
                color="white",
                alignment="left",
            )
            console_debug(
                "roofline",
                f"{cache_level}: [{cache_data[0][0]},"
                f"{cache_data[0][1]}], "
                f"[{cache_data[1][0]},"
                f"{cache_data[1][1]}], "
                f"{cache_data[2]}",
            )

        # Plot VALU and Matrix Ops Peak
        if (
            OpsSupport.VALU in SUPPORTED_DATATYPES[self.__mspec.gpu_arch][dtype]
            and self.__ceiling_data["valu"]
            and self.__ceiling_data["valu"][0] is not None
        ):
            valu_y = [
                max(self.__ceiling_data["valu"][1][0] - 0.1, 1e-9),
                max(self.__ceiling_data["valu"][1][1] - 0.1, 1e-9),
            ]
            plt.plot(
                self.__ceiling_data["valu"][0],
                valu_y,
                label=f"Peak VALU-{dtype}",
                marker="braille",
                color=get_color("valu", backend="cli"),
            )
            plt.text(
                f"{round(self.__ceiling_data['valu'][2])} G{ops_flops}/s",
                x=self.__ceiling_data["valu"][0][1] - 800,
                y=self.__ceiling_data["valu"][1][1],
                background="black",
                color="white",
                alignment="right",
            )
            console_debug(
                "roofline",
                f"VALU: [{self.__ceiling_data['valu'][0][0]},"
                f"{self.__ceiling_data['valu'][0][1]}], "
                f"[{self.__ceiling_data['valu'][1][0]},"
                f"{self.__ceiling_data['valu'][1][1]}], "
                f"{self.__ceiling_data['valu'][2]}",
            )
        else:
            console_warning(f"No PEAK measurement available for {dtype}")

        if (
            OpsSupport.MATRIX in SUPPORTED_DATATYPES[self.__mspec.gpu_arch][dtype]
            and self.__ceiling_data["matrix_ops"]
            and self.__ceiling_data["matrix_ops"][0] is not None
        ):
            matrix_ops_type = get_matrix_ops_type(
                getattr(self.__mspec, "gpu_series", "unknown_series")
            )
            matrix_y = [
                max(self.__ceiling_data["matrix_ops"][1][0] - 0.1, 1e-9),
                max(self.__ceiling_data["matrix_ops"][1][1] - 0.1, 1e-9),
            ]
            plt.plot(
                self.__ceiling_data["matrix_ops"][0],
                matrix_y,
                label=f"Peak {matrix_ops_type}-{dtype}",
                marker="braille",
                color=get_color("matrix_ops", backend="cli"),
            )
            plt.text(
                f"{round(self.__ceiling_data['matrix_ops'][2])} G{ops_flops}/s",
                x=self.__ceiling_data["matrix_ops"][0][1] - 800,
                y=self.__ceiling_data["matrix_ops"][1][1],
                background="black",
                color="white",
                alignment="right",
            )
            console_debug(
                "roofline",
                f"Matrix Ops: [{self.__ceiling_data['matrix_ops'][0][0]},"
                f"{self.__ceiling_data['matrix_ops'][0][1]}], "
                f"[{self.__ceiling_data['matrix_ops'][1][0]},"
                f"{self.__ceiling_data['matrix_ops'][1][1]}], "
                f"{self.__ceiling_data['matrix_ops'][2]}",
            )
        else:
            console_warning(f"No Matrix Ops measurement available for {dtype}")

        # Plot Application AI
        for cache_level in sanitized_cache_hierarchy:
            key = f"ai_{cache_level.lower()}"
            if key not in self.__ai_data:
                continue

            kernel_names = self.__ai_data.get("kernelNames", [])
            for i in range(len(self.__ai_data.get("kernelNames", []))):
                # Zero intensity level means no data reported for this cache level
                if i >= len(self.__ai_data[key][0]) or i >= len(self.__ai_data[key][1]):
                    console_debug(
                        "roofline",
                        f"AI_{kernel_names[i]}: array too short, skipped",
                    )
                    continue

                if self.__ai_data[key][0][i] > 0 and self.__ai_data[key][1][i] > 0:
                    plt.plot(
                        [self.__ai_data[key][0][i]],
                        [self.__ai_data[key][1][i]],
                        label=f"AI_{cache_level}_{kernel_names[i][:40]}",
                        color=get_color(cache_level, backend="cli"),
                        marker=kernel_markers[i % len(kernel_markers)],
                    )

                console_debug(
                    "roofline",
                    f"AI_{kernel_names[i]}: {self.__ai_data[key][0][i]}, "
                    f"{self.__ai_data[key][1][i]}",
                )
        plt.xlabel(f"Arithmetic Intensity ({ops_flops}s/Byte)")
        plt.ylabel("Performance (GFLOP/sec)")
        wdir = self.__run_parameters.get("workload_dir", "")
        plt.title(f"Roofline ({dtype}) - {wdir}")

        # Canvas config
        plt.theme("pro")
        plt.xscale("log")
        plt.yscale("log")

        # Build figure
        # Print plot using `plt._utility.write(self.cli_generate_plot(dtype))`
        return plt.build()

    def get_dtype(self) -> list[str]:
        """
        Return the data types requested by the user (else the default data type)
        for the roofline plot.
        """
        return self.__run_parameters["roofline_data_type"]
