# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import plotext as plt
import plotly.colors as pcolors
import plotly.graph_objects as go
from dash import dcc, html

from roofline.roofline_frame import canonical_frame
from roofline.roofline_hover import (
    build_compute_peak_hover,
    build_kernel_hover_template,
    build_roof_hover,
    format_hover_number,
    wrap_hover_name,
)
from roofline.roofline_html import (
    ALL_PEAKS_VALUE,
    ROOF_EXTRAP_MAX_AI,
    ROOF_EXTRAP_MIN_AI,
    RooflineViewModel,
    build_interactive_document,
)
from roofline.run_benchmark import (
    BENCHMARKING_SUPPORTED as ROOFLINE_SUPPORTED,  # noqa: F401
)
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
    XMAX_DEFAULT,
    XMIN,
    OpsSupport,
    construct_roof,
    machine_ceilings,
    sanitize_mem_level,
)
from utils.specs import MachineSpecs
from utils.utils_analysis import get_matrix_ops_type

_KERNEL_PALETTE: list[str] = pcolors.qualitative.Dark24 + pcolors.qualitative.Light24
DEFAULT_PEAK = "HBM"
DEFAULT_AXIS_BOUNDS = (XMIN, XMAX_DEFAULT, 1.0, 1000000.0)
ROOF_DENSE_PAD_FACTOR = 1e3
TRACE_COLORS: dict[str, dict[str, str]] = {
    "l0": {"html": "#F0E442", "cli": "brown+"},
    "l1": {"html": "#0072B2", "cli": "red+"},
    "l2": {"html": "#009E73", "cli": "green+"},
    "hbm": {"html": "#D55E00", "cli": "blue+"},
    "lds": {"html": "#E69F00", "cli": "orange+"},
    "valu": {"html": "#CC79A7", "cli": "white"},
    "matrix_ops": {"html": "#56B4E9", "cli": "magenta+"},
}

# Dense roof sampling so hover hits a vertex anywhere on the line.
_ROOF_SAMPLES_PER_DECADE = 48
_ROOF_SAMPLES_MIN = 64
_ROOF_SAMPLES_MAX = 800


def _figure_class(dtype: str) -> str:
    """Return OP or FLOP; integer datatypes use the ops figure."""
    return "OP" if str(dtype).startswith("I") else "FLOP"


def get_color(category: str, backend: str = "html") -> str:
    key = category.removeprefix("ai_").lower()

    if key not in TRACE_COLORS:
        raise RuntimeError(f"Invalid category passed to get_color(): {category}")
    if backend not in TRACE_COLORS[key]:
        raise RuntimeError(f"Invalid backend passed to get_color(): {backend}")

    return TRACE_COLORS[key][backend]


def _roof_sample_count(low_ai: float, high_ai: float) -> int:
    """Log-spaced sample count for a roof spanning [low_ai, high_ai]."""
    if not (low_ai > 0 and high_ai > low_ai):
        return _ROOF_SAMPLES_MIN
    decades = math.log10(high_ai / low_ai)
    samples = round(decades * _ROOF_SAMPLES_PER_DECADE)
    return int(min(max(samples, _ROOF_SAMPLES_MIN), _ROOF_SAMPLES_MAX))


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
        self.__compute_peaks: dict[str, list[tuple[str, float]]] = {}
        self.__ceiling_by_dtype: dict[str, dict[str, Any]] = {}
        self.__frame_bounds: Optional[tuple[float, float, float, float]] = None

    def _ceiling_for_dtype(self, dtype: str) -> dict[str, Any]:
        if dtype not in self.__ceiling_by_dtype:
            self.__ceiling_by_dtype[dtype] = construct_roof(
                roofline_parameters=self.__run_parameters,
                dtype=dtype,
                mspec=self.__mspec,
            )
        return self.__ceiling_by_dtype[dtype]

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

            new_path = base_path / app_name / gpu_model_name

            if isinstance(workload_dir_val, list):
                if isinstance(workload_dir_val[0], (list, tuple)):
                    self.__run_parameters["workload_dir"][0][0] = str(new_path)
                else:
                    self.__run_parameters["workload_dir"][0] = str(new_path)
            else:
                self.__run_parameters["workload_dir"] = str(new_path)

            final_dir = str(new_path)
        else:
            final_dir = base_dir

        Path(final_dir).mkdir(parents=True, exist_ok=True)

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
        samples = _roof_sample_count(left_x, hi)
        xs = np.logspace(np.log10(left_x), np.log10(hi), samples).tolist()
        xs.append(ROOF_EXTRAP_MAX_AI)
        ys = [peak_perf] * len(xs)
        return xs, ys

    @staticmethod
    def _envelope_compute_cap(
        compute_peaks: list[tuple[str, float]],
    ) -> tuple[float, str]:
        """The single compute ceiling the roofline envelope is capped at: the
        tallest compute roof drawn on the figure, across every stacked datatype.

        Returns inf with an empty label when the figure has no compute roof,
        so callers can treat the envelope as bandwidth-only.
        """
        if not compute_peaks:
            return float("inf"), ""
        label, value = max(compute_peaks, key=lambda peak: peak[1])
        return value, label

    @staticmethod
    def _roof_knee(bandwidth: float, cap: float) -> Optional[tuple[float, float]]:
        """Where a diagonal of this bandwidth turns over into the flat compute cap."""
        if cap == float("inf") or not bandwidth > 0:
            return None
        return (cap / bandwidth, cap)

    def _add_compute_ceiling(
        self,
        fig: go.Figure,
        ceiling: list,
        ops_flops: str,
        max_bw: float,
        roof_dense_hi: float,
        *,
        key: str,
        label: str,
        dtype: str,
    ) -> None:
        """Draw a flat compute-peak line plus a hidden highlight overlay."""
        peak_perf = ceiling[1][0]
        left_x = peak_perf / max_bw if max_bw > 0 else ceiling[0][0]
        xs, ys = self._sample_ceiling(left_x, peak_perf, roof_dense_hi)
        ceiling_name = f"Peak {label}-{dtype}"
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                name=ceiling_name,
                mode="lines",
                line=dict(color=get_color(key)),
                hovertemplate=build_compute_peak_hover(
                    label, ceiling[2], ops_flops, dtype
                ),
            )
        )
        view_model = self.__view_models[ops_flops]
        view_model.compute_traces.append({
            "traceIndex": len(fig.data) - 1,
            "label": ceiling_name,
            "peakPerf": peak_perf,
            "dtype": dtype,
        })
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"{ceiling_name} (isolated)",
                mode="lines",
                showlegend=False,
                visible=False,
                line=dict(color=get_color(key), width=3),
                hoverinfo="skip",
            )
        )
        view_model.compute_overlay_traces.append({
            "traceIndex": len(fig.data) - 1,
            "peakPerf": peak_perf,
            "sourceTraceIndex": view_model.compute_traces[-1]["traceIndex"],
        })

    def _figure_compute_peaks(self, ops_flops: str) -> list[tuple[str, float]]:
        """Every compute roof drawn on this figure, each labeled with its
        datatype, computed once per figure class.
        """
        if ops_flops not in self.__compute_peaks:
            self.__compute_peaks[ops_flops] = self._collect_compute_peaks(ops_flops)
        return self.__compute_peaks[ops_flops]

    def _collect_compute_peaks(self, ops_flops: str) -> list[tuple[str, float]]:
        """Read every stacked datatype's ceilings and label each peak with it."""
        peaks: list[tuple[str, float]] = []
        for dt in self.__run_parameters.get("roofline_data_type", []):
            dt = str(dt)
            if _figure_class(dt) != ops_flops or not self._datatype_supported(dt):
                continue
            ceiling = self._ceiling_for_dtype(dt)
            for key, label in self._compute_paths(dt):
                peak = self._peak_value(ceiling, key)
                if peak and peak > 0:
                    peaks.append((f"{dt} {label}", peak))
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
        compute_cap: float,
        compute_cap_label: str,
    ) -> str:
        """Name the specific binding roof for a kernel: the roof with the lowest
        achievable performance at the kernel's operating point. The compute
        candidate is the envelope cap the diagonals are actually drawn to, so
        the limiter agrees with the drawn roof and with the percent of roofline
        the tooltip reports."""
        candidates: list[tuple[float, str]] = []
        for level_name, ai_value in level_ai.items():
            bandwidth = self._peak_value(ceiling_data, level_name.lower())
            if bandwidth and ai_value > 0:
                candidates.append((bandwidth * ai_value, level_name))

        if compute_cap != float("inf"):
            candidates.append((compute_cap, compute_cap_label))

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
        compute_peaks: list[tuple[str, float]],
    ) -> tuple[list[go.Scatter], list[dict[str, Any]]]:
        """Build one marker trace per kernel plus the matching view-model data."""
        traces: list[go.Scatter] = []
        kernels_model: list[dict[str, Any]] = []

        counts = self.__ai_data.get("counts", [])
        total_time = self.__ai_data.get("totalTime", [])
        pct_runtime = self.__ai_data.get("pctRuntime", [])
        time_unit = self.__ai_data.get("timeUnit", "")
        compute_cap, compute_cap_label = self._envelope_compute_cap(compute_peaks)

        for kernel_index, kernel_name in enumerate(kernel_names):
            points, level_ai = self._build_kernel_points(
                kernel_index=kernel_index,
                sanitized_cache_hierarchy=sanitized_cache_hierarchy,
                ceiling_data=ceiling_data,
                compute_cap=compute_cap,
            )
            if not points:
                continue

            color, count_val, time_val, pct_val = (
                values[kernel_index] if kernel_index < len(values) else None
                for values in (kernel_colors, counts, total_time, pct_runtime)
            )
            limiter = self._determine_kernel_limiter(
                level_ai, ceiling_data, compute_cap, compute_cap_label
            )

            traces.append(
                go.Scatter(
                    x=[point["ai"] for point in points],
                    y=[point["perf"] for point in points],
                    name=kernel_name,
                    mode="markers",
                    showlegend=False,
                    marker=dict(
                        color=color,
                        size=10,
                        line=dict(width=0.5, color="black"),
                    ),
                    customdata=[point["hoverCells"] for point in points],
                    hovertemplate=build_kernel_hover_template(
                        name_html=wrap_hover_name(kernel_name),
                        limiter=limiter,
                        count=count_val,
                        total_time=time_val,
                        time_unit=time_unit,
                        pct_runtime=pct_val,
                        ops_flops=ops_flops,
                    ),
                )
            )
            kernels_model.append({
                "name": kernel_name,
                "color": color,
                "points": points,
                "pctRuntime": pct_val,
            })

        return traces, kernels_model

    def _build_kernel_points(
        self,
        kernel_index: int,
        sanitized_cache_hierarchy: list[str],
        ceiling_data: dict[str, Any],
        compute_cap: float,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """One kernel's plotted points, one per memory level it has data for.

        Also returns the level -> AI map the limiter is chosen from.
        """
        points: list[dict[str, Any]] = []
        level_ai: dict[str, float] = {}

        for cache_level in CACHE_LEVELS:
            level_name = cache_level.removeprefix("ai_").upper()
            if level_name not in sanitized_cache_hierarchy:
                continue
            level_points = self.__ai_data.get(cache_level)
            if not level_points or kernel_index >= min(
                len(level_points[0]), len(level_points[1])
            ):
                continue
            ai_value = level_points[0][kernel_index]
            performance = level_points[1][kernel_index]
            if not (ai_value > 0 and performance > 0):
                continue

            roof_perf = self._roof_value_at(
                ai_value=ai_value,
                cache_key=cache_level.removeprefix("ai_"),
                ceiling_data=ceiling_data,
                cap=compute_cap,
            )
            pct_roof = 100.0 * performance / roof_perf if roof_perf else None
            points.append({
                "peak": level_name,
                "ai": ai_value,
                "perf": performance,
                "hoverCells": [
                    format_hover_number(roof_perf, ",.3f"),
                    format_hover_number(pct_roof, ".4f"),
                ],
            })
            level_ai[level_name] = ai_value

        return points, level_ai

    @demarcate
    def construct_plotly_figures(
        self, ai_data: dict[str, Any], datatypes: Optional[list[str]] = None
    ) -> tuple[Optional[go.Figure], Optional[go.Figure], str, str]:
        """
        Build raw Plotly figure objects from pre-computed AI data.

        Returns (ops_figure, flops_figure, ops_dt_list, flops_dt_list).
        No I/O or HTML wrapping. When datatypes is None, use every datatype
        supported by the profiled GPU architecture.
        """
        self.roof_setup()
        self.__view_models = {}
        self.__compute_peaks = {}
        self.__ceiling_by_dtype = {}
        self.__frame_bounds = (
            canonical_frame(*machine_ceilings(self.__run_parameters, self.__mspec))
            or DEFAULT_AXIS_BOUNDS
        )

        console_debug("roofline", f"Path: {self.__run_parameters.get('workload_dir')}")

        self.__ai_data = ai_data

        msg = "AI at each mem level:"
        for key, value in self.__ai_data.items():
            msg += f"\n\t{key} -> {value}"
        console_debug(msg)

        has_kernel_names = bool(self.__ai_data and self.__ai_data.get("kernelNames"))

        figures: dict[str, Optional[go.Figure]] = {"OP": None, "FLOP": None}
        datatype_lists: dict[str, str] = {"OP": "", "FLOP": ""}
        selected_datatypes = (
            datatypes
            if datatypes is not None
            else list(SUPPORTED_DATATYPES.get(self.__mspec.gpu_arch, {}))
        )

        for dt in selected_datatypes:
            if not self._datatype_supported(dt):
                console_error(
                    f"{dt} is not a supported datatype for roofline profiling on "
                    f"{getattr(self.__mspec, 'gpu_model', 'N/A')} "
                    f"(arch: {getattr(self.__mspec, 'gpu_arch', 'unknown_arch')})- "
                    f"cannot construct HTML plot",
                    exit=False,
                )
                continue

            ops_flops = _figure_class(dt)
            figure = self.generate_plot(
                dtype=str(dt),
                fig=figures[ops_flops],
                include_kernels=has_kernel_names,
            )
            if figure is None:
                continue
            figures[ops_flops] = figure
            datatype_lists[ops_flops] += "_" + str(dt)

        return (
            figures["OP"],
            figures["FLOP"],
            datatype_lists["OP"],
            datatype_lists["FLOP"],
        )

    def save_html_files(
        self,
        ops_figure: Optional[go.Figure],
        flops_figure: Optional[go.Figure],
        ops_dt_list: str,
        flops_dt_list: str,
    ) -> None:
        """Write one precision-selectable Plotly HTML document to disk."""
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
        figure, view_model = self._combined_html_figure(ops_figure, flops_figure)
        if figure is not None:
            document = build_interactive_document(
                figure,
                view_model,
                title="Empirical Roofline Analysis",
            )
            path = f"{workload_dir}/empirRoof_gpu-{dev_id}{kernel_list}.html"
            Path(path).write_text(document, encoding="utf-8")
            console_log("roofline", "Roofline HTML files saved.")

    def _combined_html_figure(
        self,
        ops_figure: Optional[go.Figure],
        flops_figure: Optional[go.Figure],
    ) -> tuple[Optional[go.Figure], RooflineViewModel]:
        """Combine selected floating-point and integer compute ceilings into one
        standalone figure. Bandwidth roofs and kernel points are shared, so the
        floating-point figure provides those traces whenever it is available.
        """
        source_key = "FLOP" if flops_figure is not None else "OP"
        source_figure = flops_figure or ops_figure
        if source_figure is None:
            return None, RooflineViewModel()

        figure = go.Figure(source_figure)
        source_model = self.__view_models.get(source_key, RooflineViewModel())
        view_model = RooflineViewModel(
            peaks=list(source_model.peaks),
            peak_colors=dict(source_model.peak_colors),
            default_peak=source_model.default_peak,
            kernels=list(source_model.kernels),
            kernel_trace_indices=list(source_model.kernel_trace_indices),
            roofline_traces=list(source_model.roofline_traces),
            compute_traces=list(source_model.compute_traces),
            compute_overlay_traces=list(source_model.compute_overlay_traces),
        )

        if source_key == "FLOP" and ops_figure is not None:
            ops_model = self.__view_models.get("OP", RooflineViewModel())
            overlay_by_source = {
                overlay["sourceTraceIndex"]: overlay
                for overlay in ops_model.compute_overlay_traces
            }
            for trace in ops_model.compute_traces:
                original_index = trace["traceIndex"]
                new_index = len(figure.data)
                figure.add_trace(ops_figure.data[original_index])
                merged_trace = dict(trace)
                merged_trace["traceIndex"] = new_index
                view_model.compute_traces.append(merged_trace)

                overlay = overlay_by_source.get(original_index)
                if overlay:
                    overlay_index = len(figure.data)
                    figure.add_trace(ops_figure.data[overlay["traceIndex"]])
                    view_model.compute_overlay_traces.append({
                        "traceIndex": overlay_index,
                        "peakPerf": overlay["peakPerf"],
                        "sourceTraceIndex": new_index,
                    })

        view_model.precisions = list(
            dict.fromkeys(trace["dtype"] for trace in view_model.compute_traces)
        )
        return figure, view_model

    @staticmethod
    def generate_html_section(
        ops_figure: Optional[go.Figure],
        flops_figure: Optional[go.Figure],
    ) -> Optional[html.Section]:
        """Wrap Plotly figures in Dash HTML components for WebUI embedding."""
        graphs = [
            html.Div(
                className="float-child",
                children=[
                    html.H3(
                        children=(
                            f"Empirical Roofline Analysis "
                            f"({'Ops' if ops_flops == 'OP' else 'Flops'})"
                        )
                    ),
                    dcc.Graph(figure=figure),
                ],
            )
            for ops_flops, figure in (("OP", ops_figure), ("FLOP", flops_figure))
            if figure is not None
        ]
        if not graphs:
            return None

        return html.Section(
            id="roofline",
            children=[html.Div(className="float-container", children=graphs)],
        )

    @demarcate
    def generate_plot(
        self,
        dtype: str,
        fig: Optional[go.Figure] = None,
        include_kernels: bool = False,
    ) -> Optional[go.Figure]:
        """
        Create graph object from ai_data (coordinate points) and ceiling_data
        (peak FLOP and BW) data.

        Passing an existing fig stacks this datatype's roofs onto it.
        Returns None when the datatype has no usable benchmark data, so the
        caller can drop it rather than ship a half-built figure.
        """
        is_new_figure = fig is None

        sanitized_cache_hierarchy = sanitize_mem_level(
            self.__run_parameters["mem_level"], self.__mspec.gpu_model
        )

        self.__ceiling_data = self._ceiling_for_dtype(dtype)
        console_debug("roofline", f"Ceiling data:\n{self.__ceiling_data}")

        if all(
            v is None or all(x is None for x in v) for v in self.__ceiling_data.values()
        ):
            console_warning(
                f"Unable to generate the {dtype} roofline plot due to missing or "
                "corrupted benchmark data. Skipping this datatype."
            )
            return None

        if fig is None:
            fig = go.Figure()

        ops_flops = _figure_class(dtype)
        # AI points are FLOP-derived, so integer figures are roofs only. The
        # roofs, their colors, and their panel rows are built either way.
        plot_kernels = include_kernels and is_new_figure and ops_flops == "FLOP"

        if ops_flops not in self.__view_models:
            self.__view_models[ops_flops] = RooflineViewModel(
                peak_colors={
                    level.upper(): get_color(level.lower())
                    for level in sanitized_cache_hierarchy
                },
                default_peak=ALL_PEAKS_VALUE,
            )
        compute_peaks = self._figure_compute_peaks(ops_flops)

        if plot_kernels:
            self._add_kernel_traces(
                fig,
                sanitized_cache_hierarchy,
                ops_flops,
                compute_peaks,
            )

        if self.__frame_bounds is None:
            self.__frame_bounds = (
                canonical_frame(*machine_ceilings(self.__run_parameters, self.__mspec))
                or DEFAULT_AXIS_BOUNDS
            )
        x_lo, x_hi, y_lo, y_hi = self.__frame_bounds
        # Roofs are densely sampled across so they stay hoverable
        # throughout the visible range.
        roof_dense_lo = x_lo / ROOF_DENSE_PAD_FACTOR
        roof_dense_hi = x_hi * ROOF_DENSE_PAD_FACTOR

        roof_traces, max_bw = self._build_bandwidth_roofs(
            fig,
            sanitized_cache_hierarchy,
            ops_flops,
            compute_peaks,
            roof_dense_lo,
            roof_dense_hi,
        )

        # Attach any memory roofs this pass added so the client controller can
        # isolate roofs and color their panel rows.
        self.__view_models[ops_flops].roofline_traces.extend(roof_traces)

        self._draw_compute_ceilings(fig, dtype, ops_flops, max_bw, roof_dense_hi)

        if is_new_figure:
            self._apply_plotly_layout(fig, dtype, ops_flops, (x_lo, x_hi, y_lo, y_hi))
        else:
            self._extend_stacked_title(fig, dtype)

        return fig

    def _add_kernel_traces(
        self,
        fig: go.Figure,
        sanitized_cache_hierarchy: list[str],
        ops_flops: str,
        compute_peaks: list[tuple[str, float]],
    ) -> None:
        """Add the per-kernel scatter traces and record them in the view model."""
        view_model = self.__view_models[ops_flops]
        kernel_names = self.__ai_data.get("kernelNames", [])
        kernel_traces, kernels_model = self._build_kernel_traces(
            kernel_names=kernel_names,
            kernel_colors=[
                _KERNEL_PALETTE[i % len(_KERNEL_PALETTE)]
                for i in range(len(kernel_names))
            ],
            sanitized_cache_hierarchy=sanitized_cache_hierarchy,
            ceiling_data=self.__ceiling_data,
            ops_flops=ops_flops,
            compute_peaks=compute_peaks,
        )

        first_index = len(fig.data)
        for kernel_trace in kernel_traces:
            fig.add_trace(kernel_trace)

        present_peaks = self._present_peaks(kernels_model, sanitized_cache_hierarchy)
        view_model.peaks = present_peaks
        view_model.default_peak = (
            DEFAULT_PEAK
            if DEFAULT_PEAK in present_peaks
            else (present_peaks[0] if present_peaks else ALL_PEAKS_VALUE)
        )
        view_model.kernels = kernels_model
        view_model.kernel_trace_indices = list(
            range(first_index, first_index + len(kernel_traces))
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

    def _datatype_supported(self, dtype: str) -> bool:
        """Whether this arch can be profiled for this datatype at all, which is
        the precondition for reading its op classes."""
        gpu_arch = getattr(self.__mspec, "gpu_arch", "unknown_arch")
        return (
            gpu_arch in SUPPORTED_DATATYPES
            and str(dtype) in SUPPORTED_DATATYPES[gpu_arch]
        )

    def _supports(self, dtype: str, flag: OpsSupport) -> bool:
        """Whether this datatype supports the given op class on this arch."""
        return flag in SUPPORTED_DATATYPES[self.__mspec.gpu_arch][dtype]

    def _matrix_label(self) -> str:
        """Matrix-op label (MFMA/WMMA) for this GPU series."""
        return get_matrix_ops_type(
            getattr(self.__mspec, "gpu_series", "unknown_series")
        )

    def _compute_paths(self, dtype: str) -> list[tuple[str, str]]:
        """The compute roofs this datatype reaches on this arch, as (ceiling
        key, label): its scalar path, its matrix path, or both. The key names
        the ceiling in the benchmark data and its color alike, so a roof is
        drawn, labeled, and colored from one row of this table.
        """
        paths = []
        if self._supports(dtype, OpsSupport.VALU):
            paths.append(("valu", "VALU"))
        if self._supports(dtype, OpsSupport.MATRIX):
            paths.append(("matrix_ops", self._matrix_label()))
        return paths

    def _build_bandwidth_roofs(
        self,
        fig: go.Figure,
        sanitized_cache_hierarchy: list[str],
        ops_flops: str,
        compute_peaks: list[tuple[str, float]],
        roof_dense_lo: float,
        roof_dense_hi: float,
    ) -> tuple[list[dict[str, Any]], float]:
        """Draw one diagonal bandwidth roof per memory level.

        Returns the view-model rows for the roofs this call added, plus the peak
        bandwidth across every level.
        """
        roof_traces: list[dict[str, Any]] = []
        max_bw = 0.0
        cap, _ = self._envelope_compute_cap(compute_peaks)
        for level in sanitized_cache_hierarchy:
            peak_bw_val = self._peak_value(self.__ceiling_data, level.lower())
            if peak_bw_val is None:
                continue
            max_bw = max(max_bw, peak_bw_val)
            level_key = level.upper()
            if any(trace.name == level_key for trace in fig.data):
                continue
            knee = self._roof_knee(peak_bw_val, cap)
            self._add_bandwidth_roof(
                fig,
                level,
                peak_bw_val,
                knee[0] if knee else roof_dense_hi,
                roof_dense_lo,
                compute_peaks,
                ops_flops,
            )
            roof_traces.append({
                "level": level_key,
                "traceIndex": len(fig.data) - 1,
                "bandwidth": peak_bw_val,
                "kneeAi": knee[0] if knee else None,
                "kneePerf": knee[1] if knee else None,
            })
        return roof_traces, max_bw

    def _add_bandwidth_roof(
        self,
        fig: go.Figure,
        level: str,
        peak_bw_val: float,
        stop_ai: float,
        roof_dense_lo: float,
        compute_peaks: list[tuple[str, float]],
        ops_flops: str,
    ) -> None:
        """Add the diagonal y = BW * AI roof, drawn up to stop_ai: its knee under
        the tallest compute roof, or the right edge of the sampled window when no
        compute roof caps it."""
        level_key = level.upper()
        dense_lo = min(roof_dense_lo, stop_ai)
        diag_x = [ROOF_EXTRAP_MIN_AI] + np.logspace(
            np.log10(dense_lo), np.log10(stop_ai), _roof_sample_count(dense_lo, stop_ai)
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
        for key, label in self._compute_paths(dtype):
            ceiling = self.__ceiling_data.get(key)
            if ceiling:
                self._add_compute_ceiling(
                    fig,
                    ceiling,
                    ops_flops,
                    max_bw,
                    roof_dense_hi,
                    key=key,
                    label=label,
                    dtype=dtype,
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
        )
        fig.update_yaxes(
            type="log",
            range=[float(np.log10(view_y_lo)), float(np.log10(view_y_hi))],
            title_text=f"Performance (G{ops_flops}/sec)",
        )
        fig.update_layout(
            template="plotly_white",
            title=dict(
                text=f"Empirical Roofline Analysis ({dtype})",
                x=0.5,
                xanchor="center",
                font=dict(size=15),
            ),
            autosize=True,
            dragmode="pan",
            hovermode="closest",
            margin=dict(l=82, r=40, b=62, t=62, pad=4, autoexpand=False),
            showlegend=True,
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="rgba(0, 0, 0, 0.15)",
                align="left",
                font=dict(size=13, color="#1b1f24"),
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

        if not self._datatype_supported(dtype):
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

        ops_flops = _figure_class(dtype)

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

        # Plot the compute peaks this datatype reaches, each drawn a shade below
        # its measured value so the line reads as a ceiling rather than a bound.
        for key, label in self._compute_paths(dtype):
            ceiling = self.__ceiling_data.get(key)
            if not ceiling or ceiling[0] is None:
                console_warning(f"No {label} measurement available for {dtype}")
                continue
            plt.plot(
                ceiling[0],
                [max(perf - 0.1, 1e-9) for perf in ceiling[1]],
                label=f"Peak {label}-{dtype}",
                marker="braille",
                color=get_color(key, backend="cli"),
            )
            plt.text(
                f"{round(ceiling[2])} G{ops_flops}/s",
                x=ceiling[0][1] - 800,
                y=ceiling[1][1],
                background="black",
                color="white",
                alignment="right",
            )
            console_debug(
                "roofline",
                f"{label}: [{ceiling[0][0]},{ceiling[0][1]}], "
                f"[{ceiling[1][0]},{ceiling[1][1]}], {ceiling[2]}",
            )

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
