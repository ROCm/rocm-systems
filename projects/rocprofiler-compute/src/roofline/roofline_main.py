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

from roofline.roofline_hover import (
    build_compute_peak_hover,
    build_kernel_hover_template,
    build_roof_hover,
    format_hover_number,
    wrap_hover_name,
)
from roofline.roofline_html import (
    ALL_PEAKS_VALUE,
    FRAME_MIN_DECADES,
    FRAME_PAD,
    ROOF_EXTRAP_MAX_AI,
    RooflineViewModel,
    build_interactive_document,
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

# Shared by both axes of the interactive figure.
_PLOT_GRID_COLOR = "rgba(0, 0, 0, 0.08)"

# Memory region the roofline opens on when it is present.
DEFAULT_PEAK = "HBM"

# Log-axis fallback frame when there is no data.
DEFAULT_AXIS_BOUNDS = (0.01, 1000.0, 1.0, 100000.0)

# Roofs are extrapolated down to this AI so they still span a panned view.
ROOF_EXTRAP_MIN_AI = 1e-150

# Decades beyond the data extremes the roofs stay densely sampled (each side).
ROOF_DENSE_PAD_FACTOR = 1e3

# Per memory-level / compute-roof trace colors for the HTML and CLI backends.
TRACE_COLORS: dict[str, dict[str, str]] = {
    "l0": {"html": "#F0E442", "cli": "brown+"},
    "l1": {"html": "#0072B2", "cli": "red+"},
    "l2": {"html": "#009E73", "cli": "green+"},
    "hbm": {"html": "#D55E00", "cli": "blue+"},
    "lds": {"html": "#E69F00", "cli": "orange+"},
    "valu": {"html": "#CC79A7", "cli": "white"},
    "matrix_ops": {"html": "#56B4E9", "cli": "magenta+"},
}

# Roofs are sampled so that hover snaps to a vertex anywhere along the line
_ROOF_SAMPLES_PER_DECADE = 24
_ROOF_SAMPLES_MIN = 32
_ROOF_SAMPLES_MAX = 400


def get_color(category: str, backend: str = "html") -> str:
    key = category.removeprefix("ai_").lower()

    if key not in TRACE_COLORS:
        raise RuntimeError(f"Invalid category passed to get_color(): {category}")
    if backend not in TRACE_COLORS[key]:
        raise RuntimeError(f"Invalid backend passed to get_color(): {backend}")

    return TRACE_COLORS[key][backend]


def build_kernel_colors(num_kernels: int) -> list[str]:
    """Assign one color per kernel, cycling the palette once it runs out."""
    palette_size = len(_KERNEL_PALETTE)
    return [_KERNEL_PALETTE[index % palette_size] for index in range(num_kernels)]


def _widened_to_min_decades(lo: float, hi: float) -> tuple[float, float]:
    """Widen a positive [lo, hi] about its midpoint to at least the minimum
    decades, so an axis never opens on a sliver."""
    log_lo = math.log10(lo)
    log_hi = math.log10(hi)
    if log_hi - log_lo >= FRAME_MIN_DECADES:
        return lo, hi
    mid = 0.5 * (log_lo + log_hi)
    return (
        10 ** (mid - 0.5 * FRAME_MIN_DECADES),
        10 ** (mid + 0.5 * FRAME_MIN_DECADES),
    )


def roofline_axis_bounds(
    ceiling_data: dict[str, Any],
    ai_data: dict[str, Any],
    sanitized_cache_hierarchy: list[str],
) -> tuple[float, float, float, float]:
    """Compute the log-axis bounds the figure opens on."""
    xs: list[float] = []
    ys: list[float] = []

    for level in sanitized_cache_hierarchy:
        line = ceiling_data.get(level.lower())
        if line and len(line) >= 2 and line[0] and line[1]:
            knee = (line[0][-1], line[1][-1])
            if all(v is not None and v > 0 for v in knee):
                xs.append(knee[0])
                ys.append(knee[1])

    # Compute ceilings are horizontal and run off the right edge, so they bound
    # throughput only.
    for key in ("valu", "matrix_ops"):
        line = ceiling_data.get(key)
        if line and len(line) >= 2 and line[1]:
            ys.extend(v for v in line[1] if v is not None and v > 0)

    for cache_level in CACHE_LEVELS:
        points = ai_data.get(cache_level)
        if not points:
            continue
        xs.extend(v for v in points[0] if v is not None and v > 0)
        ys.extend(v for v in points[1] if v is not None and v > 0)

    if not xs or not ys:
        return DEFAULT_AXIS_BOUNDS

    x_lo, x_hi = _widened_to_min_decades(min(xs) / FRAME_PAD, max(xs) * FRAME_PAD)
    y_lo, y_hi = _widened_to_min_decades(min(ys) / FRAME_PAD, max(ys) * FRAME_PAD)
    return (x_lo, x_hi, y_lo, y_hi)


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
        # Compute peaks per figure class ("OP"/"FLOP"). Each entry costs one
        # roofline.csv read per requested datatype, and every datatype's pass
        # over a figure needs the same answer.
        self.__compute_peaks: dict[str, list[tuple[str, float]]] = {}

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

    def _add_compute_ceiling(
        self,
        fig: go.Figure,
        ceiling: list,
        ops_flops: str,
        max_bw: float,
        roof_dense_hi: float,
        *,
        label: str,
        color_key: str,
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
                line=dict(color=get_color(color_key)),
                hovertemplate=build_compute_peak_hover(
                    label, ceiling[2], ops_flops, dtype
                ),
            )
        )
        view_model = self.__view_models.get(ops_flops)
        if view_model is None:
            return
        view_model.compute_traces.append({
            "traceIndex": len(fig.data) - 1,
            "label": ceiling_name,
            "peakPerf": peak_perf,
        })
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"{ceiling_name} (isolated)",
                mode="lines",
                showlegend=False,
                visible=False,
                # Thicker than the base ceiling it highlights.
                line=dict(color=get_color(color_key), width=3),
                hoverinfo="skip",
            )
        )
        view_model.compute_overlay_traces.append({
            "traceIndex": len(fig.data) - 1,
            "peakPerf": peak_perf,
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

        # Per-kernel stats joined, aligned index-for-index with kernel_names.
        counts = self.__ai_data.get("counts", [])
        total_time = self.__ai_data.get("totalTime", [])
        pct_runtime = self.__ai_data.get("pctRuntime", [])
        time_unit = self.__ai_data.get("timeUnit", "")
        # Every kernel is scored against the same envelope cap, so resolve the
        # cap the diagonals are drawn to once.
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

            # A list too short to reach this kernel means the stat is missing
            # rather than zero, since the stats are joined from separate tables.
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
                    # Kept whole: the template arguments are what tell two
                    # instantiations of the same function apart.
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
            # Only what the client reads back: the dispatch count, aggregate
            # time, and limiter are already rendered into the hover template
            # above, so restating them here would ship them to the browser
            # twice.
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
        self, ai_data: dict[str, Any]
    ) -> tuple[Optional[go.Figure], Optional[go.Figure], str, str]:
        """
        Build raw Plotly figure objects from pre-computed AI data.

        Returns (ops_figure, flops_figure, ops_dt_list, flops_dt_list).
        No I/O or HTML wrapping.
        """
        self.roof_setup()
        self.__view_models = {}
        self.__compute_peaks = {}

        console_debug("roofline", f"Path: {self.__run_parameters.get('workload_dir')}")

        self.__ai_data = ai_data

        msg = "AI at each mem level:"
        for key, value in self.__ai_data.items():
            msg += f"\n\t{key} -> {value}"
        console_debug(msg)

        has_kernel_names = bool(self.__ai_data and self.__ai_data.get("kernelNames"))

        # One figure per op class, each stacking every datatype of that class.
        figures: dict[str, Optional[go.Figure]] = {"Ops": None, "Flops": None}
        datatype_lists: dict[str, str] = {"Ops": "", "Flops": ""}

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
            figure = self.generate_plot(
                dtype=str(dt),
                fig=figures[ops_flops],
                include_kernels=has_kernel_names,
            )
            # A datatype with no usable benchmark data contributes nothing, so
            # it must not claim the figure: the next datatype would then be
            # treated as a stacked pass and skip the axes and the view model.
            if figure is None:
                continue
            figures[ops_flops] = figure
            datatype_lists[ops_flops] += "_" + str(dt)

        return (
            figures["Ops"],
            figures["Flops"],
            datatype_lists["Ops"],
            datatype_lists["Flops"],
        )

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
                f"Unable to generate the {dtype} roofline plot due to missing or "
                "corrupted benchmark data. Skipping this datatype."
            )
            return None

        if fig is None:
            fig = go.Figure()

        ops_flops = "OP" if dtype.startswith("I") else "FLOP"
        # AI points are FLOP-derived, so integer figures are roofs only. The
        # roofs, their colors, and their panel rows are built either way.
        plot_kernels = include_kernels and is_new_figure and ops_flops == "FLOP"

        x_lo, x_hi, y_lo, y_hi = roofline_axis_bounds(
            self.__ceiling_data, self.__ai_data or {}, sanitized_cache_hierarchy
        )
        # Roofs are densely sampled across so they stay hoverable
        # throughout the visible range.
        roof_dense_lo = x_lo / ROOF_DENSE_PAD_FACTOR
        roof_dense_hi = x_hi * ROOF_DENSE_PAD_FACTOR

        if is_new_figure:
            # Every level in the hierarchy is colored, not just the ones with
            # kernel points, because the roofline panel is the only legend and
            # renders a row for every roof drawn.
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
        view_model = self.__view_models.get(ops_flops)
        if view_model is not None:
            view_model.roofline_traces.extend(roof_traces)

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
            kernel_colors=build_kernel_colors(len(kernel_names)),
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
            roof_traces.append({
                "level": level_key,
                "traceIndex": len(fig.data) - 1,
                "bandwidth": peak_bw_val,
            })
        return roof_traces, max_bw

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
            np.log10(dense_lo), np.log10(ridge_x), _roof_sample_count(dense_lo, ridge_x)
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
        valu_data = self.__ceiling_data.get("valu")
        if self._supports(dtype, OpsSupport.VALU) and valu_data:
            self._add_compute_ceiling(
                fig,
                valu_data,
                ops_flops,
                max_bw,
                roof_dense_hi,
                label="VALU",
                color_key="valu",
                dtype=dtype,
            )
        matrix_data = self.__ceiling_data.get("matrix_ops")
        if self._supports(dtype, OpsSupport.MATRIX) and matrix_data:
            self._add_compute_ceiling(
                fig,
                matrix_data,
                ops_flops,
                max_bw,
                roof_dense_hi,
                label=self._matrix_label(),
                color_key="matrix_ops",
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
            autosize=True,
            dragmode="pan",
            hovermode="closest",
            margin=dict(l=82, r=40, b=62, t=62, pad=4, autoexpand=False),
            # Kept on for the surfaces that embed this figure bare (the Dash
            # WebUI); the standalone document hides it because its side panels
            # replace the legend.
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
            self._supports(dtype, OpsSupport.VALU)
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
            self._supports(dtype, OpsSupport.MATRIX)
            and self.__ceiling_data["matrix_ops"]
            and self.__ceiling_data["matrix_ops"][0] is not None
        ):
            matrix_ops_type = self._matrix_label()
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
