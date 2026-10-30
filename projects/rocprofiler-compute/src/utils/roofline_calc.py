# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import csv
from dataclasses import dataclass
from enum import Flag
from pathlib import Path
from typing import Any, List, Optional, Sequence, SupportsFloat, Union

import numpy as np
import pandas as pd

from utils import schema
from utils.logger import console_debug, console_error, console_warning
from utils.metrics.base_units import SCALED_UNITS
from utils.metrics.evaluation_pipeline import eval_metric
from utils.mi_gpu_spec import mi_gpu_specs, roofline_column
from utils.specs import MachineSpecs
from utils.utils_common import _workload_base_dir

# Log-axis extent for ceiling math.
XMIN = 0.01
XMAX_DEFAULT = 1000.0

# The roofline.csv device that Analyze plots, and whose ceilings the database stores.
ROOFLINE_DEVICE_ID = 0

CACHE_LEVELS = ["ai_l0", "ai_l1", "ai_l2", "ai_hbm", "ai_lds"]


# Enum class representing support of VALU and Matrix Operations
class OpsSupport(Flag):
    VALU = 1  # Peak VALU Ops supported
    MATRIX = 2  # Matrix Ops supported


# The datatypes the roofline benchmark measures per gpu arch, from mi_gpu_spec.yaml.
SUPPORTED_DATATYPES: dict[str, dict[str, OpsSupport]] = {
    gpu_arch: {
        data_type: (OpsSupport.VALU if valu else OpsSupport(0))
        | (OpsSupport.MATRIX if matrix else OpsSupport(0))
        for data_type, valu, matrix in mi_gpu_specs.get_roofline_data_types(gpu_arch)
    }
    for gpu_arch in mi_gpu_specs.get_roofline_archs()
}


# Metric labels of the "Roofline Plot Points" table, mapped to PlotPoints fields.
METRIC_TO_AI_FIELD = {
    "AI L0": "ai_l0",
    "AI L1": "ai_l1",
    "AI L2": "ai_l2",
    "AI HBM": "ai_hbm",
    "AI LDS": "ai_lds",
}
PERFORMANCE_METRIC = "Performance (GFLOPs)"

# Table ids defined in <arch>/0400_roofline.yaml and 0000_top_stats.yaml.
_ROOFLINE_RATES_TABLE_ID = 401
_ROOFLINE_POINTS_TABLE_ID = 402
_KERNEL_TOP_TABLE_ID = 1


################################################
# Helper funcs
################################################
@dataclass
class PlotPoints:
    """Data structure for storing roofline plot points."""

    ai_l0: list[list[float]]
    ai_l1: list[list[float]]
    ai_l2: list[list[float]]
    ai_hbm: list[list[float]]
    ai_lds: list[list[float]]
    kernelNames: list[str]
    counts: list[Optional[float]]
    kernelTime: list[Optional[float]]
    kernelPctRuntime: list[Optional[float]]
    totalAppTime: Optional[float]
    totalAppDispatches: Optional[float]
    timeUnit: str

    @classmethod
    def empty(cls) -> "PlotPoints":
        """Create empty plot points structure."""
        return cls(
            ai_l0=[[], []],
            ai_l1=[[], []],
            ai_l2=[[], []],
            ai_hbm=[[], []],
            ai_lds=[[], []],
            kernelNames=[],
            counts=[],
            kernelTime=[],
            kernelPctRuntime=[],
            totalAppTime=None,
            totalAppDispatches=None,
            timeUnit="",
        )


@dataclass
class GraphPoints:
    """Data structure for storing roofline graph ceiling points."""

    hbm: list[Union[list[float], float, None]]
    l2: list[Union[list[float], float, None]]
    l1: list[Union[list[float], float, None]]
    l0: list[Union[list[float], float, None]]
    lds: list[Union[list[float], float, None]]
    valu: list[Union[list[float], float, None]]
    matrix_ops: list[Union[list[float], float, None]]

    @classmethod
    def empty(cls) -> "GraphPoints":
        """Create empty graph points structure."""
        return cls(
            hbm=[None, None, None],
            l2=[None, None, None],
            l1=[None, None, None],
            l0=[None, None, None],
            lds=[None, None, None],
            valu=[None, None, None],
            matrix_ops=[None, None, None],
        )


@dataclass
class RooflineCsvData:
    """Parsed roofline.csv device IDs and benchmark columns by row order."""

    device_ids: list[int]
    columns: dict[str, list[str]]


################################################
# Helper functions
################################################
def sanitize_ai_value(value: Union[SupportsFloat, str, None]) -> float:
    """Coerce a raw AI/performance cell to a finite number.

    Cells come from an evaluated metric table, so a missing counter can surface
    as a sentinel string ("", "N/A"), as None, or as inf/NaN. Nonnumeric and
    nonfinite values become 0; finite values retain their sign.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    return numeric if np.isfinite(numeric) else 0


def roofline_mem_levels(gpu_arch: str, gpu_model: str) -> List[str]:
    """Memory levels the roofline plots, in the model's hierarchy order: those with a
    measured bandwidth ceiling, except MALL, which is measured but not plotted."""
    measured = mi_gpu_specs.get_roofline_bandwidth(gpu_arch)
    return [
        level
        for level in mi_gpu_specs.get_memory_levels(gpu_model)
        if level in measured and level != "MALL"
    ]


# -------------------------------------------------------------------------------------
#                           Plot BW at each cache level
# -------------------------------------------------------------------------------------
def calc_ceilings(
    roofline_parameters: dict[str, Any],
    dtype: str,
    benchmark_data: dict[str, list[str]],
    mspec: MachineSpecs,
    device_row_index: int,
) -> dict[str, list[Union[list[float], float, None]]]:
    """Given benchmarking data, calculate ceilings (or peak performance) for
    empirical roofline"""

    graph_points: dict[str, list[Union[list[float], float, None]]] = {
        "hbm": [],
        "l2": [],
        "l1": [],
        "l0": [],
        "lds": [],
        "valu": [],
        "matrix_ops": [],
    }

    cache_hierarchy = roofline_mem_levels(mspec.gpu_arch, mspec.gpu_model)

    x1 = y1 = x2 = y2 = -1
    x1_matrix = y1_matrix = x2_matrix = y2_matrix = -1

    valu_column = roofline_column(dtype, "valu")

    peak_ops = 0.0
    if OpsSupport.VALU in SUPPORTED_DATATYPES[mspec.gpu_arch][dtype]:
        try:
            peak_ops = sanitize_ai_value(benchmark_data[valu_column][device_row_index])
        except KeyError:
            console_warning(
                f"Missing benchmark data for {valu_column} in benchmark_results. "
                "Skipping peak operations calculation. This may indicate incomplete or "
                "corrupted benchmark data."
            )
            return GraphPoints.empty().__dict__
        if peak_ops <= 0:
            console_warning(
                f"Invalid peak operations ({peak_ops}) for {valu_column}. "
                "Unable to construct finite compute roof geometry."
            )
            return GraphPoints.empty().__dict__

    for cache_level in cache_hierarchy:
        # Plot BW line
        curr_bw = f"{cache_level}Bw"
        try:
            peak_bw = sanitize_ai_value(benchmark_data[curr_bw][device_row_index])
        except KeyError:
            console_warning(
                f"Missing benchmark data for {curr_bw} in benchmark_results. "
                f"Skipping {cache_level} cache level. This may indicate incomplete or "
                "corrupted benchmark data."
            )
            continue

        if peak_bw <= 0:
            console_debug(
                f"Invalid peak bandwidth ({peak_bw}) for {cache_level}. Skipping."
            )
            continue

        x1 = float(XMIN)
        y1 = sanitize_ai_value(float(XMIN) * peak_bw)
        if y1 <= 0:
            console_debug(
                f"Peak bandwidth geometry underflowed for {cache_level}. Skipping."
            )
            continue

        x1_matrix = float(XMIN)
        x2_matrix = 0.0
        y1_matrix = y2_matrix = 0.0

        if OpsSupport.VALU in SUPPORTED_DATATYPES[mspec.gpu_arch][dtype]:
            x2 = sanitize_ai_value(peak_ops / peak_bw)
            y2 = peak_ops  # noqa

            # Plot Matrix Ops lines (NOTE: Assuming MI200 soc)
            x1_matrix = x2
            y1_matrix = peak_ops

        peak_matrix = 0.0
        if OpsSupport.MATRIX in SUPPORTED_DATATYPES[mspec.gpu_arch][dtype]:
            matrix_key = roofline_column(dtype, roofline_parameters["matrix_ops_type"])
            try:
                peak_matrix = sanitize_ai_value(
                    benchmark_data[matrix_key][device_row_index]
                )
                if peak_matrix > 0:
                    x2_matrix = sanitize_ai_value(peak_matrix / peak_bw)
                    y2_matrix = peak_matrix
                else:
                    console_debug(
                        f"Invalid matrix peak ({peak_matrix}) for {matrix_key}. "
                        f"Skipping {roofline_parameters['matrix_ops_type']} "
                        f"calculations for {cache_level} cache level."
                    )
            except KeyError:
                console_warning(
                    f"Missing benchmark data for {matrix_key} in benchmark_results. "
                    f"Skipping {roofline_parameters['matrix_ops_type']} "
                    f"calculations for {cache_level} cache level. "
                    "This may indicate incomplete or corrupted benchmark data."
                )

        if y1_matrix <= 0 and y2_matrix <= 0:
            console_debug(
                f"No valid compute peak for {dtype} at {cache_level}. "
                "Skipping bandwidth geometry."
            )
            continue

        # Check which peak is higher for formatting bandwidth lines
        if y2_matrix > y1_matrix:  # peak_matrix
            peak_x = x2_matrix
            peak_y = y2_matrix
        else:  # peakVALU
            peak_x = x1_matrix
            peak_y = y1_matrix

        if peak_x <= 0:
            peak_x = float(XMIN)

        cache_key = cache_level.lower()
        graph_points[cache_key].extend([[x1, peak_x], [y1, peak_y], peak_bw])

    # ----------------------------------------------------------------------------------
    #                                     Plot computing roof
    # ----------------------------------------------------------------------------------
    if OpsSupport.VALU in SUPPORTED_DATATYPES[mspec.gpu_arch][dtype] and x2 > 0:
        # Plot FMA roof
        x0 = min(x2, XMAX_DEFAULT) if x2 < XMAX_DEFAULT else XMAX_DEFAULT

        graph_points["valu"].extend([
            [x0, XMAX_DEFAULT],
            [peak_ops, peak_ops],
            peak_ops,
        ])

    # Plot Matrix Ops roof
    if (
        OpsSupport.MATRIX in SUPPORTED_DATATYPES[mspec.gpu_arch][dtype]
        and x2_matrix > 0
    ):
        x0_matrix = (
            min(x2_matrix, XMAX_DEFAULT) if x2_matrix < XMAX_DEFAULT else XMAX_DEFAULT
        )

        graph_points["matrix_ops"].extend([
            [x0_matrix, XMAX_DEFAULT],
            [peak_matrix, peak_matrix],
            peak_matrix,
        ])

    return graph_points


# -------------------------------------------------------------------------------------
#                              Overlay application performance
# -------------------------------------------------------------------------------------
def _extract_ai_metrics(calc_table: Optional[pd.DataFrame]) -> dict[str, float]:
    """Read per-level AI and performance from a kernel's evaluated plot points.

    Returns one entry per ``CACHE_LEVELS`` field plus ``performance``, defaulting
    to 0.0 for any row the table does not carry.
    """
    metrics = dict.fromkeys(CACHE_LEVELS, 0.0)
    metrics["performance"] = 0.0
    if calc_table is None:
        return metrics
    for _, row in calc_table.iterrows():
        metric = row.get("Metric", "")
        value = row.get("Value", 0)
        if metric in METRIC_TO_AI_FIELD:
            metrics[METRIC_TO_AI_FIELD[metric]] = sanitize_ai_value(value)
        elif metric == PERFORMANCE_METRIC:
            metrics["performance"] = sanitize_ai_value(value)
    return metrics


def _resolve_kernel_ids(workload: schema.Workload) -> list[int]:
    """Kernel row ids to plot: an explicit kernel filter, else every top kernel."""
    if workload.filter_kernel_ids:
        return workload.filter_kernel_ids
    if _KERNEL_TOP_TABLE_ID not in workload.dfs:
        return []
    return workload.dfs[_KERNEL_TOP_TABLE_ID].index.tolist()


def _evaluate_kernel_tables(
    arch_config: schema.ArchConfig,
    workload: schema.Workload,
    kernel_pmc_df: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Evaluate the roofline arch-config tables (401/402) for one kernel."""
    kernel_dfs: dict[int, pd.DataFrame] = {}
    kernel_dfs_type: dict[int, str] = {}
    for table_id in (_ROOFLINE_RATES_TABLE_ID, _ROOFLINE_POINTS_TABLE_ID):
        if table_id in arch_config.dfs:
            kernel_dfs[table_id] = arch_config.dfs[table_id].copy()
            kernel_dfs_type[table_id] = arch_config.dfs_type[table_id]
    # eval_metric keys off kernel_dfs only.
    eval_metric(
        kernel_dfs,
        kernel_dfs_type,
        arch_config.dfs_expressions,
        workload.sys_info.iloc[0],
        workload.roofline_peaks,
        kernel_pmc_df,
        debug=False,
    )
    return kernel_dfs


def _stat_or_none(row: pd.Series, column: Optional[str]) -> Optional[float]:
    """Read a top-kernels stat as a float, or None when missing or NaN."""
    if column is None or column not in row.index:
        return None
    value = row[column]
    return float(value) if pd.notna(value) else None


def _append_kernel_point(
    plot_points: PlotPoints,
    metrics: dict[str, float],
    kernel_name: str,
    stat_row: pd.Series,
    sum_column: Optional[str],
) -> None:
    """Append one kernel's AI points plus its joined top-kernel stats."""
    for ai_field in CACHE_LEVELS:
        getattr(plot_points, ai_field)[0].append(metrics[ai_field])
        getattr(plot_points, ai_field)[1].append(metrics["performance"])
    plot_points.kernelNames.append(kernel_name)
    plot_points.counts.append(_stat_or_none(stat_row, "Count"))
    plot_points.kernelTime.append(_stat_or_none(stat_row, sum_column))
    plot_points.kernelPctRuntime.append(_stat_or_none(stat_row, "Percent"))


def calc_ai_analyze(
    workload: schema.Workload,
    pmc_df: pd.DataFrame,
    arch_config: schema.ArchConfig,
) -> dict[str, Union[list[list[float]], list[str]]]:
    """
    Calculate per-kernel metrics and AI points with Roofline yamls using eval_metric.

    Caller must apply filters (e.g. parser.apply_filters) and pass the resulting
    DataFrame as pmc_df. This function only computes AI data points.
    """
    console_debug("calc_ai_analyze", "Starting calc_ai analysis using Roofline yamls")
    plot_points = PlotPoints.empty()

    workload.roofline_metrics = {}

    # Aggregate-time column name encodes the time unit, e.g. "Sum(ns)".
    sum_column: Optional[str] = None
    top_df = workload.dfs.get(_KERNEL_TOP_TABLE_ID)
    if top_df is not None:
        sum_column = next(
            (
                column
                for column in top_df.columns
                if column.startswith("Sum(") and column.endswith(")")
            ),
            None,
        )
        if sum_column:
            plot_points.timeUnit = sum_column[len("Sum(") : -1]
            plot_points.totalAppTime = float(top_df[sum_column].sum())
            if "Count" in top_df.columns:
                plot_points.totalAppDispatches = float(top_df["Count"].sum())

    kernel_ids_to_process = _resolve_kernel_ids(workload)
    console_debug("roofline", f"Found {len(kernel_ids_to_process)} kernels to process")

    if not kernel_ids_to_process:
        console_warning("No kernels found to process for roofline")
        return plot_points.__dict__

    for kernel_id in kernel_ids_to_process:
        if top_df is None or kernel_id not in top_df.index:
            continue
        kernel_name = top_df.loc[kernel_id, "Kernel_Name"]

        console_debug("roofline", f"Processing kernel {kernel_id}: {kernel_name[:50]}")

        # filter PMC data for specific kernel
        kernel_pmc_df = pmc_df[pmc_df["Kernel_Name"] == kernel_name]

        if kernel_pmc_df.empty:
            console_debug("roofline", f"No PMC data for kernel {kernel_id}")
            continue

        kernel_dfs = _evaluate_kernel_tables(arch_config, workload, kernel_pmc_df)

        metrics = _extract_ai_metrics(kernel_dfs.get(_ROOFLINE_POINTS_TABLE_ID))
        performance = metrics["performance"]
        console_debug(
            "roofline",
            f"Kernel {kernel_id}: "
            + ", ".join(
                f"{ai_field}={metrics[ai_field]:.2f}" for ai_field in CACHE_LEVELS
            )
            + f", Performance={performance:.2e} GFLOP/s",
        )

        if performance > 0:
            _append_kernel_point(
                plot_points,
                metrics,
                kernel_name,
                top_df.loc[kernel_id],
                sum_column,
            )
            console_debug(
                "roofline",
                f"Added kernel {kernel_id}: {kernel_name[:50]} to plot points",
            )
        else:
            console_debug(
                "roofline",
                f"Skipping kernel {kernel_id}: {kernel_name[:50]}"
                f" - no performance data",
            )

        # store metrics for display
        workload.roofline_metrics[kernel_id] = {
            "name": kernel_name,
            "ai_table": kernel_dfs.get(_ROOFLINE_RATES_TABLE_ID, pd.DataFrame()),
            "calc_table": kernel_dfs.get(_ROOFLINE_POINTS_TABLE_ID, pd.DataFrame()),
        }

    console_debug("roofline", f"Generated {len(plot_points.kernelNames)} plot points")
    return plot_points.__dict__


def machine_ceilings(
    roofline_parameters: dict[str, Any],
    mspec: MachineSpecs,
    benchmark: Optional[RooflineCsvData] = None,
) -> tuple[list[float], list[float]]:
    """Return positive bandwidth and compute ceilings for one device.

    The device row is read from the parsed roofline.csv given as benchmark, or
    else from roofline.csv in the workload directory.
    """
    roofline_csv: Optional[Path] = None
    if benchmark is None:
        base_dir = _workload_base_dir(roofline_parameters.get("workload_dir"))
        if base_dir is None:
            console_warning(
                "roofline",
                "Unable to read machine ceilings: workload path is absent",
            )
            return [], []
        roofline_csv = Path(base_dir) / "roofline.csv"
    source = "the given benchmark" if benchmark is not None else roofline_csv
    try:
        csv_data = (
            benchmark if benchmark is not None else _parse_roofline_csv(roofline_csv)
        )
        row_index = _resolve_device_row_index(
            csv_data, roofline_parameters["device_id"]
        )
        bandwidths = _device_values(csv_data.columns, row_index, ("Bw",))
        peaks = _device_values(csv_data.columns, row_index, ("Flops", "Ops"))
        if not bandwidths:
            raise ValueError("no usable bandwidth ceilings in roofline.csv")
        if not peaks:
            raise ValueError("no usable compute ceilings in roofline.csv")
    except (
        OSError,
        csv.Error,
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        OverflowError,
    ) as error:
        console_warning(
            "roofline",
            f"Unable to read machine ceilings from {source}: {error}",
        )
        return [], []
    console_debug(
        "roofline",
        f"Loaded machine ceilings for {mspec.gpu_model}: "
        f"{len(bandwidths)} bandwidth, {len(peaks)} compute values",
    )
    return bandwidths, peaks


def construct_roof(
    roofline_parameters: dict[str, Any],
    dtype: str,
    mspec: MachineSpecs,
    benchmark: Optional[RooflineCsvData] = None,
) -> dict[str, list[Union[list[float], float, None]]]:
    """Compute the empirical roofline from benchmark results.

    The results are the parsed roofline.csv given as benchmark, or else are read
    from disk in the workload directory.
    """
    base_dir = _workload_base_dir(roofline_parameters.get("workload_dir"))
    source = "the given benchmark" if benchmark is not None else base_dir

    try:
        csv_data = (
            benchmark
            if benchmark is not None
            else _parse_roofline_csv(Path(base_dir) / "roofline.csv")
        )
        device_row_index = _resolve_device_row_index(
            csv_data, roofline_parameters["device_id"]
        )
    except (
        OSError,
        csv.Error,
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        OverflowError,
    ) as error:
        console_error(
            "roofline",
            f"Failed to read benchmark results from {source}: {error}",
            exit=False,
        )
        return GraphPoints.empty().__dict__

    benchmark_data = csv_data.columns
    expected_columns = _expected_benchmark_columns(roofline_parameters, dtype, mspec)
    missing_columns = [col for col in expected_columns if col not in benchmark_data]
    if missing_columns:
        console_warning(
            f"Missing expected columns in roofline.csv for datatype {dtype}: "
            f"{', '.join(missing_columns)}. "
            "The roofline plot may be incomplete. Consider regenerating "
            "benchmark data or cleaning the directory and re-running the analysis."
        )

    return calc_ceilings(
        roofline_parameters,
        dtype,
        benchmark_data,
        mspec,
        device_row_index,
    )


def _parse_roofline_csv(benchmark_results: Path) -> RooflineCsvData:
    """Parse roofline.csv into device IDs and benchmark columns by row order."""
    device_ids: list[int] = []
    seen_device_ids: set[int] = set()
    columns: dict[str, list[str]] = {}
    headers: list[str] = []
    expected_width = 0

    with open(benchmark_results, newline="", encoding="utf-8") as csvfile:
        for row_number, row in enumerate(csv.reader(csvfile, delimiter=","), start=1):
            if row_number == 1:
                if not row or len(row) < 2:
                    raise ValueError(
                        f"roofline.csv row {row_number} is missing benchmark columns"
                    )
                headers = row[1:]
                seen_headers: set[str] = set()
                for header in headers:
                    if header in seen_headers:
                        raise ValueError(
                            "roofline.csv contains duplicate benchmark "
                            f"header {header!r}"
                        )
                    seen_headers.add(header)
                columns = {header: [] for header in headers}
                expected_width = len(headers) + 1
                continue

            if not row or all(cell.strip() == "" for cell in row):
                raise ValueError(f"roofline.csv row {row_number} is empty")

            if len(row) != expected_width:
                raise ValueError(
                    f"roofline.csv row {row_number} has {len(row)} columns, "
                    f"expected {expected_width}"
                )

            device_id = _parse_integral_id(row[0], label="device id")
            if device_id in seen_device_ids:
                raise ValueError(
                    f"roofline.csv contains duplicate device id {device_id}"
                )
            seen_device_ids.add(device_id)
            device_ids.append(device_id)
            for column_index, header in enumerate(headers):
                columns[header].append(row[column_index + 1])

    return RooflineCsvData(device_ids=device_ids, columns=columns)


def _parse_requested_device_id(value: Union[int, float, str]) -> int:
    """Validate and normalize a requested roofline device id."""
    device_id = _parse_integral_id(value, label="device_id")
    if device_id < 0:
        raise ValueError(f"device_id must be non-negative, got {device_id}")
    return device_id


def _resolve_device_row_index(
    csv_data: RooflineCsvData,
    requested_device_id: Union[int, float, str],
) -> int:
    """Return the unique CSV row index for a requested semantic device id."""
    device_id = _parse_requested_device_id(requested_device_id)
    matching_rows = [
        row_index
        for row_index, csv_device_id in enumerate(csv_data.device_ids)
        if csv_device_id == device_id
    ]
    if len(matching_rows) != 1:
        raise ValueError(
            f"device id {device_id} occurs {len(matching_rows)} times in roofline.csv"
        )
    return matching_rows[0]


def _parse_integral_id(value: Union[int, float, str], *, label: str) -> int:
    """Parse an integral device id from CSV cells or request parameters."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must not be a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{label} must be an integral value, got {value}")
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{label} must be an integral value, got empty string")
        try:
            numeric = float(stripped)
        except ValueError as error:
            raise ValueError(
                f"{label} must be an integral value, got {value!r}"
            ) from error
        if not numeric.is_integer():
            raise ValueError(f"{label} must be an integral value, got {value!r}")
        return int(numeric)
    raise ValueError(f"{label} must be an integral value, got {type(value).__name__}")


def _expected_benchmark_columns(
    roofline_parameters: dict[str, Any], dtype: str, mspec: MachineSpecs
) -> list[str]:
    """Benchmark columns the roofline needs for this datatype and mem levels."""
    columns: list[str] = []

    if OpsSupport.VALU in SUPPORTED_DATATYPES[mspec.gpu_arch][dtype]:
        columns.append(roofline_column(dtype, "valu"))

    cache_hierarchy = roofline_mem_levels(mspec.gpu_arch, mspec.gpu_model)
    columns.extend(roofline_column(level, "bandwidth") for level in cache_hierarchy)

    if OpsSupport.MATRIX in SUPPORTED_DATATYPES[mspec.gpu_arch][dtype]:
        columns.append(roofline_column(dtype, roofline_parameters["matrix_ops_type"]))
    return columns


def _device_values(
    benchmark_data: dict[str, list[str]],
    row_index: int,
    column_suffixes: tuple[str, ...],
) -> list[float]:
    """Collect positive finite values from columns ending in the given suffixes."""
    values: list[float] = []
    for column, rows in benchmark_data.items():
        if not any(column.endswith(suffix) for suffix in column_suffixes):
            continue
        sanitized = sanitize_ai_value(rows[row_index])
        if sanitized > 0:
            values.append(sanitized)
    return values


# -------------------------------------------------------------------------------------
#                       Ceilings stored in the analysis database
# -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CeilingColumn:
    """The roof a roofline.csv benchmark mean column measures."""

    kind: str  # "bandwidth" or "compute"
    memory_level: Optional[str]  # bandwidth only, e.g. "HBM"
    data_type: Optional[str]  # compute only, e.g. "FP16"
    pipe: Optional[str]  # compute only: "VALU", "MFMA" or "WMMA"
    config_unit: str  # unit of the benchmark mean, a key of SCALED_UNITS


@dataclass(frozen=True)
class Ceiling:
    """One roofline ceiling of a workload, in the base unit the database stores."""

    kind: str
    memory_level: Optional[str]
    data_type: Optional[str]
    pipe: Optional[str]
    unit: str
    value: float


def _ceiling_columns(gpu_arch: str) -> dict[str, CeilingColumn]:
    """Map each roofline.csv benchmark mean column of gpu_arch to the roof it measures.

    Low and High bounds are not means, so no column of theirs gets an entry.
    """
    columns: dict[str, CeilingColumn] = {}
    for level in mi_gpu_specs.get_roofline_bandwidth(gpu_arch):
        columns[roofline_column(level, "bandwidth")] = CeilingColumn(
            kind="bandwidth",
            memory_level=level,
            data_type=None,
            pipe=None,
            config_unit="GB/s",
        )

    matrix_type = mi_gpu_specs.get_roofline_matrix_type(gpu_arch)
    compute_pipes: list[tuple[str, str]] = []  # (data type, pipe)
    for data_type, valu, matrix in mi_gpu_specs.get_roofline_data_types(gpu_arch):
        if valu:
            compute_pipes.append((data_type, "VALU"))
        if matrix:
            compute_pipes.append((data_type, matrix_type))
    compute_pipes.extend(
        (data_type, matrix_type)
        for data_type in mi_gpu_specs.get_roofline_mixed_data_types(gpu_arch)
    )
    for data_type, pipe in compute_pipes:
        column = roofline_column(data_type, "valu" if pipe == "VALU" else pipe)
        columns[column] = CeilingColumn(
            kind="compute",
            memory_level=None,
            data_type=data_type,
            pipe=pipe,
            # Integer ops are measured in GOPS, floating-point ops in GFLOPS.
            config_unit="GIOP/s" if column.endswith("Ops") else "GFLOP/s",
        )
    return columns


def device_ceilings(benchmark_results: Path, gpu_arch: str) -> list[Ceiling]:
    """Return the ceilings of the device-0 row of roofline.csv, in base units.

    Every positive benchmark mean of gpu_arch is one ceiling, MALL included. The
    caller validates the file first; a malformed file or a missing device-0 row
    raises ValueError, OSError or csv.Error.
    """
    csv_data = _parse_roofline_csv(benchmark_results)
    row_index = _resolve_device_row_index(csv_data, ROOFLINE_DEVICE_ID)
    ceilings: list[Ceiling] = []
    for column, roof in _ceiling_columns(gpu_arch).items():
        if column not in csv_data.columns:
            continue
        value = sanitize_ai_value(csv_data.columns[column][row_index])
        if value <= 0:
            continue
        unit, scale = SCALED_UNITS[roof.config_unit]
        ceilings.append(
            Ceiling(
                kind=roof.kind,
                memory_level=roof.memory_level,
                data_type=roof.data_type,
                pipe=roof.pipe,
                unit=unit,
                value=value * scale,
            )
        )
    return ceilings


def benchmark_from_ceilings(
    ceilings: Sequence[Ceiling],
    gpu_arch: str,
) -> RooflineCsvData:
    """Rebuild the device-0 roofline.csv row the device_ceilings ceilings came from.

    The cells are in the CSV's own units, so the ceiling math reads them as it reads
    the file. A benchmark column with no ceiling row is left out, as a roofline.csv
    without that column would leave it.
    """
    layout = _ceiling_columns(gpu_arch)
    column_by_roof = {
        (column.kind, column.memory_level, column.data_type, column.pipe): name
        for name, column in layout.items()
    }
    columns: dict[str, list[str]] = {}
    for ceiling in ceilings:
        roof = (ceiling.kind, ceiling.memory_level, ceiling.data_type, ceiling.pipe)
        if roof not in column_by_roof:
            raise ValueError(f"no roofline.csv column for ceiling {roof}")
        column = column_by_roof[roof]
        value = ceiling.value / SCALED_UNITS[layout[column].config_unit][1]
        columns[column] = [str(value)]
    return RooflineCsvData(device_ids=[ROOFLINE_DEVICE_ID], columns=columns)
