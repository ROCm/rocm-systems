# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Per-kernel roofline data export, alongside the interactive roofline HTML."""

from pathlib import Path
from typing import Any, Union

import pandas as pd

from utils.logger import console_log

KERNEL_ROOFLINE_CSV_FILENAME = "roofline_kernels.csv"

KERNEL_ROOFLINE_CSV_COLUMNS = [
    "Kernel_Name",
    "Cache_Level",
    "Arithmetic_Intensity",
    "Performance_GFLOPs",
    "Roofline_Peak_GFLOPs",
    "Pct_of_Roofline",
    "Bandwidth_GBps",
    "Count",
    "Total_Time",
    "Time_Unit",
    "Pct_Runtime",
]


def collect_kernel_roofline_rows(
    kernels: list[dict[str, Any]], time_unit: str
) -> list[dict[str, Any]]:
    """Flatten the FLOP figure's per-kernel view-model data into tidy rows.

    One row per kernel per memory level actually plotted, read straight from
    RooflineViewModel.kernels (built by Roofline._build_kernel_traces), so
    these numbers can never diverge from what the interactive HTML shows.
    """
    rows: list[dict[str, Any]] = []
    for kernel in kernels:
        for point in kernel["points"]:
            rows.append({
                "Kernel_Name": kernel["name"],
                "Cache_Level": point["peak"],
                "Arithmetic_Intensity": point["ai"],
                "Performance_GFLOPs": point["perf"],
                "Roofline_Peak_GFLOPs": point["roofPerf"],
                "Pct_of_Roofline": point["pctRoof"],
                "Bandwidth_GBps": point["bandwidth"],
                "Count": kernel.get("count"),
                "Total_Time": kernel.get("totalTime"),
                "Time_Unit": time_unit,
                "Pct_Runtime": kernel.get("pctRuntime"),
            })
    return rows


def build_kernel_roofline_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble tidy per-kernel roofline rows into a DataFrame with a fixed
    column order."""
    return pd.DataFrame(rows, columns=KERNEL_ROOFLINE_CSV_COLUMNS)


def write_kernel_roofline_csv(df: pd.DataFrame, workload_dir: Union[str, Path]) -> Path:
    """Write the per-kernel roofline DataFrame to roofline_kernels.csv."""
    output_file = Path(workload_dir) / KERNEL_ROOFLINE_CSV_FILENAME
    df.to_csv(output_file, index=False)
    console_log("roofline", f"Saved per-kernel roofline data to {output_file}")
    return output_file
