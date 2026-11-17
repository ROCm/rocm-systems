# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Profile-side retention of ML API trace inputs."""

import shutil
from pathlib import Path

from utils import csv_compression
from utils.logger import console_log, demarcate


@demarcate
def save_ml_api_trace_inputs(
    workload_dir: str,
    fbase: str,
    src_counter: Path,
    src_marker: Path,
) -> None:
    """Copy the counter and marker CSVs into the workload directory for analyze."""
    dst_counter = csv_compression.compressed_name(
        Path(workload_dir) / f"ml_api_trace_{fbase}_counter_collection.csv"
    )
    dst_marker = csv_compression.compressed_name(
        Path(workload_dir) / f"ml_api_trace_{fbase}_marker_api_trace.csv"
    )
    # These files are expected to exist.
    shutil.copyfile(src_counter, dst_counter)
    shutil.copyfile(src_marker, dst_marker)
    console_log(
        "ml api trace",
        "Copied counter collection and marker trace files to the workload directory.",
    )
    console_log("Counter Collection: ", str(dst_counter))
    console_log("Marker API Trace: ", str(dst_marker))
