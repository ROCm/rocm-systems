# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Resolve the run folder an analyze run writes into.

The resolver only computes a path. It reads no clock, working directory, or
environment, and creates nothing, so the caller decides when the folder is made
and how errors are reported.
"""

import itertools
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

RUN_FOLDER_PREFIX = "rocprof_compute_analyze"
REPORT_FILE_NAME = "report.txt"
ANALYSIS_DB_FILE_NAME = "analysis.db"
METRICS_SUMMARY_FILE_NAME = "metrics-summary.csv"
PC_SAMPLING_SUMMARY_FILE_NAME = "pc-sampling-summary.csv"
RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def validate_output_name(name: str) -> None:
    """Raise ValueError unless name matches RUN_NAME_PATTERN."""
    if not RUN_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Analysis output file/folder name must contain only alphanumeric "
            "characters or underscores (_), hyphens (-)."
        )


def resolve_run_folder(directory: Path, name: Optional[str], now: datetime) -> Path:
    """Return the run folder path for an analyze run.

    Args:
        directory: Parent of the run folder, taken literally. It may not exist.
        name: User-chosen folder name, or None for a timestamped name.
        now: Local time used for the timestamped name.

    Returns:
        directory/name when a name is given and available. An occupied name
        gets a timestamp suffix, with a numeric suffix added when necessary.
        Otherwise directory/rocprof_compute_analyze_<YYYYMMDD_HHMMSS>, with
        _1, _2, ... appended until the path is free.
    """
    if name is not None:
        validate_output_name(name)
        run_folder = directory / name
        if not run_folder.exists():
            return run_folder

        base_name = f"{name}_{now:%Y%m%d_%H%M%S}"
    else:
        base_name = f"{RUN_FOLDER_PREFIX}_{now:%Y%m%d_%H%M%S}"

    candidates = itertools.chain(
        [directory / base_name],
        (directory / f"{base_name}_{index}" for index in itertools.count(1)),
    )
    return next(path for path in candidates if not path.exists())


class RunLayout(NamedTuple):
    """Where an analyze run writes: the run folder and one folder per workload."""

    run_folder: Path
    # Keyed by the workload path given with -p
    workload_folders: Dict[str, Path]


def plan_run_layout(run_folder: Path, workload_paths: List[str]) -> RunLayout:
    """Return the run folder layout for the -p workload paths, in -p order."""
    return RunLayout(
        run_folder,
        {
            workload_path: run_folder
            / workload_folder_name(position, len(workload_paths))
            for position, workload_path in enumerate(workload_paths, start=1)
        },
    )


def workload_folder_name(position: int, workload_count: int) -> str:
    """Return the subfolder name for the workload at its 1-based -p position."""
    return f"workload-{_zero_padded(position, workload_count)}"


def kernel_folder_name(position: int, kernel_count: int) -> str:
    """Return the subfolder name for a kernel at its zero-based rank."""
    return f"kernel-{_zero_padded(position, kernel_count - 1)}"


def kernel_metrics_file_name(kernel_folder: str) -> str:
    """Return the metrics CSV name of a kernel folder, which repeats the folder name."""
    return f"{kernel_folder}-metrics.csv"


def roofline_html_file_name(device_id: int) -> str:
    """Return the roofline HTML name for the profiled GPU device_id."""
    return f"roofline-gpu-{device_id}.html"


def _zero_padded(number: int, largest: int) -> str:
    """Pad number to the width of largest, so the sorted names keep their order."""
    return str(number).zfill(len(str(largest)))
