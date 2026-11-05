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
from typing import Optional

RUN_FOLDER_PREFIX = "rocprof_compute_analyze"
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
        directory/name when a name is given; it must not already exist.
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
