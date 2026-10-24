# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""State for one invocation of the nightly tarball test run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunContext:
    """Paths, CLI arguments, and the fact sheet shared by every phase.

    ``facts`` is what the summary and RESULT files are rendered from. Logging
    and step timings live here too, so a failure in any phase can still write
    a summary without module-level globals.
    """

    args: argparse.Namespace
    workdir: Path
    rocm_dir: Path
    facts: dict
    run_stamp: str
    detail_log: Path
    summary_log: Path
    env: dict | None = None
    src_dir: Path | None = None
    venv_py: Path | None = None
    tiers: dict | None = None
    rocm_updated: bool = False
    source_changed: bool = False
    detail_fh: object | None = None
    step_n: int = 0
    step_title: str | None = None
    step_start: float = 0.0
    step_times: list[tuple[str, float]] = field(default_factory=list)
    run_start: float = 0.0
