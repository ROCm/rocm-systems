# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path

# NB: Creating a new module to share global vars across modules
rocprof_compute_home = Path(__file__).resolve().parent
PROJECT_NAME = "rocprofiler-compute"

HIDDEN_COLUMNS: list[str] = []
HIDDEN_COLUMNS_CLI = ["Description"]
HIDDEN_COLUMNS_TUI: list[str] = []
HIDDEN_SECTIONS = [1900, 2000]

TIME_UNITS = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}
