# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Profile-mode roofline sort integration tests.

Extracted verbatim from tests/test_profile_general.py during the test-suite
refactor; no logic changes.
"""

import csv
import inspect
import os
import re
import socket
import subprocess
import sqlite3
import sys
import time
from pathlib import Path

import common
import numpy as np
import pandas as pd
import pytest
import yaml

from _profile_helpers import (
    AVAILABLE_SETS,
    CSVS,
    DEFAULT_ABS_DIFF,
    DEFAULT_REL_DIFF,
    GPU_ARCH,
    GPU_MODEL,
    MAX_REOCCURING_COUNT,
    METRIC_THRESHOLDS,
    MockMachineSpecs,
    MockProfiler,
    MockSoc,
    ROOF_ONLY_FILES,
    SLURM_RANK_VAR,
    SLURM_SIZE_VAR,
    are_deterministic_counters_equal,
    are_stochastic_counters_similar,
    arch_config,
    attach_detach_interval_msec_no_delay,
    attach_detach_interval_msec_with_delay,
    baseline_compare_metric,
    clear_rank_env,
    config,
    counter_compare,
    get_available_sets_for_arch,
    gpu_arch,
    is_rdna35_halo_soc,
    mock_generate_machine_specs,
    mock_load_soc_specs,
    num_devices,
    num_kernels,
    skip_unsupported_roofline_soc,
    soc,
    validate,
)

@pytest.mark.sort
def test_roof_sort_dispatches(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    """Profile creates CSV; analyze with --sort dispatches generates output."""
    skip_unsupported_roofline_soc()

    profile_options = ["--device", "0", "--roof-only"]
    workload_dir = common.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, profile_options, check_success=False, roof=True
    )
    assert returncode == 0

    file_dict = common.check_csv_files(workload_dir, 1, num_kernels)
    assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--sort",
        "dispatches",
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    validate(inspect.stack()[0][3], workload_dir, file_dict)
    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.sort
def test_roof_sort_kernels(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    """Profile creates CSV; analyze with --sort kernels generates output."""
    skip_unsupported_roofline_soc()

    profile_options = ["--device", "0", "--roof-only"]
    workload_dir = common.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, profile_options, check_success=False, roof=True
    )
    assert returncode == 0

    file_dict = common.check_csv_files(workload_dir, 1, num_kernels)
    assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--sort",
        "kernels",
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    validate(inspect.stack()[0][3], workload_dir, file_dict)
    common.clean_output_dir(config["cleanup"], workload_dir)
