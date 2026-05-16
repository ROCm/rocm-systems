# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Profile-mode miscellaneous integration tests.

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

@pytest.mark.misc
def test_analyze_rocpd(
    binary_handler_profile_rocprof_compute, binary_handler_analyze_rocprof_compute
):
    skip_unsupported_roofline_soc()

    workload_dir = common.get_output_dir()
    options = ["--device", "0", "--format-rocprof-output", "rocpd"]
    binary_handler_profile_rocprof_compute(config, workload_dir, options, roof=True)

    db_name = "test"
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--output-format",
        "db",
        "--output-name",
        f"{db_name}",
        "--path",
        workload_dir,
    ])
    assert code == 0
    assert os.path.isfile(f"{db_name}.db")

    # Open the sqlite database and assert the schema
    # Import Kernel from analysis_orm.py

    from utils.analysis_orm import (
        Dispatch,
        Kernel,
        KernelMetricValue,
        KernelRooflineData,
        Metadata,
        MetricDefinition,
        Workload,
        WorkloadMetricValue,
        WorkloadRooflineData,
    )

    table_name_map = {
        "compute_workload": Workload,
        "compute_metric_definition": MetricDefinition,
        "compute_kernel_roofline_data": KernelRooflineData,
        "compute_workload_roofline_data": WorkloadRooflineData,
        "compute_dispatch": Dispatch,
        "compute_kernel": Kernel,
        "compute_kernel_metric_value": KernelMetricValue,
        "compute_workload_metric_value": WorkloadMetricValue,
        "compute_metadata": Metadata,
    }

    def check_cols(table_name, orm_obj):
        conn = sqlite3.connect(f"{db_name}.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        expected_columns = [col.name for col in orm_obj.__table__.columns]
        assert column_names == expected_columns
        conn.close()

    for table_name, orm_obj in table_name_map.items():
        check_cols(table_name, orm_obj)

    os.remove(f"{db_name}.db")
    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.misc
def test_save_csv(
    binary_handler_profile_rocprof_compute, binary_handler_analyze_rocprof_compute
):
    workload_dir = common.get_output_dir(param_id="profile")
    analysis_workload_dir = common.get_output_dir(param_id="analysis")
    options = ["--format-rocprof-output", "rocpd"]
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--output-format",
        "csv",
        "--output-name",
        analysis_workload_dir,
        "--path",
        workload_dir,
    ])
    assert code == 0

    csv_dir = Path(analysis_workload_dir)
    assert csv_dir.is_dir()

    expected_view_csvs = ["kernel.csv", "kernel_metric.csv", "workload_metric.csv"]
    for csv_name in expected_view_csvs:
        csv_path = csv_dir / csv_name
        assert csv_path.is_file(), f"Missing per-view CSV: {csv_path}"
        df = pd.read_csv(csv_path)
        assert len(df.index) >= 1, f"Per-view CSV is empty: {csv_path}"

    assert not Path(f"{analysis_workload_dir}.db").exists()

    common.clean_output_dir(config["cleanup"], analysis_workload_dir)
    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.misc
def test_device_filter(binary_handler_profile_rocprof_compute):
    options = ["--device", "0"]
    workload_dir = common.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = common.check_csv_files(workload_dir, 1, num_kernels)
    assert sorted(list(file_dict.keys())) == CSVS

    # TODO - verify expected device id in results

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.parametrize(
    "workload_cmd, expected_exit",
    [
        pytest.param(
            ["python3", "nonexistent_script_abc.py"],
            1,
            id="missing_script",
        ),
        pytest.param(
            ["python3"],
            1,
            id="bare_interpreter",
        ),
        pytest.param(
            ["python3", "-u", "-v"],
            1,
            id="flags_only",
        ),
        pytest.param(
            ["python3", "-u", "nonexistent_script_abc.py"],
            1,
            id="missing_script_after_flags",
        ),
        pytest.param(
            ["nonexistentpython3", "script.py"],
            1,
            id="nonexistent_executable",
        ),
        pytest.param(
            ["./no_such_binary"],
            1,
            id="nonexistent_binary",
        ),
        pytest.param(
            ["python3", "-c", "print('hello')"],
            0,
            id="non_gpu_workload",
        ),
    ],
)
def test_profile_invalid_workloads_no_torch_trace(
    binary_handler_profile_rocprof_compute,
    workload_cmd,
    expected_exit,
    request,
):
    """Integration test: workload validation exit codes without --torch-trace."""
    app_name = "test_invalid_workload"
    test_config = {**config, app_name: workload_cmd}

    workload_dir = common.get_output_dir(
        param_id=f"invalid_wl_{request.node.callspec.id}"
    )

    returncode, stdout, stderr = binary_handler_profile_rocprof_compute(
        test_config,
        workload_dir,
        options=[],
        check_success=False,
        app_name=app_name,
        capture_output=True,
    )

    assert returncode == expected_exit, (
        f"Expected exit code {expected_exit} for {workload_cmd}, "
        f"got {returncode}.\nstdout: {stdout}\nstderr: {stderr}"
    )

    common.clean_output_dir(config["cleanup"], workload_dir)
