# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for assorted profile CLI options."""

import inspect
import os
import sqlite3
from pathlib import Path

import common
import pandas as pd
import pytest

from tests.integration import common as integration_common
from tests.integration.common import (
    CSVS,
    config,
    num_kernels,
    skip_unsupported_roofline_soc,
    validate,
)


@pytest.mark.misc
def test_analyze_rocpd(
    binary_handler_profile_rocprof_compute, binary_handler_analyze_rocprof_compute
):
    skip_unsupported_roofline_soc()

    workload_dir = common.get_output_dir()
    options = ["--device", "0"]
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
    binary_handler_profile_rocprof_compute(config, workload_dir)

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


def test_pc_sampling_requires_experimental(binary_handler_profile_rocprof_compute):
    """
    --pc-sampling must be rejected at argparse time when --experimental is
    not also passed (ExperimentalAction gating). This fires before hardware
    detection, so it is intentionally not gated on SoC support.
    """
    options = ["--pc-sampling"]
    workload_dir = common.get_output_dir()

    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=False
    )

    assert returncode != 0, (
        "Expected --pc-sampling without --experimental to fail, "
        f"but command exited with {returncode}"
    )

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_device_filter(binary_handler_profile_rocprof_compute):
    options = ["--device", "0"]
    workload_dir = common.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = integration_common.check_csv_files(workload_dir, 1, num_kernels)
    assert sorted(list(file_dict.keys())) == CSVS

    # TODO - verify expected device id in results

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    common.clean_output_dir(config["cleanup"], workload_dir)
