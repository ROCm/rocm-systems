# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for kernel iteration filtering during profiling."""

import inspect

import common
import pytest

from tests.integration import common as integration_common
from tests.integration.common import (
    CSVS,
    config,
    num_devices,
    validate,
)


def test_dispatch_0(binary_handler_profile_rocprof_compute):
    options = ["--kernel-iteration-range", "1"]
    workload_dir = common.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = integration_common.check_csv_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == CSVS

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
        [
            "--dispatch",
            "1",
        ],
    )

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_dispatch_0_1(binary_handler_profile_rocprof_compute):
    options = ["--kernel-iteration-range", "1:2"]
    workload_dir = common.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = integration_common.check_csv_files(workload_dir, num_devices, 2)
    assert sorted(list(file_dict.keys())) == CSVS

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
        ["--dispatch", "1", "2"],
    )

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_dispatch_2(binary_handler_profile_rocprof_compute):
    options = ["--kernel-iteration-range", "1"]
    workload_dir = common.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = integration_common.check_csv_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == CSVS

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
        [
            "--dispatch",
            "1",
        ],
    )

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.parametrize(
    "bad_value",
    ["0", "-1", "abc", "1:0", "5:3", "1:", ":3", "1:2:3"],
)
def test_dispatch_invalid_rejected(binary_handler_profile_rocprof_compute, bad_value):
    workload_dir = common.get_output_dir(
        param_id=f"dispatch_{bad_value}".replace(":", "_")
    )
    returncode, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        ["--kernel-iteration-range", bad_value],
        check_success=False,
        capture_output=True,
    )
    assert returncode == 1
    output = stdout + stderr
    assert f"Invalid --kernel-iteration-range value '{bad_value}'" in output
    common.clean_output_dir(config["cleanup"], workload_dir)
