# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for kernel-name filtering during profiling."""

import inspect

import common

from tests.integration import common as integration_common
from tests.integration.common import (
    CSVS,
    config,
    num_devices,
    num_kernels,
    validate,
)


def test_kernel(binary_handler_profile_rocprof_compute):
    options = ["--kernel", config["kernel_name_1"]]
    workload_dir = common.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = integration_common.check_csv_files(
        workload_dir, num_devices, num_kernels
    )
    assert sorted(list(file_dict.keys())) == CSVS

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    common.clean_output_dir(config["cleanup"], workload_dir)
