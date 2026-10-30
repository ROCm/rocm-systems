# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for kernel-name filtering during profiling."""

from pathlib import Path

import common

from tests.integration import common as integration_common
from tests.integration.common import (
    CSVS,
    config,
    num_devices,
)
from utils.utils_common import load_yaml


def test_kernel(binary_handler_profile_rocprof_compute):
    custom_config = dict(config)
    custom_config["kernel_name_1"] = "global_write"
    custom_config["app_1"] = ["./tests/vmem"]
    options = ["--kernel", custom_config["kernel_name_1"]]
    workload_dir = common.get_output_dir()
    binary_handler_profile_rocprof_compute(
        custom_config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = integration_common.check_csv_files(
        workload_dir, num_devices, num_kernels=1
    )
    assert sorted(list(file_dict.keys())) == CSVS

    profiling_config = load_yaml(Path(workload_dir) / "profiling_config.yaml")
    assert profiling_config["kernel"] == ["global_write"]

    counters = integration_common.read_counter_results(workload_dir)
    kernel_names = common.normalize_kernel_names(set(counters["Kernel_Name"]))
    assert kernel_names and all("global_write" in name for name in kernel_names), (
        kernel_names
    )

    common.clean_output_dir(config["cleanup"], workload_dir)
