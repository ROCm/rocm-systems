# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for block filtering and metric listing."""

from pathlib import Path

import common

from tests.integration import common as integration_common
from tests.integration.common import (
    CSVS,
    config,
    num_devices,
    num_kernels,
)
from utils.utils_common import load_yaml


def test_block_filter(binary_handler_profile_rocprof_compute):
    # Memory Chart is block 3 across the supported architecture configurations.
    blocks = ["3"]
    options = ["--block", *blocks]
    workload_dir = common.get_output_dir()
    binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    output_files = integration_common.check_csv_files(
        workload_dir, num_devices, num_kernels
    )
    assert sorted(output_files) == CSVS
    profiling_config = load_yaml(Path(workload_dir) / "profiling_config.yaml")
    assert profiling_config["filter_blocks"] == blocks

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_list_metrics(binary_handler_profile_rocprof_compute):
    options = ["--list-metrics", "--debug-list-metrics-gfx-id", "gfx90a"]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )
    # workload dir should not exist
    assert not Path(workload_dir).exists()
    common.clean_output_dir(config["cleanup"], workload_dir)


def test_list_metrics_with_block(binary_handler_profile_rocprof_compute):
    options = [
        "--list-metrics",
        "--debug-list-metrics-gfx-id",
        "gfx90a",
        "--block",
        "10",
    ]
    workload_dir = common.get_output_dir()
    code = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=False
    )
    # Should return code 1 since --block cannot be used with --list-metrics
    assert code == 1
    # workload dir should not exist
    assert not Path(workload_dir).exists()
    common.clean_output_dir(config["cleanup"], workload_dir)


def test_list_available_metrics(binary_handler_profile_rocprof_compute, capsys):
    options = ["--list-available-metrics"]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )
    # workload dir should not exist
    assert not Path(workload_dir).exists()
    common.clean_output_dir(config["cleanup"], workload_dir)

    # Test output
    output = capsys.readouterr().out
    assert "--list-available-metrics is deprecated" in output
    assert "0 -> Top Stats" in output
    assert "1 -> System Info" in output


def test_list_available_metrics_with_block(
    binary_handler_profile_rocprof_compute, capsys
):
    options = ["--list-available-metrics", "--block", "10"]
    workload_dir = common.get_output_dir()
    code = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=False
    )
    # Should return code 1 since --block cannot be used with --list-available-metrics
    assert code == 1
    # workload dir should not exist
    assert not Path(workload_dir).exists()
    common.clean_output_dir(config["cleanup"], workload_dir)
