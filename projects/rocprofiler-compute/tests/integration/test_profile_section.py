# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for block filtering and metric listing."""

import sys
from math import isfinite
from pathlib import Path

import common
import pandas as pd

from tests.integration import common as integration_common
from tests.integration.common import (
    CSVS,
    config,
    num_devices,
    num_kernels,
)
from utils.utils_common import load_yaml


def profile_block_selection(binary_handler, workload_dir, blocks):
    options = ["--block", *blocks]
    binary_handler(config, workload_dir, options, check_success=True, roof=False)

    output_files = integration_common.check_csv_files(
        workload_dir, num_devices, num_kernels
    )
    assert sorted(output_files) == CSVS
    profiling_config = load_yaml(Path(workload_dir) / "profiling_config.yaml")
    assert profiling_config["filter_blocks"] == blocks

    counter_results = integration_common.read_counter_results(workload_dir)
    selection = f"Block {' '.join(blocks)}"
    assert not counter_results.empty, f"{selection} produced no counter rows"
    counter_names = counter_results["Counter_Name"]
    assert counter_names.notna().all(), f"{selection} produced missing counter names"
    assert counter_names.str.strip().ne("").all(), (
        f"{selection} produced empty counter names"
    )
    counter_values = pd.to_numeric(counter_results["Counter_Value"], errors="coerce")
    assert counter_values.map(isfinite).all(), (
        f"{selection} produced invalid counter values"
    )
    return output_files, set(counter_names)


def test_block_filter(binary_handler_profile_rocprof_compute, capfd):
    # Memory Chart is block 3 across the supported architecture configurations.
    workload_dir = common.get_output_dir()
    output_files, counter_names = profile_block_selection(
        binary_handler_profile_rocprof_compute, workload_dir, ["3"]
    )
    captured = capfd.readouterr()
    # Preserve profiling diagnostics for normal pytest output and saved run logs.
    print(captured.out, end="")
    print(captured.err, end="", file=sys.stderr)
    gpu_arch = output_files["sysinfo.csv"]["gpu_arch"].iloc[0]
    tables = integration_common.metric_table_counters(gpu_arch, 3)
    expected_counters = set().union(*tables.values())
    integration_common.assert_selected_counters(
        counter_names, expected_counters, "Block 3", captured.out + "\n" + captured.err
    )
    common.clean_output_dir(config["cleanup"], workload_dir)


def test_list_metrics(binary_handler_profile_rocprof_compute):
    options = ["--list-metrics", "gfx90a"]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )
    # workload dir should not exist
    assert not Path(workload_dir).exists()
    common.clean_output_dir(config["cleanup"], workload_dir)


def test_list_metrics_with_block(binary_handler_profile_rocprof_compute):
    options = ["--list-metrics", "gfx90a", "--block", "10"]
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
