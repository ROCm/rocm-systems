# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for per-section profiling and metric listing."""

import inspect
from pathlib import Path

import common

from tests.integration import common as integration_common
from tests.integration.common import (
    config,
    is_gfx115x_soc,
    is_gfx1250_soc,
    num_kernels,
    validate,
)
from utils import csv_compression


def test_lds_section(binary_handler_profile_rocprof_compute):
    lds_block = "3" if is_gfx115x_soc() else ("9" if is_gfx1250_soc() else "12")
    options = ["--block", lds_block]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = integration_common.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    assert common.check_file_pattern(
        f"- '{lds_block}'", f"{workload_dir}/profiling_config.yaml"
    )
    lds_counter = "TX_VMW_LDS_INPUT_ACTIVE" if is_gfx1250_soc() else "SQ_INSTS_LDS"
    results_files = csv_compression.find_csvs(workload_dir, "results_*.csv")
    assert any(common.check_file_pattern(lds_counter, str(f)) for f in results_files)
    common.clean_output_dir(config["cleanup"], workload_dir)


def test_instmix_memchart_section(binary_handler_profile_rocprof_compute):
    rdna_or_gfx1250 = is_gfx115x_soc() or is_gfx1250_soc()
    instmix_block = "7" if rdna_or_gfx1250 else "10"
    options = ["--block", instmix_block, "3"]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = integration_common.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    assert common.check_file_pattern(
        f"- '{instmix_block}'", f"{workload_dir}/profiling_config.yaml"
    )
    assert common.check_file_pattern("- '3'", f"{workload_dir}/profiling_config.yaml")
    instmix_counter = "SQ_INSTS_FLAT" if rdna_or_gfx1250 else "TA_FLAT_WAVEFRONTS"
    results_files = csv_compression.find_csvs(workload_dir, "results_*.csv")
    assert any(
        common.check_file_pattern(instmix_counter, str(f)) for f in results_files
    )
    results_files = csv_compression.find_csvs(workload_dir, "results_*.csv")
    assert any(
        common.check_file_pattern("SQC_TC_DATA_READ_REQ", str(f)) for f in results_files
    )
    common.clean_output_dir(config["cleanup"], workload_dir)


def test_lds_sol_section(binary_handler_profile_rocprof_compute):
    lds_sol_block = "3" if is_gfx115x_soc() else ("9.4" if is_gfx1250_soc() else "12.1")
    options = ["--block", lds_sol_block]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = integration_common.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    assert common.check_file_pattern(
        f"- '{lds_sol_block}'", f"{workload_dir}/profiling_config.yaml"
    )
    if is_gfx115x_soc():
        lds_sol_counter = "SQC_LDS_IDX_ACTIVE"
    elif is_gfx1250_soc():
        lds_sol_counter = "TX_VMW_LDS_INPUT_ACTIVE"
    else:
        lds_sol_counter = "SQ_ACTIVE_INST_LDS"
    results_files = csv_compression.find_csvs(workload_dir, "results_*.csv")
    assert any(
        common.check_file_pattern(lds_sol_counter, str(f)) for f in results_files
    )
    common.clean_output_dir(config["cleanup"], workload_dir)


def test_instmix_section_global_write_kernel(binary_handler_profile_rocprof_compute):
    rdna_or_gfx1250 = is_gfx115x_soc() or is_gfx1250_soc()
    instmix_block = "7" if rdna_or_gfx1250 else "10"
    options = ["-k", "global_write", "--block", instmix_block]
    custom_config = dict(config)
    custom_config["kernel_name_1"] = "global_write"
    custom_config["app_1"] = ["./tests/vmem"]
    num_kernels = 1

    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        custom_config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = integration_common.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    assert common.check_file_pattern(
        f"- '{instmix_block}'", f"{workload_dir}/profiling_config.yaml"
    )
    assert common.check_file_pattern(
        "- global_write", f"{workload_dir}/profiling_config.yaml"
    )
    kernel_counter = "SQ_INSTS_FLAT_STORE" if rdna_or_gfx1250 else "TA_FLAT_WAVEFRONTS"
    results_files = csv_compression.find_csvs(workload_dir, "results_*.csv")
    assert any(common.check_file_pattern(kernel_counter, str(f)) for f in results_files)
    results_files = csv_compression.find_csvs(workload_dir, "results_*.csv")
    assert any(common.check_file_pattern("global_write", str(f)) for f in results_files)
    results_files = csv_compression.find_csvs(workload_dir, "results_*.csv")
    assert not any(
        common.check_file_pattern("global_read", str(f)) for f in results_files
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
