# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for iteration multiplexing."""

import inspect
import os

import common
import pandas as pd
import pytest

from tests.integration import common as integration_common
from tests.integration.common import (
    CSVS,
    are_deterministic_counters_equal,
    are_stochastic_counters_similar,
    config,
    num_devices,
    num_kernels,
    skip_unsupported_roofline_soc,
    validate,
)


@pytest.mark.iteration_multiplexing_1
def test_profiler_options(binary_handler_profile_rocprof_compute):
    options = ["--no-native-tool", "--iteration-multiplexing"]
    workload_dir = common.get_output_dir()
    code = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=False
    )
    assert code == 1


@pytest.mark.iteration_multiplexing_1
def test_iteration_multiplexing(binary_handler_profile_rocprof_compute):
    options = ["--iteration-multiplexing"]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

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


@pytest.mark.iteration_multiplexing_1
def test_iteration_multiplexing_kernel(binary_handler_profile_rocprof_compute):
    options = ["--iteration-multiplexing", "kernel"]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

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


@pytest.mark.iteration_multiplexing_1
def test_iteration_multiplexing_kernel_launch_params(
    binary_handler_profile_rocprof_compute,
):
    options = ["--iteration-multiplexing", "kernel_launch_params"]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

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


@pytest.mark.iteration_multiplexing_2
@pytest.mark.xfail(
    reason="Multiple profiling workloads mapped to the same GPU corrupts the counters"
)
def test_iteration_multiplexing_deterministic_counter_accuracy(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    skip_unsupported_roofline_soc()

    # These metrics should cover the deterministic counters being checked
    # Block 4 (roofline) included to verify roofline counters under multiplexing
    options = ["--block", "4", "6.1.5", "6.1.6", "7.2.2", "10.1"]
    workload_dir = common.get_output_dir(param_id="no_iter_mplx")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn",
    )
    integration_common.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_no_multiplexing = pd.read_csv(common.pmc_perf_path(workload_dir))
    common.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
        "4",
        "6.1.5",
        "6.1.6",
        "7.2.2",
        "10.1",
        "--iteration-multiplexing",
        "kernel",
    ]
    workload_dir = common.get_output_dir(param_id="iter_mplx_kernel")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    integration_common.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_kernel = pd.read_csv(common.pmc_perf_path(workload_dir))
    common.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
        "4",
        "6.1.5",
        "6.1.6",
        "7.2.2",
        "10.1",
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir_klp = common.get_output_dir(param_id="iter_mplx_params")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir_klp,
        options,
        check_success=True,
        roof=True,
        app_name="app_laplace_eqn_iter",
    )
    integration_common.check_csv_files(workload_dir_klp, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir_klp])
    counters_kernel_launch_params = pd.read_csv(common.pmc_perf_path(workload_dir_klp))

    assert are_deterministic_counters_equal(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )

    assert os.path.exists(f"{workload_dir_klp}/roofline.csv")
    roofline_df = pd.read_csv(f"{workload_dir_klp}/roofline.csv")
    assert len(roofline_df) >= num_devices

    common.clean_output_dir(config["cleanup"], workload_dir_klp)


@pytest.mark.iteration_multiplexing_stochastic
def test_iteration_multiplexing_stochastic_counter_accuracy(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    soc,
):
    skip_unsupported_roofline_soc()

    if not soc or ("MI300" not in soc and "MI350" not in soc):
        pytest.skip("Skipping stochastic iteration multiplexing for non-CDNA socs.")

    workload_dir = common.get_output_dir(param_id="no_iter_mplx")
    # These metrics should cover the L1 cache stochastic counters
    # Block 4 (roofline) included to verify roofline counters under multiplexing
    options = ["--block", "4", "16.1", "16.3"]
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn",
    )
    integration_common.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_no_multiplexing = pd.read_csv(common.pmc_perf_path(workload_dir))
    common.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
        "4",
        "16.1",
        "16.3",
        "--iteration-multiplexing",
        "kernel",
    ]
    workload_dir = common.get_output_dir(param_id="iter_mplx_kernel")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    integration_common.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_kernel = pd.read_csv(common.pmc_perf_path(workload_dir))
    common.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
        "4",
        "16.1",
        "16.3",
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir_klp = common.get_output_dir(param_id="iter_mplx_params")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir_klp,
        options,
        check_success=True,
        roof=True,
        app_name="app_laplace_eqn_iter",
    )
    integration_common.check_csv_files(workload_dir_klp, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir_klp])
    counters_kernel_launch_params = pd.read_csv(common.pmc_perf_path(workload_dir_klp))

    assert are_stochastic_counters_similar(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )

    assert os.path.exists(f"{workload_dir_klp}/roofline.csv")
    roofline_df = pd.read_csv(f"{workload_dir_klp}/roofline.csv")
    assert len(roofline_df) >= num_devices

    common.clean_output_dir(config["cleanup"], workload_dir_klp)


# Not part of automated test runs since testing all counters is expensive


def test_iteration_multiplexing_all_counter_accuracy(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    workload_dir = common.get_output_dir(param_id="no_iter_mplx")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn",
    )
    integration_common.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_no_multiplexing = pd.read_csv(common.pmc_perf_path(workload_dir))
    common.clean_output_dir(config["cleanup"], workload_dir)

    options = ["--iteration-multiplexing", "kernel"]
    workload_dir = common.get_output_dir(param_id="iter_mplx_kernel")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    integration_common.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_kernel = pd.read_csv(common.pmc_perf_path(workload_dir))
    common.clean_output_dir(config["cleanup"], workload_dir)

    options = ["--iteration-multiplexing", "kernel_launch_params"]
    workload_dir = common.get_output_dir(param_id="iter_mplx_params")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    integration_common.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_kernel_launch_params = pd.read_csv(common.pmc_perf_path(workload_dir))
    common.clean_output_dir(config["cleanup"], workload_dir)

    assert are_deterministic_counters_equal(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )
    assert are_stochastic_counters_similar(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )


@pytest.mark.iteration_multiplexing_2
def test_iteration_multiplexing_insufficient_dispatches(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
):
    """Verify graceful degradation when dispatches are too few for full
    counter coverage under iteration multiplexing.
    """
    options = [
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir = common.get_output_dir(param_id="iter_mplx_insufficient")
    binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_insufficient",
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    captured = capsys.readouterr()
    assert "missing counter data" in captured.out

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.iteration_multiplexing_2
def test_iteration_multiplexing_data_types(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    """Verify roofline analysis with different data types (FP32 and FP16)
    on iteration-multiplexed profiling data.
    """
    skip_unsupported_roofline_soc()

    options = [
        "--block",
        "4",
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir = common.get_output_dir(param_id="iter_mplx_dtypes")
    binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=True,
        app_name="app_laplace_eqn",
    )

    assert os.path.exists(f"{workload_dir}/roofline.csv")
    roofline_df = pd.read_csv(f"{workload_dir}/roofline.csv")
    assert len(roofline_df) >= num_devices

    code_fp32 = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "FP32",
    ])
    assert code_fp32 == 0

    code_fp16 = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "FP16",
    ])
    assert code_fp16 == 0

    common.clean_output_dir(config["cleanup"], workload_dir)
