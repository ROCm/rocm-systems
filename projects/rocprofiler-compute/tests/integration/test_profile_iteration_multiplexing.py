# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for iteration multiplexing."""

import os

import common
import pandas as pd
import pytest

from tests.integration import common as integration_common
from tests.integration.common import (
    CSVS,
    DETERMINISTIC_ACCURACY_BLOCKS,
    STOCHASTIC_ACCURACY_BLOCKS,
    are_deterministic_counters_equal,
    are_stochastic_counters_similar,
    config,
    num_devices,
    num_kernels,
    skip_unsupported_roofline_soc,
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

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.iteration_multiplexing_2
@pytest.mark.xfail(
    reason="Multiple profiling workloads mapped to the same GPU corrupts the counters"
)
def test_iteration_multiplexing_deterministic_counter_accuracy(
    binary_handler_profile_rocprof_compute,
):
    skip_unsupported_roofline_soc()

    # Common panels cover instruction, dispatch and roofline counters.
    options = ["--block", *DETERMINISTIC_ACCURACY_BLOCKS]
    workload_dir = common.get_output_dir(param_id="no_iter_mplx")
    workload_dirs = [workload_dir]
    try:
        _ = binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=True,
            roof=False,
            app_name="app_laplace_eqn",
        )
        integration_common.check_csv_files(workload_dir, num_devices, num_kernels)
        counters_no_multiplexing = integration_common.read_counter_results(workload_dir)

        options = [
            "--block",
            *DETERMINISTIC_ACCURACY_BLOCKS,
            "--iteration-multiplexing",
            "kernel",
        ]
        workload_dir = common.get_output_dir(param_id="iter_mplx_kernel")
        workload_dirs.append(workload_dir)
        _ = binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=True,
            roof=False,
            app_name="app_laplace_eqn_iter",
        )
        integration_common.check_csv_files(workload_dir, num_devices, num_kernels)
        counters_kernel = integration_common.read_counter_results(workload_dir)

        options = [
            "--block",
            *DETERMINISTIC_ACCURACY_BLOCKS,
            "--iteration-multiplexing",
            "kernel_launch_params",
        ]
        workload_dir_klp = common.get_output_dir(param_id="iter_mplx_params")
        workload_dirs.append(workload_dir_klp)
        _ = binary_handler_profile_rocprof_compute(
            config,
            workload_dir_klp,
            options,
            check_success=True,
            roof=True,
            app_name="app_laplace_eqn_iter",
        )
        integration_common.check_csv_files(workload_dir_klp, num_devices, num_kernels)
        counters_kernel_launch_params = integration_common.read_counter_results(
            workload_dir_klp
        )

        comparison_failures = []
        for policy, counters in (
            ("kernel", counters_kernel),
            ("kernel_launch_params", counters_kernel_launch_params),
        ):
            counters_equal, comparison_reason = are_deterministic_counters_equal(
                [counters], counters_no_multiplexing
            )
            print(f"{policy}: {comparison_reason}")
            if not counters_equal:
                comparison_failures.append(f"{policy}: {comparison_reason}")
        assert not comparison_failures, comparison_failures

        assert os.path.exists(f"{workload_dir_klp}/roofline.csv")
        roofline_df = pd.read_csv(f"{workload_dir_klp}/roofline.csv")
        assert len(roofline_df) >= num_devices
    finally:
        for output_dir in workload_dirs:
            common.clean_output_dir(config["cleanup"], output_dir)


@pytest.mark.iteration_multiplexing_stochastic
def test_iteration_multiplexing_stochastic_counter_accuracy(
    binary_handler_profile_rocprof_compute,
):
    skip_unsupported_roofline_soc()

    workload_dir = common.get_output_dir(param_id="no_iter_mplx")
    workload_dirs = [workload_dir]
    # Memory Chart and Roofline cover cache traffic across architectures.
    options = ["--block", *STOCHASTIC_ACCURACY_BLOCKS]
    try:
        _ = binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=True,
            roof=False,
            app_name="app_laplace_eqn",
        )
        integration_common.check_csv_files(workload_dir, num_devices, num_kernels)
        counters_no_multiplexing = integration_common.read_counter_results(workload_dir)

        options = [
            "--block",
            *STOCHASTIC_ACCURACY_BLOCKS,
            "--iteration-multiplexing",
            "kernel",
        ]
        workload_dir = common.get_output_dir(param_id="iter_mplx_kernel")
        workload_dirs.append(workload_dir)
        _ = binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=True,
            roof=False,
            app_name="app_laplace_eqn_iter",
        )
        integration_common.check_csv_files(workload_dir, num_devices, num_kernels)
        counters_kernel = integration_common.read_counter_results(workload_dir)

        options = [
            "--block",
            *STOCHASTIC_ACCURACY_BLOCKS,
            "--iteration-multiplexing",
            "kernel_launch_params",
        ]
        workload_dir_klp = common.get_output_dir(param_id="iter_mplx_params")
        workload_dirs.append(workload_dir_klp)
        _ = binary_handler_profile_rocprof_compute(
            config,
            workload_dir_klp,
            options,
            check_success=True,
            roof=True,
            app_name="app_laplace_eqn_iter",
        )
        integration_common.check_csv_files(workload_dir_klp, num_devices, num_kernels)
        counters_kernel_launch_params = integration_common.read_counter_results(
            workload_dir_klp
        )

        comparison_failures = []
        for policy, counters in (
            ("kernel", counters_kernel),
            ("kernel_launch_params", counters_kernel_launch_params),
        ):
            print(f"Stochastic comparison policy: {policy}")
            if not are_stochastic_counters_similar(
                [counters], counters_no_multiplexing
            ):
                comparison_failures.append(policy)
        assert not comparison_failures, comparison_failures

        assert os.path.exists(f"{workload_dir_klp}/roofline.csv")
        roofline_df = pd.read_csv(f"{workload_dir_klp}/roofline.csv")
        assert len(roofline_df) >= num_devices
    finally:
        for output_dir in workload_dirs:
            common.clean_output_dir(config["cleanup"], output_dir)


# Not part of automated test runs since testing all counters is expensive


def test_iteration_multiplexing_all_counter_accuracy(
    binary_handler_profile_rocprof_compute,
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
    counters_no_multiplexing = integration_common.read_counter_results(workload_dir)
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
    counters_kernel = integration_common.read_counter_results(workload_dir)
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
    counters_kernel_launch_params = integration_common.read_counter_results(
        workload_dir
    )
    common.clean_output_dir(config["cleanup"], workload_dir)

    counters_equal, comparison_reason = are_deterministic_counters_equal(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )
    assert counters_equal, comparison_reason
    print(comparison_reason)
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
def test_iteration_multiplexing_roofline(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    """Verify roofline analysis on iteration-multiplexed profiling data."""
    skip_unsupported_roofline_soc()

    options = [
        "--block",
        "4",
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir = common.get_output_dir(param_id="iter_mplx_roofline")
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

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    common.clean_output_dir(config["cleanup"], workload_dir)
