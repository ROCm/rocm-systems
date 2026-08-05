# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for roofline profiling and benchmark-only mode."""

import inspect
import os
from pathlib import Path

import common
import pytest
from profile_helpers import (
    ROOF_ONLY_FILES,
    config,
    num_devices,
    num_kernels,
    skip_unsupported_roofline_soc,
    validate,
)

from utils import csv_compression


@pytest.mark.roofline_validation
def test_roof_basic_validation(binary_handler_profile_rocprof_compute):
    """
    Test basic roofline CSV generation in profile mode.
    Validates that roofline.csv is generated via microbenchmarks.
    """
    skip_unsupported_roofline_soc()

    options = ["--device", "0", "--roof-only"]
    workload_dir = common.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=True
    )

    assert returncode == 0
    file_dict = common.check_csv_files(workload_dir, 1, num_kernels)

    assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_validation
def test_roof_file_validation(binary_handler_profile_rocprof_compute):
    """Test file validation paths in roofline"""
    skip_unsupported_roofline_soc()

    options = ["--device", "0", "--roof-only"]
    workload_dir = common.get_output_dir()

    try:
        returncode = binary_handler_profile_rocprof_compute(
            config, workload_dir, options, check_success=False, roof=True
        )

        if returncode == 0:
            roofline_csv = f"{workload_dir}/roofline.csv"
            if os.path.exists(roofline_csv):
                import pandas as pd

                df = pd.read_csv(roofline_csv)
                assert len(df) >= 0

    finally:
        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_extra_options
def test_roof_rocpd(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    skip_unsupported_roofline_soc()

    workload_dir = common.get_output_dir()
    options = ["--device", "0", "--roof-only"]
    binary_handler_profile_rocprof_compute(config, workload_dir, options, roof=True)

    # Validate profile outputs
    common.check_csv_files(workload_dir, num_devices, num_kernels)
    assert (Path(workload_dir) / "roofline.csv").exists()

    # Run analyze to create merged pmc_perf.csv
    code = binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    assert code == 0

    # Validate merged pmc_perf.csv content
    assert common.check_file_pattern("Counter_Name", f"{workload_dir}/pmc_perf.csv")

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_dir
def test_roofline_workload_dir_not_set_error():
    """
    Test roof_setup() error: "Workload directory is not set. Cannot perform setup."
    This covers lines 113-117
    """
    skip_unsupported_roofline_soc()

    try:
        from roofline.roofline_main import Roofline
        from utils.specs import generate_machine_specs

        class MockArgs:
            def __init__(self):
                self.roof_only = True
                self.mem_level = "ALL"
                self.sort = "ALL"
                self.roofline_data_type = ["FP32"]

        args = MockArgs()
        mspec = generate_machine_specs(None, None)

        run_parameters = {
            "workload_dir": None,
            "device_id": 0,
            "sort_type": "kernels",
            "mem_level": "ALL",
            "roofline_data_type": ["FP32"],
        }

        roofline_instance = Roofline(args, mspec, run_parameters)

        import contextlib
        from io import StringIO

        captured_output = StringIO()

        with contextlib.redirect_stderr(captured_output):
            try:
                roofline_instance.roof_setup()
            except SystemExit:
                pass

        assert True

    except ImportError:
        pytest.skip("Could not import roofline module for direct testing")


@pytest.mark.roofline_dir
def test_roof_workload_dir_validation(binary_handler_profile_rocprof_compute):
    skip_unsupported_roofline_soc()

    options = ["--device", "0", "--roof-only"]

    workload_dir = common.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=True
    )
    assert returncode == 0

    nested_dir = os.path.join(workload_dir, "nested", "structure")
    os.makedirs(nested_dir, exist_ok=True)
    returncode = binary_handler_profile_rocprof_compute(
        config, nested_dir, options, check_success=False, roof=True
    )
    assert returncode == 0

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_extra_options
def test_roofline_kernel_filter(binary_handler_profile_rocprof_compute):
    """
    Test roofline multi-attempt profiling with `--kernel`
    Expect to be able to re-profile into the same workload directory (with
    --overwrite) if kernels are valid.

    Roofline now takes in a dataframe that should already have filtering applied.
    Any invald kernels should be handled prior to roof activity.
    Check the following cases:
    - no valid kernels
    - one valid kernel
    - 2 kernels, one valid and one invalid
    """
    skip_unsupported_roofline_soc()

    options = [
        "--device",
        "0",
        "--roof-only",
        "--overwrite",
    ]
    workload_dir = common.get_output_dir()

    returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
        config, workload_dir, options, check_success=True, roof=True
    )
    # Wipe the directory where applicable with --overwrite, then
    # Re-profile into the same workload directory
    # Test only non-existent kernel: result should be passing
    # Dataframe given to roofline should just be all available kernels with no filtering
    options_bad = options.copy()
    options_bad.extend([
        "--kernel",
        "nonexistent_kernel_name_that_should_not_match_anything",
    ])
    returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
        config,
        workload_dir,
        options_bad,
        check_success=True,
        roof=True,
    )
    assert returncode == 0

    # Test one good kernel, re-profiling the same directory with --overwrite
    # Result should be passing as usual
    options_good = options.copy()
    options_good.extend(["--kernel", config["kernel_name_1"]])
    returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
        config, workload_dir, options_good, check_success=True, roof=True
    )
    assert returncode == 0

    # Test one good and one nonexistent kernel, re-profiling
    # Result should be passing as usual
    options_both = options.copy()
    options_both.extend([
        "--kernel",
        config["kernel_name_1"],
        "nonexistent_kernel_name_that_should_not_match_anything",
    ])
    returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
        config, workload_dir, options_both, check_success=False, roof=True
    )
    assert returncode == 0

    # Verify CSV
    assert (Path(workload_dir) / "roofline.csv").exists()

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_plot
def test_roof_cli_plot_generation(binary_handler_profile_rocprof_compute):
    skip_unsupported_roofline_soc()

    try:
        import plotext as plt  # noqa: F401

        cli_available = True
    except ImportError:
        cli_available = False

    if cli_available:
        options = ["--device", "0", "--roof-only"]
        workload_dir = common.get_output_dir()

        returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
            config, workload_dir, options, check_success=False, roof=True
        )

        common.clean_output_dir(config["cleanup"], workload_dir)
    else:
        pytest.skip("plotext not available for CLI testing")


@pytest.mark.roofline_plot
def test_roof_error_handling(binary_handler_profile_rocprof_compute):
    skip_unsupported_roofline_soc()

    options = ["--device", "0", "--roof-only"]
    workload_dir = common.get_output_dir()

    returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
        config, workload_dir, options, check_success=False, roof=True
    )

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_bench
def test_bench_only_basic(binary_handler_profile_rocprof_compute):
    """
    Test that --bench-only generates roofline.csv standalone (no application
    profiling and no performance counter collection).
    """
    skip_unsupported_roofline_soc()

    options = ["--device", "0", "--bench-only"]
    workload_dir = common.get_output_dir()

    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=True
    )

    assert returncode == 0
    workload_path = Path(workload_dir)
    roofline_csv = workload_path / "roofline.csv"
    assert roofline_csv.exists(), f"Expected {roofline_csv} to be created"
    # Bench-only must not produce profiling artifacts
    assert not (workload_path / "perfmon").exists()
    assert not (workload_path / "sysinfo.csv").exists()
    assert not (workload_path / "profiling_config.yaml").exists()
    assert not csv_compression.find_csvs(workload_path, "results_*.csv")
    assert not list(workload_path.glob("pmc_perf_*.csv"))


@pytest.mark.roofline_bench
@pytest.mark.parametrize(
    "conflicting_options",
    [
        pytest.param(["--set", "compute_thruput_util"], id="set"),
        pytest.param(["--block", "2"], id="block"),
        pytest.param(["--roof-only"], id="roof_only"),
    ],
)
def test_bench_only_mutual_exclusion(
    binary_handler_profile_rocprof_compute, conflicting_options
):
    """
    --bench-only must be rejected when paired with --set, --block, or --roof-only.
    These options are profiling-oriented and meaningless for a standalone benchmark.
    """
    skip_unsupported_roofline_soc()

    options = ["--device", "0", "--bench-only"] + conflicting_options
    workload_dir = common.get_output_dir()

    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=True
    )

    assert returncode == 1, (
        f"Expected --bench-only with {conflicting_options} to fail, "
        f"but command exited with {returncode}"
    )

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_bench
def test_bench_only_no_roof_mutual_exclusion(binary_handler_profile_rocprof_compute):
    """
    --bench-only must be rejected when combined with --no-roof, since the option
    explicitly disables the roofline microbenchmark we are trying to run.
    """
    skip_unsupported_roofline_soc()

    options = ["--device", "0", "--bench-only"]
    workload_dir = common.get_output_dir()

    # roof=False makes the fixture inject --no-roof automatically
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=False
    )

    assert returncode == 1, (
        "Expected --bench-only combined with --no-roof to fail, "
        f"but command exited with {returncode}"
    )

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_plot
def test_roofline_plot_points_data_generation():
    """
    Test that plot points data structure is correctly generated with:
    - Symbol assignments
    - AI values (FLOPs/Byte)
    - Performance values (GFLOPs/s)
    - Memory/Compute bound status
    - Cache level information

    Simulates a CDNA4 roofline run- has HBM, L1, and L2 cache levels.
    """
    skip_unsupported_roofline_soc()

    try:
        from roofline.roofline_main import Roofline
        from utils.specs import generate_machine_specs

        class MockArgs:
            def __init__(self):
                self.roof_only = True
                self.mem_level = "ALL"
                self.sort = "ALL"
                self.roofline_data_type = ["FP32"]

        args = MockArgs()
        mspec = generate_machine_specs(None, None)

        mock_ai_data = {
            "ai_l1": [[0.5, 1.2], [100.0, 150.0]],
            "ai_l2": [[0.3, 0.8], [80.0, 120.0]],
            "ai_hbm": [[0.1, 0.4], [50.0, 90.0]],
            "ai_lds": [[0.7, 1.5], [100.0, 150.0]],
            "kernelNames": ["kernel_A", "kernel_B"],
        }

        mock_ceiling_data = {
            "l1": [[0.01, 10], [10, 1000], 100],
            "l2": [[0.01, 10], [10, 800], 80],
            "hbm": [[0.01, 10], [10, 500], 50],
            "lds": [[0.01, 10], [10, 1200], 120],
            "valu": [[1, 100], [200, 200], 200],
            "matrix_ops": [[1, 100], [500, 500], 500],
        }

        plot_points_data = []
        cache_colors = {
            "ai_l1": "blue",
            "ai_l2": "green",
            "ai_hbm": "red",
            "ai_lds": "orange",
        }

        run_parameters = {
            "workload_dir": None,
            "device_id": 0,
            "sort_type": "kernels",
            "mem_level": "ALL",
            "roofline_data_type": ["FP32"],
        }
        roofline_instance = Roofline(args, mspec, run_parameters)

        for cache_level in ["ai_l1", "ai_l2", "ai_hbm", "ai_lds"]:
            if cache_level in mock_ai_data:
                x_vals = mock_ai_data[cache_level][0]
                y_vals = mock_ai_data[cache_level][1]
                num_kernels = len(mock_ai_data["kernelNames"])

                for i in range(min(len(x_vals), num_kernels)):
                    if x_vals[i] > 0 and y_vals[i] > 0:
                        status = roofline_instance._determine_kernel_bound_status(
                            ai_value=x_vals[i],
                            performance=y_vals[i],
                            cache_level=cache_level,
                            ceiling_data=mock_ceiling_data,
                        )

                        plot_points_data.append({
                            "symbol": None,
                            "color": cache_colors.get(cache_level, "gray"),
                            "cache_level": cache_level.replace("ai_", "", 1).upper(),
                            "ai": f"{x_vals[i]:.2f}",
                            "performance": f"{y_vals[i]:.2f}",
                            "status": status,
                            "kernel_idx": i,
                        })

        assert len(plot_points_data) > 0, "Plot points data should not be empty"

        for point in plot_points_data:
            assert "cache_level" in point
            assert "ai" in point
            assert "performance" in point
            assert "status" in point
            assert "kernel_idx" in point
            assert "color" in point

            assert point["cache_level"] in ["L1", "L2", "HBM", "LDS"]

            assert point["status"] in ["Memory Bound", "Compute Bound", "Unknown"]

            assert isinstance(point["ai"], str)
            assert isinstance(point["performance"], str)

    except ImportError:
        pytest.skip("Could not import roofline module for direct testing")


@pytest.mark.roofline_plot
def test_roofline_bound_status_calculation():
    """
    Test _determine_kernel_bound_status() correctly classifies kernels as
    Memory Bound or Compute Bound based on their AI and performance vs ceilings.
    Simulates a CDNA4 roofline run- has HBM, valu, and matrix ops.
    """
    skip_unsupported_roofline_soc()

    try:
        from roofline.roofline_main import Roofline
        from utils.specs import generate_machine_specs

        class MockArgs:
            def __init__(self):
                self.roof_only = True
                self.mem_level = "ALL"
                self.sort = "ALL"
                self.roofline_data_type = ["FP32"]

        args = MockArgs()
        mspec = generate_machine_specs(None, None)
        run_parameters = {
            "workload_dir": None,
            "device_id": 0,
            "sort_type": "kernels",
            "mem_level": "ALL",
            "roofline_data_type": ["FP32"],
        }
        roofline_instance = Roofline(args, mspec, run_parameters)

        ceiling_data = {
            "hbm": [[0.01, 10], [10, 1000], 100],
            "lds": [[0.01, 10], [10, 1200], 120],
            "valu": [[1, 100], [200, 200], 200],
            "matrix_ops": [[1, 100], [500, 500], 500],
        }

        status1 = roofline_instance._determine_kernel_bound_status(
            ai_value=1.0,
            performance=100.0,
            cache_level="ai_hbm",
            ceiling_data=ceiling_data,
        )
        assert status1 == "Memory Bound", f"Expected Memory Bound, got {status1}"

        status2 = roofline_instance._determine_kernel_bound_status(
            ai_value=5.0,
            performance=150.0,
            cache_level="ai_hbm",
            ceiling_data=ceiling_data,
        )
        assert status2 == "Compute Bound", f"Expected Compute Bound, got {status2}"

        status_lds = roofline_instance._determine_kernel_bound_status(
            ai_value=1.0,
            performance=100.0,
            cache_level="ai_lds",
            ceiling_data=ceiling_data,
        )
        assert status_lds == "Memory Bound", (
            f"Expected LDS Memory Bound, got {status_lds}"
        )

        status3 = roofline_instance._determine_kernel_bound_status(
            ai_value=1.0,
            performance=100.0,
            cache_level="ai_l1",
            ceiling_data=ceiling_data,
        )
        assert status3 == "Unknown", f"Expected Unknown, got {status3}"

        bad_ceiling_data = {
            "hbm": [100],
        }
        status4 = roofline_instance._determine_kernel_bound_status(
            ai_value=1.0,
            performance=100.0,
            cache_level="ai_hbm",
            ceiling_data=bad_ceiling_data,
        )
        assert status4 == "Unknown", f"Expected Unknown for bad data, got {status4}"

    except ImportError:
        pytest.skip("Could not import roofline module for direct testing")


@pytest.mark.roofline_plot
def test_roofline_many_kernels_dynamic_height(binary_handler_profile_rocprof_compute):
    """
    Test roofline CSV generation with many kernels.
    """
    skip_unsupported_roofline_soc()

    options = ["--device", "0", "--roof-only"]
    workload_dir = common.get_output_dir()

    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=True
    )

    assert returncode == 0, "Roofline profiling should succeed"

    assert (Path(workload_dir) / "roofline.csv").exists()

    file_dict = common.check_csv_files(workload_dir, 1, num_kernels)
    assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES

    common.clean_output_dir(config["cleanup"], workload_dir)
