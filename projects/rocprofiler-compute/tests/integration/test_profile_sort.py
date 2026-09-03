# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for the profile --sort option."""

import inspect
from pathlib import Path

import common

from tests.integration import common as integration_common
from tests.integration.common import (
    ROOF_ONLY_FILES,
    config,
    num_kernels,
    skip_unsupported_roofline_soc,
    validate,
)


def test_roof_sort_dispatches(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    """Profile creates CSV; analyze with --sort dispatches generates output."""
    skip_unsupported_roofline_soc()

    profile_options = ["--device", "0", "--roof-only"]
    workload_dir = common.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, profile_options, check_success=False, roof=True
    )
    assert returncode == 0

    file_dict = integration_common.check_csv_files(workload_dir, 1, num_kernels)
    assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--sort",
        "dispatches",
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    validate(inspect.stack()[0][3], workload_dir, file_dict)
    common.clean_output_dir(config["cleanup"], workload_dir)


def test_roof_sort_kernels(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    """Profile creates CSV; analyze with --sort kernels generates output."""
    skip_unsupported_roofline_soc()

    profile_options = ["--device", "0", "--roof-only"]
    workload_dir = common.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, profile_options, check_success=False, roof=True
    )
    assert returncode == 0

    file_dict = integration_common.check_csv_files(workload_dir, 1, num_kernels)
    assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--sort",
        "kernels",
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    validate(inspect.stack()[0][3], workload_dir, file_dict)
    common.clean_output_dir(config["cleanup"], workload_dir)
