##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el
# Common helper routines for testing collateral

import inspect
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


def check_resource_allocation():
    """Check if CTEST resource allocation is enabled for parallel testing and set
    HIP_VISIBLE_DEVICES variable accordingly with assigned gpu index.
    """

    if "CTEST_RESOURCE_GROUP_COUNT" not in os.environ:
        return

    if "CTEST_RESOURCE_GROUP_0_GPUS" in os.environ:
        resource = os.environ["CTEST_RESOURCE_GROUP_0_GPUS"]
        # extract assigned gpu id from env var: example format -> 'id:0,slots:1'
        for item in resource.split(","):
            key, value = item.split(":")
            if key == "id":
                os.environ["HIP_VISIBLE_DEVICES"] = value
                return

    return


def get_output_dir(suffix="_output", clean_existing=True):
    """Provides a unique output directory based on the name of the calling test function with a suffix applied.

    Args:
        suffix (str, optional): suffix to append to output_dir. Defaults to "_output".
        clean_existing (bool, optional): Whether to remove existing directory if exists. Defaults to True.
    """

    output_dir = inspect.stack()[1].function + suffix
    if clean_existing:
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)
    return output_dir


def setup_workload_dir(input_dir, suffix="_tmp", clean_existing=True):
    """Provides a unique input workoad directory with contents of input_dir
    based on the name of the calling test function.

    Setup is a NOOP when tests run serially.
    """

    if "PYTEST_XDIST_WORKER_COUNT" not in os.environ:
        return input_dir

    output_dir = inspect.stack()[1].function + suffix
    if clean_existing:
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)

    shutil.copytree(input_dir, output_dir)
    return output_dir


def clean_output_dir(cleanup, output_dir):
    """Remove output directory generated from rocprofiler-compute execution

    Args:
        cleanup (boolean): flag to enable/disable directory cleanup
        output_dir (string): name of directory to remove
    """
    if cleanup:
        if Path(output_dir).exists():
            try:
                shutil.rmtree(output_dir)
            except OSError as e:
                print("WARNING: shutil.rmdir(output_dir): directory may not be empty...")
    return


def check_csv_files(output_dir, num_devices, num_kernels):
    """Check profiling output csv files for expected number of entries (based on kernel invocations)

    Args:
        output_dir (string): output directory containing csv files
        num_kernels (int): number of kernels expected to have been profiled

    Returns:
        dict: dictionary housing file contents as pandas dataframe
    """

    file_dict = {}
    files_in_workload = os.listdir(output_dir)
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(output_dir + "/" + file)
            if "roofline" in file:
                assert len(file_dict[file].index) >= num_devices
            elif not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
        elif file.endswith(".pdf"):
            file_dict[file] = "pdf"
    return file_dict


def launch_rocprof_compute(config, options, workload_dir, check_success=True):
    """Launch ROCm Compute Profiler with command-line optoins

    Args:
        config (list): runtime configuration settings
        options (list): command line options to provide to rocprofiler-compute
        workload_dir (string): desired output directory
        check_success (bool, optional): Whether to verify successful exit condition. Defaults to True.

    Returns:
       exception: SystemExit exception
    """
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", options + ["--path", workload_dir, "--"] + config["app_1"]
        ):
            config["rocprofiler-compute"].main()

    # verify run status
    if check_success:
        assert e.value.code == 0

    return e
