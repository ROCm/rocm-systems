##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################


import subprocess
from importlib.machinery import SourceFileLoader
from unittest.mock import patch

import pytest

rocprof_compute = SourceFileLoader(
    "rocprof-compute", "src/rocprof-compute"
).load_module()


def pytest_addoption(parser):
    parser.addoption(
        "--call-binary",
        action="store_true",
        default=False,
        help="Call standalone binary instead of main function during tests",
    )

    parser.addoption(
        "--rocprofiler-sdk-library-path",
        type=str,
        default="/opt/rocm/lib/librocprofiler-sdk.so",
        help="Path to the rocprofiler-sdk library",
    )


@pytest.fixture
def binary_handler_profile_rocprof_compute(request):
    def _handler(
        config,
        workload_dir,
        options=[],
        check_success=True,
        roof=False,
        app_name="app_1",
    ):
        if request.config.getoption("--rocprofiler-sdk-library-path"):
            options.extend(
                [
                    "--rocprofiler-sdk-library-path",
                    request.config.getoption("--rocprofiler-sdk-library-path"),
                ],
            )
        if request.config.getoption("--call-binary"):
            baseline_opts = [
                "build/rocprof-compute.bin",
                "profile",
                "-n",
                app_name,
                "-VVV",
            ]
            if not roof:
                baseline_opts.append("--no-roof")
            process = subprocess.run(
                baseline_opts
                + options
                + ["--path", workload_dir, "--"]
                + config[app_name],
                text=True,
            )
            # verify run status
            if check_success:
                assert process.returncode == 0
            return process.returncode
        else:
            baseline_opts = ["rocprof-compute", "profile", "-n", app_name, "-VVV"]
            if not roof:
                baseline_opts.append("--no-roof")
            with pytest.raises(SystemExit) as e:
                with patch(
                    "sys.argv",
                    baseline_opts
                    + options
                    + ["--path", workload_dir, "--"]
                    + config[app_name],
                ):
                    rocprof_compute.main()
            # verify run status
            if check_success:
                assert e.value.code == 0
            return e.value.code

    return _handler


@pytest.fixture
def binary_handler_analyze_rocprof_compute(request):
    def _handler(arguments):
        if request.config.getoption("--call-binary"):
            process = subprocess.run(
                ["build/rocprof-compute.bin", *arguments],
                text=True,
            )
            return process.returncode
        else:
            with pytest.raises(SystemExit) as e:
                with patch(
                    "sys.argv",
                    ["rocprof-compute", *arguments],
                ):
                    rocprof_compute.main()
            return e.value.code

    return _handler
