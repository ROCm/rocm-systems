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

import os
import shutil
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    rocprof_compute = SourceFileLoader(
        "rocprof-compute", "src/rocprof-compute"
    ).load_module()
except Exception:
    rocprof_compute = SourceFileLoader(
        "rocprof-compute", "rocprof-compute"
    ).load_module()


def pytest_addoption(parser):
    parser.addoption(
        "--call-binary",
        action="store_true",
        default=False,
        help="Call standalone binary instead of main function during tests",
    )

    parser.addoption(
        "--rocprofiler-sdk-tool-path",
        type=str,
        default=str(
            Path(os.getenv("ROCM_PATH", "/opt/rocm"))
            / "lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"
        ),
        help="Path to the rocprofiler-sdk tool",
    )


@pytest.fixture(autouse=True)
def skip_monkeypatch_with_binary(request):
    """Auto-skip tests using monkeypatch when --call-binary is used.

    Tests that use monkeypatch to patch Python functions/classes/modules
    cannot work with --call-binary mode because the binary runs in a separate
    process where Python patches don't apply.
    """
    if (
        request.config.getoption("--call-binary")
        and "monkeypatch" in request.fixturenames
    ):
        pytest.skip(
            "Test uses monkeypatch which is incompatible with --call-binary mode"
        )


@pytest.fixture
def binary_handler_profile_rocprof_compute(request):
    """
    Fixture to run rocprof-compute profile command.

    Args:
        config: Test configuration dictionary containing app commands.
        workload_dir: Directory to store profiling output.
        options: Additional command-line options.
        check_success: If True, assert that the command succeeds.
        roof: If True, enable roofline.
        app_name: Key in config dict for the application command.
        attach_detach_para: Parameters for attach/detach mode.
        skip_app_name: If True, skip adding --name option.
        workload_dir_type: "output_directory" or "default".
        num_ranks: Number of MPI ranks (1 = no MPI, >1 = use mpirun).
        capture_output: If True, capture stdout/stderr and return
            (returncode, stdout, stderr) tuple instead of just returncode.

    Returns:
        If capture_output is False: returncode (int)
        If capture_output is True: (returncode, stdout, stderr) tuple
    """

    def _handler(
        config,
        workload_dir,
        options=[],
        check_success=True,
        roof=False,
        app_name="app_1",
        attach_detach_para=None,
        skip_app_name=False,
        workload_dir_type="output_directory",
        num_ranks=1,
        capture_output=False,
    ):
        # Skip test if multiple ranks are requested but mpirun is not available
        if num_ranks > 1 and shutil.which("mpirun") is None:
            pytest.skip(f"mpirun not found, skipping {request.node.name}")

        if request.config.getoption("--rocprofiler-sdk-tool-path"):
            options.extend(
                [
                    "--rocprofiler-sdk-tool-path",
                    request.config.getoption("--rocprofiler-sdk-tool-path"),
                ],
            )
        if request.config.getoption("--call-binary"):
            baseline_opts = [
                "./rocprof-compute.bin",
                "profile",
                "-VVV",
            ]
            if not skip_app_name:
                baseline_opts.extend(["-n", app_name])
            if not roof:
                baseline_opts.append("--no-roof")

            command_rocprof_compute = baseline_opts + options

            if workload_dir_type == "output_directory":
                command_rocprof_compute = command_rocprof_compute + [
                    "--output-directory",
                    workload_dir,
                ]

            if not attach_detach_para:
                command_rocprof_compute = (
                    command_rocprof_compute + ["--"] + config[app_name]
                )
            else:
                command_rocprof_compute = command_rocprof_compute + [
                    "--attach-pid",
                    str(attach_detach_para["attach_pid"]),
                ]
                if attach_detach_para["attach-duration-msec"]:
                    command_rocprof_compute = command_rocprof_compute + [
                        "--attach-duration-msec",
                        str(attach_detach_para["attach-duration-msec"]),
                    ]

            # Wrap with mpirun if num_ranks > 1
            if num_ranks > 1:
                command_rocprof_compute = [
                    "mpirun",
                    "-n",
                    str(num_ranks),
                ] + command_rocprof_compute

            process = subprocess.run(
                command_rocprof_compute,
                text=True,
                capture_output=True,
            )
            # Print output so capsys can capture it
            if process.stdout:
                print(process.stdout, end="")
            if process.stderr:
                print(process.stderr, end="", file=sys.stderr)
            # verify run status
            if check_success:
                assert process.returncode == 0

            # Return output tuple if capture_output is enabled
            if capture_output:
                return process.returncode, process.stdout, process.stderr

            return process.returncode
        else:
            # Non-binary mode: use Python module directly or subprocess
            baseline_opts = [
                "rocprof-compute",
                "profile",
                "-VVV",
            ]
            if not skip_app_name:
                baseline_opts.extend(["-n", app_name])
            if not roof:
                baseline_opts.append("--no-roof")

            command_rocprof_compute = baseline_opts + options

            if workload_dir_type == "output_directory":
                command_rocprof_compute = command_rocprof_compute + [
                    "--output-directory",
                    workload_dir,
                ]

            if not attach_detach_para:
                command_rocprof_compute = (
                    command_rocprof_compute + ["--"] + config[app_name]
                )
            else:
                command_rocprof_compute = command_rocprof_compute + [
                    "--attach-pid",
                    str(attach_detach_para["attach_pid"]),
                ]
                if attach_detach_para["attach-duration-msec"]:
                    command_rocprof_compute = command_rocprof_compute + [
                        "--attach-duration-msec",
                        str(attach_detach_para["attach-duration-msec"]),
                    ]

            # For multi-rank, use mpirun to run the command
            if num_ranks > 1:
                # Use src/rocprof-compute instead of rocprof-compute
                command_rocprof_compute[0] = "src/rocprof-compute"
                command_rocprof_compute = [
                    "mpirun",
                    "-n",
                    str(num_ranks),
                ] + command_rocprof_compute

            # For capture_output or multi-rank, run the command with subprocess
            if capture_output or num_ranks > 1:
                # Use src/rocprof-compute instead of rocprof-compute
                if num_ranks == 1:
                    command_rocprof_compute[0] = "src/rocprof-compute"

                process = subprocess.run(
                    command_rocprof_compute,
                    text=True,
                    capture_output=capture_output,
                )

                # Verify run status
                if check_success:
                    assert process.returncode == 0

                # Return output tuple if capture_output is enabled
                if capture_output:
                    return process.returncode, process.stdout, process.stderr

                return process.returncode

            # Default single-rank mode: patch sys.argv and call main() directly
            with pytest.raises(SystemExit) as e:
                with patch(
                    "sys.argv",
                    command_rocprof_compute,
                ):
                    rocprof_compute.main()
            # verify run status
            if check_success:
                assert e.value.code == 0
            return e.value.code

    return _handler


@pytest.fixture
def binary_handler_analyze_rocprof_compute(request):
    """
    Fixture to run rocprof-compute analyze command.

    Args:
        arguments: Command-line arguments for the analyze command.

    Returns:
        returncode (int): Exit code from the command.
    """

    def _handler(arguments):
        if request.config.getoption("--call-binary"):
            process = subprocess.run(
                ["./rocprof-compute.bin", *arguments],
                text=True,
                capture_output=True,
            )
            # Print output so capsys can capture it
            if process.stdout:
                print(process.stdout, end="")
            if process.stderr:
                print(process.stderr, end="", file=sys.stderr)
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
