# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Fixtures that drive the rocprof-compute CLI or need a GPU.

Pytest resolves fixtures by directory, so nothing here is visible to
tests/unit/. A unit test that requests one fails at collection, which is the
isolation this layout exists to provide.
"""

import shutil
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import patch

import common
import pytest
from common import ROOT
from conftest import ProfileModeImportGuard

from tests.integration.common import gpu_soc, inject_mpirun

rocprof_compute_script_path = Path(ROOT) / "src/rocprof-compute"
if not rocprof_compute_script_path.exists():
    rocprof_compute_script_path = Path(ROOT) / "rocprof-compute"
if not rocprof_compute_script_path.exists():
    raise FileNotFoundError("Cannot find rocprof-compute script")
rocprof_compute_script_path = str(rocprof_compute_script_path)


@pytest.fixture(scope="session")
def gpu_soc_info():
    """(arch, model) probed from rocminfo once per session, ('', '') with no GPU."""
    return gpu_soc()


@pytest.fixture(scope="session")
def gpu_arch(gpu_soc_info):
    """GPU architecture string, e.g. 'gfx942'."""
    return gpu_soc_info[0]


@pytest.fixture(scope="session")
def soc(gpu_soc_info):
    """GPU model string, e.g. 'MI300'."""
    return gpu_soc_info[1]


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
        stream: If True, echo child output line by line as it is produced
            (requires capture_output).

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
        stream=False,
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
            # Use rocprof_compute_script_path instead of rocprof-compute
            command_rocprof_compute[0] = rocprof_compute_script_path
            command_rocprof_compute = inject_mpirun(command_rocprof_compute, num_ranks)

        # For capture_output or multi-rank, run the command with subprocess
        if capture_output or num_ranks > 1:
            # Use rocprof_compute_script_path instead of rocprof-compute
            if num_ranks == 1:
                command_rocprof_compute[0] = rocprof_compute_script_path

            process = common.run_subprocess(
                command_rocprof_compute,
                capture_output=capture_output,
                stream=stream,
            )

            # Verify run status
            if check_success:
                assert process.returncode == 0

            # Return output tuple if capture_output is enabled
            if capture_output:
                return process.returncode, process.stdout, process.stderr

            return process.returncode

        # Default single-rank mode: patch sys.argv and call main() directly
        # Guard imports during profile execution (test-time enforcement)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                command_rocprof_compute,
            ):
                with ProfileModeImportGuard():
                    rocprof_compute = SourceFileLoader(
                        "rocprof-compute", rocprof_compute_script_path
                    ).load_module()
                    rocprof_compute.main()
        # verify run status
        if check_success:
            assert e.value.code == 0
        return e.value.code

    return _handler


@pytest.fixture
def binary_handler_analyze_rocprof_compute():
    """
    Fixture to run rocprof-compute analyze command.

    Args:
        arguments: Command-line arguments for the analyze command.

    Returns:
        returncode (int): Exit code from the command.
    """

    def _handler(arguments):
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                ["rocprof-compute", *arguments],
            ):
                # Load module (no guard needed for analyze mode)
                rocprof_compute = SourceFileLoader(
                    "rocprof-compute", rocprof_compute_script_path
                ).load_module()
                rocprof_compute.main()
        return e.value.code

    return _handler
