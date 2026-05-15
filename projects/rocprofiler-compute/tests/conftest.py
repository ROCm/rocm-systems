# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

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

# Determine script path
rocprof_compute_script_path = Path(ROOT) / "src/rocprof-compute"
if not rocprof_compute_script_path.exists():
    rocprof_compute_script_path = Path(ROOT) / "rocprof-compute"
if not rocprof_compute_script_path.exists():
    raise FileNotFoundError("Cannot find rocprof-compute script")
rocprof_compute_script_path = str(rocprof_compute_script_path)


class ProfileModeImportGuard:
    """
    Import guard using sys.meta_path to enforce stdlib-only imports in profile mode.

    Python Version Compatibility:
        - Python 3.10+: Full enforcement (uses sys.stdlib_module_names)
        - Python 3.8-3.9: No-op mode (enforcement disabled, warning issued)

    Usage:
        with ProfileModeImportGuard():
            # Python 3.10+: Import checking active, non-stdlib imports raise ImportError
            # Python 3.8-3.9: No-op, all imports allowed (with warning)

    Context Manager Protocol:
        __enter__: Registers guard with Python's import system (sys.meta_path)
        __exit__: Unregisters guard after code execution completes
    """

    # Project modules that are allowed (non-stdlib)
    ALLOWED_PROJECT_MODULES = frozenset([
        "rocprof_compute",
        "rocprof_compute_profile",
        "rocprof_compute_analyze",
        "rocprof_compute_soc",
        "rocprof_compute_tui",
        "utils",
        "vendored",
        "roofline",
        "config",
        "argparser",  # src/argparser.py, not stdlib argparse
        "rocprof_compute_base",
    ])

    # ROCm system libraries (not pip packages)
    ALLOWED_ROCM_MODULES = frozenset([
        "amdsmi",  # AMD System Management Interface
        "hip",  # HIP runtime Python bindings
        "rocprofv3",  # rocprofv3 python modules such as avail
        "rocprofv3_avail_module",  # Alternative avail module for backward compatibility
    ])

    def __enter__(self):
        """
        Register import guard with Python's import system.

        Called automatically when entering 'with' block.
        Adds this object to sys.meta_path so Python calls our find_spec()
        for every import during the with block.
        """
        if sys.version_info >= (3, 10):
            sys.meta_path.insert(0, self)
        else:
            print(
                "\n" + "=" * 70 + "\n"
                "WARNING: ProfileModeImportGuard requires Python 3.10+\n"
                "(sys.stdlib_module_names unavailable).\n"
                "Import enforcement DISABLED for this test run.\n" + "=" * 70 + "\n",
                file=sys.stderr,
            )
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """
        Unregister import guard from Python's import system.

        Called automatically when exiting 'with' block.
        Removes this object from sys.meta_path, disabling import checking.
        """
        if sys.version_info >= (3, 10) and self in sys.meta_path:
            sys.meta_path.remove(self)

    def find_spec(self, fullname, path, target=None):
        """
        PEP 451 import hook - called automatically by Python during imports.

        Python's import system calls this method for every import statement.
        We check if the module is allowed, and raise ImportError if not.
        """
        top_level = fullname.split(".")[0]

        # Check stdlib
        if top_level in sys.stdlib_module_names:
            return None

        # Check ROCm modules
        if top_level in self.ALLOWED_ROCM_MODULES:
            return None

        # Check project modules (validate origin to prevent third-party modules
        # with same name, e.g., "utils" from site-packages)
        if top_level in self.ALLOWED_PROJECT_MODULES:
            if self._is_from_project(top_level):
                return None

        # Forbidden module
        raise ImportError(
            f"\n{'=' * 70}\n"
            "PROFILE MODE DEPENDENCY VIOLATION\n"
            f"{'=' * 70}\n"
            f"Forbidden package: {top_level}\n\n"
            "Profile mode must use ONLY Python stdlib + ROCm libraries.\n"
            "Fix: Move import to analyze mode or use stdlib alternative.\n"
            "See CONTRIBUTING.md 'Profile Mode Dependency Policy'\n"
            f"{'=' * 70}\n"
        )

    def _is_from_project(self, module_name):
        """Check if module exists in project directory, not site-packages."""
        project_root = Path(__file__).parent.parent
        for base in [project_root / "src", project_root]:
            # Check for: module.py, module/__init__.py, or module/ (namespace pkg)
            candidates = [
                base / f"{module_name}.py",
                base / module_name / "__init__.py",
                base / module_name,  # namespace package (dir without __init__.py)
            ]
            for p in candidates:
                if p.is_file() or (p.is_dir() and p.exists()):
                    return True
        return False


def inject_mpirun(command, num_ranks):
    """
    Wrap a command with mpirun for multi-rank execution.
    """
    mpirun_cmd = ["mpirun"]
    # Add --allow-run-as-root only when running as root
    # (needed for OpenMPI in containers)
    # This flag is OpenMPI-specific and would cause errors
    # with other MPI implementations
    if os.geteuid() == 0:
        mpirun_cmd.append("--allow-run-as-root")
    mpirun_cmd.extend(["-n", str(num_ranks)])
    return mpirun_cmd + command


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
                command_rocprof_compute = inject_mpirun(
                    command_rocprof_compute, num_ranks
                )

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
                # Use rocprof_compute_script_path instead of rocprof-compute
                command_rocprof_compute[0] = rocprof_compute_script_path
                command_rocprof_compute = inject_mpirun(
                    command_rocprof_compute, num_ranks
                )

            # For capture_output or multi-rank, run the command with subprocess
            if capture_output or num_ranks > 1:
                # Use rocprof_compute_script_path instead of rocprof-compute
                if num_ranks == 1:
                    command_rocprof_compute[0] = rocprof_compute_script_path

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
                    # Load module (no guard needed for analyze mode)
                    rocprof_compute = SourceFileLoader(
                        "rocprof-compute", rocprof_compute_script_path
                    ).load_module()
                    rocprof_compute.main()
            return e.value.code

    return _handler
