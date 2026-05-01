#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Test Executor Module
Handles test execution, build processes, and result tracking
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import datetime
import copy
from enum import IntEnum, Enum
from pathlib import Path

# Make stdout unbuffered to prevent output ordering issues with subprocesses
sys.stdout.reconfigure(line_buffering=True)


class ExitCode(IntEnum):
    """Exit codes for processes"""
    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1
    EXIT_TIMEOUT = 124


class TestResult(str, Enum):
    """Test result statuses"""
    RESULT_PASSED = "PASSED"
    RESULT_FAILED = "FAILED"
    RESULT_TIMEOUT = "TIMEOUT"
    RESULT_SKIPPED = "SKIPPED"


def infer_gtest_result_from_output(captured_output: str, returncode: int) -> str:
    """
    Map gtest process exit + stdout/stderr to a TestResult string.

    Google Test returns exit 0 when failures are absent, including when all
    selected tests are SKIPPED. Prefer ``infer_gtest_result_from_json_file`` when
    ``--gtest_output=json:…`` is used; this function is the stdout fallback (e.g.
    ``[  SKIPPED ]`` / ``[  OK ]`` patterns).
    """
    if returncode == ExitCode.EXIT_TIMEOUT:
        return TestResult.RESULT_TIMEOUT.value
    if returncode != ExitCode.EXIT_SUCCESS:
        return TestResult.RESULT_FAILED.value

    out = captured_output or ""
    if re.search(r"\[\s+FAILED\s+\]", out):
        return TestResult.RESULT_FAILED.value
    has_ok = re.search(r"\[\s+OK\s+\]", out) is not None
    has_skipped = re.search(r"\[\s+SKIPPED\s+\]", out) is not None
    if has_ok:
        return TestResult.RESULT_PASSED.value
    if has_skipped:
        return TestResult.RESULT_SKIPPED.value
    return TestResult.RESULT_PASSED.value


def _gtest_json_accumulate(obj, stats):
    """Walk gtest JSON (--gtest_output=json); set stats keys failed/passed/skipped."""
    if isinstance(obj, dict):
        if "result" in obj and isinstance(obj.get("name"), str):
            fails = obj.get("failures")
            if isinstance(fails, list) and len(fails) > 0:
                stats["failed"] = True
            else:
                res = obj.get("result")
                if res == "SKIPPED":
                    stats["skipped"] = True
                elif res == "COMPLETED":
                    stats["passed"] = True
        for v in obj.values():
            _gtest_json_accumulate(v, stats)
    elif isinstance(obj, list):
        for item in obj:
            _gtest_json_accumulate(item, stats)


def infer_gtest_result_from_json_file(json_path: str, returncode: int) -> str:
    """
    Map gtest exit code + JSON report to TestResult.

    When returncode is 0, inspects leaf tests for failures/SKIPPED/COMPLETED.
    Falls back to infer_gtest_result_from_output(\"\", rc) if the file is missing
    or invalid JSON.
    """
    if returncode == ExitCode.EXIT_TIMEOUT:
        return TestResult.RESULT_TIMEOUT.value
    if returncode != ExitCode.EXIT_SUCCESS:
        return TestResult.RESULT_FAILED.value
    if not json_path or not os.path.isfile(json_path):
        return infer_gtest_result_from_output("", returncode)
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return infer_gtest_result_from_output("", returncode)
    stats = {"failed": False, "passed": False, "skipped": False}
    _gtest_json_accumulate(data, stats)
    if stats["failed"]:
        return TestResult.RESULT_FAILED.value
    if stats["passed"]:
        return TestResult.RESULT_PASSED.value
    if stats["skipped"]:
        return TestResult.RESULT_SKIPPED.value
    return TestResult.RESULT_PASSED.value


def _distinct_host_count(mpi_hosts: dict) -> int:
    """
    Count distinct hosts from SLURM host_list or Open MPI hostfile.
    Returns 0 if unknown (no host list / file), so callers skip the insufficient-nodes
    check when topology cannot be determined.
    """
    if not mpi_hosts:
        return 0
    if "host_list" in mpi_hosts:
        seen = set()
        for part in mpi_hosts["host_list"].split(","):
            part = part.strip()
            if not part:
                continue
            host = part.split(":")[0].strip()
            if host:
                seen.add(host)
        return len(seen)
    if "hostfile" in mpi_hosts:
        path = mpi_hosts["hostfile"]
        seen = set()
        try:
            with open(path, encoding="utf-8", errors="replace") as hf:
                for line in hf:
                    line = line.split("#")[0].strip()
                    if not line:
                        continue
                    host = line.split()[0].strip()
                    if host:
                        seen.add(host)
        except OSError:
            return 0
        return len(seen)
    return 0


class TestExecutor:
    """
    Executes tests and manages build/test workflows
    """

    MPI_IMPL_CONFIG = {
        "openmpi": {
            "env_format": "-x {key}='{value}'",
            "default_args": "--mca btl ^vader,openib --mca pml ucx --bind-to none",
        },
        "mpich": {
            "env_format": "-env {key} '{value}'",
            "default_args": "-bind-to none",
        },
    }

    def __init__(self, config_processor, args):
        """
        Initialize TestExecutor

        Args:
            config_processor: TestConfigProcessor instance
            args: Parsed command-line arguments
        """
        self.config_processor = config_processor
        self.args = args
        self.system_config = config_processor.get_system_config()
        self.paths = config_processor.get_paths()
        self.global_env = dict(config_processor.get_env_variables())

        # Merge system-specific env overrides if --system is specified
        system = getattr(args, 'system', '') or ''
        if system:
            system_env = config_processor.config.get("system_env_variables", {})
            if isinstance(system_env, dict) and system in system_env:
                self.global_env.update(system_env[system])
            elif system_env and system not in system_env:
                available = list(system_env.keys()) if isinstance(system_env, dict) else []
                print(f"WARNING: No system_env_variables for '{system}'. Available: {available}")
        self.build_config = config_processor.get_build_config()

        # Setup directories
        self.setup_directories()

        # MPI hostfile is detected lazily on first MPI test
        self._mpi_hostfile = None
        self._mpi_hostfile_detected = False

        # MPI implementation: openmpi (default) or mpich (via --mpich flag)
        self.mpi_impl = "mpich" if getattr(args, 'mpich', False) else "openmpi"
        self.mpi_config = self.MPI_IMPL_CONFIG[self.mpi_impl]

        # Detect MPI hosts: auto-detect from SLURM if "auto_detect_hosts" is true in config, otherwise use hostfile
        self.mpi_hosts = self._detect_mpi_hosts()

        # Test tracking
        self.test_results = []
        self.test_names = []
        self.test_durations = []
        self.test_suites = []

        # Rerun tracking
        self.failed_test_info = []  # Store info needed to rerun failed tests
        self.rerun_results = []
        self.rerun_names = []
        self.rerun_durations = []

    def setup_directories(self):
        """Setup build and log directories"""
        workdir = self.paths.get("workdir", os.getcwd())

        # Determine workspace name for logs/reports (always timestamped)
        suffix_part = f"_{self.args.report_suffix}" if self.args.report_suffix else ""
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
        workspace_name = f"rccl_test_artifacts{suffix_part}_{timestamp}"

        # Create workspace directory path
        self.workspace_dir = os.path.join(workdir, workspace_name)

        # Determine build directory (priority: --build-dir > env var > default)
        custom_rccl_path = os.environ.get('RCCL_LIB_PATH') or os.environ.get('RCCL_BUILD_DIR')

        if self.args.build_dir:
            # Use custom build directory from command line
            self.build_dir = os.path.expanduser(os.path.expandvars(self.args.build_dir))
            self.using_custom_lib = True
            if self.args.verbose:
                print(f"Using custom build directory from --build-dir: {self.build_dir}")
        elif custom_rccl_path:
            # Use custom library path from environment variable
            self.build_dir = os.path.expanduser(os.path.expandvars(custom_rccl_path))
            self.using_custom_lib = True
            if self.args.verbose:
                print(f"Using custom RCCL library path from environment: {self.build_dir}")
        else:
            # Use default build directory matching install.sh convention
            self.using_custom_lib = False
            install_flags = self.build_config.get("install_flags", [])
            if "--debug" in install_flags or "--debug-fast" in install_flags:
                build_type = "debug"
            else:
                build_type = "release"
            self.build_dir = os.path.join(workdir, "build", build_type)

        # Set log and report directories under workspace
        self.log_dir = os.path.join(self.workspace_dir, "logs")
        self.report_dir = os.path.join(self.workspace_dir, "report")

        # Create directories (skip build_dir if using custom lib)
        if not self.using_custom_lib:
            os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

        if self.args.verbose:
            print(f"Work directory:   {workdir}")
            print(f"Workspace directory: {self.workspace_dir}")
            print(f"Build directory:  {self.build_dir}")
            if self.using_custom_lib:
                print(f"  (Custom path via {'--build-dir' if self.args.build_dir else 'RCCL_LIB_PATH/RCCL_BUILD_DIR'})")
            print(f"Log directory:    {self.log_dir}")
            print(f"Report directory: {self.report_dir}")

    @property
    def mpi_hostfile(self):
        """Lazy MPI hostfile detection -- only runs on first access."""
        if not self._mpi_hostfile_detected:
            self._mpi_hostfile = self._detect_mpi_hostfile()
            self._mpi_hostfile_detected = True
        return self._mpi_hostfile

    def _detect_mpi_hostfile(self):
        """
        Detect MPI hostfile.
        Checks RCCL_TEST_MPI_HOSTFILE env var, then ~/.mpi_hostfile default.

        Returns:
            str: Path to hostfile, or None if not found
        """
        hostfile = os.environ.get('RCCL_TEST_MPI_HOSTFILE')
        if hostfile and os.path.isfile(hostfile):
            print(f"Using MPI hostfile from RCCL_TEST_MPI_HOSTFILE: {hostfile}")
            return hostfile

        # Check default hostfile
        default_hostfile = os.path.expanduser('~/.mpi_hostfile')
        if os.path.isfile(default_hostfile):
            print(f"Using default MPI hostfile: {default_hostfile}")
            return default_hostfile

        if self.args.verbose:
            print("No MPI hostfile found (checked RCCL_TEST_MPI_HOSTFILE env var and ~/.mpi_hostfile)")
        return None

    def _detect_mpi_hosts(self):
        """
        Detect MPI host list once during initialization.

        If "auto_detect_hosts" is true in the system profile (or top-level config)
        and a SLURM allocation is active, uses scontrol to get the host list.
        Otherwise falls back to the hostfile detected by mpi_hostfile property.

        Returns:
            dict with 'host_list', 'hostfile', or empty dict
        """
        system = getattr(self.args, 'system', '') or ''
        auto_detect = self.config_processor.config.get("auto_detect_hosts", False)

        if auto_detect and os.environ.get('SLURM_JOB_ID'):
            try:
                result = subprocess.run(
                    ['scontrol', 'show', 'hostnames'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    hosts = ','.join(result.stdout.strip().split('\n'))
                    print(f"Using SLURM hosts: {hosts}")
                    return {'host_list': hosts}
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        hostfile = self.mpi_hostfile
        if hostfile:
            return {'hostfile': hostfile}

        return {}

    def check_environment(self):
        """
        Check that required environment and tools are available

        Returns:
            bool: True if environment is valid
        """
        errors = []

        # Check ROCm
        rocm_path = self.paths.get("rocm_path", "/opt/rocm")
        if not os.path.isdir(rocm_path):
            errors.append(f"ROCm not found at {rocm_path}")

        # Check MPI (unless --skip-mpi-check is set)
        if not self.args.skip_mpi_check:
            mpi_path = self.paths.get("mpi_path")
            if mpi_path:
                if not os.path.isdir(mpi_path):
                    print(f"WARNING: MPI path not found: {mpi_path}")
                elif not os.path.isfile(os.path.join(mpi_path, "bin", "mpirun")):
                    print(f"WARNING: mpirun not found in {mpi_path}/bin/")
        elif self.args.verbose:
            print("SKIP: MPI check skipped (--skip-mpi-check)")

        # Check RCCL library (if not building or using custom lib)
        if self.args.no_build or self.using_custom_lib:
            lib_path = os.path.join(self.build_dir, "librccl.so")
            if not os.path.isfile(lib_path):
                errors.append(f"RCCL library not found: {lib_path}")
            elif self.args.verbose:
                print(f"Found RCCL library: {lib_path}")

        if errors:
            print("ERROR: Environment check failed:")
            for error in errors:
                print(f"  - {error}")
            return False

        if self.args.verbose:
            print("Environment validation passed")
        return True

    def build_rccl(self):
        """
        Build RCCL using install.sh with configurable build settings.

        The build_configuration in the JSON config specifies:
        - install_flags: List of install.sh command-line flags
        - cmake_options: Optional string of additional CMake options (passed via --cmake-options)
        - env_variables: Environment variables to set during the build
        - parallel_jobs: Number of parallel compilation jobs (passed via -j)

        Returns:
            bool: True if build succeeded
        """
        # Skip build if using custom library from environment variable
        if self.using_custom_lib:
            if self.args.verbose:
                print("SKIP: Build step skipped (using custom RCCL library from environment)")
            return True

        if self.args.no_build:
            if self.args.verbose:
                print("SKIP: Build step skipped (--no-build)")
            return True

        print("="*80)
        print("BUILDING RCCL")
        print("="*80)

        workdir = self.paths.get("workdir", os.getcwd())
        rocm_path = self.paths.get("rocm_path", "/opt/rocm")
        mpi_path = self.paths.get("mpi_path", "")

        install_flags = list(self.build_config.get("install_flags", []))
        cmake_options = self.build_config.get("cmake_options", "")
        build_env_vars = self.build_config.get("env_variables", {})
        parallel_jobs = self.build_config.get("parallel_jobs")

        if self.args.skip_mpi_check:
            if "--enable-mpi-tests" in install_flags:
                install_flags.remove("--enable-mpi-tests")
            # Explicitly disable to override any cached CMake value from prior builds
            if cmake_options:
                cmake_options += " -DENABLE_MPI_TESTS=OFF"
            else:
                cmake_options = "-DENABLE_MPI_TESTS=OFF"
            print("NOTE: MPI tests disabled in build (--skip-mpi-check)")

        # Build install.sh command
        install_script = os.path.join(workdir, "install.sh")
        cmd = [install_script] + install_flags

        if parallel_jobs:
            cmd.extend(["-j", str(parallel_jobs)])

        if cmake_options:
            cmd.extend(["--cmake-options", cmake_options])

        # Setup environment
        env = os.environ.copy()
        env['ROCM_PATH'] = rocm_path
        if mpi_path:
            env['MPI_PATH'] = mpi_path

        for key, value in build_env_vars.items():
            env[key] = str(value)

        if self.args.verbose:
            print(f"Work directory:  {workdir}")
            print(f"ROCm path:       {rocm_path}")
            print(f"MPI path:        {mpi_path}")
            print(f"Build directory: {self.build_dir}")
            print(f"Install script:  {install_script}")
            print(f"Install flags:   {' '.join(install_flags)}")
            if cmake_options:
                print(f"CMake options:   {cmake_options}")
            if parallel_jobs:
                print(f"Parallel jobs:   {parallel_jobs}")
            print(f"Command: {' '.join(cmd)}")
            if build_env_vars:
                print("Build environment variables:")
                for key, value in build_env_vars.items():
                    print(f"  {key}={value}")

        try:
            result = subprocess.run(
                cmd,
                cwd=workdir,
                env=env,
                capture_output=False
            )

            if result.returncode != 0:
                print("ERROR: install.sh build failed")
                return False

            print("Build completed successfully")
            return True

        except Exception as e:
            print(f"ERROR: Build failed with exception: {e}")
            return False

    def _resolve_binary_path(self, binary, test_config):
        """
        Resolve the test binary path using multiple strategies:
        1. If binary is an absolute path -> use it directly
        2. If test_binary_dir is specified in config -> use as base directory
        3. If binary contains ${VAR} -> expand environment variables
        4. Otherwise -> use default build_dir/test/binary

        Args:
            binary: Binary name or path from config
            test_config: Test configuration dict

        Returns:
            str: Resolved absolute path to the binary
        """
        # Strategy 1: Check if binary is already an absolute path
        if os.path.isabs(binary):
            expanded_path = os.path.expandvars(binary)
            resolved = os.path.expanduser(expanded_path)
            if self.args.verbose:
                print(f"  Binary resolved via absolute path: {resolved}")
            return resolved

        # Strategy 2: Expand environment variables in binary path
        if '$' in binary or '~' in binary:
            expanded_path = os.path.expandvars(binary)
            expanded_path = os.path.expanduser(expanded_path)
            # If after expansion it becomes absolute, use it
            if os.path.isabs(expanded_path):
                if self.args.verbose:
                    print(f"  Binary resolved via env expansion: {expanded_path}")
                return expanded_path
            # Otherwise treat as relative to test_binary_dir or build_dir
            binary = expanded_path

        # Strategy 3: Check for custom test_binary_dir in config
        test_binary_dir = test_config.get("test_binary_dir", "")
        if test_binary_dir:
            # Expand environment variables in test_binary_dir
            test_binary_dir = os.path.expandvars(test_binary_dir)
            test_binary_dir = os.path.expanduser(test_binary_dir)
            resolved = os.path.join(test_binary_dir, binary)
            if self.args.verbose:
                print(f"  Binary resolved via test config test_binary_dir: {resolved}")
            return resolved

        # Strategy 4: Check for test_binary_dir in paths config
        if "test_binary_dir" in self.paths:
            test_binary_dir = self.paths["test_binary_dir"]
            # Expand environment variables in test_binary_dir
            test_binary_dir = os.path.expandvars(test_binary_dir)
            test_binary_dir = os.path.expanduser(test_binary_dir)
            resolved = os.path.join(test_binary_dir, binary)
            if self.args.verbose:
                print(f"  Binary resolved via paths config test_binary_dir: {resolved}")
            return resolved

        # Strategy 5: Default - use build_dir/test/binary
        resolved = os.path.join(self.build_dir, "test", binary)
        if self.args.verbose:
            print(f"  Binary resolved via default build_dir/test: {resolved}")
        return resolved

    def run_test(self, test_config, suite_config):
        """
        Run a single test

        Args:
            test_config: Test configuration dict
            suite_config: Test suite configuration dict

        Returns:
            dict: Test result
        """
        test_name = test_config.get("name")
        is_gtest = test_config.get("is_gtest", True)  # Default to True for backward compatibility
        description = test_config.get("description", "")
        binary = test_config.get("binary", "rccl-UnitTestsMPI")

        # Use test_filter for all test types
        test_filter = test_config.get("test_filter", "*")

        num_ranks = test_config.get("num_ranks", 1)
        num_nodes = test_config.get("num_nodes", 1)
        num_gpus = test_config.get("num_gpus", 8)  # GPUs per node (default: 8)
        timeout = test_config.get("timeout", 0)
        env_vars = test_config.get("env_variables", {})

        # Support custom command arguments for non-gtest or specialized tests
        custom_args = test_config.get("command_args", "")

        # Merge environment variables
        merged_env = {
            **self.global_env,
            **suite_config.get("env_variables", {}),
            **env_vars
        }

        print(f"\n{'='*80}")
        print(f"Test: {test_name}")
        print(f"{'='*80}")

        if self.args.verbose:
            if description:
                print(f"  Description: {description}")
            print(f"  Type:    {'gtest' if is_gtest else 'non-gtest'}")
            print(f"  Binary:  {binary}")
            print(f"  Filter:  {test_filter}")
            print(f"  Ranks:   {num_ranks}")
            print(f"  Nodes:   {num_nodes}")
            print(f"  GPUs/node: {num_gpus}")
            print(f"  Timeout: {timeout if timeout > 0 else 'unlimited'}")
            if custom_args:
                print(f"  Custom args: {custom_args}")
            if merged_env:
                print(f"  Environment variables ({len(merged_env)}):")
                for key, value in merged_env.items():
                    print(f"    {key}={value}")
            print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Resolve binary path using flexible strategies
        test_binary_path = self._resolve_binary_path(binary, test_config)

        if self.args.verbose:
            print(f"  Binary path: {test_binary_path}")

        if not os.path.isfile(test_binary_path):
            if num_ranks > 1:
                print(f"SKIP: MPI test binary not found: {test_binary_path} (build may not have --enable-mpi-tests)")
                return {
                    "name": test_name,
                    "result": TestResult.RESULT_SKIPPED.value,
                    "duration": 0,
                    "error": f"MPI binary not found: {test_binary_path}"
                }
            print(f"ERROR: Test binary not found: {test_binary_path}")
            return {
                "name": test_name,
                "result": TestResult.RESULT_FAILED.value,
                "duration": 0,
                "error": f"Binary not found: {test_binary_path}"
            }

        # For MPI tests, verify mpirun is available
        if num_ranks > 1:
            mpi_path = self.paths.get("mpi_path", "")
            mpirun = os.path.join(mpi_path, "bin", "mpirun") if mpi_path else shutil.which("mpirun")
            if mpi_path and not os.path.isfile(os.path.join(mpi_path, "bin", "mpirun")):
                mpirun = None
            if not mpirun:
                print(f"SKIP: mpirun not found, cannot run MPI test '{test_name}'")
                return {
                    "name": test_name,
                    "result": TestResult.RESULT_SKIPPED.value,
                    "duration": 0,
                    "error": "mpirun not available"
                }

        # Multi-node tests: skip if hostfile / SLURM provides fewer hosts than required
        if num_ranks > 1 and num_nodes > 1:
            avail = _distinct_host_count(self.mpi_hosts)
            if avail > 0 and avail < num_nodes:
                msg = (
                    f"SKIP: test needs {num_nodes} distinct host(s), "
                    f"hostfile/SLURM has {avail}"
                )
                print(msg)
                return {
                    "name": test_name,
                    "result": TestResult.RESULT_SKIPPED.value,
                    "duration": 0,
                    "error": msg,
                }

        # Setup environment
        env = os.environ.copy()

        # Build LD_LIBRARY_PATH with build dir and MPI lib (if available)
        mpi_path = self.paths.get("mpi_path", "")
        ld_library_path_parts = [self.build_dir]
        if mpi_path:
            ld_library_path_parts.append(os.path.join(mpi_path, "lib"))
        if env.get('LD_LIBRARY_PATH'):
            ld_library_path_parts.append(env.get('LD_LIBRARY_PATH'))
        env['LD_LIBRARY_PATH'] = ":".join(ld_library_path_parts)

        # Set LLVM_PROFILE_FILE for code coverage (prevents default.profraw collision)
        env['LLVM_PROFILE_FILE'] = "rccl_tests_%p_%m.profraw"

        # Add test-specific env vars
        for key, value in merged_env.items():
            env[key] = str(value)

        # Build command based on test type
        if num_ranks == 1:
            # Non-MPI test - prepend environment variables to the command
            env_prefix = ""
            for key, value in merged_env.items():
                env_prefix += f"{key}={value} "

            if is_gtest:
                # GTest-based test - use --gtest_filter syntax
                if test_filter == "ALL" or test_filter == "*":
                    cmd = f"{env_prefix}./{binary}"
                else:
                    cmd = f"{env_prefix}./{binary} --gtest_filter={test_filter}"

                # Add custom arguments if provided
                if custom_args:
                    cmd += f" {custom_args}"
            else:
                # Non-gtest test (perf, custom, etc.) - run binary with args
                cmd = f"{env_prefix}./{binary}"
                if custom_args:
                    cmd += f" {custom_args}"

        else:
            # MPI test
            mpi_path = self.paths.get("mpi_path", "")
            mpi_cmd = f"{mpi_path}/bin/mpirun" if mpi_path else "mpirun"

            # Allow running as root (common in Docker containers)
            if os.getuid() == 0:
                mpi_cmd += " --allow-run-as-root"

            # Use cached host detection from initialization
            if 'host_list' in self.mpi_hosts:
                # SLURM mode: use --host with slot counts instead of --map-by ppr
                # Repeat each host num_gpus times to place that many ranks per node
                hosts = self.mpi_hosts['host_list'].split(',')
                expanded = ','.join(f"{h}:{num_gpus}" for h in hosts)
                host_arg = f"--host {expanded} "
                map_by_arg = ""
            elif 'hostfile' in self.mpi_hosts:
                host_arg = f"--hostfile {self.mpi_hosts['hostfile']} "
                # Use PPR to control ranks per node with hostfile
                map_by_arg = f"--map-by ppr:{num_gpus}:node " if num_nodes > 1 else ""
            else:
                if num_nodes > 1:
                    print("WARNING: Multi-node test without hostfile or SLURM allocation")
                host_arg = ""
                map_by_arg = ""

            # MCA params priority: --system profile lookup > test-level "mpi_args" string > default
            default_mca = self.mpi_config["default_args"]
            system = getattr(self.args, 'system', '') or ''
            mpi_args_config = self.config_processor.config.get("mpi_args", {})

            if system:
                if isinstance(mpi_args_config, dict) and system in mpi_args_config:
                    mca_params = mpi_args_config[system]
                else:
                    mca_params = default_mca
            elif isinstance(mpi_args_config, str) and mpi_args_config:
                mca_params = mpi_args_config
            else:
                config_mpi_args = test_config.get("mpi_args", "")
                mca_params = config_mpi_args if config_mpi_args else default_mca

            mpi_args = (
                f"-np {num_ranks} "
                f"{host_arg}"
                f"{map_by_arg}"
                f"{mca_params}"
            )

            # Add environment variables for MPI (quote values to handle shell metacharacters like ;)
            env_fmt = self.mpi_config["env_format"]
            for key, value in merged_env.items():
                mpi_args += " " + env_fmt.format(key=key, value=value)

            mpi_args += " " + env_fmt.format(key="LD_LIBRARY_PATH", value=env['LD_LIBRARY_PATH'])
            mpi_args += " " + env_fmt.format(key="LLVM_PROFILE_FILE", value="rccl_tests_%p_%m.profraw")

            # Forward LD_PRELOAD so UCX core libraries are preloaded with
            # global visibility on remote ranks (required for UCX PML)
            ld_preload = os.environ.get("LD_PRELOAD", "")
            if ld_preload:
                mpi_args += " " + env_fmt.format(key="LD_PRELOAD", value=ld_preload)

            # Build test command based on type
            if is_gtest:
                # GTest-based test - use --gtest_filter syntax
                if test_filter == "ALL" or test_filter == "*":
                    cmd = f"{mpi_cmd} {mpi_args} ./{binary}"
                else:
                    cmd = f"{mpi_cmd} {mpi_args} ./{binary} --gtest_filter={test_filter}"

                if custom_args:
                    cmd += f" {custom_args}"
            else:
                # Non-gtest test (perf, custom, etc.) - run binary with args
                cmd = f"{mpi_cmd} {mpi_args} ./{binary}"
                if custom_args:
                    cmd += f" {custom_args}"

        gtest_json_path = None
        if is_gtest:
            fd, gtest_json_path = tempfile.mkstemp(
                prefix="rccl_gtest_", suffix=".json", dir=tempfile.gettempdir()
            )
            os.close(fd)
            cmd += f" --gtest_output=json:{shlex.quote(gtest_json_path)}"

        if self.args.verbose:
            print(f"\n  Command: {cmd}")
            print(f"  Working directory: {os.path.join(self.build_dir, 'test')}")
            print(f"  LD_LIBRARY_PATH: {env.get('LD_LIBRARY_PATH', '')}")
            print(f"  LLVM_PROFILE_FILE: {env.get('LLVM_PROFILE_FILE', 'Not set')}\n")

        # Inherit stdout/stderr (no PIPE capture). For gtest, --gtest_output=json:…
        # (temp file, removed in finally) supplies reliable SKIPPED vs PASSED on exit 0.
        start_time = time.time()
        run_kwargs = {
            "shell": True,
            "cwd": os.path.join(self.build_dir, "test"),
            "env": env,
            "capture_output": False,
        }
        if timeout > 0:
            run_kwargs["timeout"] = timeout
        try:
            try:
                result = subprocess.run(cmd, **run_kwargs)
            except subprocess.TimeoutExpired as e:
                duration = time.time() - start_time
                parts = []
                if getattr(e, "stdout", None):
                    parts.append(e.stdout)
                if getattr(e, "stderr", None):
                    parts.append(e.stderr)
                combined = "".join(parts)
                if combined:
                    print(combined, end="" if combined.endswith("\n") else "\n")
                print(f"\n  Result: {TestResult.RESULT_TIMEOUT.value} after {timeout} seconds")
                return {
                    "name": test_name,
                    "result": TestResult.RESULT_TIMEOUT.value,
                    "duration": duration,
                    "error": f"Test timed out after {timeout} seconds",
                }
            except Exception as e:
                duration = time.time() - start_time
                print(f"\n  ERROR: {e}")
                return {
                    "name": test_name,
                    "result": TestResult.RESULT_FAILED.value,
                    "duration": duration,
                    "error": str(e)
                }

            duration = time.time() - start_time

            if is_gtest:
                rc = result.returncode if result.returncode is not None else -1
                test_result = infer_gtest_result_from_json_file(gtest_json_path or "", rc)
            else:
                if result.returncode == ExitCode.EXIT_SUCCESS:
                    test_result = TestResult.RESULT_PASSED.value
                elif result.returncode == ExitCode.EXIT_TIMEOUT:
                    test_result = TestResult.RESULT_TIMEOUT.value
                else:
                    test_result = TestResult.RESULT_FAILED.value

            print(f"\n  Result: {test_result} ({duration:.3f} seconds)")

            return {
                "name": test_name,
                "result": test_result,
                "duration": duration,
                "exit_code": int(result.returncode) if result.returncode is not None else -1,
            }
        finally:
            if gtest_json_path:
                try:
                    os.unlink(gtest_json_path)
                except OSError:
                    pass

    def run_test_suite(self, suite_config):
        """
        Run all tests in a test suite

        Args:
            suite_config: Test suite configuration dict

        Returns:
            list: List of test results
        """
        suite_name = suite_config["suite_details"]["name"]

        if self.args.verbose:
            print(f"\n{'='*80}")
            print(f"TEST SUITE: {suite_name}")
            print(f"{'='*80}")

        tests = suite_config.get("tests", [])
        if not tests:
            print(f"WARNING: No tests defined for test suite '{suite_name}'")
            return []

        results = []
        skipped_count = 0
        should_stop = False  # Track if we should stop due to rerun failure

        for test in tests:
            # Check if we should stop due to previous rerun failure
            if should_stop:
                if self.args.verbose:
                    print(f"\nStopping test suite execution due to rerun failure (--stop-on-rerun-failure)")
                break

            # Filter by test name if specified
            test_name = test.get("name")
            if self.args.test_name and test_name != self.args.test_name:
                skipped_count += 1
                continue

            # Skip MPI tests when --skip-mpi-check is set
            test_ranks = test.get("num_ranks", suite_config.get("num_ranks", 1))
            if self.args.skip_mpi_check and test_ranks > 1:
                skipped_count += 1
                if self.args.verbose:
                    print(f"  SKIP: '{test_name}' requires {test_ranks} ranks (--skip-mpi-check)")
                continue

            result = self.run_test(test, suite_config)
            results.append(result)

            self.test_names.append(test_name)
            self.test_results.append(result["result"])
            self.test_durations.append(result["duration"])
            self.test_suites.append(suite_name)

            # If test failed and rerun flag is set, rerun immediately
            if self.args.rerun_failed and result["result"] in [TestResult.RESULT_FAILED.value, TestResult.RESULT_TIMEOUT.value]:
                # Get rerun_env_variables from suite config or test config
                rerun_env = suite_config.get("rerun_env_variables", {})
                test_rerun_env = test.get("rerun_env_variables", {})

                # Merge rerun environments (test-level overrides suite-level)
                merged_rerun_env = {**rerun_env, **test_rerun_env}

                if merged_rerun_env:
                    print(f"\n{'='*80}")
                    print(f"RERUNNING FAILED TEST IMMEDIATELY")
                    print(f"{'='*80}")

                    # Create a modified test config with merged environment variables
                    rerun_test_config = copy.deepcopy(test)

                    # Merge original env_variables with rerun_env_variables
                    original_env = test.get("env_variables", {})
                    rerun_test_config["env_variables"] = {**original_env, **merged_rerun_env}

                    print(f"\nRerunning test: {test_name}")
                    print(f"  Original result: {result['result']}")
                    print(f"  Additional env variables:")
                    for key, value in merged_rerun_env.items():
                        print(f"    {key}={value}")
                    if self.args.verbose:
                        print(f"  Final merged env_variables for rerun:")
                        for key, value in rerun_test_config["env_variables"].items():
                            print(f"    {key}={value}")

                    # Run the test with merged environment
                    rerun_result = self.run_test(rerun_test_config, suite_config)

                    # Track rerun results
                    self.rerun_names.append(test_name)
                    self.rerun_results.append(rerun_result["result"])
                    self.rerun_durations.append(rerun_result["duration"])

                    print(f"  Rerun result: {rerun_result['result']}")
                    if "exit_code" in rerun_result:
                        print(f"  Rerun exit code: {rerun_result['exit_code']}")
                    print(f"{'='*80}\n")

                    # Check if rerun also failed and we should stop
                    if self.args.stop_on_rerun_failure and rerun_result["result"] in [TestResult.RESULT_FAILED.value, TestResult.RESULT_TIMEOUT.value]:
                        print(f"\nERROR: Rerun failed for test '{test_name}'")
                        print(f"Stopping test execution (--stop-on-rerun-failure)")
                        should_stop = True
                    # Otherwise continue to next test regardless of rerun result
                else:
                    if self.args.verbose:
                        print(f"SKIP: No rerun_env_variables defined for failed test '{test_name}'")

        if self.args.verbose and skipped_count > 0:
            print(f"  Skipped {skipped_count} test(s) due to filters")

        return results

    def _format_duration(self, seconds):
        """
        Format duration in a human-readable format

        Args:
            seconds: Duration in seconds

        Returns:
            str: Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes} min {secs:.2f} sec"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours} hr {minutes} min {secs:.2f} sec"

    def print_summary(self):
        """Print test execution summary"""
        total_tests = len(self.test_results)
        passed = self.test_results.count(TestResult.RESULT_PASSED.value)
        failed = self.test_results.count(TestResult.RESULT_FAILED.value)
        timeout = self.test_results.count(TestResult.RESULT_TIMEOUT.value)
        skipped = self.test_results.count(TestResult.RESULT_SKIPPED.value)

        # Calculate total test time
        total_time_seconds = sum(self.test_durations) if self.test_durations else 0

        # Get unique test suites that were run
        unique_suites = sorted(set(self.test_suites)) if self.test_suites else []

        if total_tests > 0:
            print("\nDetailed Results:")
            print("-"*120)
            print(f"{'Test Suite':<40} {'Test Name':<40} {'Result':<10} {'Duration'}")
            print("-"*120)
            for i in range(total_tests):
                print(
                    f"{self.test_suites[i]:<40} "
                    f"{self.test_names[i]:<40} "
                    f"{self.test_results[i]:<10} "
                    f"{self.test_durations[i]:.3f} seconds"
                )
            print("-"*120)
            print(f"Total Tests:   {total_tests}")
            print(f"Passed:        {passed}")
            print(f"Failed:        {failed}")
            print(f"Skipped:       {skipped}")
            print(f"Timeout:       {timeout}")
            if skipped > 0:
                print(f"Skipped:       {skipped}")
            print(f"Total Time:    {self._format_duration(total_time_seconds)}")
            print("="*120)

        # Print rerun results if any
        if self.rerun_results:
            total_reruns = len(self.rerun_results)
            rerun_passed = self.rerun_results.count(TestResult.RESULT_PASSED.value)
            rerun_failed = self.rerun_results.count(TestResult.RESULT_FAILED.value)
            rerun_timeout = self.rerun_results.count(TestResult.RESULT_TIMEOUT.value)
            rerun_time_seconds = sum(self.rerun_durations) if self.rerun_durations else 0

            print("\nRerun Results (with additional environment variables):")
            print("-"*120)
            print(f"{'Test Name':<60} {'Result':<10} {'Duration'}")
            print("-"*120)
            for i in range(total_reruns):
                print(
                    f"{self.rerun_names[i]:<60} "
                    f"{self.rerun_results[i]:<10} "
                    f"{self.rerun_durations[i]:.3f} seconds"
                )
            print("-"*120)
            print(f"Total Reruns:  {total_reruns}")
            print(f"Passed:        {rerun_passed}")
            print(f"Failed:        {rerun_failed}")
            print(f"Timeout:       {rerun_timeout}")
            print(f"Total Time:    {self._format_duration(rerun_time_seconds)}")
            print("="*120)

    def generate_coverage_report(self):
        """Generate code coverage report"""
        if not self.args.coverage_report:
            return

        print(f"\n{'='*80}")
        print("GENERATING COVERAGE REPORT")
        print(f"{'='*80}")

        # Check for profraw files
        import glob
        import shutil

        profraw_files = glob.glob(os.path.join(self.build_dir, "**/*.profraw"), recursive=True)

        if not profraw_files:
            print("WARNING: No profraw files found. Cannot generate coverage report.")
            return

        print(f"Found {len(profraw_files)} profraw files")

        os.makedirs(self.report_dir, exist_ok=True)

        # Create rawfiles directory
        rawfiles_dir = os.path.join(self.log_dir, "rawfiles")
        os.makedirs(rawfiles_dir, exist_ok=True)

        # Move all profraw files into a single location
        print("Copying profraw files...")
        for profraw in profraw_files:
            shutil.copy(profraw, rawfiles_dir)

        # Create a list of raw files to merge
        rawprofiles_list = os.path.join(self.log_dir, "rawprofiles.list")
        with open(rawprofiles_list, 'w') as f:
            for profraw in glob.glob(os.path.join(rawfiles_dir, "*.profraw")):
                f.write(f"{profraw}\n")

        # Get ROCm path for LLVM tools
        rocm_path = self.paths.get("rocm_path", "/opt/rocm")
        llvm_profdata = os.path.join(rocm_path, "lib", "llvm", "bin", "llvm-profdata")
        llvm_cov = os.path.join(rocm_path, "lib", "llvm", "bin", "llvm-cov")

        if self.args.verbose:
            print(f"ROCm path:      {rocm_path}")
            print(f"llvm-profdata:  {llvm_profdata} (exists: {os.path.isfile(llvm_profdata)})")
            print(f"llvm-cov:       {llvm_cov} (exists: {os.path.isfile(llvm_cov)})")
            print(f"Rawfiles dir:   {rawfiles_dir}")

        # Create the merged profdata
        print("Merging profraw files...")
        merged_profdata = os.path.join(self.log_dir, "merged.profdata")

        merge_cmd = [
            llvm_profdata,
            "merge",
            "--sparse",
            f"--input-files={rawprofiles_list}",
            f"--output={merged_profdata}"
        ]

        if self.args.verbose:
            print(f"Merge command: {' '.join(merge_cmd)}")

        try:
            result = subprocess.run(
                merge_cmd,
                capture_output=False,
                text=True,
                check=True
            )
            print("Profraw files merged successfully")
            if self.args.verbose:
                print(f"Merged profdata file: {merged_profdata}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to merge profraw files")
            print(f"Command: {' '.join(merge_cmd)}")
            print(f"Error: {e.stderr}")
            return

        # Build list of object files
        object_files = []

        librccl_so = os.path.join(self.build_dir, "librccl.so")
        if os.path.isfile(librccl_so):
            object_files.extend(["--object", librccl_so])
            if self.args.verbose:
                print(f"Found library: {librccl_so}")

        # Add test binaries
        test_dir = os.path.join(self.build_dir, "test")
        for binary in ["rccl-UnitTestsFixtures", "rccl-UnitTests", "rccl-UnitTestsMPI"]:
            binary_path = os.path.join(test_dir, binary)
            if os.path.isfile(binary_path):
                object_files.extend(["--object", binary_path])
                if self.args.verbose:
                    print(f"Found binary: {binary_path}")

        if not object_files:
            print("WARNING: No object files found for coverage report")
            return

        if self.args.verbose:
            print(f"Total object files for coverage: {len(object_files) // 2}")

        # Ignore patterns for non-relevant files
        ignore_regex = (
            ".*tuner_v.*|.*profiler_v.*|.*net_v.*|.*_deps.*|ext.*|"
            ".*coll_net.*|.*nvls.*|.*nvml.*|.*nvtx.*|test/|.*gtest.*"
        )

        if self.args.verbose:
            print(f"Ignore regex: {ignore_regex}")

        # Create the HTML report
        print("Generating HTML coverage report...")
        html_cmd = [
            llvm_cov,
            "show",
            f"--instr-profile={merged_profdata}",
            "--format=html",
            "--Xdemangler=c++filt",
            f"--output-dir={self.report_dir}",
            "--project-title=RCCL_Lib_Coverage_Report",
            f"--ignore-filename-regex={ignore_regex}"
        ]
        html_cmd.extend(object_files)

        if self.args.verbose:
            print(f"HTML coverage command: {' '.join(html_cmd)}")

        try:
            result = subprocess.run(
                html_cmd,
                capture_output=False,
                text=True,
                check=True
            )
            print(f"HTML coverage report generated: {self.report_dir}/index.html")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to generate HTML coverage report")
            print(f"Error: {e.stderr}")
            if self.args.verbose:
                print(f"Command was: {' '.join(html_cmd)}")

        # Generate function coverage summary (text report)
        print("Generating text coverage report...")
        text_report = os.path.join(self.report_dir, "function_coverage_report.txt")

        # Build command matching bash script exactly
        text_cmd = [
            llvm_cov,
            "report",
            f"--instr-profile={merged_profdata}",
            "--Xdemangler=c++filt"
        ]
        # Add object files first
        text_cmd.extend(object_files)
        # Add remaining options - matching bash script order
        text_cmd.extend([
            f"--ignore-filename-regex={ignore_regex}",
            "--show-functions",
            "--sources",
            self.build_dir
        ])

        if self.args.verbose:
            print(f"Text coverage command: {' '.join(text_cmd)}")

        try:
            with open(text_report, 'w') as f:
                result = subprocess.run(
                    text_cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
            print(f"Function coverage report generated: {text_report}")

        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to generate text coverage report")
            print(f"Error: {e.stderr}")
            if self.args.verbose:
                print(f"Command was: {' '.join(text_cmd)}")

        print(f"\n{'='*80}")
        print("COVERAGE REPORT GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"Report directory: {self.report_dir}")
        print(f"HTML report: {self.report_dir}/index.html")
        print(f"Text report: {text_report}")

