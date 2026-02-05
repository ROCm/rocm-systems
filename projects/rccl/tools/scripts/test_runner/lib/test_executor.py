#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Test Executor Module
Handles test execution, build processes, and result tracking
"""

import os
import subprocess
import sys
import time
import datetime
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


class TestExecutor:
    """
    Executes tests and manages build/test workflows
    """

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
        self.global_env = config_processor.get_env_variables()
        self.build_config = config_processor.get_build_config()

        # Setup directories
        self.setup_directories()

        # Detect MPI hostfile once during initialization
        self.mpi_hostfile = self._detect_mpi_hostfile()

        # Test tracking
        self.test_results = []
        self.test_names = []
        self.test_durations = []
        self.test_suites = []

    def setup_directories(self):
        """Setup build and log directories"""
        workdir = self.paths.get("workdir", os.getcwd())

        # Determine workspace name (with or without timestamp)
        suffix_part = f"_{self.args.report_suffix}" if self.args.report_suffix else ""
        if self.args.overwrite:
            workspace_name = f"rccl_test_artifacts{suffix_part}"
            timestamp_suffix = ""
        else:
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
            workspace_name = f"rccl_test_artifacts{suffix_part}_{timestamp}"
            timestamp_suffix = f"_{timestamp}"

        # Create workspace directory path
        self.workspace_dir = os.path.join(workdir, workspace_name)

        # Check for custom RCCL library path from environment variable
        custom_rccl_path = os.environ.get('RCCL_LIB_PATH') or os.environ.get('RCCL_BUILD_DIR')

        if custom_rccl_path:
            # Use custom library path from environment variable
            self.build_dir = os.path.expanduser(os.path.expandvars(custom_rccl_path))
            self.using_custom_lib = True
            if self.args.verbose:
                print(f"Using custom RCCL library path from environment: {self.build_dir}")
        else:
            # Use default build directory
            self.using_custom_lib = False
            self.build_dir = os.path.join(
                workdir,
                f"build_debug_cov_on_tests_on{timestamp_suffix}"
            )

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
                print(f"  (Using custom library from RCCL_LIB_PATH/RCCL_BUILD_DIR)")
            print(f"Log directory:    {self.log_dir}")
            print(f"Report directory: {self.report_dir}")

    def _detect_mpi_hostfile(self):
        """
        Detect MPI hostfile once during initialization.
        Checks RCCL_TEST_MPI_HOSTFILE env var, then ~/.mpi_hostfile default.
        Prints detection message only once.

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

        # No hostfile found
        return None

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

        # Check MPI
        mpi_path = self.paths.get("mpi_path")
        if mpi_path:
            if not os.path.isdir(mpi_path):
                print(f"WARNING: MPI path not found: {mpi_path}")
            elif not os.path.isfile(os.path.join(mpi_path, "bin", "mpirun")):
                print(f"WARNING: mpirun not found in {mpi_path}/bin/")

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
        Build RCCL with test support using configurable build settings

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

        # Get build configuration (with defaults)
        cmake_options = self.build_config.get("cmake_options", {})
        build_env_vars = self.build_config.get("env_variables", {})
        parallel_jobs = self.build_config.get("parallel_jobs", 64)
        generator = self.build_config.get("generator", "Unix Makefiles")

        if self.args.verbose:
            print(f"Work directory:  {workdir}")
            print(f"ROCm path:       {rocm_path}")
            print(f"MPI path:        {mpi_path}")
            print(f"Build directory: {self.build_dir}")
            print(f"Parallel jobs:   {parallel_jobs}")
            print(f"Generator:       {generator}")

        # Setup environment for build
        env = os.environ.copy()

        # Apply default environment variables for code coverage
        default_env = {
            'HIPCC_COMPILE_FLAGS_APPEND': (
                "-g -Wno-format-nonliteral -Xarch_host -fprofile-instr-generate "
                "-Xarch_host -fcoverage-mapping -parallel-jobs=16"
            ),
            'HIPCC_LINK_FLAGS_APPEND': (
                "-fprofile-instr-generate -fcoverage-mapping -parallel-jobs=16"
            ),
            'LLVM_PROFILE_FILE': "rccl_tests_%p_%m.profraw",
            'CXX': f"{rocm_path}/bin/amdclang++"
        }

        # Merge with user-provided build environment variables (user values override defaults)
        for key, value in default_env.items():
            env[key] = value
        for key, value in build_env_vars.items():
            env[key] = str(value)

        # Build CMake configuration command with defaults
        default_cmake_options = {
            "CMAKE_CXX_FLAGS": "-Wl,--build-id=sha1",
            "CMAKE_EXE_LINKER_FLAGS": "-Wl,--build-id=sha1",
            "CMAKE_BUILD_TYPE": "Debug",
            "ENABLE_CODE_COVERAGE": "ON",
            "BUILD_TESTS": "ON",
            "BUILD_LOCAL_GPU_TARGET_ONLY": "ON",
            "TRACE": "ON",
            "COLLTRACE": "ON",
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
            "CMAKE_VERBOSE_MAKEFILE": "1",
            "ENABLE_MPI_TESTS": "ON",
            "MPI_PATH": mpi_path
        }

        # Merge with user-provided CMake options (user values override defaults)
        merged_cmake_options = {**default_cmake_options, **cmake_options}

        # Build CMake command
        cmake_cmd = [
            "cmake",
            "-S", workdir,
            "-B", self.build_dir
        ]

        # Add CMake options as -D flags
        for key, value in merged_cmake_options.items():
            cmake_cmd.append(f"-D{key}={value}")

        # Add generator
        cmake_cmd.append(f"-G{generator}")

        try:
            print("Running CMake configuration...")
            if self.args.verbose:
                print(f"CMake command: {' '.join(cmake_cmd)}")
                print(f"Build environment variables:")
                for key, value in build_env_vars.items():
                    print(f"  {key}={value}")

            result = subprocess.run(
                cmake_cmd,
                cwd=workdir,
                env=env,
                capture_output=False
            )

            if result.returncode != 0:
                print(f"ERROR: CMake configuration failed")
                return False

            print("\nRunning CMake build...")
            build_cmd = f"cmake --build {self.build_dir} --parallel {parallel_jobs}"
            if self.args.verbose:
                print(f"Build command: {build_cmd}")

            result = subprocess.run(
                build_cmd,
                shell=True,
                cwd=workdir,
                env=env,
                capture_output=False
            )

            if result.returncode != 0:
                print(f"ERROR: CMake build failed")
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
            return os.path.expanduser(expanded_path)

        # Strategy 2: Expand environment variables in binary path
        if '$' in binary or '~' in binary:
            expanded_path = os.path.expandvars(binary)
            expanded_path = os.path.expanduser(expanded_path)
            # If after expansion it becomes absolute, use it
            if os.path.isabs(expanded_path):
                return expanded_path
            # Otherwise treat as relative to test_binary_dir or build_dir
            binary = expanded_path

        # Strategy 3: Check for custom test_binary_dir in config
        test_binary_dir = test_config.get("test_binary_dir", "")
        if test_binary_dir:
            # Expand environment variables in test_binary_dir
            test_binary_dir = os.path.expandvars(test_binary_dir)
            test_binary_dir = os.path.expanduser(test_binary_dir)
            return os.path.join(test_binary_dir, binary)

        # Strategy 4: Check for test_binary_dir in paths config
        if "test_binary_dir" in self.paths:
            test_binary_dir = self.paths["test_binary_dir"]
            # Expand environment variables in test_binary_dir
            test_binary_dir = os.path.expandvars(test_binary_dir)
            test_binary_dir = os.path.expanduser(test_binary_dir)
            return os.path.join(test_binary_dir, binary)

        # Strategy 5: Default - use build_dir/test/binary
        return os.path.join(self.build_dir, "test", binary)

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

        if self.args.verbose:
            print(f"\n{'='*80}")
            print(f"Test: {test_name}")
            print(f"{'='*80}")
            if description:
                print(f"  Description: {description}")
            print(f"  Type:    {'gtest' if is_gtest else 'non-gtest'}")
            print(f"  Binary:  {binary}")
            print(f"  Filter:  {test_filter}")
            print(f"  Ranks:   {num_ranks}")
            print(f"  Nodes:   {num_nodes}")
            print(f"  GPUs/node: {num_gpus}")
            print(f"  Timeout: {timeout if timeout > 0 else 'unlimited'}")
            print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Resolve binary path using flexible strategies
        test_binary_path = self._resolve_binary_path(binary, test_config)

        if self.args.verbose:
            print(f"  Binary path: {test_binary_path}")

        if not os.path.isfile(test_binary_path):
            print(f"ERROR: Test binary not found: {test_binary_path}")
            return {
                "name": test_name,
                "result": TestResult.RESULT_FAILED.value,
                "duration": 0,
                "error": f"Binary not found: {test_binary_path}"
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

            # Use cached hostfile detected during initialization
            hostfile = self.mpi_hostfile

            # Warn if multi-node test without hostfile
            if hostfile is None and num_nodes > 1:
                print("WARNING: Multi-node test without hostfile")

            hostfile_arg = f"--hostfile {hostfile} " if hostfile else ""

            # Determine mapping strategy based on num_gpus and num_nodes
            # Use PPR (processes per resource) to place num_gpus ranks per node
            # This ignores the slots specification in the hostfile
            if num_nodes > 1:
                # Multi-node test: use ppr to control ranks per node
                map_by_arg = f"--map-by ppr:{num_gpus}:node "
            else:
                # Single node: use default mapping (no need for ppr)
                map_by_arg = ""

            mpi_args = (
                f"-np {num_ranks} "
                f"{hostfile_arg}"
                f"{map_by_arg}"
                f"--mca btl ^vader,openib "
                f"--mca pml ucx "
                f"--bind-to none"
            )

            # Add environment variables for MPI
            for key, value in merged_env.items():
                mpi_args += f" -x {key}={value}"

            # Pass the LD_LIBRARY_PATH
            mpi_args += f" -x LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}"

            # Pass LLVM_PROFILE_FILE to MPI ranks for code coverage (prevents default.profraw collision)
            mpi_args += f" -x LLVM_PROFILE_FILE=rccl_tests_%p_%m.profraw"

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


        if self.args.verbose:
            print(f"\n  Command: {cmd}")
            print(f"  Working directory: {os.path.join(self.build_dir, 'test')}")
            print(f"  LD_LIBRARY_PATH: {env.get('LD_LIBRARY_PATH', '')}")
            print(f"  LLVM_PROFILE_FILE: {env.get('LLVM_PROFILE_FILE', 'Not set')}\n")

        # Execute test
        start_time = time.time()
        try:
            if timeout > 0:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=os.path.join(self.build_dir, "test"),
                    env=env,
                    capture_output=False,
                    timeout=timeout
                )
            else:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=os.path.join(self.build_dir, "test"),
                    env=env,
                    capture_output=False
                )

            duration = time.time() - start_time

            # Determine result
            if result.returncode == ExitCode.EXIT_SUCCESS:
                test_result = TestResult.RESULT_PASSED.value
            elif result.returncode == ExitCode.EXIT_TIMEOUT:
                test_result = TestResult.RESULT_TIMEOUT.value
            else:
                test_result = TestResult.RESULT_FAILED.value

            if self.args.verbose:
                print(f"\n  Result: {test_result} ({duration:.3f} seconds)")

            return {
                "name": test_name,
                "result": test_result,
                "duration": duration,
                "exit_code": result.returncode
            }

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            if self.args.verbose:
                print(f"\n  Result: {TestResult.RESULT_TIMEOUT.value} after {timeout} seconds")
            return {
                "name": test_name,
                "result": TestResult.RESULT_TIMEOUT.value,
                "duration": duration,
                "error": f"Test timed out after {timeout} seconds"
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
            if self.args.verbose:
                print(f"WARNING: No tests defined for test suite '{suite_name}'")
            return []

        results = []
        for test in tests:
            # Filter by test name if specified
            test_name = test.get("name")
            if self.args.test_name and test_name != self.args.test_name:
                continue

            result = self.run_test(test, suite_config)
            results.append(result)

            self.test_names.append(test_name)
            self.test_results.append(result["result"])
            self.test_durations.append(result["duration"])
            self.test_suites.append(suite_name)  # Track suite name

        return results

    def print_summary(self):
        """Print test execution summary"""
        total_tests = len(self.test_results)
        passed = self.test_results.count(TestResult.RESULT_PASSED.value)
        failed = self.test_results.count(TestResult.RESULT_FAILED.value)
        timeout = self.test_results.count(TestResult.RESULT_TIMEOUT.value)

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
            print(f"Timeout:       {timeout}")
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
                capture_output=True,
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
                capture_output=True,
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

