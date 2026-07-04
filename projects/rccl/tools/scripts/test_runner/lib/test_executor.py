#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Test Executor Module
Handles test execution, build processes, and result tracking
"""

import glob
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import datetime
import copy
import xml.etree.ElementTree as ET
from enum import IntEnum, Enum
from pathlib import Path

# Bash-style env-var expander (supports ${VAR:-default}); shared with the config
# processor so binary/path resolution honors the same syntax used elsewhere.
try:
    from lib.test_config import expand_env_vars
except ImportError:
    from test_config import expand_env_vars

# Make stdout unbuffered to prevent output ordering issues with subprocesses
sys.stdout.reconfigure(line_buffering=True)


def glob_filter_matches(name: str, pattern_str: str) -> bool:
    """Return True if *name* matches the gtest-style glob filter *pattern_str*.

    Syntax (same as GTest's --gtest_filter):
      *     matches any substring
      ?     matches any single character
      :     separates patterns (OR)
      -     prefix on a token negates it (exclude)

    Matching is anchored to the full name and is case-sensitive.

    Examples:
      glob_filter_matches("P2P_AllTests", "P2P_*")          → True
      glob_filter_matches("SHM_Basic",    "P2P_*")          → False
      glob_filter_matches("P2P_AllTests", "*:-P2P*")        → False  (excluded)
      glob_filter_matches("SHM_Basic",    "P2P_*:SHM_*")   → True   (OR)
    """
    def _to_re(pat: str) -> re.Pattern:
        escaped = re.sub(r'([.+^${}()|\\])', r'\\\1', pat)
        return re.compile('^' + escaped.replace('*', '.*').replace('?', '.') + '$')

    pos, neg = [], []
    for token in pattern_str.split(':'):
        (neg if token.startswith('-') else pos).append(_to_re(token.lstrip('-')))

    if neg and any(p.match(name) for p in neg):
        return False
    return (not pos) or any(p.match(name) for p in pos)


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


def infer_pytest_result_from_junit(junit_path: str, returncode: int) -> str:
    """Map a pytest run (JUnit XML + exit code) to a TestResult, preferring the
    report so a fully-skipped harness reports SKIPPED rather than PASSED."""
    if returncode == ExitCode.EXIT_TIMEOUT:
        return TestResult.RESULT_TIMEOUT.value
    # pytest exit 5 = no tests collected.
    if returncode == 5:
        return TestResult.RESULT_SKIPPED.value
    if not junit_path or not os.path.isfile(junit_path):
        return (TestResult.RESULT_PASSED.value if returncode == ExitCode.EXIT_SUCCESS
                else TestResult.RESULT_FAILED.value)
    try:
        root = ET.parse(junit_path).getroot()
    except (OSError, ET.ParseError):
        return (TestResult.RESULT_PASSED.value if returncode == ExitCode.EXIT_SUCCESS
                else TestResult.RESULT_FAILED.value)

    total = passed = skipped = failed = 0
    for tc in root.iter("testcase"):
        total += 1
        kinds = {child.tag for child in tc}
        if kinds & {"failure", "error"}:
            failed += 1
        elif "skipped" in kinds:
            skipped += 1
        else:
            passed += 1

    if failed:
        return TestResult.RESULT_FAILED.value
    # Non-clean exit with no per-test failure (collection/internal error).
    if returncode not in (0,):
        return TestResult.RESULT_FAILED.value
    if total == 0 or (passed == 0 and skipped > 0):
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
            "default_args": "--mca btl ^vader,openib --bind-to none",
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
        self.rccl_tests_build_config = config_processor.get_rccl_tests_build_config()

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

        # GPUs-per-node is detected lazily on first use (see gpus_per_node property)
        self._gpus_per_node = 0
        self._gpus_per_node_detected = False

        # Test tracking
        self.test_results = []
        self.test_names = []
        self.test_durations = []
        self.test_suites = []

        # Structured result emission (dashboard). Enabling either --emit-results or
        # --db-push turns on per-test log capture so perf output can be parsed.
        self.emit_enabled = bool(
            getattr(args, "emit_results", False) or getattr(args, "db_push", False)
        )
        self.test_records = []       # rich per-test records for the emitter
        self._emit_log_counter = 0   # keeps captured-log filenames unique

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
            self.build_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(self.args.build_dir)))
            self.using_custom_lib = True
            if self.args.verbose:
                print(f"Using custom build directory from --build-dir: {self.build_dir}")
        elif custom_rccl_path:
            # Use custom library path from environment variable
            self.build_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(custom_rccl_path)))
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

    def _emit_log_path(self, test_name):
        """Unique per-test captured-log path under log_dir (used when result
        emission is enabled)."""
        self._emit_log_counter += 1
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(test_name or "test"))
        return os.path.join(self.log_dir, f"{self._emit_log_counter:04d}_{safe}.log")

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

    @property
    def gpus_per_node(self):
        """
        Number of GPUs available on a node (lazy, detected once).

        Returns 0 when the count cannot be determined, in which case "auto"
        sizing falls back to 8 GPUs/node and GPU-count-based skipping is disabled.
        """
        if not self._gpus_per_node_detected:
            self._gpus_per_node = self._detect_gpus_per_node()
            self._gpus_per_node_detected = True
            if self._gpus_per_node:
                if self.args.verbose:
                    print(f"Detected GPUs per node: {self._gpus_per_node}")
            else:
                # Unconditional: silent fallback to 8 ranks on a smaller node is
                # very hard to debug, so always surface the detection failure.
                print("WARNING: could not detect GPU count; 'auto' sizing falls "
                      "back to 8 GPUs/node and GPU-count skipping is disabled")
        return self._gpus_per_node

    def _detect_gpus_per_node(self):
        """
        Detect the number of GPUs usable on a node.

        Priority:
          1. RCCL_TEST_GPUS_PER_NODE env override
          2. Visible-device masks (HIP/ROCR/CUDA_VISIBLE_DEVICES)
          3. rocminfo GPU agent count

        Returns:
            int: GPU count, or 0 if it cannot be determined.
        """
        override = os.environ.get('RCCL_TEST_GPUS_PER_NODE', '').strip()
        if override.isdigit() and int(override) > 0:
            return int(override)

        for var in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
            mask = os.environ.get(var)
            if mask:
                ids = [tok for tok in mask.split(',') if tok.strip() != '']
                if ids:
                    return len(ids)

        rocm_path = self.paths.get('rocm_path', '/opt/rocm')
        rocminfo = os.path.join(rocm_path, 'bin', 'rocminfo')
        if not os.path.isfile(rocminfo):
            rocminfo = shutil.which('rocminfo')
        if rocminfo:
            try:
                result = subprocess.run(
                    [rocminfo], capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0:
                    count = sum(
                        1 for line in result.stdout.splitlines()
                        if 'Device Type:' in line and 'GPU' in line
                    )
                    if count > 0:
                        return count
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

        return 0

    def check_environment(self):
        """
        Check that required environment and tools are available

        Returns:
            bool: True if environment is valid
        """
        errors = []

        # Check ROCm
        rocm_path = self._rocm_root()
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
            else:
                if self.args.verbose:
                    print(f"Found RCCL library: {lib_path}")
                # Fail fast: --coverage-report requires an instrumented library,
                # but with --no-build / custom lib we cannot rebuild it ourselves.
                if self.args.coverage_report and not self._check_coverage_instrumentation(lib_path):
                    self._print_coverage_missing_error(lib_path)
                    return False

        if errors:
            print("ERROR: Environment check failed:")
            for error in errors:
                print(f"  - {error}")
            return False

        if self.args.verbose:
            print("Environment validation passed")
        return True

    def _check_coverage_instrumentation(self, lib_path):
        """
        Verify librccl.so was built with LLVM source-based code coverage
        instrumentation by looking for the ``__llvm_prf_*`` ELF sections that
        clang adds when ``-fprofile-instr-generate -fcoverage-mapping`` is in
        effect (i.e. when RCCL was built with -DENABLE_CODE_COVERAGE=ON).

        Returns:
            bool: True if instrumented; True (with warning) if the check could
                not be performed (e.g. readelf missing); False if confirmed
                non-instrumented.
        """
        try:
            result = subprocess.run(
                ['readelf', '-S', lib_path],
                capture_output=True, text=True, timeout=30
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"WARNING: Could not run 'readelf -S {lib_path}' to verify "
                  f"coverage instrumentation: {e}")
            print("         Proceeding under the assumption it is instrumented.")
            return True
        if result.returncode != 0:
            print(f"WARNING: 'readelf -S {lib_path}' exited with "
                  f"{result.returncode}; skipping coverage instrumentation check.")
            return True
        return '__llvm_prf_' in result.stdout

    def _print_coverage_missing_error(self, lib_path):
        """Emit a clear, actionable message when --coverage-report is requested
        but the RCCL library lacks LLVM coverage instrumentation."""
        print("=" * 80)
        print("ERROR: --coverage-report was requested, but the RCCL library is")
        print("       not instrumented for LLVM source-based code coverage:")
        print(f"           {lib_path}")
        print()
        print("Rebuild RCCL with coverage instrumentation, then retry. Options:")
        print("  - Use install.sh directly:")
        print("        ./install.sh --enable-code-coverage <other flags...>")
        print("  - Or have the runner build it by adding the flag to your")
        print("    config's 'build_configuration.install_flags':")
        print('        "--enable-code-coverage"')
        print("  - Or pass the CMake option through install.sh:")
        print('        ./install.sh --cmake-options "-DENABLE_CODE_COVERAGE=ON" ...')
        print("=" * 80)

    def build_rccl(self):
        """
        Build RCCL using install.sh with configurable build settings.

        The build_configuration in the JSON config specifies:
        - install_flags: List of install.sh command-line flags
        - cmake_options: Optional CMake options, either a string (e.g. "-DFOO=BAR") or a
          dict (e.g. {"FOO": "BAR"}); dicts are converted to "-DKEY=VAL" form (passed via --cmake-options)
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
        rocm_path = self._rocm_root()
        mpi_path = self.paths.get("mpi_path", "")

        install_flags = list(self.build_config.get("install_flags", []))
        cmake_options = self.build_config.get("cmake_options", "")
        if isinstance(cmake_options, dict):
            cmake_options = " ".join(f"-D{k}={v}" for k, v in cmake_options.items())
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

        # Drop code coverage instrumentation when no coverage report is requested.
        # Coverage instrumentation slows down both runtime (counter updates) and
        # process exit (profraw file write per PID), which is significant when
        # many short-lived test processes are spawned.
        if not self.args.coverage_report:
            if "--enable-code-coverage" in install_flags:
                install_flags.remove("--enable-code-coverage")
            # Explicitly disable to override any cached CMake value from prior builds
            if cmake_options:
                cmake_options += " -DENABLE_CODE_COVERAGE=OFF"
            else:
                cmake_options = "-DENABLE_CODE_COVERAGE=OFF"
            print("NOTE: Code coverage instrumentation disabled in build "
                  "(use --coverage-report to enable)")

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

            # If --coverage-report was requested, verify the freshly built library
            # actually contains coverage instrumentation. The user's build_configuration
            # in the JSON config may not include --enable-code-coverage, in which
            # case we must abort before running tests (otherwise no profraw files
            # would be produced and the coverage step would silently produce nothing).
            if self.args.coverage_report:
                lib_path = os.path.join(self.build_dir, "librccl.so")
                if os.path.isfile(lib_path) and not self._check_coverage_instrumentation(lib_path):
                    self._print_coverage_missing_error(lib_path)
                    return False

            return True

        except Exception as e:
            print(f"ERROR: Build failed with exception: {e}")
            return False

    def build_rccl_tests(self):
        """
        Build rccl-tests (the perf binaries: all_reduce_perf, all_gather_perf, ...)
        using its own build system, mirroring build_rccl().

        The rccl_tests_build_configuration in the JSON config specifies:
        - enabled:        Set to false to skip this step entirely (default true if the
                          section is present; absent section => skipped).
        - source_dir:     Path to the rccl-tests checkout (env vars/~ expanded).
        - install_script: Build script relative to source_dir (default "install.sh").
        - install_flags:  List of flags passed to the build script (e.g. ["--mpi"]).
        - build_command:  Optional full shell command that overrides install_script/
                          install_flags (run with cwd=source_dir).
        - rccl_home:      Path to the RCCL install/build to link against. Defaults to
                          the RCCL build_dir produced by build_rccl(). Exported as both
                          NCCL_HOME and RCCL_HOME for the build.
        - env_variables:  Extra environment variables to set during the build.

        Returns:
            bool: True if the build succeeded or was intentionally skipped.
        """
        cfg = self.rccl_tests_build_config

        # No section => nothing to build (backward compatible with existing configs).
        if not cfg:
            return True

        if not cfg.get("enabled", True):
            if self.args.verbose:
                print("SKIP: rccl-tests build disabled (rccl_tests_build_configuration.enabled=false)")
            return True

        if self.args.no_build:
            if self.args.verbose:
                print("SKIP: rccl-tests build skipped (--no-build)")
            return True

        print("="*80)
        print("BUILDING rccl-tests")
        print("="*80)

        workdir = self.paths.get("workdir", os.getcwd())
        rocm_path = self._rocm_root()
        mpi_path = self.paths.get("mpi_path", "")

        # Use the bash-aware expander so ${VAR:-default} (e.g. the computed
        # RCCL_TESTS_DIR with a fallback) resolves; os.path.expandvars cannot
        # handle the ":-" default syntax and would leave the path literal.
        def _expand(p):
            return os.path.expanduser(expand_env_vars(str(p)))

        source_dir = _expand(cfg.get("source_dir", os.path.join(workdir, "rccl-tests")))
        if not os.path.isdir(source_dir):
            print(f"ERROR: rccl-tests source directory not found: {source_dir}")
            print("       Set rccl_tests_build_configuration.source_dir or the "
                  "RCCL_TESTS_DIR environment variable.")
            return False

        # RCCL to link against: explicit rccl_home, else the RCCL build_dir we built.
        rccl_home = _expand(cfg.get("rccl_home", self.build_dir))

        # Expand env vars / ~ in each flag so values like
        # "--hip_compiler ${HIP_COMPILER:-$HOME/.local/llvm/bin/amdclang++}"
        # resolve before being passed to install.sh (argv is not shell-expanded).
        install_flags = [_expand(f) for f in cfg.get("install_flags", [])]
        build_env_vars = cfg.get("env_variables", {})

        # Build the command: explicit build_command wins, otherwise install.sh + flags.
        # build_command may be a shell string or an argv array (per schema). An argv
        # array must run without a shell; a string runs through the shell. In both
        # forms expand env vars / ~ for consistency with the other resolved paths.
        build_command = cfg.get("build_command")
        if build_command:
            if isinstance(build_command, (list, tuple)):
                cmd = [_expand(arg) for arg in build_command]
                use_shell = False
            else:
                cmd = _expand(build_command)
                use_shell = True
        else:
            install_script = os.path.join(source_dir, cfg.get("install_script", "install.sh"))
            if not os.path.isfile(install_script):
                print(f"ERROR: rccl-tests build script not found: {install_script}")
                print("       Provide rccl_tests_build_configuration.build_command or "
                      "install_script.")
                return False
            # rccl-tests/install.sh parses args with getopt (short opts: hmt) and
            # parallelizes internally with -j$(nproc); it does NOT accept a -j flag.
            # It also ignores NCCL_HOME/RCCL_HOME/MPI_HOME from the environment, so
            # the RCCL and MPI locations must be passed as explicit flags (it does
            # read ROCM_PATH and HIP_COMPILER from the env, but not those homes).
            if rccl_home and "--rccl_home" not in install_flags:
                install_flags += ["--rccl_home", rccl_home]
            if rocm_path and "--rocm_home" not in install_flags:
                install_flags += ["--rocm_home", rocm_path]
            mpi_requested = any(f in ("--mpi", "-m") for f in install_flags)
            if mpi_requested and mpi_path and "--mpi_home" not in install_flags:
                install_flags += ["--mpi_home", mpi_path]
            cmd = [install_script] + install_flags
            use_shell = False

        # Setup environment: point rccl-tests at the RCCL we just built.
        env = os.environ.copy()
        env['ROCM_PATH'] = rocm_path
        if mpi_path:
            env['MPI_PATH'] = mpi_path
            env['MPI_HOME'] = mpi_path
        env['NCCL_HOME'] = rccl_home
        env['RCCL_HOME'] = rccl_home
        for key, value in build_env_vars.items():
            env[key] = str(value)

        if self.args.verbose:
            print(f"Source directory: {source_dir}")
            print(f"ROCm path:        {rocm_path}")
            print(f"MPI path:         {mpi_path}")
            print(f"RCCL home:        {rccl_home}")
            print(f"Command:          {cmd if use_shell else ' '.join(cmd)}")
            if build_env_vars:
                print("Build environment variables:")
                for key, value in build_env_vars.items():
                    print(f"  {key}={value}")

        try:
            result = subprocess.run(
                cmd,
                cwd=source_dir,
                env=env,
                shell=use_shell,
                capture_output=False
            )

            if result.returncode != 0:
                print("ERROR: rccl-tests build failed")
                return False

            print("rccl-tests build completed successfully")
            return True

        except Exception as e:
            print(f"ERROR: rccl-tests build failed with exception: {e}")
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
            expanded_path = expand_env_vars(binary)
            resolved = os.path.expanduser(expanded_path)
            if self.args.verbose:
                print(f"  Binary resolved via absolute path: {resolved}")
            return resolved

        # Strategy 2: Expand environment variables in binary path
        if '$' in binary or '~' in binary:
            expanded_path = expand_env_vars(binary)
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
            test_binary_dir = expand_env_vars(test_binary_dir)
            test_binary_dir = os.path.expanduser(test_binary_dir)
            resolved = os.path.join(test_binary_dir, binary)
            if self.args.verbose:
                print(f"  Binary resolved via test config test_binary_dir: {resolved}")
            return resolved

        # Strategy 4: Check for test_binary_dir in paths config
        if "test_binary_dir" in self.paths:
            test_binary_dir = self.paths["test_binary_dir"]
            # Expand environment variables in test_binary_dir
            test_binary_dir = expand_env_vars(test_binary_dir)
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

    def _terminate_process_group(self, proc):
        """Tear down the entire process group of ``proc`` (SIGTERM, then SIGKILL).

        Tests are launched with ``start_new_session=True`` so the shell, mpirun,
        orted and all spawned ranks share one process group (pgid == proc.pid).
        Signalling the group -- rather than just ``proc`` -- guarantees a timed-out
        or interrupted MPI job does not leave orphaned ranks holding the GPUs.
        """
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            return  # already gone

        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, OSError):
                return  # group already gone
            try:
                # Give the group a short grace period to exit on SIGTERM before
                # escalating to SIGKILL.
                proc.wait(timeout=10)
                return
            except subprocess.TimeoutExpired:
                continue
        # Reap to avoid a zombie even if it ignored SIGKILL (should not happen).
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    @staticmethod
    def _normalize_mpi_args(value):
        """
        Normalize an "mpi_args" config value into a single argument string.

        Accepts a string (returned trimmed) or a list of strings (joined with
        spaces). Returns "" for None/empty values so callers can skip it.
        """
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return " ".join(str(item).strip() for item in value if str(item).strip())
        return str(value).strip()

    @staticmethod
    def _resolve_gpu_count(value, detected):
        """
        Resolve a num_gpus value into a concrete integer.

        "auto" (or None) resolves to the detected GPU count, or 8 when detection
        failed (detected == 0), preserving the historical default. Numeric values
        are returned as-is.
        """
        if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
            return detected if detected else 8
        return int(value)

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

        # num_gpus / num_ranks support the literal "auto" (and num_gpus defaults to
        # "auto" when omitted): "auto" resolves to the detected GPUs per node so
        # tests adapt to whatever hardware they run on. Bad/overridden values that
        # bypass schema validation fail this single test rather than aborting the
        # whole run.
        detected_gpus = self.gpus_per_node
        try:
            num_nodes = int(test_config.get("num_nodes", 1))
            num_gpus = self._resolve_gpu_count(test_config.get("num_gpus", "auto"), detected_gpus)
            raw_ranks = test_config.get("num_ranks", 1)
            if isinstance(raw_ranks, str) and raw_ranks.strip().lower() == "auto":
                # All GPUs across all nodes
                num_ranks = num_gpus * num_nodes
            else:
                num_ranks = int(raw_ranks)
        except (ValueError, TypeError) as e:
            msg = f"Invalid num_nodes/num_gpus/num_ranks for test '{test_name}': {e}"
            print(f"ERROR: {msg}")
            return {
                "name": test_name,
                "result": TestResult.RESULT_FAILED.value,
                "duration": 0,
                "error": msg,
            }
        timeout = test_config.get("timeout", 0)
        env_vars = test_config.get("env_variables", {})

        # Support custom command arguments for non-gtest or specialized tests.
        # The suite-level "command_args" is a shared base; a test's own
        # "command_args" is appended to it, so a test can add flags
        # (e.g. "-R 1 -G 2") without restating the shared base args.
        base_args = suite_config.get("command_args", "")
        test_args = test_config.get("command_args", "")
        custom_args = f"{base_args} {test_args}".strip()

        # rccl-tests dtype flag (-d <type>) for result emission; None for gtests.
        _dtype_match = re.search(r'(?:^|\s)-d\s+(\S+)', custom_args)
        perf_dtype = _dtype_match.group(1) if _dtype_match else None

        # Execution mode: MPI (>1 rank -> mpirun) vs single-process multithreaded
        # (-t N) vs single. Recorded per test so results are attributable.
        _t_match = re.search(r'(?:^|\s)-t\s+(\d+)', custom_args)
        perf_nthreads = int(_t_match.group(1)) if _t_match else 1
        exec_mode = "mpi" if num_ranks > 1 else ("threaded" if perf_nthreads > 1 else "single")

        # Merge environment variables
        merged_env = {
            **self.global_env,
            **suite_config.get("env_variables", {}),
            **env_vars
        }

        print(f"\n{'='*80}")
        print(f"Test: {test_name}")
        print(f"{'='*80}")

        # Pytest-harness suites use a dedicated runner; the gtest/MPI path below
        # is left untouched.
        if test_config.get("is_pytest", False):
            return self._run_pytest_test(test_config, merged_env)

        if self.args.verbose:
            if description:
                print(f"  Description: {description}")
            print(f"  Type:    {'gtest' if is_gtest else 'non-gtest'}")
            print(f"  Binary:  {os.path.expanduser(expand_env_vars(str(binary)))}")
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

        # GPU-count skip: when the per-node GPU count is known, a test that needs
        # more GPUs than the node provides is SKIPPED (not failed) so fixed-size
        # tests can live in a shared config and self-skip on smaller nodes. For
        # single-node tests the requirement is num_ranks; for multi-node it is the
        # per-node rank count (num_gpus). Detection failure (detected_gpus == 0)
        # disables this check. See README "Automatic skipping on insufficient GPUs".
        if detected_gpus > 0:
            gpus_needed = num_ranks if num_nodes <= 1 else num_gpus
            if gpus_needed > detected_gpus:
                msg = (
                    f"SKIP: test needs {gpus_needed} GPU(s)/node, "
                    f"node has {detected_gpus}"
                )
                print(msg)
                return {
                    "name": test_name,
                    "result": TestResult.RESULT_SKIPPED.value,
                    "duration": 0,
                    "error": msg,
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

        # Build LD_LIBRARY_PATH.  Priority order (highest → lowest):
        #   1. build_dir          – always first so the test-built librccl.so wins
        #   2. test JSON value    – per-test custom lib dir (e.g. a backport HIP stack)
        #   3. caller environment – LD_LIBRARY_PATH already set in the shell lets
        #                           users point at a custom libamdhip64.so.7 without
        #                           any special variable:
        #                             LD_LIBRARY_PATH=/path/to/hip python3 test_runner.py ...
        #   4. {rocm_path}/lib    – default ROCm/HIP fallback
        #   5. mpi_path/lib       – MPI runtime
        #
        # LD_LIBRARY_PATH from the JSON config is consumed here (not in the
        # merged_env loop below) so that build_dir always stays first.
        mpi_path  = self.paths.get("mpi_path", "")
        rocm_path = self._rocm_root()

        ld_library_path_parts = [self.build_dir]
        test_ld = merged_env.get('LD_LIBRARY_PATH')
        if test_ld:
            ld_library_path_parts.append(str(test_ld))
        if env.get('LD_LIBRARY_PATH'):
            ld_library_path_parts.append(env['LD_LIBRARY_PATH'])
        ld_library_path_parts.append(os.path.join(rocm_path, "lib"))
        if mpi_path:
            ld_library_path_parts.append(os.path.join(mpi_path, "lib"))
        env['LD_LIBRARY_PATH'] = ":".join(ld_library_path_parts)

        # Set LLVM_PROFILE_FILE for code coverage (prevents default.profraw
        # collision). Only do this when coverage reporting is requested -- with
        # an instrumented binary, writing per-PID profraw files on every process
        # exit is a significant overhead when many short-lived test processes
        # are spawned.
        #
        # %p  — unique per child PID (ProcessIsolatedTestRunner re-execs each
        #        test as a separate process, so each gets its own file).
        # %m  — binary/module signature (keeps test-binary and librccl.so
        #        profiles in separate files since each has its own runtime).
        if self.args.coverage_report:
            env['LLVM_PROFILE_FILE'] = "rccl_tests_%p_%m.profraw"

        # Add test-specific env vars.  LD_LIBRARY_PATH is already merged above.
        for key, value in merged_env.items():
            if key != 'LD_LIBRARY_PATH':
                env[key] = str(value)

        # Executable to invoke. Use the fully-resolved absolute path (not
        # "./<binary>") so out-of-tree binaries work: gtest binaries live in
        # cwd=<build_dir>/test, but rccl-tests perf binaries live under the
        # rccl-tests build dir. test_binary_path was resolved above and verified
        # to exist; shlex.quote handles any spaces.
        exe = shlex.quote(test_binary_path)

        # Build command based on test type
        if num_ranks == 1:
            # Non-MPI test - prepend environment variables to the command.
            # LD_LIBRARY_PATH is already merged with correct priority order above,
            # so skip it in the merged_env loop and use the final env value instead.
            env_prefix = ""
            for key, value in merged_env.items():
                if key != 'LD_LIBRARY_PATH':
                    env_prefix += f"{key}={value} "
            env_prefix += f"LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']} "

            if is_gtest:
                # GTest-based test - use --gtest_filter syntax
                if test_filter == "ALL" or test_filter == "*":
                    cmd = f"{env_prefix}{exe}"
                else:
                    cmd = f"{env_prefix}{exe} --gtest_filter={test_filter}"

                # Add custom arguments if provided
                if custom_args:
                    cmd += f" {custom_args}"
            else:
                # Non-gtest test (perf, custom, etc.) - run binary with args
                cmd = f"{env_prefix}{exe}"
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

            # Resolve the mpirun/MCA arguments. Each "mpi_args" value may be a
            # string or a list of strings (lists are joined with spaces).
            #
            # The BASE set replaces the built-in defaults, with this priority:
            #   1. top-level "mpi_args" dict entry for the active --system
            #   2. top-level "mpi_args" string/list (applies to all systems)
            #   3. built-in default_args
            #
            # Suite-level and test-level "mpi_args" are then APPENDED on top of
            # the base, so they add flags regardless of --system.
            default_mca = self.mpi_config["default_args"]
            system = getattr(self.args, 'system', '') or ''
            top_level_config = self.config_processor.config.get("mpi_args", "")

            if isinstance(top_level_config, dict):
                base_args = self._normalize_mpi_args(top_level_config.get(system)) if system else ""
            else:
                base_args = self._normalize_mpi_args(top_level_config)

            if not base_args:
                base_args = default_mca

            suite_args = self._normalize_mpi_args(suite_config.get("mpi_args"))
            test_args = self._normalize_mpi_args(test_config.get("mpi_args"))

            # User-supplied extra mpi args, appended last so they override/extend
            # whatever the config provides. Both the --mpi-args CLI flag and the
            # RCCL_TEST_MPI_ARGS env var are honored (CLI first, then env).
            cli_extra_args = self._normalize_mpi_args(getattr(self.args, 'mpi_args', ''))
            env_extra_args = self._normalize_mpi_args(os.environ.get('RCCL_TEST_MPI_ARGS', ''))

            mca_params = " ".join(
                p for p in (base_args, suite_args, test_args, cli_extra_args, env_extra_args) if p
            )

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
            if self.args.coverage_report:
                mpi_args += " " + env_fmt.format(
                    key="LLVM_PROFILE_FILE", value="rccl_tests_%p_%m.profraw"
                )

            # Forward LD_PRELOAD so UCX core libraries are preloaded with
            # global visibility on remote ranks (required for UCX PML)
            ld_preload = os.environ.get("LD_PRELOAD", "")
            if ld_preload:
                mpi_args += " " + env_fmt.format(key="LD_PRELOAD", value=ld_preload)

            # Build test command based on type
            if is_gtest:
                # GTest-based test - use --gtest_filter syntax
                if test_filter == "ALL" or test_filter == "*":
                    cmd = f"{mpi_cmd} {mpi_args} {exe}"
                else:
                    cmd = f"{mpi_cmd} {mpi_args} {exe} --gtest_filter={test_filter}"

                if custom_args:
                    cmd += f" {custom_args}"
            else:
                # Non-gtest test (perf, custom, etc.) - run binary with args
                cmd = f"{mpi_cmd} {mpi_args} {exe}"
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
        #
        # Launch the test in its own session (start_new_session=True) so the
        # shell AND every descendant -- mpirun, orted, and the spawned ranks /
        # perf binaries -- share a single process group we can signal as a unit.
        # subprocess.run(timeout=...) only SIGKILLs the immediate /bin/sh child,
        # leaving mpirun and all of its ranks running (and holding the GPUs),
        # which is exactly the orphaned-process behaviour seen on timeout.
        # When result emission is enabled, tee output to a per-test log so perf
        # (busbw/algbw) numbers can be parsed afterwards, while still streaming to
        # the console. ``set -o pipefail`` keeps the pipeline's exit status equal to
        # the test's (not tee's). Default behaviour (inherited stdout, no capture)
        # is unchanged when emission is off.
        emit_log_path = None
        start_time = time.time()
        if self.emit_enabled:
            emit_log_path = self._emit_log_path(test_name)
            wrapped = f"set -o pipefail; ({cmd}) 2>&1 | tee {shlex.quote(emit_log_path)}"
            proc = subprocess.Popen(
                ["bash", "-c", wrapped],
                cwd=os.path.join(self.build_dir, "test"),
                env=env,
                start_new_session=True,
            )
        else:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=os.path.join(self.build_dir, "test"),
                env=env,
                start_new_session=True,
            )
        try:
            try:
                returncode = proc.wait(timeout=timeout if timeout > 0 else None)
            except subprocess.TimeoutExpired:
                duration = time.time() - start_time
                print(f"\n  Result: {TestResult.RESULT_TIMEOUT.value} after {timeout} seconds")
                print("  Killing process group (mpirun and all ranks)...")
                self._terminate_process_group(proc)
                return {
                    "name": test_name,
                    "result": TestResult.RESULT_TIMEOUT.value,
                    "duration": duration,
                    "error": f"Test timed out after {timeout} seconds",
                    "binary": binary, "is_gtest": is_gtest, "dtype": perf_dtype,
                    "num_nodes": num_nodes, "num_gpus": num_gpus,
                    "num_ranks": num_ranks, "log_file": emit_log_path,
                    "exec_mode": exec_mode, "nthreads": perf_nthreads,
                }
            except KeyboardInterrupt:
                # Make sure Ctrl-C tears down the whole MPI job, not just the shell.
                print("\n  Interrupted -- killing process group (mpirun and all ranks)...")
                self._terminate_process_group(proc)
                raise
            except Exception as e:
                duration = time.time() - start_time
                self._terminate_process_group(proc)
                print(f"\n  ERROR: {e}")
                return {
                    "name": test_name,
                    "result": TestResult.RESULT_FAILED.value,
                    "duration": duration,
                    "error": str(e)
                }

            duration = time.time() - start_time

            if is_gtest:
                rc = returncode if returncode is not None else -1
                test_result = infer_gtest_result_from_json_file(gtest_json_path or "", rc)
            else:
                if returncode == ExitCode.EXIT_SUCCESS:
                    test_result = TestResult.RESULT_PASSED.value
                elif returncode == ExitCode.EXIT_TIMEOUT:
                    test_result = TestResult.RESULT_TIMEOUT.value
                else:
                    test_result = TestResult.RESULT_FAILED.value

            print(f"\n  Result: {test_result} ({duration:.3f} seconds)")

            return {
                "name": test_name,
                "result": test_result,
                "duration": duration,
                "exit_code": int(returncode) if returncode is not None else -1,
                "binary": binary, "is_gtest": is_gtest, "dtype": perf_dtype,
                "num_nodes": num_nodes, "num_gpus": num_gpus,
                "num_ranks": num_ranks, "log_file": emit_log_path,
                "exec_mode": exec_mode, "nthreads": perf_nthreads,
            }
        finally:
            if gtest_json_path:
                try:
                    os.unlink(gtest_json_path)
                except OSError:
                    pass

    def _setup_pytest_venv(self, test_dir, test_config):
        """Create/reuse a venv and install ONLY the given requirements file into
        it (opt-in via setup_venv); nothing is installed outside the venv. Keys:
        venv_dir (default <test_dir>/venv), requirements (default
        <test_dir>/requirements.txt), python_bin (base interpreter).
        Returns (venv_python, error) where error is "" on success."""
        venv_dir = test_config.get("venv_dir", "") or os.path.join(test_dir, "venv")
        venv_dir = os.path.expanduser(os.path.expandvars(venv_dir))
        if not os.path.isabs(venv_dir):
            venv_dir = os.path.join(test_dir, venv_dir)
        venv_py = os.path.join(venv_dir, "bin", "python")
        base_python = test_config.get("python_bin", "") or "python3"

        req = test_config.get("requirements", "") or os.path.join(test_dir, "requirements.txt")
        req = os.path.expanduser(os.path.expandvars(req))
        if not os.path.isabs(req):
            req = os.path.join(test_dir, req)
        if not os.path.isfile(req):
            return venv_py, f"requirements file not found: {req}"

        def _run(argv, label):
            print(f"  [setup_venv] {label}: {' '.join(argv)}")
            r = subprocess.run(argv, cwd=test_dir, capture_output=True, text=True)
            if r.returncode != 0:
                return f"{label} failed (rc={r.returncode}): {(r.stderr or r.stdout).strip()[:500]}"
            return ""

        if not os.path.isfile(venv_py):
            err = _run([base_python, "-m", "venv", venv_dir], "create venv")
            if err:
                return venv_py, err

        # Install ONLY the requirements file, into the venv (no extra packages).
        err = _run([venv_py, "-m", "pip", "install", "-r", req], "install requirements")
        return venv_py, err

    def _run_pytest_test(self, test_config, merged_env):
        """Run a pytest-harness test in its source dir (no mpirun); result is
        derived from a JUnit XML report. Keys: test_dir (required), test_filter
        (leading '-' = raw pytest args, else -k expr; '*'/'ALL'/empty = all),
        python_bin (default <test_dir>/venv else python3), timeout."""
        test_name = test_config.get("name")
        timeout = test_config.get("timeout", 0)

        # Resolve harness dir (absolute, or relative to workdir).
        test_dir = test_config.get("test_dir", "")
        if not test_dir:
            print(f"ERROR: pytest test '{test_name}' is missing required 'test_dir'")
            return {
                "name": test_name,
                "result": TestResult.RESULT_FAILED.value,
                "duration": 0,
                "error": "pytest test missing 'test_dir'",
            }
        test_dir = os.path.expanduser(os.path.expandvars(test_dir))
        if not os.path.isabs(test_dir):
            workdir = self.paths.get("workdir", os.getcwd())
            test_dir = os.path.join(workdir, test_dir)

        if not os.path.isdir(test_dir):
            print(f"SKIP: pytest directory not found: {test_dir}")
            return {
                "name": test_name,
                "result": TestResult.RESULT_SKIPPED.value,
                "duration": 0,
                "error": f"pytest directory not found: {test_dir}",
            }

        # Interpreter selection. With setup_venv, create/reuse a managed venv and
        # install requirements; otherwise: explicit override > local venv > python3.
        if test_config.get("setup_venv", False):
            python_bin, setup_err = self._setup_pytest_venv(test_dir, test_config)
            if setup_err:
                print(f"\n  Result: {TestResult.RESULT_FAILED.value} ({setup_err})")
                return {
                    "name": test_name,
                    "result": TestResult.RESULT_FAILED.value,
                    "duration": 0,
                    "error": setup_err,
                }
        else:
            python_bin = test_config.get("python_bin", "")
            if not python_bin:
                venv_py = os.path.join(test_dir, "venv", "bin", "python")
                python_bin = venv_py if os.path.isfile(venv_py) else "python3"

        # test_filter -> pytest selection args (raw args if leading '-', else -k expr).
        test_filter = test_config.get("test_filter", "*")
        select_args = []
        if test_filter and test_filter not in ("*", "ALL"):
            if test_filter.lstrip().startswith("-"):
                select_args = shlex.split(test_filter)
            else:
                select_args = ["-k", test_filter]

        # Env: build_dir first on LD_LIBRARY_PATH, then a per-test LD_LIBRARY_PATH
        # from the JSON env (mirrors the gtest/MPI path), then rocm/mpi libs.
        # RCCL_BUILD defaults to build_dir; remaining config env is applied as-is.
        env = os.environ.copy()
        rocm_path = self._rocm_root()
        mpi_path = self.paths.get("mpi_path", "")
        test_ld = merged_env.get("LD_LIBRARY_PATH")
        ld_parts = [self.build_dir]
        if test_ld:
            ld_parts.append(str(test_ld))
        if env.get("LD_LIBRARY_PATH"):
            ld_parts.append(env["LD_LIBRARY_PATH"])
        ld_parts.append(os.path.join(rocm_path, "lib"))
        if mpi_path:
            ld_parts.append(os.path.join(mpi_path, "lib"))
        env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
        env.setdefault("RCCL_BUILD", self.build_dir)
        if self.args.coverage_report:
            env["LLVM_PROFILE_FILE"] = "rccl_tests_%p_%m.profraw"
        for key, value in merged_env.items():
            if key != "LD_LIBRARY_PATH":
                env[key] = str(value)

        # Keep the JUnit report as an artifact under the runner's log dir (it is
        # the only structured record of the pytest run), one file per test.
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", test_name or "pytest")
        junit_path = os.path.join(self.log_dir, f"pytest_{safe_name}.xml")
        cmd = [
            python_bin, "-m", "pytest", "-v", "-p", "no:cacheprovider",
            f"--junitxml={junit_path}",
        ] + select_args

        if self.args.verbose:
            print(f"  Description: {test_config.get('description', '')}")
            print(f"  Type:    pytest")
            print(f"  Dir:     {test_dir}")
            print(f"  Filter:  {test_filter}")
            print(f"  Timeout: {timeout if timeout > 0 else 'unlimited'}")
            print(f"\n  Command: {' '.join(cmd)}")
            print(f"  Working directory: {test_dir}")
            print(f"  LD_LIBRARY_PATH: {env.get('LD_LIBRARY_PATH', '')}\n")

        start_time = time.time()
        run_kwargs = {
            "cwd": test_dir,
            "env": env,
            "capture_output": False,
        }
        if timeout > 0:
            run_kwargs["timeout"] = timeout

        try:
            result = subprocess.run(cmd, **run_kwargs)
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
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
                "error": str(e),
            }

        duration = time.time() - start_time
        rc = result.returncode if result.returncode is not None else -1
        test_result = infer_pytest_result_from_junit(junit_path, rc)
        print(f"\n  Result: {test_result} ({duration:.3f} seconds)")
        if self.args.verbose:
            print(f"  JUnit report: {junit_path}")
        return {
            "name": test_name,
            "result": test_result,
            "duration": duration,
            "exit_code": int(rc),
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

            # Filter by test name if specified (gtest-style glob; see glob_filter_matches).
            test_name = test.get("name")
            if self.args.test_name and not glob_filter_matches(test_name, self.args.test_name):
                skipped_count += 1
                continue

            # Skip MPI tests when --skip-mpi-check is set ("auto" implies multi-rank)
            test_ranks = test.get("num_ranks", suite_config.get("num_ranks", 1))
            is_auto_ranks = isinstance(test_ranks, str) and test_ranks.strip().lower() == "auto"
            is_mpi_test = is_auto_ranks or (not isinstance(test_ranks, str) and test_ranks > 1)
            if self.args.skip_mpi_check and is_mpi_test:
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

            if self.emit_enabled:
                record = dict(result)
                record["suite"] = suite_name
                record["test_name"] = test_name
                self.test_records.append(record)

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

    def _rccl_tests_build_dir(self):
        """Resolve the rccl-tests build directory (``<source_dir>/build``).

        Mirrors the path logic in ``build_rccl_tests`` so coverage discovery and
        the build use the same location. Env vars and ``~`` are expanded with the
        bash-aware expander (``${VAR:-default}`` support). Returns None when
        rccl-tests is not configured.
        """
        cfg = self.rccl_tests_build_config
        if not cfg:
            return None
        workdir = self.paths.get("workdir", os.getcwd())
        source_dir = cfg.get("source_dir", os.path.join(workdir, "rccl-tests"))
        source_dir = os.path.expanduser(expand_env_vars(str(source_dir)))
        return os.path.join(source_dir, "build")

    # LLVM tool layout under a ROCm root, most-specific first.
    _LLVM_BIN_SUBDIRS = (("lib", "llvm", "bin"), ("llvm", "bin"), ("bin",))

    def _candidate_rocm_roots(self):
        """Ordered, de-duplicated list of ROCm roots to consider, most-trusted first.

        A `module load rocm/...` sets ROCM_PATH/HIP_PATH to a versioned tree
        (e.g. /cluster/.../rocm-7.13.0...) and puts its compiler on PATH, while the
        configured rocm_path may still be the bare ${ROCM_PATH:-/opt/rocm} default.
        """
        roots = []

        def add(root):
            if root:
                root = os.path.normpath(root)
                if root not in roots:
                    roots.append(root)

        # 1. Explicitly configured rocm_path (paths.rocm_path / --rocm_home).
        add(self.paths.get("rocm_path"))
        # 2. The loaded module's environment.
        for env_var in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
            add(os.environ.get(env_var))
        # 3. The root inferred from a ROCm compiler/tool on PATH. The module may add
        #    <root>/bin or <root>/lib/llvm/bin to PATH; map either back to <root>.
        for binname in ("amdclang++", "hipcc", "rocminfo", "rocm_agent_enumerator"):
            binpath = shutil.which(binname)
            if not binpath:
                continue
            bindir = os.path.dirname(os.path.realpath(binpath))
            add(os.path.dirname(bindir))  # <root>/bin -> <root>
            add(os.path.dirname(os.path.dirname(os.path.dirname(bindir))))  # <root>/lib/llvm/bin -> <root>
        # 4. Last-resort default.
        add("/opt/rocm")
        return roots

    def _rocm_root(self):
        """Single source of truth for the ROCm toolchain root, shared by build, test,
        and coverage so all three phases use an identical toolchain.

        The build compiles RCCL with <root>/.../amdclang++ and coverage MUST read the
        profraw files with the matching <root>/lib/llvm/bin/llvm-profdata, so the same
        root has to drive every phase. A root is preferred only if it actually contains
        the LLVM toolchain (llvm-profdata present); this lets a stale `/opt/rocm`
        default yield to the loaded module. Cached after first resolution.
        """
        cached = getattr(self, "_rocm_root_cache", None)
        if cached:
            return cached

        candidates = self._candidate_rocm_roots()

        def has_llvm_toolchain(root):
            return any(os.path.isfile(os.path.join(root, *sub, "llvm-profdata"))
                       for sub in self._LLVM_BIN_SUBDIRS)

        # Prefer the first root with a complete LLVM toolchain (matches the compiler
        # and provides the coverage tools); otherwise the first existing directory.
        chosen = next((r for r in candidates if has_llvm_toolchain(r)), None)
        if chosen is None:
            chosen = next((r for r in candidates if os.path.isdir(r)), None)
        if chosen is None:
            chosen = self.paths.get("rocm_path") or "/opt/rocm"

        self._rocm_root_cache = chosen
        if getattr(self.args, "verbose", False):
            print(f"Resolved ROCm toolchain root: {chosen}")
        return chosen

    def _resolve_llvm_tool(self, name):
        """Resolve an LLVM tool (e.g. llvm-profdata, llvm-cov) from the SAME ROCm root
        used to build and run, so the profile format matches the compiler.

        Order:
          1. Explicit config override in self.paths (key == tool name).
          2. The resolved ROCm root (_rocm_root) — preferred over a bare PATH hit so a
             different LLVM on PATH cannot shadow the build's toolchain.
          3. PATH lookup via shutil.which as a final fallback.
        Raises FileNotFoundError with a clear, searched-path message if not found.
        """
        override = self.paths.get(name)
        if override and os.path.isfile(override):
            return override

        searched = []
        rocm_root = self._rocm_root()
        for sub in self._LLVM_BIN_SUBDIRS:
            candidate = os.path.join(rocm_root, *sub, name)
            searched.append(candidate)
            if os.path.isfile(candidate):
                return candidate

        found = shutil.which(name)
        if found:
            return found

        raise FileNotFoundError(
            f"{name} not found under the resolved ROCm root ({rocm_root}) or on PATH. "
            f"Searched: {', '.join(searched)}. "
            f"Load a ROCm module or set rocm_path/--rocm_home."
        )

    def generate_coverage_report(self):
        """Generate code coverage report.

        Supports the report-only workflow: when invoked with
        ``--no-build --skip-tests --coverage-report`` after a previous run that
        executed an instrumented binary, this method picks up the existing
        ``*.profraw`` files left in ``<build_dir>/test/`` and produces a fresh
        HTML/text report under the current run's workspace ``report/``.
        """
        if not self.args.coverage_report:
            return

        print(f"\n{'='*80}")
        print("GENERATING COVERAGE REPORT")
        print(f"{'='*80}")

        # Check for profraw files
        import glob
        import shutil

        # Tests run with cwd=<build_dir>/test, so RCCL profraw files are written
        # there. The rccl-tests perf binaries are instrumented separately and
        # their profraw files live under the rccl-tests build tree, so search
        # both roots. A recursive glob also picks up files written elsewhere
        # under either tree (e.g. by ad-hoc runs).
        profraw_search_roots = [self.build_dir]
        rccl_tests_build_dir = self._rccl_tests_build_dir()
        if rccl_tests_build_dir:
            profraw_search_roots.append(rccl_tests_build_dir)

        profraw_files = []
        for root in profraw_search_roots:
            profraw_files.extend(
                glob.glob(os.path.join(root, "**/*.profraw"), recursive=True)
            )
        # De-duplicate in case the roots overlap or are nested.
        profraw_files = sorted({os.path.abspath(p) for p in profraw_files})

        if not profraw_files:
            print("ERROR: No .profraw files found under the build directories:")
            for root in profraw_search_roots:
                print(f"           {root}")
            if self.args.skip_tests:
                print()
                print("--coverage-report --skip-tests was requested, so this run did")
                print("not execute any tests. Run the tests at least once with")
                print("--coverage-report (without --skip-tests) so the instrumented")
                print("binary writes .profraw files, then re-run with")
                print("--no-build --skip-tests --coverage-report to regenerate the")
                print("report from those files.")
            else:
                print()
                print("Tests ran but produced no coverage data. Confirm that the RCCL")
                print("library and test binaries were built with coverage instrumentation")
                print("(--enable-code-coverage / -DENABLE_CODE_COVERAGE=ON) and that the")
                print("test processes terminated cleanly so the runtime could flush the")
                print("profraw files.")
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

        # Resolve LLVM tools from the same ROCm root used to build/run so the
        # profraw format matches the compiler that produced it.
        rocm_path = self._rocm_root()
        try:
            llvm_profdata = self._resolve_llvm_tool("llvm-profdata")
            llvm_cov = self._resolve_llvm_tool("llvm-cov")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return

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

        # Add rccl-tests perf binaries so their host coverage mapping is attributed.
        if rccl_tests_build_dir and os.path.isdir(rccl_tests_build_dir):
            perf_binaries = sorted(glob.glob(os.path.join(rccl_tests_build_dir, "*_perf")))
            for perf_binary in perf_binaries:
                if os.path.isfile(perf_binary):
                    object_files.extend(["--object", perf_binary])
                    if self.args.verbose:
                        print(f"Found perf binary: {perf_binary}")

        if not object_files:
            print("WARNING: No object files found for coverage report")
            return

        if self.args.verbose:
            print(f"Total object files for coverage: {len(object_files) // 2}")

        # Ignore patterns for non-relevant files
        ignore_regex = (
            ".*tuner_v.*|.*profiler_v.*|.*net_v.*|.*_deps.*|ext.*|"
            ".*coll_net.*|.*nvls.*|.*nvml.*|.*nvtx.*|test/|.*gtest.*|"
            ".*gensrc.*|.*rccl-tests.*"
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

    def emit_results(self):
        """Emit structured results for the results dashboard.

        Always writes local JSON/JSONL + a .tar.gz snapshot (the durable source of
        truth). When --db-push is set, additionally pushes to PostgreSQL on a best
        effort basis -- a DB failure/timeout is logged but never fails the run.

        No-op unless --emit-results or --db-push was passed.
        """
        if not self.emit_enabled:
            return

        import socket
        import uuid
        from lib import results_emitter as re_mod
        from lib import host_metadata

        print(f"\n{'='*80}")
        print("EMITTING RESULTS")
        print(f"{'='*80}")

        stamp = datetime.datetime.now(datetime.timezone.utc)
        run_id = f"{stamp.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

        sha, branch = re_mod.git_info(os.path.dirname(os.path.abspath(__file__)))

        # num_nodes: largest node count seen across recorded tests (falls back to 1).
        node_counts = [r.get("num_nodes") for r in self.test_records
                       if isinstance(r.get("num_nodes"), int)]
        num_nodes = max(node_counts) if node_counts else 1

        sys_cfg = {}
        try:
            sys_cfg = self.config_processor.config.get("system_configurations", {}) or {}
        except AttributeError:
            sys_cfg = {}

        def _redact_sensitive_env(env):
            # Persist env for debugging, but mask values of secret-like keys so
            # tokens/passwords/DSNs never land in the tarball or DB.
            if not env:
                return None
            sensitive = re.compile(r"(?i)(pass|secret|token|api[_-]?key|access[_-]?key|credential|dsn)")
            return {k: ("***redacted***" if sensitive.search(k) else v) for k, v in env.items()}

        # Run tags (from --tag / --tags), de-duplicated in order.
        run_tags = list(getattr(self.args, "tag", []) or [])
        for _t in (getattr(self.args, "tags", "") or "").split(","):
            _t = _t.strip()
            if _t and _t not in run_tags:
                run_tags.append(_t)

        manifest = {
            "run_id": run_id,
            "created_at": stamp.isoformat(),
            "started_at": stamp.isoformat(),
            "rccl_sha": sha,
            "rccl_branch": branch,
            "rocm_version": re_mod._rocm_version(self._rocm_root()),
            "host": socket.gethostname(),
            "hosts": self.mpi_hosts or None,
            "gpu_arch": re_mod.gpu_arch(),
            "node_names": re_mod.node_names_from_mpi_hosts(self.mpi_hosts) or [socket.gethostname()],
            "num_nodes": num_nodes,
            "gpus_per_node": self.gpus_per_node or None,
            "config_name": sys_cfg.get("name"),
            "config_description": sys_cfg.get("description"),
            "label": getattr(self.args, "run_label", "") or None,
            "tags": run_tags or None,
            "env": _redact_sensitive_env(self.global_env),
        }

        # Rich host/telemetry snapshot (best-effort; scale-up fabric gated on capability).
        try:
            md = host_metadata.collect(rocm_version=re_mod._rocm_version(self._rocm_root()))

            # RCCL build type (perf configs should use a Release build).
            install_flags = self.build_config.get("install_flags", []) if isinstance(self.build_config, dict) else []
            if getattr(self, "using_custom_lib", False):
                build_type = "custom"
            elif any(f in install_flags for f in ("--debug", "--debug-fast")):
                build_type = "debug"
            else:
                build_type = "release"
            md["rccl_build_type"] = build_type
            md["mpi_impl"] = self.mpi_impl
            md.setdefault("checks", {})["release_build"] = {
                "status": "OK" if build_type == "release" else ("SKIP" if build_type == "custom" else "WARN"),
                "value": build_type + (" (perf should use release)" if build_type == "debug" else ""),
            }
            manifest["metadata"] = md
        except Exception as e:
            print(f"WARNING: host metadata collection failed: {e}")

        results_dir = (getattr(self.args, "results_dir", "") or
                       os.path.join(self.workspace_dir, "results"))

        emitter = re_mod.ResultsEmitter(manifest, results_dir)
        for record in self.test_records:
            emitter.add_test(record)

        # Attach coverage if a report was generated this run.
        cov = re_mod.parse_coverage_report(
            os.path.join(self.report_dir, "function_coverage_report.txt")
        )
        emitter.set_coverage(cov)

        summary = {
            "total": len(self.test_results),
            "passed": self.test_results.count(TestResult.RESULT_PASSED.value),
            "failed": self.test_results.count(TestResult.RESULT_FAILED.value),
            "timeout": self.test_results.count(TestResult.RESULT_TIMEOUT.value),
            "skipped": self.test_results.count(TestResult.RESULT_SKIPPED.value),
            "duration_s": sum(self.test_durations) if self.test_durations else 0,
        }
        emitter.finalize_summary(summary)

        emitter.write_local(log_dir=self.log_dir, report_dir=self.report_dir)

        if getattr(self.args, "db_push", False):
            emitter.push_postgres(timeout=getattr(self.args, "db_timeout", 10))

