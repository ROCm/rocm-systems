# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Pytest configuration and fixtures for rocprofiler-systems tests.

This module provides shared fixtures and configuration for all test modules.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from functools import lru_cache
from typing import Callable, Generator, Optional

import re
import os
import sys
import shutil

# Add the pytest directory to Python path for rocprofsys package
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from pytest import StashKey

from rocprofsys import (
    RocprofsysConfig,
    discover_build_config,
    GPUInfo,
    get_rocminfo,
    detect_gpu,
    get_offload_extractor,
    get_target_gpu_arch,
    get_xnack_support,
    TestResult,
    validate_regex,
    validate_file_regex,
    validate_perfetto_trace,
    validate_rocpd_database,
    validate_timemory_json,
    validate_causal_json,
    validate_file_exists,
    BaselineRunner,
    SamplingRunner,
    BinaryRewriteRunner,
    RuntimeInstrumentRunner,
    SysRunRunner,
    CausalRunner,
    PythonRunner,
)

# Key for storing the single test result on pytest items
_result_key: StashKey = StashKey()
# Key for tracking subtest failures (for pytest-subtests plugin compatibility when pytest < 9.0.0)
_subtest_failures_key: StashKey[list] = StashKey()
# Key to prevent duplicate output printing
_output_printed_key: StashKey[bool] = StashKey()

ROCPROFSYS_RUNNER_NAMES = [
    "baseline",
    "sampling",
    "binary_rewrite",
    "runtime_instrument",
    "sys_run",
    "python",
]

ROCPROFSYS_RUNNER_CLASSES = {
    "baseline": BaselineRunner,
    "sampling": SamplingRunner,
    "binary_rewrite": BinaryRewriteRunner,
    "runtime_instrument": RuntimeInstrumentRunner,
    "sys_run": SysRunRunner,
    "causal": CausalRunner,
    "python": PythonRunner,
}


# ============================================================================
#
# Pytest Hooks (Placed in the general order they are called)
#
# ============================================================================

# ----------------------------------------------------------------------------
# Initialization hooks
# ----------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command-line options."""
    group = parser.getgroup("rocprofsys", "rocprofiler-systems test options")
    group.addoption(
        "--show-output",
        action="store_true",
        default=False,
        help="Show runner output on test pass",
    )
    group.addoption(
        "--show-output-on-subtest-fail",
        action="store_true",
        default=False,
        help="Show runner output only when subtests fail",
    )
    group.addoption(
        "--show-config",
        action="store_true",
        default=False,
        help="Show the test configuration at the beginning of the session",
    )
    group.addoption(
        "--show-config-only",
        action="store_true",
        default=False,
        help="Show the test configuration and exit without running any tests",
    )
    group.addoption(
        "--output-dir",
        action="store",
        default=None,
        help="Set the test output directory (default: <build_dir>/rocprof-sys-pytest-output in build mode, /tmp/<user>/rocprof-sys-pytest-output in install mode)",
    )
    # @output_dir@ is replaced with the value of --output-dir (or default) in the log file path
    group.addoption(
        "--output-log",
        action="store",
        default="@output_dir@/pytest-output.txt",
        help="Write log output to the specified file (use 'none' to disable)",
    )
    group.addoption(
        "--group-by-module",
        action="store_true",
        default=False,
        help="Test outputs are grouped by module name (default off)",
    )
    group.addoption(
        "--parallelize",
        action="store_true",
        default=False,
        help="Enables certain tests to be parallelized (default off)",
    )
    group.addoption(
        "--num-processes",
        action="store",
        type=int,
        default=2,
        help="Set the number of processes to use for transpose MPI tests (default 2)",
    )
    group.addoption(
        "--monochrome",
        action="store_true",
        default=False,
        help="Runners use ROCPROFSYS_MONOCHROME=ON and pytest color output is disabled",
    )
    group.addoption(
        "--ci-mode",
        action="store_true",
        default=False,
        help="Enable CI mode (developer flag : default off)",
    )
    group.addoption(
        "--allow-disabled",
        action="store_true",
        default=False,
        help="Allow disabled subtests to run (CI/CTest integration mode developer flag : default off)",
    )
    # "--collect-only" does not capture the markers associated with the tests. We need this for CTest labels
    group.addoption(
        "--ctest-mode",
        action="store",
        default="off",
        choices=("off", "collect", "run"),
        help="CTest integration mode (developer flag): 'off' (default), 'collect', or 'run'",
    )
    group.addoption(
        "--dev",
        action="store_true",
        default=False,
        help="Enables some QOL flags (developer flag : default off)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and configure pytest"""

    configure_mode(config)

    if config.getoption("--show-config-only", default=False):
        header = _generate_rocprofsys_config_header(config)
        for line in header:
            print(line)
        pytest.exit(reason="Header generated", returncode=0)

    is_monochrome = config.getoption("--monochrome", default=False)
    if is_monochrome:
        config.option.color = "no"

    # Functional markers (use arguments or do more than just label a test)

    config.addinivalue_line(
        "markers",
        "gpu: mark test as requiring a GPU",
    )  # required for run_test to check if the target supports the current system architectures
    config.addinivalue_line(
        "markers",
        "mpi_optional(target): If MPI is available and the target supports MPI, uses MPI to run the test",
    )
    config.addinivalue_line(
        "markers",
        "run_if_gpu_category(expr): run test only if GPU category expression is true "
        "(e.g., 'apu and not instinct', 'instinct or radeon')",
    )
    config.addinivalue_line(
        "markers",
        "rocm_min_version(version): mark test as requiring minimum ROCm version",
    )
    config.addinivalue_line(
        "markers",
        "rocpd(env): mark test as using ROCpd and inject ROCpd env into given env",
    )
    config.addinivalue_line(
        "markers",
        "ci_enable: Full test will be run when in CI mode. To disable a subtest, use ci_disable(name) (CI mode only)",
    )
    config.addinivalue_line(
        "markers",
        "ci_disable(name): Use 'all' to skip entire test, or assertion name (e.g., 'assert_rocpd') to disable subtest. Overrides ci_enable (CI mode only)",
    )
    config.addinivalue_line(
        "markers",
        "mpi_implementation(implementation): mark test as requiring specific MPI implementation",
    )
    config.addinivalue_line(
        "markers",
        "python_versions: Test will be parametrized by Python version",
    )

    # See pytest_collection_modifyitems
    generic_functional_markers = [
        "ucx",
        "overflow",
        "attach",
        "mpi",
        "rocm",
        "python",
        "annotate",
        "julia",
        "xnack",
    ]

    # Non-functional informational markers

    config.addinivalue_line(
        "markers", "rocprofiler: mark test as using ROCProfiler counters"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "loops: mark test as testing loop instrumentation")

    # Can be described using generic desc below
    non_functional_markers = [
        "avail",
        "instrument",
        "baseline",
        "sampling",
        "binary_rewrite",
        "runtime_instrument",
        "sys_run",
        "decode",
        "videodecode",
        "jpegdecode",
        "rocprof_binary",
        "rocprof_config",
        "xgmi",
        "group_by_queue",
        "group_by_stream",
        "openmp",
        "openmp_target",
        "ompvv",
        "sampling_duration",
        "no_tmp_files",
        "rccl",
        "roctx",
        "time_window",
        "transpose",
        "nic",
        "network",
        "fork",
        "user_api",
        "thread_limit",
        "pthreads",
        "rewrite_caller",
        "locks",
        "caller_include",
        "causal",
        "causal_e2e",
        "papi",
        "code_coverage",
        "lulesh",
        "unit_tests",
        "hip_stream",
        "presets",
        "hpc",
        "hip",
    ]
    for label in non_functional_markers + generic_functional_markers:
        config.addinivalue_line("markers", f"{label}: label test as {label}")

    # Ignore unknown marker warnings when plugins are not installed
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Unknown pytest.mark.xdist_group:pytest.PytestUnknownMarkWarning",
    )

    # Check if xdist is being used without loadgroup distribution
    try:
        numprocesses = config.getoption("numprocesses", default=None)
        dist_mode = config.getoption("dist", default=None)
        if numprocesses and numprocesses != 0 and dist_mode != "loadgroup":
            pytest.exit(
                f"Running with xdist (-n {numprocesses}) but --dist={dist_mode}. "
                "For proper test grouping with @pytest.mark.xdist_group, use --dist=loadgroup",
                returncode=1,
            )
    except (ValueError, AttributeError):
        pass  # xdist not installed or option not available

    # Save flags to pytest
    pytest._show_output_flag = config.getoption("--show-output", default=False)
    pytest._show_output_on_subtest_fail_flag = config.getoption(
        "--show-output-on-subtest-fail", default=False
    )

    # Store config reference for hooks that need terminal reporter access
    pytest._config_ref = config


# ----------------------------------------------------------------------------
# Session start hooks
# ----------------------------------------------------------------------------


def pytest_sessionstart(session):
    """Set up terminal output redirection after plugins are loaded."""
    config = session.config

    try:
        rocprof_config = get_rocprof_config()
    except Exception as e:
        pytest.exit(f"{e}")

    log_file = config.getoption("--output-log", default="@output_dir@/pytest-output.txt")

    # With pytest-xdist, we require that only the master writes to the log file.
    if log_file.lower() == "none" or hasattr(config, "workerinput"):
        config._output_log_path = None
        config._log_file_handle = None
    else:
        log_file = log_file.replace("@output_dir@", str(rocprof_config.test_output_dir))
        log_path = Path(log_file)
        config._output_log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        config._log_file_handle = open(log_path, "w")

        terminal = config.pluginmanager.get_plugin("terminalreporter")
        if terminal:
            tw = terminal._tw
            file_handle = config._log_file_handle

            original_write = tw.write

            def redirect_to_file(s, **kwargs):
                original_write(s, **kwargs)
                file_handle.write(str(s))
                file_handle.flush()

            tw.write = redirect_to_file


def pytest_report_header(config) -> list[str]:
    if not config.getoption("--show-config", default=False):
        return []
    return _generate_rocprofsys_config_header(config)


# ----------------------------------------------------------------------------
# Collection hooks
# ----------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests based on markers."""
    # ----------------------------------------------------------------------------
    # Generate an instance of the test for each Python version
    # Only if the python_versions marker is present
    marker = metafunc.definition.get_closest_marker("python_versions")
    if marker is not None:
        rocprof_config = get_rocprof_config()
        versions = rocprof_config.python_versions or []
        if versions:
            metafunc.parametrize("python_version", versions)
        else:
            # Skip if no Python versions available
            metafunc.parametrize(
                "python_version",
                [pytest.param(None, marks=pytest.mark.skip("No Python versions found"))],
            )
    # ----------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items) -> None:
    """Skip tests based on markers and available resources."""

    selected_tests = []
    deselected_tests = []

    try:
        rocprof_config = get_rocprof_config()
    except Exception as e:
        pytest.exit(f"{e}")
    gpu_info = get_gpu_info()

    gpu_category_eval_context = {
        "instinct": "instinct" in gpu_info.categories,
        "radeon": "radeon" in gpu_info.categories,
        "apu": "apu" in gpu_info.categories,
    }
    skip_gpu = pytest.mark.skip(reason="No valid GPU available")
    skip_ucx = pytest.mark.skip(reason="UCX not available")
    skip_mpi = pytest.mark.skip(reason="MPI not available")
    skip_overflow = pytest.mark.skip(
        reason=f"Requires either perf_event_paranoid <= 3, CAP_SYS_ADMIN, or CAP_PERFMON to be available"
    )
    skip_attach = pytest.mark.skip(
        reason=f"Requires ptrace_scope to be 0. "
        "Run 'echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope' to enable attaching to process"
    )
    skip_rocm = pytest.mark.skip(reason="ROCm not available")
    skip_python = pytest.mark.skip(reason="Python not available")
    skip_julia = pytest.mark.skip(reason="Julia not available")
    skip_xnack = pytest.mark.skip(reason="XNACK not supported")

    # Conditions
    overflow_available = (
        rocprof_config.capabilities.perf_event_paranoid <= 3
        or rocprof_config.capabilities.cap_sys_admin
        or rocprof_config.capabilities.cap_perfmon
    )
    annotate_available = (
        rocprof_config.capabilities.papi_availability and overflow_available
    )
    attach_available = rocprof_config.capabilities.ptrace_scope == 0
    mpi_available = rocprof_config.mpiexec is not None and not config.getoption(
        "--ci-mode", default=False
    )
    python_available = (
        rocprof_config.python_versions is not None
        and os.environ.get("ROCPROFSYS_USE_PYTHON", "ON").upper() == "ON"
    )
    xnack_available = (
        get_xnack_support(rocprof_config.rocm_path) if rocprof_config.rocm_path else False
    )

    for item in items:
        _standardize_test_name(item)
        # ----------------------------------------------------------------------------
        # Handle <name>_optional markers
        # If <name>_optional passes, then <name> marker is added
        if "mpi_optional" in item.keywords and mpi_available:
            target = item.get_closest_marker("mpi_optional").args[0]
            try:
                target_path = rocprof_config.get_target_executable(target)
                if rocprof_config.capabilities.target_support_mpi(target_path):
                    item.add_marker(pytest.mark.mpi)
            except FileNotFoundError:
                pass
        # ----------------------------------------------------------------------------
        # Add marker dependencies
        add_marker_if(item, "papi", cond=annotate_available, req_mark="annotate")
        add_marker_if(item, "mpi", req_mark="mpi_implementation")
        add_marker_if(item, "python", req_mark="python_versions")
        # ----------------------------------------------------------------------------
        # Add corresponding runner type markers based on parametrized values ("mode")
        # Ex: If "sampling" runner is used, then sampling marker is added
        detected_runners: set[str] = set()
        if hasattr(item, "callspec") and item.callspec:
            params = item.callspec.params
            for param_name in ["runner", "mode", "instrumentation_mode"]:
                if param_name in params:
                    value = str(params[param_name])
                    if value in ROCPROFSYS_RUNNER_NAMES:
                        detected_runners.add(value)
        for runner in detected_runners:
            marker_name = runner.replace("-", "_")
            item.add_marker(getattr(pytest.mark, marker_name))
        # ----------------------------------------------------------------------------
        # Marker checks
        if "gpu" in item.keywords and not gpu_info.available:
            item.add_marker(skip_gpu)
        if "ucx" in item.keywords and not rocprof_config.capabilities.ucx_availability:
            item.add_marker(skip_ucx)
        if "mpi" in item.keywords and not mpi_available:
            item.add_marker(skip_mpi)
        if "mpi_implementation" in item.keywords:
            req_impl = item.get_closest_marker("mpi_implementation").args[0]
            if req_impl != rocprof_config.capabilities.mpi_implementation:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"Requires {req_impl}, but {rocprof_config.capabilities.mpi_implementation} found"
                    )
                )
        if "overflow" in item.keywords and not overflow_available:
            item.add_marker(skip_overflow)
        if "attach" in item.keywords and not attach_available:
            item.add_marker(skip_attach)
        if "rocm" in item.keywords and not rocprof_config.rocm_path:
            item.add_marker(skip_rocm)
        if "python" in item.keywords and not python_available:
            item.add_marker(skip_python)
        if "julia" in item.keywords and not rocprof_config.julia:
            item.add_marker(skip_julia)
        if "xnack" in item.keywords and not xnack_available:
            item.add_marker(skip_xnack)
        if "rocm_min_version" in item.keywords:
            req_version = item.get_closest_marker("rocm_min_version").args[0]
            system_version = rocprof_config.rocm_version
            if system_version is None:
                item.add_marker(pytest.mark.skip(reason="ROCm version not found"))
            else:
                # Parse min_version and compare
                min_parts = req_version.split(".")
                min_tuple = tuple(int(p) for p in (min_parts + ["0", "0"])[:3])
                if system_version < min_tuple:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"ROCm {'.'.join(map(str, system_version))} < required {req_version}"
                        )
                    )
        if "run_if_gpu_category" in item.keywords:
            if not gpu_info.available:
                item.add_marker(skip_gpu)
            expr = item.get_closest_marker("run_if_gpu_category").args[0]
            try:
                result = eval(expr, {"__builtins__": {}}, gpu_category_eval_context)
                if not result:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"GPU category condition '{expr}' not met, "
                            f"GPU has categories {gpu_info.categories}"
                        )
                    )
            except Exception as e:
                pytest.exit(
                    f"Invalid run_if_gpu_category marker expression: {e}", returncode=1
                )
        # ----------------------------------------------------------------------------
        # Deselect tests for CI mode (TheRock)
        # Only tests explicitly marked with @pytest.mark.ci_enable are selected.
        # Note that ci_disable("all") overrides ci_enable.
        if config.getoption("--ci-mode", default=False) and not config.getoption(
            "--allow-disabled", default=False
        ):
            disable_marker = item.get_closest_marker("ci_disable")
            ci_disabled = disable_marker and "all" in disable_marker.args
            if item.get_closest_marker("ci_enable") and not ci_disabled:
                selected_tests.append(item)
            else:
                deselected_tests.append(item)

    # Apply deselection
    if deselected_tests:
        config.hook.pytest_deselected(items=deselected_tests)
        items[:] = selected_tests


# ----------------------------------------------------------------------------
# Test execution hooks
# ----------------------------------------------------------------------------


@pytest.hookimpl(hookwrapper=True)  # Allows yield
def pytest_runtest_makereport(item, call):
    """Build runner output and attach to report."""
    outcome = yield
    rep = outcome.get_result()

    # Relevant flags
    show_output_flag = getattr(pytest, "_show_output_flag", False)
    show_on_subfail_flag = getattr(pytest, "_show_output_on_subtest_fail_flag", False)

    has_subtest_failures = len(item.stash.get(_subtest_failures_key, [])) > 0
    show_runner_output = (show_output_flag and not rep.failed) or (
        show_on_subfail_flag and has_subtest_failures
    )

    if (
        rep.when != "call"
        or item.stash.get(_output_printed_key, False)
        or not (show_runner_output)
    ):
        return

    # A test should only call run_test once
    result = item.stash.get(_result_key, None)
    if not result:
        return

    output_parts = []

    # Build the output
    if show_runner_output:
        item.stash[_output_printed_key] = True
        cmd = " ".join(str(c) for c in getattr(result, "command", []))
        if cmd:
            output_parts.append(f"{'='*70}")
            output_parts.append(f"Command: {cmd}")
        result_env = getattr(result, "environment", None)
        if isinstance(result_env, dict) and result_env:
            env_lines = [f"  {k}={v}" for k, v in sorted(result_env.items())]
            output_parts.append("Environment:\n\n" + "\n".join(env_lines) + "\n")
            output_parts.append(f"{'='*70}")
        output_parts.append("Test Output:\n")
        test_out = getattr(result, "test_output", "")
        if test_out:
            output_parts.append(test_out)

    if not output_parts:
        return

    output_text = "\n".join(output_parts) + "\n\n"
    rep.sections.append(("Runner Output", output_text))


def pytest_runtest_logreport(report):
    """Handle output display for passing tests."""
    # Determine if we should show runner output
    show_output_flag = getattr(pytest, "_show_output_flag", False)
    if show_output_flag and report.when == "call" and report.passed:
        config = getattr(pytest, "_config_ref", None)
        terminal = config.pluginmanager.get_plugin("terminalreporter") if config else None
        if terminal:
            for section_name, section_content in report.sections:
                if section_name == "Runner Output":
                    terminal.write_line(f"\n--- {section_name} ---")
                    for line in section_content.splitlines():
                        terminal.write_line(line)


# ----------------------------------------------------------------------------
# Session End hooks
# ----------------------------------------------------------------------------


def pytest_sessionfinish(session, exitstatus):
    """Code that runs after all tests complete

    If ROCPROFSYS_KEEP_TEST_OUTPUT is not set to OFF, this code cleans up:
    - Temporary buffered storage files
    - Temporary metadata files
    - Perfetto temp files
    - HSA/ROCm temp files
    - Instrumented binaries
    - Causal profiling temp files
    - Empty pytest output directories
    - Test config directories
    """

    # Disallow xdist workers from executing code after this call
    # Only the master process should run this code
    if hasattr(session.config, "workerinput"):
        return

    if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "1") == "1":
        return

    import glob

    # Clean up temp files matching patterns
    for pattern in _cleanup_temp_patterns():
        for filepath in glob.glob(pattern):
            _safe_remove_file(Path(filepath))

    # Clean up empty directories in test output areas
    try:
        config = get_rocprof_config()
        build_dir = config.rocprofsys_build_dir
    except Exception:
        return  # Can't get config, skip directory cleanup

    for dir_path in _cleanup_directory_patterns(build_dir):
        if dir_path.exists():
            # First pass: remove empty subdirectories
            for child in list(dir_path.iterdir()):
                _safe_remove_directory(child, remove_if_empty=True)
            # Second pass: remove parent if now empty
            _safe_remove_directory(dir_path, remove_if_empty=True)


def pytest_unconfigure(config):
    """Clean up resources at end of session."""
    log_handle = getattr(config, "_log_file_handle", None)
    if log_handle:
        log_handle.close()


# ============================================================================
#
# Helper functions
#
# ============================================================================


def configure_mode(config: pytest.Config) -> None:
    """Configure the mode based on the command line options.

    Modes:
     - --ci-mode: CI mode
     - --ctest-mode: CTest integration mode
     - --dev: Developer mode
    """

    # MPI is disabled in CI mode, this is done in collection_modifyitems
    ci_mode = config.getoption("--ci-mode", default=False)
    ctest_mode = config.getoption("--ctest-mode", default="off") == "run"
    dev_mode = config.getoption("--dev", default=False)

    if ci_mode or ctest_mode:
        config.option.output_log = "none"  # Already reported to dashboard
        config.option.show_output_on_subtest_fail = True
        config.option.verbose = max(config.option.verbose, 1)  # -v
        config.option.tbstyle = "short"  # --tb=short
        if "s" not in config.option.reportchars:  # -rs
            config.option.reportchars += "s"

    if ci_mode:
        config.option.show_config = True

    if dev_mode:
        config.option.show_config = True
        config.option.show_output_on_subtest_fail = True
        config.option.verbose = max(config.option.verbose, 1)  # -v
        config.option.tbstyle = "short"  # --tb=short
        if "s" not in config.option.reportchars:  # -rs
            config.option.reportchars += "s"


def _standardize_test_name(item: pytest.Item) -> None:
    if "-]" in item._nodeid or "--" in item._nodeid:
        # Fix trailing dashes
        item._nodeid = re.sub(r"-+\]", "]", item._nodeid)
        item._nodeid = re.sub(r"--+", "-", item._nodeid)
    if "-]" in item.name or "--" in item.name:
        # Also update the display name
        item.name = re.sub(r"-+\]", "]", item.name)
        item.name = re.sub(r"--+", "-", item.name)


def _generate_rocprofsys_config_header(config: pytest.Config) -> list[str]:
    try:
        rocprof_config = get_rocprof_config()
    except Exception as e:
        return [f"{e}"]

    gpu_info = get_gpu_info()

    if rocprof_config.rocm_path:
        # Rocm version
        rocm_version = (
            ".".join(map(str, rocprof_config.rocm_version))
            if rocprof_config.rocm_version
            else "Not found"
        )

        # Rocminfo
        rocminfo_path = get_rocminfo(rocprof_config.rocm_path)
        if not rocminfo_path:
            rocminfo_err_msg = "Not found - Ensure rocminfo is in ROCM_PATH or PATH - Assuming no GPU configuration"

        # Offload extractor
        offload_msg = None
        tool_path, is_llvm_too_old = get_offload_extractor(rocprof_config.rocm_path)
        if tool_path:
            if tool_path.name == "llvm-objdump":
                offload_msg = f"{tool_path}"
            elif tool_path.name == "roc-obj-ls":
                if not is_llvm_too_old:
                    offload_msg = f"Using deprecated {tool_path} - Set ROCM_LLVM_OBJDUMP to use llvm-objdump instead"
                else:
                    offload_msg = f"{tool_path}"

        if not offload_msg:
            offload_msg = (
                "Not found - Set ROCM_LLVM_OBJDUMP to path of llvm-objdump (v20+), "
                "or to path of roc-obj-ls if llvm-objdump < v20"
            )
        xnack_support = get_xnack_support(rocprof_config.rocm_path)
    else:
        rocm_version = "Not found"
        rocminfo_err_msg = "ROCm not found"
        offload_msg = "ROCm not found"
        rocminfo_path = None
        xnack_support = False

    header = [
        "",
        "=" * 70,
        "Test Configuration:",
        "=" * 70,
        f"  ROCm version:         {rocm_version}",
        f"  ROCm path:            {rocprof_config.rocm_path}",
        f"  Is installed:         {rocprof_config.is_installed}",
        f"  Output dir:           {rocprof_config.test_output_dir}",
        f"  Log file:             {getattr(config, '_output_log_path', None) or 'Disabled'}",
        f"  Validate ROCPD:       {check_use_rocpd()}",
        f"  Validate Perfetto:    {check_use_perfetto()}",
        "-" * 70,
        "System Capabilities:",
        f"  Detected num procs:   {rocprof_config.capabilities.num_procs}",
        f"  MPI impl:             {rocprof_config.capabilities.mpi_implementation}",
        f"  UCX available:        {rocprof_config.capabilities.ucx_availability}",
        f"  Default NIC:          {rocprof_config.capabilities.default_nic}",
        f"  PAPI available:       {rocprof_config.capabilities.papi_availability}",
        f"  PAPI NIC events:      {rocprof_config.capabilities.papi_nic_events}",
        f"  Perf event paranoid:  {rocprof_config.capabilities.perf_event_paranoid}",
        f"  CAP_SYS_ADMIN:        {rocprof_config.capabilities.cap_sys_admin}",
        f"  CAP_PERFMON:          {rocprof_config.capabilities.cap_perfmon}",
        f"  Ptrace scope:         {rocprof_config.capabilities.ptrace_scope}",
        "-" * 70,
        "GPU Information:",
        f"  rocminfo:             {rocminfo_path if rocminfo_path else rocminfo_err_msg}",
        f"  Available:            {gpu_info.available}",
        f"  Architectures:        {gpu_info.architectures if gpu_info.architectures else 'None'}",
        f"  Device count:         {gpu_info.device_count}",
        f"  Categories:           {gpu_info.categories if gpu_info.categories else 'None'}",
        f"  XNACK support:        {xnack_support}",
        "-" * 70,
        "Directories:",
        f"  Build dir:            {rocprof_config.rocprofsys_build_dir}",
        f"  Lib dir:              {rocprof_config.rocprofsys_lib_dir}",
        f"  Bin dir:              {rocprof_config.rocprofsys_bin_dir}",
        f"  Tests dir:            {rocprof_config.rocprofsys_tests_dir}",
        f"  Examples dir:         {rocprof_config.rocprofsys_examples_dir}",
        f"  Validation dir:       {rocprof_config.rocpd_validation_rules}",
        "-" * 70,
        "Executables:",
        f"  Instrument:           {rocprof_config.rocprofsys_instrument}",
        f"  Run:                  {rocprof_config.rocprofsys_run}",
        f"  Sample:               {rocprof_config.rocprofsys_sample}",
        f"  Avail:                {rocprof_config.rocprofsys_avail}",
        f"  Causal:               {rocprof_config.rocprofsys_causal}",
        f"  MPI exec:             {rocprof_config.mpiexec}",
        f"  Julia:                {rocprof_config.julia}",
        f"  Offload tool:         {offload_msg}",
        "-" * 70,
        "Python:",
        f"  Module Path:          {rocprof_config.python_module_path or '(none)'}",
    ]
    if rocprof_config.python_versions and rocprof_config.python_executables:
        for version, exe in zip(
            rocprof_config.python_versions, rocprof_config.python_executables
        ):
            header.append(f"  {version}:                 {exe}")
    else:
        header.append("  Executables:          (none found)")
    header.extend(
        [
            "-" * 70,
            "System Environment:",
        ]
    )
    fundamental_env = rocprof_config.get_fundamental_environment()
    for key, value in sorted(fundamental_env.items()):
        header.append(f"  {key}:{' ' * (17 - len(key))}{value}")
    header.extend(["=" * 70, ""])
    return header


def add_marker_if(
    item,
    marker_to_add: str,
    cond: bool = True,
    req_mark: Optional[str] = None,
    skip_reason: Optional[str] = None,
) -> None:
    """Add a marker to a test item if:
        - target_marker is present (or not specified)
        - AND condition is True
    If condition is False and skip_reason is provided, add a skip marker instead.
    """
    if req_mark and not item.get_closest_marker(req_mark):
        return

    if cond:
        item.add_marker(getattr(pytest.mark, marker_to_add))
    elif skip_reason:
        item.add_marker(pytest.mark.skip(reason=skip_reason))


@lru_cache(maxsize=1)
def check_use_rocpd() -> bool:
    """Whether ROCpd is available for tests.

    ROCpd requires:
    - ROCPROFSYS_USE_ROCPD not set to OFF (default: ON)
    - A valid GPU
    - ROCm >= 7.0
    """
    if os.environ.get("ROCPROFSYS_USE_ROCPD", "ON").upper() != "ON":
        return False
    try:
        rocprof_config = get_rocprof_config()
    except Exception as e:
        pytest.exit(f"{e}")
    gpu_info = get_gpu_info()
    if not gpu_info.available:
        return False
    rocm_version = rocprof_config.rocm_version
    return rocm_version is not None and rocm_version >= (7, 0, 0)


@lru_cache(maxsize=1)
def check_use_perfetto() -> bool:
    """Whether Perfetto is available for tests.

    Perfetto requires:
    - ROCPROFSYS_VALIDATE_PERFETTO not set to OFF (default: ON)
    - Perfetto Python module installed
    """
    if os.environ.get("ROCPROFSYS_VALIDATE_PERFETTO", "ON").upper() != "ON":
        return False
    try:
        import perfetto  # noqa

        return True
    except ImportError:
        return False


# The first call to this function MUST be performed in pytest_sessionstart
# as we need the --python-versions and --python-root-dirs options to be set
@lru_cache(maxsize=1)
def get_rocprof_config() -> RocprofsysConfig:
    """Return the rocprofiler-systems configuration."""
    try:
        pytest_config = getattr(pytest, "_config_ref", None)
        custom_output_dir = None
        if pytest_config:
            custom_output_dir = pytest_config.getoption("--output-dir", default=None)

        return discover_build_config(
            output_dir=Path(custom_output_dir) if custom_output_dir else None
        )
    except Exception as e:
        raise RuntimeError(f"Failed to get rocprofiler-systems configuration: {e}")


@lru_cache(maxsize=1)
def get_gpu_info() -> GPUInfo:
    """Return the GPU information."""
    try:
        rocprof_config = get_rocprof_config()
    except Exception as e:
        pytest.exit(f"{e}")
    return detect_gpu(rocprof_config.rocm_path)


def _cleanup_temp_patterns() -> list[str]:
    """Return list of temp file patterns to clean up."""
    patterns = []

    # For CTest impl, these should not be deleted
    patterns.extend(
        [
            "/tmp/buffered_storage*.bin",
            "/tmp/metadata*.json",
        ]
    )

    # Other rocprofiler-systems temp files (always cleaned)
    patterns.extend(
        [
            "/tmp/rocprof-sys-*.tmp",
            "/tmp/rocprofsys-*.tmp",
            # Perfetto temp files
            "/tmp/perfetto-*.proto",
            "/tmp/perfetto_trace*.proto",
            # HSA/ROCm temp files
            "/tmp/hsa-*.tmp",
            "/tmp/rocm-*.tmp",
            "/tmp/hip-*.tmp",
            # Instrumented binaries that might be left over
            "/tmp/*.inst",
            # Causal profiling temp files
            "/tmp/causal-*.json",
            "/tmp/experiments-*.coz",
            # Core dumps (if any)
            "/tmp/core.*",
        ]
    )

    return patterns


def _cleanup_directory_patterns(build_dir: Path) -> list[Path]:
    """Return list of directories to check for cleanup."""
    patterns = []
    # For CTest impl, these should not be deleted
    patterns.extend(
        [
            build_dir / "rocprof-sys-pytest-output",
            build_dir / "rocprof-sys-tests-output",
        ]
    )

    return patterns


def _safe_remove_file(filepath: Path) -> None:
    """Safely remove a file, ignoring errors."""
    try:
        if filepath.is_file():
            filepath.unlink()
    except OSError:
        pass


def _safe_remove_directory(dirpath: Path, remove_if_empty: bool = True) -> None:
    """Safely remove a directory.

    Args:
        dirpath: Path to directory
        remove_if_empty: If True, only remove if empty. If False, remove recursively.
    """
    try:
        if not dirpath.exists():
            return
        if remove_if_empty:
            if dirpath.is_dir() and not any(dirpath.iterdir()):
                dirpath.rmdir()
        else:
            if dirpath.is_dir():
                shutil.rmtree(dirpath)
    except OSError:
        pass


# ============================================================================
#
# Fixtures
#
# ============================================================================

# ----------------------------------------------------------------------------
# Environment Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def base_env(rocprof_config) -> dict[str, str]:
    """Get base environment variables for test execution."""
    return rocprof_config.get_base_environment()


@pytest.fixture
def flat_env(base_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for flat profile tests."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_COUT_OUTPUT": "ON",
        "ROCPROFSYS_FLAT_PROFILE": "ON",
        "ROCPROFSYS_TIMELINE_PROFILE": "OFF",
        "ROCPROFSYS_COLLAPSE_PROCESSES": "ON",
        "ROCPROFSYS_COLLAPSE_THREADS": "ON",
        "ROCPROFSYS_SAMPLING_FREQ": "50",
        "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count",
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "threads",
        "OMP_NUM_THREADS": "2",
        "LD_LIBRARY_PATH": base_env.get("LD_LIBRARY_PATH", ""),
    }


@pytest.fixture
def lock_env(base_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for thread lock tracing tests."""
    return {
        "ROCPROFSYS_USE_SAMPLING": "ON",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
        "ROCPROFSYS_SAMPLING_FREQ": "750",
        "ROCPROFSYS_COLLAPSE_THREADS": "ON",
        "ROCPROFSYS_TRACE_THREAD_LOCKS": "ON",
        "ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS": "ON",
        "ROCPROFSYS_TRACE_THREAD_RW_LOCKS": "ON",
        "ROCPROFSYS_COUT_OUTPUT": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_TIMELINE_PROFILE": "OFF",
        "ROCPROFSYS_LOG_LEVEL": "trace",
        "LD_LIBRARY_PATH": base_env.get("LD_LIBRARY_PATH", ""),
    }


@pytest.fixture
def perfetto_env(base_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for perfetto-only tests."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "OFF",
        "ROCPROFSYS_USE_SAMPLING": "ON",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_PERFETTO_BACKEND": "inprocess",
        "ROCPROFSYS_PERFETTO_FILL_POLICY": "ring_buffer",
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "threads",
        "OMP_NUM_THREADS": "2",
        "LD_LIBRARY_PATH": base_env.get("LD_LIBRARY_PATH", ""),
    }


@pytest.fixture
def timemory_env(base_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for timemory-only tests."""
    return {
        "ROCPROFSYS_TRACE": "OFF",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_USE_SAMPLING": "ON",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count,peak_rss",
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "threads",
        "OMP_NUM_THREADS": "2",
        "LD_LIBRARY_PATH": base_env.get("LD_LIBRARY_PATH", ""),
    }


# ----------------------------------------------------------------------------
# Session-scoped Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def is_xdist_used(request) -> bool:
    """Whether xdist is actively being used (parallel mode) for the test session."""
    # workerinput only exists on xdist worker processes
    return hasattr(request.config, "workerinput")


@pytest.fixture(scope="session")
def num_processes(request) -> int:
    """Get the number of processes for the test."""
    return request.config.getoption("--num-processes", default=2)


@pytest.fixture(scope="session")
def get_test_num_threads(rocprof_config) -> int:
    """Get the number of threads for the test."""
    num_threads = rocprof_config.capabilities.num_procs + (
        rocprof_config.capabilities.num_procs // 2
    )
    if num_threads > 12:
        return 12
    return num_threads


@pytest.fixture(scope="session")
def rocprof_config() -> RocprofsysConfig:
    """Session-wide rocprofiler-systems configuration.

    Discovers build directory and creates configuration object.
    Can be overridden with ROCPROFSYS_BUILD_DIR environment variable.
    """
    return get_rocprof_config()


@pytest.fixture(scope="session")
def gpu_info() -> GPUInfo:
    """Session-wide GPU information.

    Detects available GPUs and their capabilities.
    """
    return get_gpu_info()


@pytest.fixture(scope="session")
def tests_dir(rocprof_config) -> Path:
    """Path to tests directory."""
    return rocprof_config.rocprofsys_tests_dir


@pytest.fixture(scope="session")
def validation_rules_dir(rocprof_config) -> Path:
    """Path to validation rules directory."""
    return rocprof_config.rocpd_validation_rules


# ----------------------------------------------------------------------------
# Module-scoped Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_output_base(rocprof_config, request) -> Path:
    """Base directory for test outputs"""
    group_by_module = request.config.getoption("--group-by-module", default=False)
    if group_by_module:
        module_name = request.module.__name__
        if module_name.startswith("test_"):
            module_name = module_name[5:]
        module_name = module_name.replace("_", "-")
        output_dir = rocprof_config.test_output_dir / module_name
    else:
        output_dir = rocprof_config.test_output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(scope="module", autouse=True)
def cleanup_module_temp_files(
    rocprof_config, request: pytest.FixtureRequest, is_xdist_used
):
    """Module-scoped cleanup that runs AFTER each test module completes.

    Execution Order:
        1. Module starts
        2. All tests in module run (with their validations)
        3. Module ends
        4. This cleanup runs (after yield)

    Cleans up instrumented binaries and intermediate files created during module tests.
    This does NOT interfere with individual test validations.
    """
    yield  # All tests in module run here

    if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "1") == "1":
        return

    import glob

    # Clean up instrumented binaries in build directory
    for pattern in ["*.inst", "*.inst.orig"]:
        for filepath in glob.glob(str(rocprof_config.rocprofsys_build_dir / pattern)):
            _safe_remove_file(Path(filepath))

    # Defer below cleanup to end of session
    if is_xdist_used:
        return

    # Clean up trace cache temp files
    # For CTest impl, these should not be deleted
    for pattern in ["/tmp/buffered_storage*.bin", "/tmp/metadata*.json"]:
        for filepath in glob.glob(pattern):
            _safe_remove_file(Path(filepath))


# ----------------------------------------------------------------------------
# Class-scoped Fixtures
# ----------------------------------------------------------------------------


class OutputPathStore:
    """Simple key-value store for sharing output paths between tests in a class.

    Usage:
        collect_output_path.store("my-key", some_path)
        path = collect_output_path.get("my-key")
        all_paths = collect_output_path.all()
    """

    def __init__(self):
        self._storage: dict[str, Path] = {}

    def store(self, key: str, value: Path) -> Path:
        """Store a path under the given key. Returns the stored path."""
        self._storage[key] = value
        return value

    def get(self, key: str) -> Optional[Path]:
        """Retrieve a path by key, or None if not found."""
        return self._storage.get(key)

    def all(self) -> dict[str, Path]:
        """Return a copy of all stored paths."""
        return self._storage.copy()


# Used to collect paths from different tests to be used in another test
@pytest.fixture(scope="class")
def collect_output_path():
    """Class-scoped fixture for collecting and retrieving output paths."""
    return OutputPathStore()


# ----------------------------------------------------------------------------
# Function-scoped Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture(scope="function")
def parallelize(request):
    """Run assertion functions in parallel.

    This should only be used for assertions that take a long time to run.
    User must pass --parallelize for this to have any effect.

    Usage:
        parallelize(
            lambda: self.<assertion_fixture>(<args>),
            ...
        )
    """
    use_parallel = request.config.getoption("--parallelize", default=False)

    def _parallel(*assert_fns: Callable[[], None]) -> None:
        if use_parallel:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(fn) for fn in assert_fns]
                for future in futures:
                    future.result()
        else:
            for fn in assert_fns:
                fn()

    return _parallel


@pytest.fixture
def create_config_file(test_output_dir) -> Path:
    """Create a config file for a test based on env vars and return Path.

    Filters out environment-only settings that should not be written to config files
    """
    # Settings that should only be in environment, not config files
    env_only_pattern = re.compile(
        r"ROCPROFSYS_(CI|CI_TIMEOUT|MODE|USE_MPIP|DEBUG_[A-Z_]+|"
        r"FORCE_ROCPROFILER_INIT|DEFAULT_MIN_INSTRUCTIONS|MONOCHROME|VERBOSE)$"
    )

    def _create_config_file(
        env: dict[str, str],
        name: Optional[str] = "config.cfg",
        skip_filter: bool = False,
    ) -> Path:
        config_file = test_output_dir / name
        content = "# auto-generated by pytest\n\n"

        if skip_filter:
            config_vars = {k: v for k, v in env.items() if k != "ROCPROFSYS_CONFIG_FILE"}
        else:
            # Only write ROCPROFSYS_* settings to config file, excluding env-only settings
            # Non-ROCPROFSYS vars (OMP_*, LD_LIBRARY_PATH, etc.) should stay as env vars only
            # Also exclude ROCPROFSYS_CONFIG_FILE to avoid self-reference
            config_vars = {
                k: v
                for k, v in env.items()
                if k.startswith("ROCPROFSYS_")
                and not env_only_pattern.match(k)
                and k != "ROCPROFSYS_CONFIG_FILE"
            }

        content += "\n".join(f"{k}={v}" for k, v in config_vars.items())
        config_file.write_text(content)
        return config_file

    return _create_config_file


@pytest.fixture
def collect_result(request) -> Callable:
    """Fixture to collect test results for display.

    Handled by the `run_test` fixture

    Manual usage in tests:
        result = runner.run()
        collect_result(result)
    """

    def _collect(result):
        request.node.stash[_result_key] = result

    return _collect


@pytest.fixture
def test_output_dir(
    test_output_base: Path,
    request: pytest.FixtureRequest,
) -> Generator[Path, None, None]:
    """Unique output directory for each test.

    Creates a directory named after the test and cleans up on success.
    On failure, the directory is preserved for debugging.

    Cleanup Order:
        1. Test setup: Directory is created
        2. Test body: Runner executes, output files are written
        3. Test body: Validation happens on output files
        4. Test body: Assertions complete
        5. Test teardown: This fixture cleans up the directory (AFTER yield)

    This ensures validation always has access to output files.
    """
    class_name = request.node.cls.__name__ if request.node.cls else None
    # Remove "Test" prefix from class name
    if class_name and class_name.startswith("Test"):
        class_name = class_name[4:]
    test_name = request.node.name
    # Remove "test_/-" prefix from test name
    if test_name.startswith("test"):
        test_name = test_name[4:]
        if test_name.startswith("_") or test_name.startswith("-"):
            test_name = test_name[1:]
    full_name = f"{class_name}__{test_name}" if class_name else test_name
    # Replace non-alphanumeric (excluding ".") with dash and strip trailing dashes
    safe_name = "".join(c if c.isalnum() or c == "." else "-" for c in full_name)
    while "--" in safe_name:
        safe_name = safe_name.replace("--", "-")
    safe_name = safe_name.strip("-")
    output_dir = test_output_base / safe_name

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    yield output_dir  # Test body executes here (including validation)

    # === CLEANUP PHASE (runs AFTER test body completes) ===
    # Cleanup on success unless ROCPROFSYS_KEEP_TEST_OUTPUT is set
    keep_output = os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "1") == "1"
    test_failed = hasattr(request.node, "rep_call") and request.node.rep_call.failed

    if not keep_output and not test_failed and output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture(scope="function", autouse=True)
def apply_rocpd_marker(request):
    """Automatically add ROCpd env vars based on marker.

    Usage:
        @pytest.mark.rocpd("<env name>")
    """
    if not check_use_rocpd():
        return

    marker = request.node.get_closest_marker("rocpd")
    if not marker or not marker.args:
        return

    # First arg is fixture name
    env_fixture_name = marker.args[0]

    try:
        env = request.getfixturevalue(env_fixture_name)
    except pytest.FixtureLookupError:
        return

    # Add ROCpd base env
    env["ROCPROFSYS_USE_ROCPD"] = "ON"


@pytest.fixture
def cleanup_instrumented_binary(
    rocprof_config,
    test_output_dir: Path,
) -> Generator[None, None, None]:
    """Function-scoped cleanup for instrumented binaries.

    Use this fixture in tests that create instrumented binaries to ensure
    they are cleaned up after the test completes.
    """
    # Track files before test
    pre_existing = (
        set(test_output_dir.glob("*.inst")) if test_output_dir.exists() else set()
    )

    yield

    if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "1") == "1":
        return

    # Clean up any new .inst files
    if test_output_dir.exists():
        for inst_file in test_output_dir.glob("*.inst"):
            if inst_file not in pre_existing:
                _safe_remove_file(inst_file)

    # Also clean from build directory
    for inst_file in rocprof_config.rocprofsys_build_dir.glob("*.inst"):
        _safe_remove_file(inst_file)


# This is needed for pytest-subtests plugin compatibility when pytest < 9.0.0
@pytest.fixture
def record_subtest_failure(request):
    """Fixture to record subtest failures for --show-output-on-subtest-fail.

    Used by assert fixtures to track failures with pytest-subtests plugin.
    """

    def _record(name: str):
        request.node.stash.setdefault(_subtest_failures_key, []).append(name)

    return _record


def _is_assert_disabled(request: pytest.FixtureRequest, subtest_name: str) -> bool:
    """Check if a subtest is disabled via ci_disable marker in CI mode."""
    if not request.config.getoption(
        "--ci-mode", default=False
    ) or request.config.getoption("--allow-disabled", default=False):
        return False
    for marker in request.node.iter_markers("ci_disable"):
        if subtest_name in marker.args:
            return True
    return False


# ============================================================================
# Base Test Class
# ============================================================================


class RocprofsysTest:
    """Base class that auto-captures parametrized values and common fixtures onto self."""

    # Capture common fixtures and parametrized values onto self
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        run_test,
        assert_regex,
        assert_perfetto,
        assert_rocpd,
        assert_causal_json,
        assert_file_exists,
        assert_timemory,
        assert_file_regex,
        parallelize,
        get_test_num_threads,
        test_output_dir,
    ):

        self.run_test = run_test
        self.assert_regex = assert_regex
        self.assert_perfetto = assert_perfetto
        self.assert_rocpd = assert_rocpd
        self.assert_causal_json = assert_causal_json
        self.assert_file_exists = assert_file_exists
        self.assert_timemory = assert_timemory
        self.assert_file_regex = assert_file_regex
        self.parallelize = parallelize
        self.num_threads = get_test_num_threads
        self.test_output_dir = test_output_dir


# ============================================================================
# Test run and assertion fixtures
# ============================================================================


@pytest.fixture
def run_test(
    request,
    collect_result,
    rocprof_config,
    gpu_info,
    test_output_dir,
):
    """Unified fixture to run any test runner type and handle pytest logic.
    If a rocprof-sys binary is provided, uses "base_binary_environment" instead of "base_environment".

    Args:
        runner_type: One of "baseline", "sampling", "binary_rewrite",
                     "runtime_instrument", "sys_run"
        target: Target executable name
        run_args: Arguments passed to the target executable
        env: Environment variables dict
        timeout: Test timeout in seconds
        mpi_ranks: Number of MPI ranks (0 = disabled)
        working_directory: Custom working directory
        check_target_arch: If True, checks if the target supports the current system architectures (default: False)
                           Note: This requires @pytest.mark.gpu to be present
        skip_on_error: If True, pytest.skip on non-zero return code (default: False = fail)
        fail_on_pass: If True, pytest.fail on success and pytest.pass on failure (default: False)
        fail_on_not_found: If True, pytest.fail when binary not found (default: False = skip)
        fail_message: Custom failure message (default: "{runner_type} test failed: {output}")
        no_base_env: If true, don't use the base environment (default: False)
        **kwargs: Additional runner-specific arguments (sample_args, rewrite_args, etc.)

    Returns:
        TestResult for further assertions
    """

    def _run_test(
        runner_type: str,
        target: str,
        env: Optional[dict[str, str]] = None,
        run_args: Optional[list[str]] = None,
        pre_run_args: Optional[list[str]] = None,
        timeout: int = 300,
        mpi_ranks: int = 0,
        working_directory: Optional[Path] = None,
        check_target_arch: bool = False,
        skip_on_error: bool = False,
        fail_on_pass: bool = False,
        fail_on_not_found: bool = False,
        fail_message: Optional[str] = None,
        **kwargs,
    ) -> TestResult:
        # Check for mode-specific timeout
        # Normalize runner_type for lookup (hyphens to underscores)
        timeout_key = f"{runner_type.replace('-', '_')}_timeout"
        if timeout_key in kwargs:
            timeout = kwargs.pop(timeout_key)

        # Filter kwargs to only pass runner-specific args that each runner accepts.
        runner_specific_args = {
            "baseline": {"command"},
            "sampling": {"sample_args"},
            "binary_rewrite": {"rewrite_args", "cleanup_on_success"},
            "runtime_instrument": {"runtime_args"},
            "sys_run": {"sysrun_args"},
            "causal": {"causal_args", "causal_mode"},
            "python": {"python_version", "profile_args", "annotated", "standalone"},
        }
        allowed_args = runner_specific_args.get(runner_type, set())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_args}

        if runner_type == "causal" and "causal_mode" not in filtered_kwargs:
            pytest.exit("causal_mode is required for causal tests", returncode=1)

        runner_class = ROCPROFSYS_RUNNER_CLASSES.get(runner_type)
        if not runner_class:
            pytest.fail(
                f"Invalid runner type: {runner_type}. Use: {list(ROCPROFSYS_RUNNER_CLASSES.keys())}"
            )

        # For GPU tests, ensure that the target supports at least one of the current system architectures
        if request.node.get_closest_marker("gpu") and check_target_arch:
            try:
                target_path = rocprof_config.get_target_executable(target)
                target_archs = get_target_gpu_arch(rocprof_config.rocm_path, target_path)
                system_archs = gpu_info.architectures
                if not any(arch in target_archs for arch in system_archs):
                    pytest.skip(
                        f"{target} does not support any of the current system architectures. "
                        f"{target} architectures: {target_archs}, system architectures: {system_archs}"
                    )
            except FileNotFoundError:
                pass

        # Verify that MPI is available for "mpi_optional" tests
        if request.node.get_closest_marker("mpi_optional") and mpi_ranks > 0:
            if not request.node.get_closest_marker("mpi"):
                mpi_ranks = 0

        # Apply --monochrome option if set
        if request.config.getoption("--monochrome", default=False):
            env = env.copy() if env else {}
            env["ROCPROFSYS_MONOCHROME"] = "ON"

        try:
            runner = runner_class(
                config=rocprof_config,
                target=target,
                output_dir=test_output_dir,
                run_args=run_args,
                pre_run_args=pre_run_args,
                env=env,
                timeout=timeout,
                mpi_ranks=mpi_ranks,
                working_directory=working_directory,
                **filtered_kwargs,
            )
        except FileNotFoundError:
            if fail_on_not_found:
                pytest.fail(f"{target} binary not found")
            else:
                pytest.skip(f"{target} binary not found")

        result = runner.run()
        collect_result(result)
        output = (
            f"{result.test_output}\n{result.extra_output}"
            if result.extra_output
            else result.test_output
        )

        if not result.success and not fail_on_pass:
            # Makereport is only for subtest failures, so we also build output here
            cmd_str = " ".join(str(c) for c in getattr(result, "command", []))
            env_dict = getattr(result, "environment", {})
            env_str = (
                "\n".join(f"  {k}={v}" for k, v in sorted(env_dict.items()))
                if env_dict
                else ""
            )

            details = []
            if cmd_str:
                details.append(f"Command: {cmd_str}")
            if env_str:
                details.append(f"Environment:\n{env_str}")
            details.append(f"Runner Output:\n{output}")

            detail_text = "\n\n".join(details)
            if fail_message:
                msg = f"{fail_message}\n\n{detail_text}"
            else:
                msg = f"{runner_type} test failed\n\n{detail_text}"
            if skip_on_error:
                pytest.skip(msg)
            else:
                pytest.fail(msg)

        if fail_on_pass and result.success:
            pytest.fail(f"{runner_type} test passed unexpectedly: {result.test_output}")

        return result

    return _run_test


@pytest.fixture
def assert_regex(subtests, record_subtest_failure, request):
    """Fixture that returns an assert_regex function.

    Args:
        result: TestResult from run_test
        mode: Optional runner type (e.g., "binary_rewrite", "sys_run"). If provided, looks up
              mode-specific regexes from kwargs (e.g., rewrite_pass_regex, sys_run_pass_regex)
        subtest_name: Name shown in subtest output (defaults to "Regex validation")
        pass_regex: Explicit list of pass regex patterns (used if mode is None or no mode-specific found)
        fail_regex: Explicit list of fail regex patterns (used if mode is None or no mode-specific found)
        skip_on_fail: If True, skip instead of fail when validation fails
        fail_message: Custom message for failure (defaults to validation message)
        **kwargs: Mode-specific regexes like rewrite_pass_regex, sys_run_fail_regex, etc.
    """
    if _is_assert_disabled(request, "assert_regex"):
        return lambda *args, **kwargs: None

    def _assert_regex(
        result: TestResult,
        mode: Optional[str] = None,
        subtest_name: str = "Regex validation",
        pass_regex: Optional[list[str]] = None,
        fail_regex: Optional[list[str]] = None,
        use_abort_fail_regex: bool = True,
        skip_on_fail: bool = False,
        fail_message: Optional[str] = None,
        **kwargs,
    ) -> None:
        # If mode is provided, look up mode-specific regexes from kwargs
        if mode is not None:
            # Normalize mode name (hyphens to underscores)
            mode_key = mode.replace("-", "_")
            pass_regex = kwargs.get(f"{mode_key}_pass_regex") or pass_regex
            fail_regex = kwargs.get(f"{mode_key}_fail_regex") or fail_regex

        with subtests.test(subtest_name):
            validation = validate_regex(
                result, pass_regex, fail_regex, use_abort_fail_regex
            )
            if not validation.is_valid:
                msg = fail_message or f"Regex validation failed: {validation.message}"
                if skip_on_fail:
                    pytest.skip(msg)
                else:
                    record_subtest_failure(subtest_name)
                    pytest.fail(msg)

    return _assert_regex


@pytest.fixture
def assert_file_regex(subtests, record_subtest_failure, request):
    """Variant of assert_regex that validates against a file."""
    if _is_assert_disabled(request, "assert_file_regex"):
        return lambda *args, **kwargs: None

    def _assert_file_regex(
        file_path: Path,
        subtest_name: str = "File regex validation",
        pass_regex: Optional[list[str]] = None,
        fail_regex: Optional[list[str]] = None,
        use_abort_fail_regex: bool = True,
        skip_on_fail: bool = False,
        fail_message: Optional[str] = None,
    ) -> None:
        with subtests.test(subtest_name):
            validation = validate_file_regex(
                file_path,
                pass_regex,
                fail_regex,
                use_abort_fail_regex,
            )

            if not validation.is_valid:
                msg = (
                    fail_message or f"File regex validation failed: {validation.message}"
                )
                if skip_on_fail:
                    pytest.skip(msg)
                else:
                    record_subtest_failure(subtest_name)
                    pytest.fail(msg)

    return _assert_file_regex


@pytest.fixture
def assert_perfetto(
    subtests, tests_dir, record_subtest_failure, request, test_output_dir
):
    """Fixture that returns an assert_perfetto function.

    Args not from validate_perfetto_trace:
        subtest_name: Name shown in subtest output (defaults to "Perfetto validation")
        perfetto_file: (Optional) Name of the perfetto file in the test output directory (e.g., for merged.proto)
        pass_regex: (Optional) Regex patterns that must be found in validation.stdout
        fail_regex: (Optional) Regex patterns that must NOT be found in validation.stdout
        skip_on_fail: If True, skip instead of fail when validation fails
        fail_message: Custom message for failure (defaults to validation message)
    """
    if _is_assert_disabled(request, "assert_perfetto"):
        return lambda *args, **kwargs: None

    def _assert_perfetto(
        result: TestResult,
        subtest_name: str = "Perfetto validation",
        perfetto_file: Optional[Path] = None,
        categories: Optional[list[str]] = None,
        labels: Optional[list[str]] = None,
        counts: Optional[list[int]] = None,
        depths: Optional[list[int]] = None,
        label_substrings: Optional[list[str]] = None,
        counter_names: Optional[list[str]] = None,
        key_names: Optional[list[str]] = None,
        key_counts: Optional[list[int]] = None,
        trace_processor_path: Optional[Path] = None,
        print_output: bool = True,
        timeout: int = 120,
        pass_regex: Optional[list[str]] = None,
        fail_regex: Optional[list[str]] = None,
        skip_on_fail: bool = False,
        fail_message: Optional[str] = None,
    ) -> None:
        with subtests.test(subtest_name):
            if not check_use_perfetto():
                pytest.skip("Perfetto is disabled")

            # Perfetto file check
            if perfetto_file is not None:
                perfetto = Path(test_output_dir) / perfetto_file
            else:
                perfetto = result.perfetto_file
            if not perfetto.exists():
                record_subtest_failure(subtest_name)
                pytest.fail(f"Perfetto trace file {perfetto} not found")

            validation = validate_perfetto_trace(
                perfetto,
                tests_dir=tests_dir,
                categories=categories,
                labels=labels,
                counts=counts,
                depths=depths,
                label_substrings=label_substrings,
                counter_names=counter_names,
                key_names=key_names,
                key_counts=key_counts,
                trace_processor_path=trace_processor_path,
                print_output=print_output,
                timeout=timeout,
            )
            output = f"Command: {validation.command}\n\n{validation.message}"
            if not validation.is_valid:
                msg = fail_message or f"Perfetto validation failed:\n{output}"
                if skip_on_fail:
                    pytest.skip(msg)
                else:
                    record_subtest_failure(subtest_name)
                    pytest.fail(msg)
            if pass_regex:
                for pattern in pass_regex:
                    if not re.search(pattern, validation.stdout):
                        record_subtest_failure(subtest_name)
                        pytest.fail(f"Pass regex not found: {pattern}\n{output}")
            if fail_regex:
                for pattern in fail_regex:
                    if re.search(pattern, validation.stdout):
                        record_subtest_failure(subtest_name)
                        pytest.fail(f"Fail regex found: {pattern}\n{output}")

    return _assert_perfetto


@pytest.fixture
def assert_rocpd(subtests, tests_dir, record_subtest_failure, request):
    """Fixture that returns an assert_rocpd function.

    Must be used with @pytest.mark.rocpd("<env fixture name>")

    Args not from validate_rocpd_database:
        subtest_name: Name shown in subtest output (defaults to "ROCpd validation")
        pass_regex: (Optional) Regex patterns that must be found in validation.stdout
        fail_regex: (Optional) Regex patterns that must NOT be found in validation.stdout
        skip_on_fail: If True, skip instead of fail when validation fails
        fail_message: Custom message for failure (defaults to validation message)
    """
    if _is_assert_disabled(request, "assert_rocpd"):
        return lambda *args, **kwargs: None

    def _assert_rocpd(
        result: TestResult,
        subtest_name: str = "ROCpd validation",
        rules_files: Optional[list[Path]] = None,
        timeout: int = 60,
        pass_regex: Optional[list[str]] = None,
        fail_regex: Optional[list[str]] = None,
        skip_on_fail: bool = False,
        fail_message: Optional[str] = None,
    ) -> None:
        with subtests.test(subtest_name):
            if not check_use_rocpd():
                pytest.skip("ROCpd is disabled")
            rocpd_file = result.rocpd_file
            if rocpd_file is None:
                record_subtest_failure(subtest_name)
                pytest.fail("ROCpd database not created")

            existing_rules = None
            if rules_files is not None:
                existing_rules = [r for r in rules_files if r.exists()]
                if not existing_rules:
                    record_subtest_failure(subtest_name)
                    pytest.fail("No validation rules found")

            validation = validate_rocpd_database(
                rocpd_file,
                tests_dir=tests_dir,
                rules_files=existing_rules,
                timeout=timeout,
            )
            output = f"Command: {validation.command}\n\n{validation.message}"
            if not validation.is_valid:
                msg = fail_message or f"ROCpd validation failed:\n{output}"
                if skip_on_fail:
                    pytest.skip(msg)
                else:
                    record_subtest_failure(subtest_name)
                    pytest.fail(msg)
            if pass_regex:
                for pattern in pass_regex:
                    if not re.search(pattern, validation.stdout):
                        record_subtest_failure(subtest_name)
                        pytest.fail(f"Pass regex not found: {pattern}\n{output}")
            if fail_regex:
                for pattern in fail_regex:
                    if re.search(pattern, validation.stdout):
                        record_subtest_failure(subtest_name)
                        pytest.fail(f"Fail regex found: {pattern}\n{output}")

    return _assert_rocpd


@pytest.fixture
def assert_timemory(subtests, tests_dir, record_subtest_failure, request):
    """Fixture that returns an assert_timemory function.

    Args not from validate_timemory_json:
        subtest_name: Name shown in subtest output (defaults to "Timemory validation")
        pass_regex: (Optional) Regex patterns that must be found in validation.stdout
        fail_regex: (Optional) Regex patterns that must NOT be found in validation.stdout
        skip_on_fail: If True, skip instead of fail when validation fails
        fail_message: Custom message for failure (defaults to validation message)
    """
    if _is_assert_disabled(request, "assert_timemory"):
        return lambda *args, **kwargs: None

    def _assert_timemory(
        result: TestResult,
        file_name: str,
        metric: str,
        subtest_name: str = "Timemory validation",
        labels: Optional[list[str]] = None,
        counts: Optional[list[int]] = None,
        depths: Optional[list[int]] = None,
        print_output: bool = True,
        timeout: int = 60,
        pass_regex: Optional[list[str]] = None,
        fail_regex: Optional[list[str]] = None,
        skip_on_fail: bool = False,
        fail_message: Optional[str] = None,
    ) -> None:
        with subtests.test(subtest_name):
            timemory_file = result.output_dir / file_name
            if not timemory_file.exists():
                record_subtest_failure(subtest_name)
                pytest.fail(f"Timemory file not found: {timemory_file}")
            validation = validate_timemory_json(
                json_path=timemory_file,
                tests_dir=tests_dir,
                metric=metric,
                labels=labels,
                counts=counts,
                depths=depths,
                print_output=print_output,
                timeout=timeout,
            )
            output = f"Command: {validation.command}\n\n{validation.message}"
            if not validation.is_valid:
                msg = fail_message or f"Timemory validation failed:\n{output}"
                if skip_on_fail:
                    pytest.skip(msg)
                else:
                    record_subtest_failure(subtest_name)
                    pytest.fail(msg)
            if pass_regex:
                for pattern in pass_regex:
                    if not re.search(pattern, validation.stdout):
                        record_subtest_failure(subtest_name)
                        pytest.fail(f"Pass regex not found: {pattern}\n{output}")
            if fail_regex:
                for pattern in fail_regex:
                    if re.search(pattern, validation.stdout):
                        record_subtest_failure(subtest_name)
                        pytest.fail(f"Fail regex found: {pattern}\n{output}")

    return _assert_timemory


@pytest.fixture
def assert_file_exists(subtests, record_subtest_failure, request):
    """Fixture that returns an assert_file_exists function.

    Args not from validate_file_exists:
        subtest_name: Name shown in subtest output (defaults to "File existence validation")
        skip_on_fail: If True, skip instead of fail when validation fails
        fail_message: Custom message for failure (defaults to validation message)
    """
    if _is_assert_disabled(request, "assert_file_exists"):
        return lambda *args, **kwargs: None

    def _assert_file_exists(
        path: Path | list[Path],
        description: str = "File",
        subtest_name: str = "File existence validation",
        skip_on_fail: bool = False,
        fail_message: Optional[str] = None,
    ) -> None:
        paths = [path] if isinstance(path, Path) else path
        with subtests.test(subtest_name):
            for p in paths:
                validation = validate_file_exists(p, description)
                if not validation.is_valid:
                    msg = (
                        fail_message
                        or f"File existence validation failed: {validation.message}"
                    )
                    if skip_on_fail:
                        pytest.skip(msg)
                    else:
                        record_subtest_failure(subtest_name)
                        pytest.fail(msg)

    return _assert_file_exists


@pytest.fixture
def assert_causal_json(subtests, tests_dir, record_subtest_failure, request):
    """Fixture that returns an assert_causal_json function.

    Args not from validate_causal_json:
        pass_regex: (Optional) Regex patterns that must be found in validation.stdout
        fail_regex: (Optional) Regex patterns that must NOT be found in validation.stdout
        skip_on_fail: If True, skip instead of fail when validation fails
        fail_message: Custom message for failure (defaults to validation message)
    """
    if _is_assert_disabled(request, "assert_causal_json"):
        return lambda *args, **kwargs: None

    def _assert_causal_json(
        result: TestResult,
        file_name: str,
        subtest_name: str = "Causal JSON validation",
        ci_mode: bool = False,
        additional_args: Optional[list[str]] = None,
        timeout: int = 60,
        pass_regex: Optional[list[str]] = None,
        fail_regex: Optional[list[str]] = None,
        skip_on_fail: bool = False,
        fail_message: Optional[str] = None,
    ) -> None:
        with subtests.test(subtest_name):
            causal_file = result.output_dir / file_name
            if not causal_file.exists():
                record_subtest_failure(subtest_name)
                pytest.fail(f"Causal JSON file not found: {causal_file}")

            validation = validate_causal_json(
                json_path=causal_file,
                tests_dir=tests_dir,
                ci_mode=ci_mode,
                additional_args=additional_args,
                timeout=timeout,
            )
            output = f"Command: {validation.command}\n\n{validation.message}"
            if not validation.is_valid:
                if fail_message:
                    msg = f"{fail_message}:\n{output}"
                else:
                    msg = f"Causal JSON validation failed:\n{output}"
                if skip_on_fail:
                    pytest.skip(msg)
                else:
                    record_subtest_failure(subtest_name)
                    pytest.fail(msg)

            if pass_regex:
                for pattern in pass_regex:
                    if not re.search(pattern, validation.stdout):
                        record_subtest_failure(subtest_name)
                        pytest.fail(f"Pass regex not found: {pattern}\n{output}")

            if fail_regex:
                for pattern in fail_regex:
                    if re.search(pattern, validation.stdout):
                        record_subtest_failure(subtest_name)
                        pytest.fail(f"Fail regex found: {pattern}\n{output}")

    return _assert_causal_json
