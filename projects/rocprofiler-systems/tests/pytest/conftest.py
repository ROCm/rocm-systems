# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Pytest configuration and fixtures for rocprofiler-systems tests.

This module provides shared fixtures and configuration for all test modules.
"""

from __future__ import annotations
import os
import sys
import shutil
import re
from pathlib import Path
from functools import lru_cache
from typing import Callable, Generator, Optional

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
    TestResult,
    validate_regex,
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
)

# Key for storing the single test result on pytest items
_result_key: StashKey = StashKey()
# Key for tracking subtest failures (for pytest-subtests plugin compatibility when pytest < 9.0.0)
_subtest_failures_key: StashKey[list] = StashKey()
# Key to prevent duplicate output printing
_output_printed_key: StashKey[bool] = StashKey()


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
        "--ctest-integration",
        action="store_true",
        default=False,
        help="Enable CTest integration (developer flag : default off)",
    )
    group.addoption(
        "--allow-disabled",
        action="store_true",
        default=False,
        help="Allow disabled subtests to run (CI mode only, developer flag : default off)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and configure pytest"""

    # Enable CI configuration
    if config.getoption("--ci-mode", default=False):
        config.option.output_log = "none"  # Already reported to dashboard
        config.option.show_config = True
        config.option.show_output_on_subtest_fail = True
        config.option.verbose = max(config.option.verbose, 1)  # -v
        config.option.tbstyle = "short"  # --tb=short
        if "s" not in config.option.reportchars:  # -rs
            config.option.reportchars += "s"

    is_monochrome = config.getoption("--monochrome", default=False)
    if is_monochrome:
        config.option.color = "no"

    # Functional markers (use arguments or do more than just label a test)

    config.addinivalue_line(
        "markers",
        "gpu: mark test as requiring a GPU (default: any available GPU)",
    )  # triggers GPU check in run_test unless no_check_target_arch=True
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
        "disable(name): Use 'all' to skip entire test, or assertion name (e.g., 'assert_rocpd') to disable subtest (CI mode only).",
    )

    # Non-functional informational markers

    config.addinivalue_line("markers", "mpi: mark test as requiring MPI")
    config.addinivalue_line("markers", "rocm: mark test as requiring ROCm")
    config.addinivalue_line(
        "markers", "rocprofiler: mark test as using ROCProfiler counters"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "loops: mark test as testing loop instrumentation")

    # Can be described using generic desc below
    label_list = [
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
    ]
    for label in label_list:
        config.addinivalue_line("markers", f"{label}: label test as {label}")

    # Save flags to pytest
    pytest._show_output_flag = config.getoption("--show-output", default=False)
    pytest._show_output_on_subtest_fail_flag = config.getoption(
        "--show-output-on-subtest-fail", default=False
    )
    pytest._ctest_integration_flag = config.getoption(
        "--ctest-integration", default=False
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

    if log_file.lower() == "none":
        config._output_log_path = None
        config._log_file_handle = None
    else:
        log_file = log_file.replace("@output_dir@", str(rocprof_config.test_output_dir))
        config._output_log_path = Path(log_file)

        log_path = config._output_log_path
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
    """Add test configuration to pytest header output."""

    try:
        rocprof_config = get_rocprof_config()
    except Exception as e:
        return [f"{e}"]

    try:
        gpuInfo = detect_gpu(rocprof_config.rocm_path)
    except Exception as e:
        return [f"rocprofiler-systems: GPU detection error - {e}"]

    if not config.getoption("--show-config", default=False):
        return []

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
            "or ROC_OBJ_LS to path of roc-obj-ls if llvm-objdump < v20"
        )

    rocm_version = (
        ".".join(map(str, rocprof_config.rocm_version))
        if rocprof_config.rocm_version
        else "Not found"
    )

    lines = [
        "",
        "=" * 70,
        "Test Configuration:",
        "=" * 70,
        f"  ROCm version:      {rocm_version}",
        f"  ROCm path:         {rocprof_config.rocm_path}",
        f"  Is installed:      {rocprof_config.is_installed}",
        f"  Output dir:        {rocprof_config.test_output_dir}",
        f"  Log file:          {getattr(config, '_output_log_path', None) or 'Disabled'}",
        f"  Validate ROCPD:    {check_use_rocpd()}",
        f"  Validate Perfetto: {check_use_perfetto()}",
        "-" * 70,
        "GPU Information:",
        f"  rocminfo:          {rocminfo_path if rocminfo_path else rocminfo_err_msg}",
        f"  Available:         {gpuInfo.available}",
        f"  Architectures:     {gpuInfo.architectures}",
        f"  Device count:      {gpuInfo.device_count}",
        f"  Categories:        {gpuInfo.categories}",
        "-" * 70,
        "Directories:",
        f"  Build dir:         {rocprof_config.rocprofsys_build_dir}",
        f"  Lib dir:           {rocprof_config.rocprofsys_lib_dir}",
        f"  Bin dir:           {rocprof_config.rocprofsys_bin_dir}",
        f"  Tests dir:         {rocprof_config.rocprofsys_tests_dir}",
        f"  Examples dir:      {rocprof_config.rocprofsys_examples_dir}",
        f"  Validation dir:    {rocprof_config.rocpd_validation_rules}",
        "-" * 70,
        "Executables:",
        f"  Instrument:        {rocprof_config.rocprofsys_instrument}",
        f"  Run:               {rocprof_config.rocprofsys_run}",
        f"  Sample:            {rocprof_config.rocprofsys_sample}",
        f"  Avail:             {rocprof_config.rocprofsys_avail}",
        f"  Causal:            {rocprof_config.rocprofsys_causal}",
        f"  MPI exec:          {rocprof_config.mpiexec}",
        f"  Offload tool:      {offload_msg}",
        "-" * 70,
        "System Environment:",
    ]
    fundamental_env = rocprof_config.get_fundamental_environment()
    for key, value in sorted(fundamental_env.items()):
        lines.append(f"  {key}:{' ' * (17 - len(key))}{value}")
    lines.extend(["=" * 70, ""])
    return lines


# ----------------------------------------------------------------------------
# Collection hooks
# ----------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items) -> None:
    """Skip tests based on markers and available resources."""
    try:
        rocprof_config = get_rocprof_config()
    except Exception as e:
        pytest.exit(f"{e}")
    gpu_info = detect_gpu(rocprof_config.rocm_path)

    skip_gpu = pytest.mark.skip(reason="No valid GPU available")
    skip_mpi = pytest.mark.skip(reason="MPI not available")

    mpi_available = rocprof_config.mpiexec is not None

    for item in items:
        if "gpu" in item.keywords and not gpu_info.available:
            item.add_marker(skip_gpu)

        if "mpi" in item.keywords and not mpi_available:
            item.add_marker(skip_mpi)

        # Check rocm_min_version marker
        rocm_min_marker = item.get_closest_marker("rocm_min_version")
        if rocm_min_marker:
            min_version = rocm_min_marker.args[0] if rocm_min_marker.args else None
            rocm_version = rocprof_config.rocm_version
            if rocm_version is None:
                item.add_marker(pytest.mark.skip(reason="ROCm not found"))
            else:
                # Parse min_version and compare
                min_parts = min_version.split(".")
                min_tuple = tuple(int(p) for p in (min_parts + ["0", "0"])[:3])
                if rocm_version < min_tuple:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"ROCm {'.'.join(map(str, rocm_version))} < required {min_version}"
                        )
                    )

        # Check run_if_gpu_category marker
        run_if_gpu_category_marker = item.get_closest_marker("run_if_gpu_category")
        if run_if_gpu_category_marker and gpu_info.available:
            expr = run_if_gpu_category_marker.args[0]

            # Build evaluation context: each category is True/False
            eval_context = {
                "instinct": "instinct" in gpu_info.categories,
                "radeon": "radeon" in gpu_info.categories,
                "apu": "apu" in gpu_info.categories,
            }

            try:
                result = eval(expr, {"__builtins__": {}}, eval_context)
                if not result:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"GPU category condition '{expr}' not met, "
                            f"GPU has categories {gpu_info.categories}"
                        )
                    )
            except Exception as e:
                item.add_marker(
                    pytest.mark.fail(
                        reason=f"Invalid run_if_gpu_category marker expression: {e}"
                    )
                )

    # Deselect tests marked with @pytest.mark.disable("all") (CI mode)
    if config.getoption("--ci-mode", default=False) and not config.getoption(
        "--allow-disabled", default=False
    ):
        selected = []
        deselected = []
        for item in items:
            marker = item.get_closest_marker("disable")
            if marker and "all" in marker.args:
                deselected.append(item)
            else:
                selected.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = selected


# ----------------------------------------------------------------------------
# Test execution hooks
# ----------------------------------------------------------------------------


@pytest.hookimpl(hookwrapper=True)  # Allows yield
def pytest_runtest_makereport(item, call):
    """Build runner output and attach to report."""
    outcome = yield
    rep = outcome.get_result()
    config = getattr(pytest, "_config_ref", None)

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


@lru_cache(maxsize=1)
def check_use_rocpd() -> bool:
    """Whether ROCpd is available for tests.

    ROCpd requires:
    - ROCPROFSYS_USE_ROCPD not set to OFF (default: ON)
    - A valid GPU
    - ROCm >= 7.0
    """
    if os.environ.get("ROCPROFSYS_USE_ROCPD", "").upper() == "OFF":
        return False
    try:
        rocprof_config = get_rocprof_config()
    except Exception as e:
        pytest.exit(f"{e}")
    gpu_info = detect_gpu(rocprof_config.rocm_path)
    if not gpu_info.available:
        return False
    rocm_version = rocprof_config.rocm_version
    return rocm_version is not None and rocm_version >= (7, 0, 0)


@lru_cache(maxsize=1)
def check_use_perfetto() -> bool:
    """Whether Perfetto is available for tests.

    Perfetto requires:
    - Perfetto Python module installed
    - ROCPROFSYS_VALIDATE_PERFETTO not set to OFF (default: ON)
    """
    if os.environ.get("ROCPROFSYS_VALIDATE_PERFETTO", "").upper() == "OFF":
        return False
    try:
        import perfetto  # noqa

        return True
    except ImportError:
        return False


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


def _cleanup_temp_patterns() -> list[str]:
    """Return list of temp file patterns to clean up."""
    patterns = []

    if not getattr(pytest, "_ctest_integration_flag", False):
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
    if not getattr(pytest, "_ctest_integration_flag", False):
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
def rocprof_config() -> RocprofsysConfig:
    """Session-wide rocprofiler-systems configuration.

    Discovers build directory and creates configuration object.
    Can be overridden with ROCPROFSYS_BUILD_DIR environment variable.
    """
    return get_rocprof_config()


@pytest.fixture(scope="session")
def gpu_info(rocprof_config) -> GPUInfo:
    """Session-wide GPU information.

    Detects available GPUs and their capabilities.
    """
    return detect_gpu(rocprof_config.rocm_path)


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
def test_output_base(rocprof_config) -> Path:
    """Base directory for test outputs (module-scoped).

    All test outputs for a module are stored under this directory.
    """
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
    if not getattr(pytest, "_ctest_integration_flag", False):
        for pattern in ["/tmp/buffered_storage*.bin", "/tmp/metadata*.json"]:
            for filepath in glob.glob(pattern):
                _safe_remove_file(Path(filepath))


# ----------------------------------------------------------------------------
# Function-scoped Fixtures
# ----------------------------------------------------------------------------


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
    test_name = request.node.name
    full_name = f"{class_name}__{test_name}" if class_name else test_name
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in full_name)
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
        no_check_target_arch: If True, bypasses checking if the target supports the current
                              system architectures when @pytest.mark.gpu is present (default: False)
        skip_on_error: If True, pytest.skip on non-zero return code (default: False = fail)
        fail_on_pass: If True, pytest.fail on success and pytest.pass on failure (default: False)
        fail_on_not_found: If True, pytest.fail when binary not found (default: False = skip)
        fail_message: Custom failure message (default: "{runner_type} test failed: {output}")
        no_base_env: If true, don't use the base environment (default: False)
        **kwargs: Additional runner-specific arguments (sample_args, rewrite_args, etc.)

    Returns:
        TestResult for further assertions
    """
    RUNNERS = {
        "baseline": BaselineRunner,
        "sampling": SamplingRunner,
        "binary_rewrite": BinaryRewriteRunner,
        "runtime_instrument": RuntimeInstrumentRunner,
        "sys_run": SysRunRunner,
    }

    def _run_test(
        runner_type: str,
        target: str,
        run_args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        timeout: int = 300,
        mpi_ranks: int = 0,
        working_directory: Optional[Path] = None,
        no_check_target_arch: bool = False,
        skip_on_error: bool = False,
        fail_on_pass: bool = False,
        fail_on_not_found: bool = False,
        fail_message: Optional[str] = None,
        **kwargs,
    ) -> TestResult:
        runner_class = RUNNERS.get(runner_type)
        if not runner_class:
            pytest.fail(
                f"Invalid runner type: {runner_type}. Use: {list(RUNNERS.keys())}"
            )

        # For GPU tests, ensure that the target supports at least one of the current system architectures
        if request.node.get_closest_marker("gpu") and not no_check_target_arch:
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
                env=env,
                timeout=timeout,
                mpi_ranks=mpi_ranks,
                working_directory=working_directory,
                **kwargs,
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
            if fail_message:
                msg = f"{fail_message}: {output}"
            else:
                msg = f"{runner_type} test failed: {output}"
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

    Args not from validate_regex:
        subtest_name: Name shown in subtest output (defaults to "Regex validation")
        skip_on_fail: If True, skip instead of fail when validation fails
        fail_message: Custom message for failure (defaults to validation message)
    """
    disabled_subtests: set[str] = set()
    if request.config.getoption(
        "--ci-mode", default=False
    ) and not request.config.getoption("--allow-disabled", default=False):
        for marker in request.node.iter_markers("disable"):
            disabled_subtests.update(marker.args)

    def _assert_regex(
        result: TestResult,
        subtest_name: str = "Regex validation",
        pass_regex: Optional[list[str]] = None,
        fail_regex: Optional[list[str]] = None,
        use_abort_fail_regex: bool = True,
        skip_on_fail: bool = False,
        fail_message: Optional[str] = None,
    ) -> None:
        if "assert_regex" in disabled_subtests:
            return

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
def assert_perfetto(subtests, tests_dir, record_subtest_failure, request):
    """Fixture that returns an assert_perfetto function.

    Args not from validate_perfetto_trace:
        subtest_name: Name shown in subtest output (defaults to "Perfetto validation")
        pass_regex: (Optional) Regex patterns that must be found in validation.stdout
        fail_regex: (Optional) Regex patterns that must NOT be found in validation.stdout
        skip_on_fail: If True, skip instead of fail when validation fails
        fail_message: Custom message for failure (defaults to validation message)
    """
    disabled_subtests: set[str] = set()
    if request.config.getoption(
        "--ci-mode", default=False
    ) and not request.config.getoption("--allow-disabled", default=False):
        for marker in request.node.iter_markers("disable"):
            disabled_subtests.update(marker.args)

    def _assert_perfetto(
        result: TestResult,
        subtest_name: str = "Perfetto validation",
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
        if "assert_perfetto" in disabled_subtests:
            return

        with subtests.test(subtest_name):
            if not check_use_perfetto():
                pytest.skip("Perfetto is disabled")
            perfetto = result.perfetto_file
            if perfetto is None:
                record_subtest_failure(subtest_name)
                pytest.fail("Perfetto trace not created")
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
    disabled_subtests: set[str] = set()
    if request.config.getoption(
        "--ci-mode", default=False
    ) and not request.config.getoption("--allow-disabled", default=False):
        for marker in request.node.iter_markers("disable"):
            disabled_subtests.update(marker.args)

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
        if "assert_rocpd" in disabled_subtests:
            return

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
    disabled_subtests: set[str] = set()
    if request.config.getoption(
        "--ci-mode", default=False
    ) and not request.config.getoption("--allow-disabled", default=False):
        for marker in request.node.iter_markers("disable"):
            disabled_subtests.update(marker.args)

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
        if "assert_timemory" in disabled_subtests:
            return

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
    disabled_subtests: set[str] = set()
    if request.config.getoption(
        "--ci-mode", default=False
    ) and not request.config.getoption("--allow-disabled", default=False):
        for marker in request.node.iter_markers("disable"):
            disabled_subtests.update(marker.args)

    def _assert_file_exists(
        path: Path | list[Path],
        description: str = "File",
        subtest_name: str = "File existence validation",
        skip_on_fail: bool = False,
        fail_message: Optional[str] = None,
    ) -> None:
        if "assert_file_exists" in disabled_subtests:
            return

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
    disabled_subtests: set[str] = set()
    if request.config.getoption(
        "--ci-mode", default=False
    ) and not request.config.getoption("--allow-disabled", default=False):
        for marker in request.node.iter_markers("disable"):
            disabled_subtests.update(marker.args)

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
        if "assert_causal_json" in disabled_subtests:
            return

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
