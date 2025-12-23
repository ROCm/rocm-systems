# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Pytest configuration and fixtures for rocprofiler-systems tests.

This module provides shared fixtures and configuration for all test modules.
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from typing import Generator, Callable

# Add the pytest directory to Python path for rocprofsys package
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from pytest import StashKey

from rocprofsys import (
    RocprofsysConfig,
    discover_build_config,
    GPUInfo,
    detect_gpu,
    lookup_gpu_category,
    _get_rocm_version,
    _check_rocm_version_geq,
)

# Key for storing test results on pytest items
_results_key: StashKey[list] = StashKey()


# ============================================================================
# Pytest Configuration
# ============================================================================


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
        "--no-output",
        action="store_true",
        default=False,
        help="Suppress all output including failure details",
    )
    group.addoption(
        "--show-output-on-subtest-fail",
        action="store_true",
        default=False,
        help="Show runner output only when subtests fail",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring a GPU")
    config.addinivalue_line(
        "markers",
        "gpu_category_exclude(categories): exclude tests that require a specific GPU category (instinct, radeon, apu)",
    )
    config.addinivalue_line("markers", "mpi: mark test as requiring MPI")
    config.addinivalue_line("markers", "rocm: mark test as requiring ROCm")
    config.addinivalue_line(
        "markers", "rocprofiler: mark test as using ROCProfiler counters"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "loops: mark test as testing loop instrumentation")
    config.addinivalue_line(
        "markers",
        "rocm_min_version(version): mark test as requiring minimum ROCm version (e.g., '7.0', '6.2.1')",
    )
    # Register plugin to handle --no-output flag
    config.pluginmanager.register(NoOutputPlugin(config), "no_output_plugin")


class NoOutputPlugin:
    """Plugin to suppress failure details when --no-output is specified."""

    def __init__(self, config: pytest.Config):
        self.no_output = config.getoption("--no-output", default=False)

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if self.no_output and report.failed:
            # Use empty string instead of None to remain compatible with junitxml plugin
            report.longrepr = ""


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests based on markers and available resources."""
    gpu_info = detect_gpu()

    skip_gpu = pytest.mark.skip(reason="No valid GPU available")
    skip_mpi = pytest.mark.skip(reason="MPI not available")

    mpi_available = (
        shutil.which("mpiexec") is not None or shutil.which("mpirun") is not None
    )

    for item in items:
        if "gpu" in item.keywords and not gpu_info.available:
            item.add_marker(skip_gpu)

        if "mpi" in item.keywords and not mpi_available:
            item.add_marker(skip_mpi)

        # Check rocm_min_version marker
        rocm_min_marker = item.get_closest_marker("rocm_min_version")
        if rocm_min_marker:
            min_version = rocm_min_marker.args[0] if rocm_min_marker.args else None
            if min_version and not _check_rocm_version_geq(min_version):
                rocm_version = _get_rocm_version()
                if rocm_version is None:
                    reason = "ROCm not found"
                else:
                    reason = f"ROCm {rocm_version[0]}.{rocm_version[1]}.{rocm_version[2]} < required {min_version}"
                item.add_marker(pytest.mark.skip(reason=reason))

        # Check gpu_category_exclude marker
        gpu_category_marker = item.get_closest_marker("gpu_category_exclude")
        if gpu_category_marker and gpu_info.available:
            # Get excluded categories from marker args (can be multiple)
            excluded_categories = set(gpu_category_marker.args)

            # Get system GPU categories (check all detected architectures)
            system_categories: set[str] = set()
            for arch in gpu_info.architectures:
                system_categories.update(lookup_gpu_category(arch))

            # Skip if any system GPU category is in the excluded list
            matching_categories = system_categories & excluded_categories
            if matching_categories:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"Test does not support GPU categories {excluded_categories}, "
                        f"system has {system_categories}"
                    )
                )


# ============================================================================
# Session-scoped Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def use_rocpd(gpu_info: GPUInfo) -> bool:
    """Whether ROCpd is available for tests.

    ROCpd requires:
    - ROCPROFSYS_USE_ROCPD not set to OFF (default: ON)
    - A valid GPU
    - ROCm >= 7.0
    """

    if os.environ.get("ROCPROFSYS_USE_ROCPD", "").upper() == "OFF":
        return False
    if not gpu_info.available:
        return False
    return _check_rocm_version_geq("7.0")


@pytest.fixture(scope="session")
def use_perfetto() -> bool:
    """Whether Perfetto is available for tests.

    Perfetto requires:
    - Perfetto Python module installed
    - ROCPROFSYS_TRACE not set to OFF (default: ON)
    """
    if os.environ.get("ROCPROFSYS_TRACE", "").upper() == "OFF":
        return False
    try:
        import perfetto  # noqa

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def rocprof_config() -> RocprofsysConfig:
    """Session-wide rocprofiler-systems configuration.

    Discovers build directory and creates configuration object.
    Can be overridden with ROCPROFSYS_BUILD_DIR environment variable.
    """
    return discover_build_config()


@pytest.fixture(scope="session")
def gpu_info() -> GPUInfo:
    """Session-wide GPU information.

    Detects available GPUs and their capabilities.
    """
    return detect_gpu()


@pytest.fixture(scope="session")
def root_dir(rocprof_config: RocprofsysConfig) -> Path:
    """Path to the rocprofiler-systems root directory (or install directory)."""
    return rocprof_config.rocprofsys_root_dir


@pytest.fixture(scope="session")
def build_dir(rocprof_config: RocprofsysConfig) -> Path:
    """Path to rocprofiler-systems build directory (or install directory)."""
    return rocprof_config.rocprofsys_build_dir


@pytest.fixture(scope="session")
def tests_dir(rocprof_config: RocprofsysConfig) -> Path:
    """Path to tests directory."""
    return rocprof_config.rocprofsys_tests_dir


@pytest.fixture(scope="session")
def validation_rules_dir(rocprof_config: RocprofsysConfig) -> Path:
    """Path to validation rules directory."""
    return rocprof_config.rocpd_validation_rules


# ============================================================================
# Module-scoped Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def test_output_base(rocprof_config: RocprofsysConfig) -> Path:
    """Base directory for test outputs (module-scoped).

    All test outputs for a module are stored under this directory.
    """
    output_dir = rocprof_config.test_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# Function-scoped Fixtures
# ============================================================================


@pytest.fixture(scope="function", autouse=True)
def _result_output(request):
    """Auto-use fixture to handle result collection and --show-output display.

    Output display logic:
    - Default (no flags): Test fails → pytest shows test_output. No extra printing.
    - --show-output-on-subtest-fail: Print test_output only when subtests fail
    - --show-output: Implies --show-output-on-subtest-fail, plus prints on test pass
    - --no-output: Suppresses all output

    Main test body failures are always handled by pytest's captured output.
    """
    results = []
    request.node.stash[_results_key] = results
    yield
    if not results:
        return

    show_output = request.config.getoption("--show-output", default=False)
    show_on_subtest_fail = request.config.getoption(
        "--show-output-on-subtest-fail", default=False
    )
    if show_output:
        show_on_subtest_fail = True
    if not show_output and not show_on_subtest_fail:
        return

    # Check test status
    test_failed = hasattr(request.node, "rep_call") and request.node.rep_call.failed

    # Check if the test even uses subtests fixture
    # If subtests fixture is not used, there can be no subtest failures
    uses_subtests = False
    if hasattr(request, "fixturenames"):
        uses_subtests = "subtests" in request.fixturenames

    # Detect probable subtest failures (heuristic: test uses subtests, runners passed, but test failed)
    has_subtest_failures = False
    if uses_subtests and test_failed:
        all_results_succeeded = all(getattr(r, "success", True) for r in results)
        if all_results_succeeded:
            has_subtest_failures = True

    should_show = False
    if has_subtest_failures and show_on_subtest_fail:
        # Actual subtest failed - show output
        should_show = True
    elif not test_failed and show_output:
        # Test passed and --show-output is set - show output
        should_show = True
    # Main test body failure (no subtests): pytest handles it via captured output, don't duplicate
    if not should_show:
        return

    for result in results:
        cmd_str = " ".join(str(c) for c in getattr(result, "command", []))
        if cmd_str:
            print(f"\n{'='*70}")
            print(f"Command: {cmd_str}")
            print(f"{'='*70}")

        test_output = getattr(result, "test_output", "")
        extra_output = getattr(result, "extra_output", "")

        if test_output:
            print(f"--- TEST OUTPUT ---\n{test_output}")
        if extra_output:
            print(f"--- EXTRA OUTPUT ---\n{extra_output}")


@pytest.fixture
def collect_result(request) -> Callable:
    """Fixture to collect test results for --show-output display.

    Usage in tests:
        result = runner.run()
        collect_result(result)
        assert result.success, f"Failed: {result.failure_reason}"
    """

    def _collect(result):
        request.node.stash[_results_key].append(result)

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
    test_name = request.node.name
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in test_name)
    output_dir = test_output_base / safe_name

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    yield output_dir  # Test body executes here (including validation)

    # === CLEANUP PHASE (runs AFTER test body completes) ===
    # Cleanup on success unless ROCPROFSYS_KEEP_TEST_OUTPUT is set
    keep_output = os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "0") == "1"
    test_failed = hasattr(request.node, "rep_call") and request.node.rep_call.failed

    if not keep_output and not test_failed and output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture
def base_env(rocprof_config: RocprofsysConfig) -> dict[str, str]:
    """Base environment variables for test execution."""
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


# ============================================================================
# Cleanup Fixtures
# ============================================================================


def _cleanup_temp_patterns() -> list[str]:
    """Return list of temp file patterns to clean up."""
    return [
        # rocprofiler-systems temp files
        "/tmp/buffered_storage*.bin",
        "/tmp/metadata*.json",
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


def _cleanup_directory_patterns(build_dir: Path) -> list[Path]:
    """Return list of directories to check for cleanup."""
    return [
        build_dir / "rocprof-sys-pytest-output",
        build_dir / "rocprof-sys-tests-output",
        build_dir / "rocprof-sys-tests-config",
    ]


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


@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_files(rocprof_config: RocprofsysConfig):
    """Session-scoped cleanup fixture that runs AFTER ALL tests complete.

    Execution Order:
        1. Session starts
        2. All test modules run (with their validations)
        3. Session ends
        4. This cleanup runs (after yield)

    Cleans up:
    - Temporary buffered storage files
    - Temporary metadata files
    - Perfetto temp files
    - HSA/ROCm temp files
    - Instrumented binaries
    - Causal profiling temp files
    - Empty pytest output directories
    - Test config directories
    """
    yield  # All tests run here

    if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "0") == "1":
        return

    import glob

    # Clean up temp files matching patterns
    for pattern in _cleanup_temp_patterns():
        for filepath in glob.glob(pattern):
            _safe_remove_file(Path(filepath))

    # Clean up empty directories in test output areas
    for dir_path in _cleanup_directory_patterns(rocprof_config.rocprofsys_build_dir):
        if dir_path.exists():
            # First pass: remove empty subdirectories
            for child in list(dir_path.iterdir()):
                _safe_remove_directory(child, remove_if_empty=True)
            # Second pass: remove parent if now empty
            _safe_remove_directory(dir_path, remove_if_empty=True)


@pytest.fixture(scope="module", autouse=True)
def cleanup_module_temp_files(
    rocprof_config: RocprofsysConfig, request: pytest.FixtureRequest
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

    if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "0") == "1":
        return

    import glob

    # Clean up instrumented binaries in build directory
    for pattern in ["*.inst", "*.inst.orig"]:
        for filepath in glob.glob(str(rocprof_config.rocprofsys_build_dir / pattern)):
            _safe_remove_file(Path(filepath))

    # Clean up any temp files in /tmp that match session patterns
    temp_patterns = [
        "/tmp/buffered_storage*.bin",
        "/tmp/metadata*.json",
    ]
    for pattern in temp_patterns:
        for filepath in glob.glob(pattern):
            _safe_remove_file(Path(filepath))


@pytest.fixture
def cleanup_instrumented_binary(
    rocprof_config: RocprofsysConfig,
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

    if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "0") == "1":
        return

    # Clean up any new .inst files
    if test_output_dir.exists():
        for inst_file in test_output_dir.glob("*.inst"):
            if inst_file not in pre_existing:
                _safe_remove_file(inst_file)

    # Also clean from build directory
    for inst_file in rocprof_config.rocprofsys_build_dir.glob("*.inst"):
        _safe_remove_file(inst_file)


# ============================================================================
# Pytest Hooks for Result Tracking
# ============================================================================


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Track test results for cleanup decisions."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


# Debug helper
def pytest_report_header(config: pytest.Config) -> list[str]:
    """Add test configuration to pytest header output."""
    try:
        rocprof_config = discover_build_config()
    except Exception as e:
        return [f"rocprofiler-systems: Configuration error - {e}"]

    try:
        gpuInfo = detect_gpu()
    except Exception as e:
        return [f"rocprofiler-systems: GPU detection error - {e}"]

    rocm_ver = _get_rocm_version()
    rocm_ver_str = (
        f"{rocm_ver[0]}.{rocm_ver[1]}.{rocm_ver[2]}" if rocm_ver else "Not found"
    )

    lines = [
        "",
        "=" * 70,
        "Test Configuration:",
        "=" * 70,
        f"  ROCm version:   {rocm_ver_str}",
        f"  ROCm path:      {rocprof_config.rocm_path}",
        f"  Is installed:   {rocprof_config.is_installed}",
        "-" * 70,
        "GPU Information:",
        f"  Available:      {gpuInfo.available}",
        f"  Architectures:  {gpuInfo.architectures}",
        f"  Device count:   {gpuInfo.device_count}",
        f"  Is Navi:        {gpuInfo.is_navi}",
        f"  Is Mi300:       {gpuInfo.is_mi300}",
        "-" * 70,
        "Directories:",
        f"  Root dir:       {rocprof_config.rocprofsys_root_dir}",
        f"  Build dir:      {rocprof_config.rocprofsys_build_dir}",
        f"  Lib dir:        {rocprof_config.rocprofsys_lib_dir}",
        f"  Bin dir:        {rocprof_config.rocprofsys_bin_dir}",
        f"  Tests dir:      {rocprof_config.rocprofsys_tests_dir}",
        f"  Examples dir:   {rocprof_config.rocprofsys_examples_dir}",
        f"  Output dir:     {rocprof_config.test_output_dir}",
        f"  Validation dir: {rocprof_config.rocpd_validation_rules}",
        "-" * 70,
        "Executables:",
        f"  Instrument:     {rocprof_config.rocprofsys_instrument}",
        f"  Run:            {rocprof_config.rocprofsys_run}",
        f"  Sample:         {rocprof_config.rocprofsys_sample}",
        f"  Avail:          {rocprof_config.rocprofsys_avail}",
        f"  Causal:         {rocprof_config.rocprofsys_causal}",
        f"  MPI exec:       {rocprof_config.mpiexec}",
        "=" * 70,
        "",
    ]
    return lines
