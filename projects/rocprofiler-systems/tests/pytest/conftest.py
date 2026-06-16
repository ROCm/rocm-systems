# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Pytest configuration and fixtures for rocprofiler-systems tests.

This module provides shared fixtures and configuration for all test modules.
"""

from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from typing import Any, Callable, Generator, Optional

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
    validate_unified_memory_outputs,
    validate_file_exists,
    BaselineRunner,
    SamplingRunner,
    BinaryRewriteRunner,
    RuntimeInstrumentRunner,
    SysRunRunner,
    CausalRunner,
    PythonRunner,
    safe_remove,
)

# Key for storing the single test result on pytest items
# Item-level stash keys
_result_key: StashKey = StashKey()
_subtest_failures_key: StashKey[list] = StashKey()
_output_printed_key: StashKey[bool] = StashKey()
_original_nodeid_key: StashKey[str] = StashKey()

# Config-level stash keys
_show_output_key: StashKey[bool] = StashKey()
_show_on_subfail_key: StashKey[bool] = StashKey()

# GNU convention. Used for CTests
SKIP_RETURN_CODE = 77
# Default timeout for tests in seconds
DEFAULT_TIMEOUT = 300
# Extra seconds added to pytest timeout in generated CTest (flush / teardown)
CTEST_TIMEOUT_BUFFER = 30

# Accepted runner types when using parametrized "mode" marker
ROCPROFSYS_RUNNER_CLASSES = {
    "baseline": BaselineRunner,
    "sampling": SamplingRunner,
    "binary_rewrite": BinaryRewriteRunner,
    "runtime_instrument": RuntimeInstrumentRunner,
    "sys_run": SysRunRunner,
    "causal": CausalRunner,
    "python": PythonRunner,
}
# Accepted runner types when using parametrized "mode" marker
ROCPROFSYS_RUNNER_NAMES = list(ROCPROFSYS_RUNNER_CLASSES.keys())

# rocprofiler-sdk < 1.2.2 can abort on undefined KFD node IDs; product disables KFD domains.
KFD_MIN_SDK_VERSION: tuple[int, int, int] = (1, 2, 2)

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
    # TODO: Deprecate once TheRock switches to CTest and CTest based filtering
    group.addoption(
        "--show-test-output",
        action="store",
        default="subtest",
        choices=("none", "subtest", "all"),
        help="Show runner output: 'none' (no output), 'subtest' (default, on failure only), or 'all' (always)",
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
    # TODO: Deprecate once TheRock switches to CTest and CTest based filtering
    group.addoption(
        "--ci-mode",
        action="store_true",
        default=False,
        help="Enable CI mode (developer flag : default: False)",
    )
    group.addoption(
        "--ctest-mode",
        action="store",
        default="off",
        choices=("off", "generate", "run", "cleanup"),
        help="CTest integration mode (developer flag): 'off' (default), 'generate', 'run', or 'cleanup'",
    )
    group.addoption(
        "--ctest-output-path",
        action="store",
        default=None,
        help="Path to write the CTest definitions file when in CTest generate mode (default: None)",
    )
    # TODO: Deprecate once TheRock switches to CTest
    group.addoption(
        "--dev",
        action="store_true",
        default=False,
        help="Enables some QOL flags (developer flag : default off)",
    )
    group.addoption(
        "--python-versions",
        action="store",
        default=None,
        help="Semicolon-separated list of Python versions (e.g. '3.8;3.9;3.10')",
    )
    group.addoption(
        "--python-root-dirs",
        action="store",
        default=None,
        help="Semicolon-separated list of directories to search for Python interpreters in order of preference",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and configure pytest"""

    configure_mode(config)

    if config.getoption("--ctest-mode", default="off") == "cleanup":
        _run_cleanup(config)
        pytest.exit("Cleanup complete", returncode=0)

    if config.getoption("--show-config-only", default=False):
        pytest._config_ref = config
        header = _generate_rocprofsys_config_header()
        for line in header:
            print(line)
        pytest.exit("Header generated", returncode=0)

    is_monochrome = config.getoption("--monochrome", default=False)
    if is_monochrome:
        config.option.color = "no"

    # Disable pytest-timeout plugin if detected
    # It will interfere with our timeout marker
    timeout_plugin = config.pluginmanager.get_plugin("timeout")
    if timeout_plugin:
        config.pluginmanager.unregister(timeout_plugin)

    # Functional markers (do more than just label a test)
    #   See pytest_collection_modifyitems
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
        "preserve(file): prevents the file from being deleted after the test, even if ROCPROFSYS_KEEP_TEST_OUTPUT is set to OFF",
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
        "oshrun_min_version(version): mark test as requiring minimum OpenSHMEM version",
    )
    config.addinivalue_line(
        "markers",
        "rocpd(env): mark test as using ROCpd and inject ROCpd env into given env",
    )
    # TODO: Deprecate once TheRock switches to CTest and CTest based filtering
    config.addinivalue_line(
        "markers",
        "ci_enable: Full test will be run when in CI mode. To disable a subtest, use ci_disable(name) (CI mode only)",
    )
    # TODO: Deprecate once TheRock switches to CTest and CTest based filtering
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
    config.addinivalue_line(
        "markers",
        "timeout(seconds): mark test as having a timeout of seconds",
    )
    # Used for CTest
    config.addinivalue_line(
        "markers",
        "depends_on(*names): declare CTest dependency on the named tests"
        " (used for CTest DEPENDS)",
    )
    config.addinivalue_line(
        "markers",
        "serialize: mark test as serializable (used for CTest)",
    )
    config.addinivalue_line(
        "markers",
        "class_name(segment): hyphenated logical name for standardized / CTest test "
        "names (replaces the auto-derived class segment from TestCamelCase; "
        "e.g. 'rocprofiler-systems-instrument')",
    )
    config.addinivalue_line(
        "markers",
        "multi_gpu(num): mark test as using requiring atleast num amount of GPUs",
    )
    config.addinivalue_line(
        "markers",
        "rockoff: prevents the test from being run on TheRock",
    )

    # See pytest_collection_modifyitems
    generic_functional_markers = [
        "ucx",
        "overflow",
        "attach",
        "mpi",
        "python",
        "annotate",
        "julia",
        "xnack",
        "no_docker",
        "shmem",
        "nic",
        "ainic",
    ]

    # Informational markers, only used for test labeling

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
        "sdma",
        "group_by_queue",
        "group_by_stream",
        "openmp",
        "openmp_target",
        "fortran",
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
        "scratch_memory",
        "rocm",
        "kfd",
        "unified_memory",
        "validation_usm",
        "selective_regions",
        "minimal",
        "rank_filter",
    ]
    for label in non_functional_markers + generic_functional_markers:
        config.addinivalue_line("markers", f"{label}: label test as {label}")
    #
    _show_test_output = config.getoption("--show-test-output", default="subtest")
    config.stash[_show_output_key] = _show_test_output == "all"
    config.stash[_show_on_subfail_key] = _show_test_output == "subtest"

    # Keep a module-level ref for hooks that don't receive config directly
    pytest._config_ref = config


# ----------------------------------------------------------------------------
# Session start hooks
# ----------------------------------------------------------------------------


# TODO: Deprecate once TheRock switches to CTest and CTest based filtering
def pytest_report_header(config) -> list[str]:
    if not config.getoption("--ci-mode", default=False):
        return []
    return _generate_rocprofsys_config_header()


# ----------------------------------------------------------------------------
# Collection hooks
# ----------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests based on markers."""
    marker = metafunc.definition.get_closest_marker("python_versions")
    if marker is not None:
        rocprof_config = get_rocprof_config()
        supported = set(rocprof_config.capabilities.supported_python_versions or [])

        # When --python-versions is explicitly passed (e.g. from CTest),
        # always parametrize with those exact versions so node IDs match.
        # Unsupported versions are marked as skip.
        pytest_config = getattr(pytest, "_config_ref", None)
        requested_str = (
            pytest_config.getoption("--python-versions", default=None)
            if pytest_config
            else None
        )
        if requested_str:
            requested = [v.strip() for v in requested_str.split(";") if v.strip()]
            params = []
            for ver in requested:
                if ver in supported:
                    params.append(ver)
                else:
                    params.append(
                        pytest.param(
                            ver,
                            marks=pytest.mark.skip(f"Python {ver} not available"),
                        )
                    )
            metafunc.parametrize("python_version", params)
        elif supported:
            metafunc.parametrize("python_version", sorted(supported))
        else:
            metafunc.parametrize(
                "python_version",
                [pytest.param(None, marks=pytest.mark.skip("No Python versions found"))],
            )


# ----------------------------------------------------------------------------
# run_if_gpu_category: namespace for eval() (not an availability / skip reason helper)
# ----------------------------------------------------------------------------


def gpu_category_eval_context() -> dict[str, bool]:
    info = get_gpu_info()
    return {
        "instinct": info is not None and "instinct" in info.categories,
        "radeon": info is not None and "radeon" in info.categories,
        "apu": info is not None and "apu" in info.categories,
    }


def pytest_collection_modifyitems(config, items) -> None:
    """Modify items based on markers."""
    verbose = config.option.verbose > 0

    try:
        rocprof_config = get_rocprof_config()
    except Exception as e:
        pytest.exit(f"{e}")

    # ----------------------------------------------------------------------------
    def base_modifications(item: pytest.Item) -> None:
        """This function should be called for every item."""
        _standardize_test_name(item, config, verbose=verbose)

        # Handle optional markers
        # The general form is <name>_optional(...). If the condition is met, <name> marker is added
        if (
            "mpi_optional" in item.keywords
            and mpi_unavailable_reason(rocprof_config, config) is None
        ):
            target = item.get_closest_marker("mpi_optional").args[0]
            try:
                target_path = rocprof_config.get_target_executable(target)
                if rocprof_config.capabilities.target_support_mpi(target_path):
                    item.add_marker(pytest.mark.mpi)
            except FileNotFoundError:
                pass

        # Marker dependencies
        add_marker_if(
            item,
            "papi",
            req_mark="annotate",
            unavailable_reason=lambda: annotate_unavailable_reason(rocprof_config),
        )
        add_marker_if(item, "mpi", req_mark="mpi_implementation")
        add_marker_if(item, "python", req_mark="python_versions")
        add_marker_if(item, "gpu", req_mark="multi_gpu")

        # Add corresponding runner type markers based on parametrized values ("mode")
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

    # We will not be running tests in this mode, so marker checks are redundant
    if config.getoption("--ctest-mode", default="off") == "generate":
        for item in items:
            base_modifications(item)
        return

    # TODO: Deprecate once TheRock switches to CTest and CTest based filtering
    if config.getoption("--ci-mode", default=False):
        selected_tests = []
        deselected_tests = []
        for item in items:
            base_modifications(item)
            disable_marker = item.get_closest_marker("ci_disable")
            ci_disabled = disable_marker and "all" in disable_marker.args
            if item.get_closest_marker("ci_enable") and not ci_disabled:
                selected_tests.append(item)
            else:
                deselected_tests.append(item)
        config.hook.pytest_deselected(items=deselected_tests)
        items[:] = selected_tests
        return

    # Marker checks
    # "Skip" markers are left for runtime evaluation
    for item in items:
        base_modifications(item)
        if "gpu" in item.keywords:
            _msg = gpu_unavailable_reason()
            if _msg is not None:
                item.add_marker(pytest.mark.skip(reason=_msg))
        if "ucx" in item.keywords and not rocprof_config.capabilities.ucx_availability:
            item.add_marker(pytest.mark.skip(reason="UCX not available"))
        if "mpi" in item.keywords:
            _msg = mpi_unavailable_reason(rocprof_config, config)
            if _msg is not None:
                item.add_marker(pytest.mark.skip(reason=_msg))
        if "mpi_implementation" in item.keywords:
            req_impl = item.get_closest_marker("mpi_implementation").args[0]
            if req_impl != rocprof_config.capabilities.mpi_implementation:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"Requires {req_impl}, but {rocprof_config.capabilities.mpi_implementation} found"
                    )
                )
        if "overflow" in item.keywords:
            _msg = overflow_unavailable_reason(rocprof_config)
            if _msg is not None:
                item.add_marker(pytest.mark.skip(reason=_msg))
        if "attach" in item.keywords:
            _msg = attach_unavailable_reason(rocprof_config)
            if _msg is not None:
                item.add_marker(pytest.mark.skip(reason=_msg))
        if "python" in item.keywords:
            _msg = python_base_unavailable_reason(rocprof_config)
            if _msg is not None:
                item.add_marker(pytest.mark.skip(reason=_msg))
            _msg = python_versions_unavailable_reason(rocprof_config)
            if _msg is not None:
                item.add_marker(pytest.mark.skip(reason=_msg))
        if "julia" in item.keywords:
            _msg = julia_unavailable_reason(rocprof_config)
            if _msg is not None:
                item.add_marker(pytest.mark.skip(reason=_msg))
        if "xnack" in item.keywords:
            _msg = xnack_unavailable_reason(rocprof_config)
            if _msg is not None:
                item.add_marker(pytest.mark.skip(reason=_msg))
        if "no_docker" in item.keywords:
            _msg = no_docker_unavailable_reason(rocprof_config)
            if _msg is not None:
                item.add_marker(pytest.mark.skip(reason=_msg))
        if "shmem" in item.keywords:
            _msg = shmem_unavailable_reason(rocprof_config)
            if _msg is not None:
                item.add_marker(pytest.mark.skip(reason=_msg))
        if "nic" in item.keywords:
            _msg = nic_unavailable_reason(rocprof_config)
            if _msg is not None:
                item.add_marker(pytest.mark.skip(reason=_msg))
        if "ainic" in item.keywords:
            _msg = ainic_unavailable_reason(rocprof_config)
            if _msg is not None:
                item.add_marker(pytest.mark.skip(reason=_msg))
        if "kfd" in item.keywords or "unified_memory" in item.keywords:
            _msg = kfd_unavailable_reason(rocprof_config)
            if _msg is not None:
                item.add_marker(pytest.mark.skip(reason=_msg))
        if "rocm_min_version" in item.keywords:
            req_version = item.get_closest_marker("rocm_min_version").args[0]
            system_version = rocprof_config.rocm_version
            if system_version is None:
                item.add_marker(pytest.mark.skip(reason="ROCm version not found"))
            # Parse min_version and compare
            min_parts = req_version.split(".")
            min_tuple = tuple(int(p) for p in (min_parts + ["0", "0"])[:3])
            if system_version < min_tuple:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"ROCm {'.'.join(map(str, system_version))} < required {req_version}"
                    )
                )
        if "oshrun_min_version" in item.keywords:
            req_version = item.get_closest_marker("oshrun_min_version").args[0]
            system_version = rocprof_config.capabilities.oshrun_version
            if system_version is None:
                item.add_marker(pytest.mark.skip(reason="OpenSHMEM version not found"))
            else:
                min_parts = req_version.split(".")
                min_tuple = tuple(int(p) for p in (min_parts + ["0", "0"])[:2])
                if system_version < min_tuple:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"oshrun version {'.'.join(map(str, system_version))} < required {req_version}"
                        )
                    )
        if "run_if_gpu_category" in item.keywords:
            _gpu_msg = gpu_unavailable_reason()
            if _gpu_msg is not None:
                item.add_marker(pytest.mark.skip(reason=_gpu_msg))
            expr = item.get_closest_marker("run_if_gpu_category").args[0]
            try:
                result = eval(expr, {"__builtins__": {}}, gpu_category_eval_context())
                if not result:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"GPU category condition '{expr}' is False"
                        )
                    )
            except Exception as e:
                pytest.exit(f"Invalid run_if_gpu_category expression: {e}", returncode=1)
        if "multi_gpu" in item.keywords:
            num_gpu = item.get_closest_marker("multi_gpu").args[0]
            info = get_gpu_info()
            if info.device_count < num_gpu:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"Test requires atleast {num_gpu} GPUs but system has {info.device_count}"
                    )
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


def pytest_collection_finish(session):
    """Generate CTest definitions after collection."""
    if session.config.getoption("--ctest-mode", default="off") == "generate":
        raw_path = session.config.getoption("--ctest-output-path", default=None)
        output_path = Path(raw_path) if raw_path else None
        _ctest_generate_tests(session.items, output_path)


# ----------------------------------------------------------------------------
# Test execution hooks
# ----------------------------------------------------------------------------


# TODO: Deprecate once TheRock switches to CTest
@pytest.hookimpl(hookwrapper=True)  # Allows yield
def pytest_runtest_makereport(item, call):
    """Build runner output and attach to report."""
    outcome = yield
    rep = outcome.get_result()

    setattr(item, f"rep_{rep.when}", rep)

    # Relevant flags
    config = item.config
    show_output_flag = config.stash.get(_show_output_key, False)
    show_on_subfail_flag = config.stash.get(_show_on_subfail_key, False)

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


# TODO: Deprecate once TheRock switches to CTest
def pytest_runtest_logreport(report):
    """Handle output display for passing tests."""
    # Determine if we should show runner output
    config = getattr(pytest, "_config_ref", None)
    show_output_flag = config.stash.get(_show_output_key, False) if config else False
    if show_output_flag and report.when == "call" and report.passed:
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
    """Code that runs after tests complete

    In CTest mode, map "all skipped" to exit code SKIP_RETURN_CODE
    so that CTest can distinguish skipped from passed (via SKIP_RETURN_CODE).
    """
    if (
        session.config.getoption("--ctest-mode", default="off") == "run"
        and exitstatus == 0
    ):
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            passed = len(reporter.stats.get("passed", []))
            skipped = len(reporter.stats.get("skipped", []))
            if passed == 0 and skipped > 0:
                session.exitstatus = SKIP_RETURN_CODE


# ============================================================================
#
# Helper functions
#
# ============================================================================

# ----------------------------------------------------------------------------
# Collection-time availability: return None if OK, else a skip reason string.
# ----------------------------------------------------------------------------


def overflow_unavailable_reason(rocprof_config: RocprofsysConfig) -> Optional[str]:
    caps = rocprof_config.capabilities
    if caps.perf_event_paranoid <= 3 or caps.cap_sys_admin or caps.cap_perfmon:
        return None
    return "Requires either perf_event_paranoid <= 3, CAP_SYS_ADMIN, or CAP_PERFMON to be available"


def gpu_unavailable_reason() -> Optional[str]:
    gpu_info = get_gpu_info()
    if gpu_info is not None and gpu_info.available:
        return None
    return "No valid GPU available"


def annotate_unavailable_reason(rocprof_config: RocprofsysConfig) -> Optional[str]:
    msg = overflow_unavailable_reason(rocprof_config)
    if msg is not None:
        return msg
    if not rocprof_config.capabilities.papi_availability:
        return "PAPI not available"
    return None


def attach_unavailable_reason(rocprof_config: RocprofsysConfig) -> Optional[str]:
    if rocprof_config.capabilities.ptrace_scope == 0:
        return None
    return (
        "Requires ptrace_scope to be 0. Run 'echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope' "
        "to enable attaching to process"
    )


def nic_unavailable_reason(rocprof_config: RocprofsysConfig) -> Optional[str]:
    caps = rocprof_config.capabilities
    if caps.papi_nic_events is not None and caps.perf_event_paranoid <= 2:
        return None
    return "Requires PAPI network events and perf_event_paranoid <= 2 to be available"


def ainic_unavailable_reason(rocprof_config: RocprofsysConfig) -> Optional[str]:
    """Check if AI NIC tracking is available.

    Requires ``amd-smi static`` to report at least one NETDEV entry.
    """
    if not rocprof_config.capabilities.ai_nic_devices:
        return "No AI NIC devices found (amd-smi static reports no NETDEV entries)"
    return None


def kfd_unavailable_reason(rocprof_config: RocprofsysConfig) -> Optional[str]:
    sdk = rocprof_config.capabilities.rocprofiler_sdk_version
    if sdk is not None and sdk >= KFD_MIN_SDK_VERSION:
        return None
    _req = ".".join(map(str, KFD_MIN_SDK_VERSION))
    _found = ".".join(map(str, sdk)) if sdk is not None else "not found"
    return (
        f"Requires rocprofiler-sdk minimum {_req}, but system detected version {_found}"
    )


# TODO: Deprecate once TheRock switches to CTest and CTest based filtering
def mpi_unavailable_reason(
    rocprof_config: RocprofsysConfig, config: pytest.Config
) -> Optional[str]:
    if rocprof_config.capabilities.mpiexec_exec is None:
        return "MPI not available"
    if config.getoption("--ci-mode", default=False):
        return "MPI tests are not run in --ci-mode"
    return None


def python_base_unavailable_reason(rocprof_config: RocprofsysConfig) -> Optional[str]:
    if rocprof_config.rocprofsys_python is not None:
        return None
    return "rocprof-sys-python binary not found"


def python_versions_unavailable_reason(rocprof_config: RocprofsysConfig) -> Optional[str]:
    if (
        rocprof_config.capabilities.supported_python_versions is not None
        and os.environ.get("ROCPROFSYS_USE_PYTHON", "ON").upper() == "ON"
    ):
        return None
    return (
        "No supported Python versions. Each version needs a corresponding "
        "libpyrocprofsys.<IMPL>-<VERSION>-<ARCH>-<OS>-<ABI>.so in site-packages/rocprofsys."
    )


def julia_unavailable_reason(rocprof_config: RocprofsysConfig) -> Optional[str]:
    if rocprof_config.capabilities.julia_exec:
        return None
    return "Julia not available"


def xnack_unavailable_reason(rocprof_config: RocprofsysConfig) -> Optional[str]:
    if get_xnack_support(rocprof_config.rocm_path):
        return None
    return "XNACK not supported"


def no_docker_unavailable_reason(rocprof_config: RocprofsysConfig) -> Optional[str]:
    if not rocprof_config.capabilities.is_inside_docker:
        return None
    return "Test cannot run inside a Docker container"


def shmem_unavailable_reason(rocprof_config: RocprofsysConfig) -> Optional[str]:
    if rocprof_config.capabilities.oshrun_exec:
        return None
    return "SHMEM not available"


# ----------------------------------------------------------------------------
# CTest generator functions
# ----------------------------------------------------------------------------


def _cmake_escape(s: str) -> str:
    """Escape a string for use inside CMake double-quoted arguments."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _ctest_item_ctest_identity(item: pytest.Item) -> tuple[str, str, str]:
    """Return ``(original_nodeid, item name, CTest nodeid fragment)`` for CMake generation."""
    test_id = item.stash.get(_original_nodeid_key, item.nodeid)
    test_name = item.name
    if "::" in test_id:
        file_part, _, rest = test_id.partition("::")
        test_nodeid = f"{Path(file_part).name}::{rest}"
    else:
        test_nodeid = Path(test_id).name
    return test_id, test_name, test_nodeid


def _emit_ctest_header_block() -> list[str]:
    """CMake preamble for generated CTestTestfile.cmake (env, paths, pytest/python discovery)."""
    return [
        "# Auto-generated CTest definitions from rocprofiler-systems pytest suite",
        "# DO NOT EDIT — regenerate via: pytest <dir> --ctest-mode=generate",
        "#",
        "# Run with: ctest --test-dir <directory containing this file>",
        "#",
        "# Supported environment variables:",
        "#   ROCPROFSYS_TEST_DIR         - Path to test package directory or .pyz file",
        "#   ROCPROFSYS_TEST_EXECUTABLE  - Python or pytest executable to use",
        "#   ROCPROFSYS_PYTHON_HINTS     - Additional search paths for versioned Python interpreters",
        "#",
        "",
        "if(DEFINED ENV{ROCPROFSYS_TEST_DIR} AND NOT DEFINED ROCPROFSYS_TEST_DIR)",
        '    set(ROCPROFSYS_TEST_DIR "$ENV{ROCPROFSYS_TEST_DIR}")',
        "endif()",
        "if(DEFINED ENV{ROCPROFSYS_TEST_EXECUTABLE} AND NOT DEFINED ROCPROFSYS_TEST_EXECUTABLE)",
        '    set(ROCPROFSYS_TEST_EXECUTABLE "$ENV{ROCPROFSYS_TEST_EXECUTABLE}")',
        "endif()",
        "if(DEFINED ENV{ROCPROFSYS_PYTHON_HINTS} AND NOT DEFINED ROCPROFSYS_PYTHON_HINTS)",
        '    set(ROCPROFSYS_PYTHON_HINTS "$ENV{ROCPROFSYS_PYTHON_HINTS}")',
        "endif()",
        "",
        "execute_process(COMMAND pwd OUTPUT_VARIABLE _CTEST_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)",
        "if(NOT DEFINED ROCPROFSYS_TEST_DIR)",
        '    set(ROCPROFSYS_TEST_DIR "${_CTEST_DIR}")',
        "endif()",
        "",
        'set(_INSTALL_PATH "${ROCPROFSYS_TEST_DIR}/rocprofsys-tests.pyz")',
        'set(_BUILD_PATH "${ROCPROFSYS_TEST_DIR}/../share/rocprofiler-systems/tests/pytest/")',
        'set(_TEST_ARGS "-s" "--ctest-mode" "run")',
        "",
        'if(EXISTS "${_INSTALL_PATH}")',
        "    if(NOT DEFINED ROCPROFSYS_TEST_EXECUTABLE)",
        "        find_program(ROCPROFSYS_TEST_EXECUTABLE NAMES python3 python HINTS ${ROCPROFSYS_PYTHON_HINTS})",
        "    endif()",
        "    if(NOT ROCPROFSYS_TEST_EXECUTABLE)",
        "        message(FATAL_ERROR",
        '            "python executable not found. "',
        '            "Set ROCPROFSYS_TEST_EXECUTABLE to the correct path "',
        '            "or provide ROCPROFSYS_PYTHON_HINTS to search for the executable.")',
        "    endif()",
        '    set(_ROCPROFSYS_EXE "${ROCPROFSYS_TEST_EXECUTABLE}")',
        '    set(_ROCPROFSYS_EXE_ARGS "${_INSTALL_PATH}")',
        '    set(_ROCPROFSYS_NODEID_PFX "")',
        '    set(_ROCPROFSYS_EXTRA_ARGS "${_TEST_ARGS}")',
        'elseif(EXISTS "${_BUILD_PATH}")',
        "    if(NOT DEFINED ROCPROFSYS_TEST_EXECUTABLE)",
        "        find_program(ROCPROFSYS_TEST_EXECUTABLE NAMES pytest pytest3 HINTS ${ROCPROFSYS_PYTHON_HINTS})",
        "    endif()",
        "    if(NOT ROCPROFSYS_TEST_EXECUTABLE)",
        "        message(FATAL_ERROR",
        '            "pytest executable not found. "',
        '            "Set ROCPROFSYS_TEST_EXECUTABLE to the correct path "',
        '            "or provide ROCPROFSYS_PYTHON_HINTS to search for the executable.")',
        "    endif()",
        '    set(_ROCPROFSYS_EXE "${ROCPROFSYS_TEST_EXECUTABLE}")',
        '    set(_ROCPROFSYS_EXE_ARGS "")',
        '    set(_ROCPROFSYS_NODEID_PFX "${_BUILD_PATH}")',
        '    set(_ROCPROFSYS_EXTRA_ARGS "${_TEST_ARGS}")',
        "else()",
        '    message(FATAL_ERROR "Cannot find test package. Set ROCPROFSYS_TEST_DIR=/path/to/rocprofsys-tests.pyz")',
        "endif()",
        "",
        "if(DEFINED ENV{ROCPROFSYS_CI_TIMEOUT})",
        '    set(_ROCPROFSYS_CI_TIMEOUT "$ENV{ROCPROFSYS_CI_TIMEOUT}")',
        "endif()",
        "",
    ]


def _emit_prerequisite_block() -> list[str]:
    """``rocprofiler-systems-pytest-config`` prerequisite test (global tmp fixture setup)."""
    return [
        'add_test("rocprofiler-systems-pytest-config" "${_ROCPROFSYS_EXE}"'
        ' "${_ROCPROFSYS_EXE_ARGS}"'
        ' "${_ROCPROFSYS_NODEID_PFX}" "--show-config-only")',
        'set_tests_properties("rocprofiler-systems-pytest-config" PROPERTIES',
        '    FIXTURES_SETUP "rocprofsys-global-tmp-files"',
        '    LABELS "prerequisite;global"',
        "    TIMEOUT 10",
        ")",
        "",
    ]


def _emit_cleanup_block() -> list[str]:
    """``rocprofiler-systems-test-cleanup`` (global tmp fixture cleanup)."""
    return [
        'add_test("rocprofiler-systems-test-cleanup" "${_ROCPROFSYS_EXE}"'
        ' "${_ROCPROFSYS_EXE_ARGS}"'
        ' "${_ROCPROFSYS_NODEID_PFX}" "--ctest-mode" "cleanup")',
        'set_tests_properties("rocprofiler-systems-test-cleanup" PROPERTIES',
        '    FIXTURES_CLEANUP "rocprofsys-global-tmp-files"',
        '    LABELS "cleanup;global"',
        "    TIMEOUT 30",
        ")",
        "",
    ]


def _emit_test_timeout_block(
    item: pytest.Item, timeout_buffer: int = CTEST_TIMEOUT_BUFFER
) -> list[str]:
    """One CMake block: set ``_TEST_TIMEOUT`` from ``ROCPROFSYS_CI_TIMEOUT`` or pytest timeout (+ buffer)."""
    timeout_marker = item.get_closest_marker("timeout")
    timeout = (
        int(timeout_marker.args[0])
        if timeout_marker and timeout_marker.args
        else DEFAULT_TIMEOUT
    )
    default_timeout = timeout + timeout_buffer
    return [
        "if(DEFINED _ROCPROFSYS_CI_TIMEOUT)",
        f'    math(EXPR _TEST_TIMEOUT "${{_ROCPROFSYS_CI_TIMEOUT}} + {timeout_buffer}")',
        "else()",
        f"    set(_TEST_TIMEOUT {default_timeout})",
        "endif()",
        "",
    ]


def _emit_test_item_block(
    item: pytest.Item,
    labels: set[str],
    depends_on: list[str],
    run_serial: bool,
) -> list[str]:
    """``add_test`` + ``set_tests_properties`` for one item (timeout block emitted separately)."""
    _, test_name, test_nodeid = _ctest_item_ctest_identity(item)
    escaped_name = _cmake_escape(test_name)
    escaped_nodeid = _cmake_escape(test_nodeid)

    # Check if the test runs on a specific python version
    extra_args = ""
    if hasattr(item, "callspec") and "python_version" in item.callspec.params:
        py_ver = item.callspec.params["python_version"]
        if py_ver is not None:
            extra_args += f' "--python-versions={py_ver}"'

    lines_out: list[str] = [
        f'add_test("{escaped_name}" "${{_ROCPROFSYS_EXE}}"'
        f' "${{_ROCPROFSYS_EXE_ARGS}}"'
        f' "${{_ROCPROFSYS_NODEID_PFX}}{escaped_nodeid}"'
        f"{extra_args} ${{_ROCPROFSYS_EXTRA_ARGS}})"
    ]
    props: list[str] = []
    if labels:
        props.append(f'    LABELS "{";".join(sorted(labels))}"')
    props.append("    TIMEOUT ${_TEST_TIMEOUT}")
    props.append(f"    SKIP_RETURN_CODE {SKIP_RETURN_CODE}")
    props.append('    FIXTURES_REQUIRED "rocprofsys-global-tmp-files"')
    if run_serial:
        props.append("    RUN_SERIAL TRUE")
    if depends_on:
        deps_str = ";".join(_cmake_escape(d) for d in depends_on)
        props.append(f'    DEPENDS "{deps_str}"')

    lines_out.append(f'set_tests_properties("{escaped_name}" PROPERTIES')
    lines_out.extend(props)
    lines_out.append(")")
    lines_out.append("")
    return lines_out


def _ctest_generate_tests(
    items: list[pytest.Item], output_path: Optional[Path] = None
) -> None:
    """Generate a CTestTestfile.cmake file and print it to stdout."""

    no_report_markers = {
        "parametrize",  # Ignored, except for "mode" parameter (instrumentation mode)
        # Pytest built-in
        "usefixtures",
        "filterwarnings",
        "skipif",
        "skip",
        "xfail",
        # Internal markers
        "python_versions",
        "ci_enable",
        "ci_disable",
        "mpi_optional",
        "no_docker",
        "oshrun_min_version",
        "rocm_min_version",
        "run_if_gpu_category",
        "preserve",
        # For CTests
        "timeout",
        "depends_on",
        "serialize",
        "class_name",
    }
    no_report_args_markers = {"rocpd"}
    only_report_args_markers = {"mpi_implementation"}

    lines = _emit_ctest_header_block()
    lines.extend(_emit_prerequisite_block())

    seen_names: dict[str, str] = {}  # escaped_name -> original nodeid

    for item in items:
        test_id, test_name, _ = _ctest_item_ctest_identity(item)

        # Handle certain markers that affect how CTest is configured

        labels: set[str] = set()
        depends_on: list[str] = []
        run_serial = False

        depends_marker = item.get_closest_marker("depends_on")
        if depends_marker:
            depends_on.extend(str(arg) for arg in depends_marker.args)

        if item.get_closest_marker("serialize"):
            run_serial = True

        if hasattr(item, "callspec") and "mode" in item.callspec.params:
            labels.add(str(item.callspec.params["mode"]))

        # Translate pytest markers to CTest labels

        for marker in item.iter_markers():
            if marker.name in no_report_markers:
                continue
            if marker.name in only_report_args_markers:
                for arg in marker.args:
                    labels.add(str(arg))
                continue
            if marker.name in no_report_args_markers or not marker.args:
                labels.add(marker.name)
            else:
                args_str = ", ".join(str(a) for a in marker.args)
                labels.add(f"{marker.name}[{args_str}]")

        escaped_name = _cmake_escape(test_name)

        if escaped_name in seen_names:
            pytest.exit(
                f"\nDuplicate CTest name '{escaped_name}' generated from:\n"
                f"  1) {seen_names[escaped_name]}\n"
                f"  2) {test_id}\n"
                f"(Due to _standardize_test_name or parametrization)\n"
                f"Rework test name or parametrization to produce unique names.",
                returncode=1,
            )
        seen_names[escaped_name] = test_id

        lines.extend(_emit_test_timeout_block(item))
        lines.extend(
            _emit_test_item_block(
                item,
                labels,
                depends_on,
                run_serial,
            )
        )

    lines.extend(_emit_cleanup_block())

    content = "\n".join(lines)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"Generated {len(items)} CTest definitions -> {output_path}")
    else:
        print(content)
    pytest.exit("CTest generation complete", returncode=0)


# ----------------------------------------------------------------------------
# Other helpers
# ----------------------------------------------------------------------------


# TODO: Deprecate once TheRock switches to CTest and CTest based filtering
def configure_mode(config: pytest.Config) -> None:
    """Configure the mode based on the command line options.

    Modes:
     - --ci-mode: CI mode
     - --ctest-mode: CTest integration mode
     - --dev: Developer mode
    """

    # MPI is disabled in CI mode, this is done in collection_modifyit
    ci_mode = config.getoption("--ci-mode", default=False)
    ctest_mode = config.getoption("--ctest-mode", default="off") == "run"
    dev_mode = config.getoption("--dev", default=False)

    if ci_mode or ctest_mode:
        config.option.verbose = max(config.option.verbose, 1)  # -v
        config.option.tbstyle = "short"  # --tb=short
        if "s" not in config.option.reportchars:  # -rs
            config.option.reportchars += "s"

    if ctest_mode:
        config.option.no_header = True
        config.option.show_test_output = "all"

    if ci_mode:
        config.option.show_config = True
        config.option.show_test_output = "subtest"

    if dev_mode:
        config.option.show_config = True
        config.option.show_test_output = "subtest"
        config.option.verbose = max(config.option.verbose, 1)  # -v
        config.option.tbstyle = "short"  # --tb=short
        if "s" not in config.option.reportchars:  # -rs
            config.option.reportchars += "s"


def _standardize_test_name(
    item: pytest.Item, config: pytest.Config, verbose: bool = False
) -> None:

    # Strip test prefix from the test method name
    test_name = item.name
    if test_name.startswith("test"):
        test_name = test_name[4:]
        if test_name.startswith(("_", "-")):
            test_name = test_name[1:]

    ctest_mode = config.getoption("--ctest-mode", default="off") in ("generate", "run")
    class_name = None
    if ctest_mode:
        name_marker = item.get_closest_marker("class_name")
        if name_marker and name_marker.args:
            class_name = str(name_marker.args[0]).strip()

    if class_name:
        full_name = f"{class_name}-{test_name}"
    elif item.cls:
        py_class = item.cls.__name__
        if py_class.startswith("Test"):
            py_class = py_class[4:]
        full_name = f"{py_class}-{test_name}"
    else:
        full_name = test_name

    formatted_name = "".join(c if c.isalnum() or c == "." else "-" for c in full_name)

    if ctest_mode:
        formatted_name = formatted_name.replace("_", "-")
        while "--" in formatted_name:
            formatted_name = formatted_name.replace("--", "-")
        formatted_name = formatted_name.strip("-")
        formatted_name = formatted_name.lower()
    else:
        # TODO: Deprecate once TheRock switches to CTests
        formatted_name = formatted_name.replace("-", "_")
        while "__" in formatted_name:
            formatted_name = formatted_name.replace("__", "_")
        formatted_name = formatted_name.strip("_")

    item.stash[_original_nodeid_key] = item.nodeid
    # nodeid is what is used to display the test name in the terminal
    # By default, it groups it by module. In verbose, it shows the full path + class + method
    # To get a cleaner output in verbose mode, we modify the nodeid but only if verbose is True
    # This avoids breaking the default grouping by module in non-verbose mode
    if verbose:
        item._nodeid = formatted_name
    item.name = formatted_name

    # Allow -k filtering by the formatted name
    item.extra_keyword_matches.add(formatted_name)
    item.extra_keyword_matches.add(formatted_name.lower())


def _generate_rocprofsys_config_header() -> list[str]:
    try:
        rocprof_config = get_rocprof_config()
        cap = rocprof_config.capabilities
    except Exception as e:
        return [f"{e}"]

    gpu_info = get_gpu_info()

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

    if cap.oshrun_version is not None:
        oshrun_version_str = f"{cap.oshrun_version[0]}.{cap.oshrun_version[1]}"
    else:
        oshrun_version_str = "Not found"

    # Rocprofiler SDK version
    rocprofiler_sdk_version_str = (
        f"{cap.rocprofiler_sdk_version[0]}.{cap.rocprofiler_sdk_version[1]}.{cap.rocprofiler_sdk_version[2]}"
        if cap.rocprofiler_sdk_version
        else "Not found"
    )

    W = 22  # label width for alignment

    def _row(label: str, value) -> str:
        return f"  {label:<{W}}{value}"

    def _subrow(label: str, value) -> str:
        return f"    {label:<{W}}{value}"

    header = [
        "",
        "=" * 70,
        "Test Configuration:",
        "=" * 70,
        _row("ROCm version:", rocm_version),
        _row("ROCprof-SDK version:", rocprofiler_sdk_version_str),
        _row("ROCm path:", rocprof_config.rocm_path),
        _row("Is installed:", rocprof_config.is_installed),
        _row("Output dir:", rocprof_config.test_output_dir),
        _row("Validate ROCPD:", check_use_rocpd()),
        _row("Validate Perfetto:", check_use_perfetto()),
        "-" * 70,
        "Core Executables:",
        _row("Instrument:", rocprof_config.rocprofsys_instrument),
        _row("Run:", rocprof_config.rocprofsys_run),
        _row("Sample:", rocprof_config.rocprofsys_sample),
        _row("Avail:", rocprof_config.rocprofsys_avail),
        _row("Causal:", rocprof_config.rocprofsys_causal),
        _row("Python:", rocprof_config.rocprofsys_python),
        "-" * 70,
        "Executables:",
        _row("MPI:", cap.mpiexec_exec),
        _subrow("Implementation:", cap.mpi_implementation),
        _row("Julia:", cap.julia_exec),
        _row("Oshrun:", cap.oshrun_exec),
        _subrow("Version:", oshrun_version_str),
        _row("Offload tool:", offload_msg),
        _row("Rocminfo:", rocminfo_path if rocminfo_path else rocminfo_err_msg),
        "-" * 70,
        "System Capabilities:",
        _row("Detected num procs:", cap.num_procs),
        _row("UCX available:", cap.ucx_availability),
        _row("Perf event paranoid:", cap.perf_event_paranoid),
        _row("CAP_SYS_ADMIN:", cap.cap_sys_admin),
        _row("CAP_PERFMON:", cap.cap_perfmon),
        _row("Ptrace scope:", cap.ptrace_scope),
        _row("Is inside docker:", rocprof_config.capabilities.is_inside_docker),
        _row("PAPI available:", cap.papi_availability),
        _row("AI NIC devices:", cap.ai_nic_devices),
        _row("Default NIC:", cap.default_nic),
        *(
            lambda evts: (
                [_row("PAPI NIC events:", evts[0])] + [_row("", e) for e in evts[1:]]
                if evts
                else [_row("PAPI NIC events:", "None")]
            )
        )(cap.papi_nic_events.split() if cap.papi_nic_events else []),
        "-" * 70,
        "GPU Information:",
        _row("Available:", gpu_info.available),
        _row("Architectures:", gpu_info.architectures or "None"),
        _row("Device count:", gpu_info.device_count),
        _row("Categories:", gpu_info.categories or "None"),
        _row("XNACK support:", xnack_support),
        "-" * 70,
        "Directories:",
        _row("Build dir:", rocprof_config.rocprofsys_build_dir),
        _row("Lib dir:", rocprof_config.rocprofsys_lib_dir),
        _row("Bin dir:", rocprof_config.rocprofsys_bin_dir),
        _row("Tests dir:", rocprof_config.rocprofsys_tests_dir),
        _row("Examples dir:", rocprof_config.rocprofsys_examples_dir),
        _row("Validation dir:", rocprof_config.rocpd_validation_rules),
        "-" * 70,
        "Python:",
        _row("Site packages:", rocprof_config.rocprofsys_site_packages),
    ]
    if cap.supported_python_versions and cap.supported_python_executables:
        for version, exe in zip(
            cap.supported_python_versions,
            cap.supported_python_executables,
        ):
            header.append(_row(version, exe))
    else:
        header.append(
            _row(
                "Executables:",
                "(no supported Python versions found — each version needs a "
                "libpyrocprofsys.<IMPL>-<VERSION>-<ARCH>-<OS>-<ABI>.so in site-packages/rocprofsys)",
            )
        )
    header.extend(["-" * 70, "System Environment:"])
    for key, value in sorted(rocprof_config.get_fundamental_environment().items()):
        header.append(_row(f"{key}:", value))
    header.extend(["=" * 70, ""])
    return header


def add_marker_if(
    item,
    marker_to_add: str,
    cond: Callable[[], bool] = lambda: True,
    req_mark: Optional[str] = None,
    skip_reason: Optional[str] = None,
    unavailable_reason: Optional[Callable[[], Optional[str]]] = None,
) -> None:
    """Add a marker to a test item if:
        - target_marker is present (or not specified)
        - AND condition evaluates to True (lambda)

    If ``unavailable_reason`` is set, it is called: ``None`` means add the marker;
    a non-empty string means skip the test with that reason (preferred over cond/skip_reason).

    If condition is False and skip_reason is provided, add a skip marker instead.
    """
    if req_mark and not item.get_closest_marker(req_mark):
        return

    if unavailable_reason is not None:
        msg = unavailable_reason()
        if msg is None:
            item.add_marker(getattr(pytest.mark, marker_to_add))
        else:
            item.add_marker(pytest.mark.skip(reason=msg))
        return

    if cond():
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
        python_versions = None
        python_root_dirs = None
        custom_output_dir = None
        rocm_optional = False
        if pytest_config:
            custom_output_dir = pytest_config.getoption("--output-dir", default=None)
            ver_str = pytest_config.getoption("--python-versions", default=None)
            dir_str = pytest_config.getoption("--python-root-dirs", default=None)
            # When generating the CTestTestfile.cmake in TheRock, ROCm is not present
            rocm_optional = (
                pytest_config.getoption("--ctest-mode", default="off") == "generate"
            )
            if ver_str:
                python_versions = [v.strip() for v in ver_str.split(";") if v.strip()]
            if dir_str:
                python_root_dirs = [
                    Path(d.strip()) for d in dir_str.split(";") if d.strip()
                ]

        return discover_build_config(
            output_dir=Path(custom_output_dir) if custom_output_dir else None,
            python_versions=python_versions,
            python_root_dirs=python_root_dirs,
            rocm_optional=rocm_optional,
        )
    except Exception as e:
        raise RuntimeError("Failed to get rocprofiler-systems configuration") from e


@lru_cache(maxsize=1)
def get_gpu_info() -> GPUInfo:
    """Return the GPU information."""
    try:
        rocprof_config = get_rocprof_config()
    except Exception as e:
        pytest.exit(f"{e}")
    return detect_gpu(rocprof_config.rocm_path)


def _run_cleanup(config: pytest.Config) -> None:
    """Run cleanup of temp files and optionally the test output directory."""
    import glob
    import getpass

    # Clean up temp files
    for pattern in _cleanup_temp_patterns():
        for filepath in glob.glob(pattern):
            try:
                p = Path(filepath)
                if p.is_file() and p.owner() == getpass.getuser():
                    p.unlink()
                    print(f"Removed: {filepath}")
            except (OSError, KeyError):
                pass

    # Clean up test output directory if ROCPROFSYS_KEEP_TEST_OUTPUT=0
    if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "1") == "0":
        try:
            rocprof_config = get_rocprof_config()
            output_dir = rocprof_config.test_output_dir
            if output_dir.exists():
                shutil.rmtree(output_dir)
                print(f"Removed test output directory: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not clean test output directory: {e}")


def _cleanup_temp_patterns() -> list[str]:
    """Return list of rocprofiler-systems temp file patterns to clean up."""
    tmpdir = os.environ.get("ROCPROFSYS_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
    dirs = ["/tmp"]
    if tmpdir and not tmpdir.startswith("%") and tmpdir != "/tmp":
        dirs.append(tmpdir)

    patterns = []
    for d in dirs:
        patterns.extend(
            [
                f"{d}/rocprof-sys-*.tmp",
                f"{d}/rocprofsys-*.tmp",
                f"{d}/buffered_storage*.bin",
                f"{d}/metadata*.json",
                f"{d}/perfetto-*.proto",
                f"{d}/perfetto_trace*.proto",
                f"{d}/hsa-*.tmp",
                f"{d}/rocm-*.tmp",
                f"{d}/hip-*.tmp",
                f"{d}/*.inst",
                f"{d}/causal-*.json",
                f"{d}/experiments-*.coz",
                f"{d}/core.*",
            ]
        )
    return patterns


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
        "ROCPROFSYS_LOG_LEVEL": "info",
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
    output_dir = rocprof_config.test_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# TODO: Deprecate once TheRock switches to CTest
@pytest.fixture(scope="module", autouse=True)
def cleanup_module_temp_files(request: pytest.FixtureRequest):
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

    if not request.config.getoption("--ctest-mode", default="off") == "run":
        for pattern in ["/tmp/buffered_storage*.bin", "/tmp/metadata*.json"]:
            for filepath in glob.glob(pattern):
                safe_remove(Path(filepath))


# ----------------------------------------------------------------------------
# Function-scoped Fixtures
# ----------------------------------------------------------------------------


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
    Directory is removed if it is empty.

    Cleanup Order:
        1. Test setup: Directory is created
        2. Test body: Runner executes, output files are written
        3. Test body: Validation happens on output files
        4. Test body: Assertions complete
        5. Test teardown: This fixture cleans up the directory (AFTER yield)

    This ensures validation always has access to output files.
    """
    output_dir = test_output_base / request.node.name

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    yield output_dir  # Test body executes here (including validation)

    # === CLEANUP PHASE (runs AFTER test body completes) ===

    # If the output directory is empty, remove it
    if output_dir.exists() and not any(output_dir.iterdir()):
        shutil.rmtree(output_dir)

    keep_output = os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "1") == "1"
    test_failed = hasattr(request.node, "rep_call") and request.node.rep_call.failed

    if keep_output or test_failed or not output_dir.exists():
        return

    # Remove all files in the output directory, then the directory itself
    # unless the preserve marker is present

    to_preserve = []
    preserve_marker = request.node.get_closest_marker("preserve")
    if preserve_marker and preserve_marker.args:
        for fname in preserve_marker.args:
            preserved = output_dir / fname
            to_preserve.append(preserved)

    for entry in output_dir.iterdir():
        if entry not in to_preserve:
            safe_remove(entry)

    # Remove the output directory if empty
    if not any(output_dir.iterdir()):
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


# This is needed for pytest-subtests plugin compatibility when pytest < 9.0.0
@pytest.fixture
def record_subtest_failure(request):
    """Fixture to record subtest failures for --show-output-on-subtest-fail.

    Used by assert fixtures to track failures with pytest-subtests plugin.
    """

    def _record(name: str):
        request.node.stash.setdefault(_subtest_failures_key, []).append(name)

    return _record


# TODO: Will be default once TheRock switches to CTest based filtering
def _print_subtest_output(request, subtest_name: str, output: str) -> None:
    """Print subtest validation output for important subtests when in CTest run mode."""
    if request.config.getoption("--ctest-mode", default="off") == "run" and output:
        print(f"\n--- {subtest_name} ---\n{output}\n", flush=True)


# TODO: Deprecate once TheRock switches to CTest and CTest based filtering
def _is_assert_disabled(request: pytest.FixtureRequest, subtest_name: str) -> bool:
    """Check if a subtest is disabled via ci_disable marker in CI mode."""
    if not request.config.getoption("--ci-mode", default=False):
        return False
    for marker in request.node.iter_markers("ci_disable"):
        if subtest_name in marker.args:
            return True
    return False


# Contains a set of kwargs accepted for a given (function, mode) pair.
_FUNCTION_ALLOWED_KWARGS: dict[str, dict[str, set[str]]] = {
    "run_test": {
        "baseline": {"command"},
        "sampling": {"sampling_args"},
        "binary_rewrite": {"binary_rewrite_args", "cleanup_on_success"},
        "runtime_instrument": {"runtime_instrument_args"},
        "sys_run": {"sys_run_args"},
        "causal": {"causal_args", "causal_mode"},
        "python": {"python_version", "profile_args", "annotated", "standalone"},
    },
    "assert_regex": {
        "baseline": {"baseline_pass_regex", "baseline_fail_regex"},
        "sampling": {"sampling_pass_regex", "sampling_fail_regex"},
        "binary_rewrite": {"binary_rewrite_pass_regex", "binary_rewrite_fail_regex"},
        "runtime_instrument": {
            "runtime_instrument_pass_regex",
            "runtime_instrument_fail_regex",
        },
        "sys_run": {"sys_run_pass_regex", "sys_run_fail_regex"},
        "causal": {"causal_pass_regex", "causal_fail_regex"},
        "python": {"python_pass_regex", "python_fail_regex"},
    },
}


def _filter_kwargs(function: str, mode: str, **kwargs: Any) -> dict[str, Any]:
    """Filter ``kwargs`` to those accepted by ``function`` for ``mode``.

    This also verifies that the kwargs passed are valid for the given function.
    If a kwarg is not valid, pytest.fail is called.

    Returns:
        A new dict containing only kwargs valid for ``(function, mode)``.
    """
    allowed_per_mode = _FUNCTION_ALLOWED_KWARGS.get(function)
    if allowed_per_mode is None:
        pytest.fail(
            f"_filter_kwargs called with unknown function '{function}'. "
            f"Expected one of: {sorted(_FUNCTION_ALLOWED_KWARGS.keys())}."
        )

    mode_key = mode.replace("-", "_")
    allowed_for_mode = allowed_per_mode.get(mode_key)
    if allowed_for_mode is None:
        pytest.fail(
            f"Unknown mode '{mode}' for '{function}'. "
            f"Expected one of: {sorted(allowed_per_mode.keys())}."
        )

    # Union of every kwarg accepted by any mode of this function. Anything
    # outside this set is considered a typo and an error is raised.
    all_known_for_function: set[str] = set().union(*allowed_per_mode.values())
    unknown = set(kwargs) - all_known_for_function
    if unknown:
        pytest.fail(
            f"{function}: unknown kwargs {sorted(unknown)}. "
            f"Valid kwargs across all modes: {sorted(all_known_for_function)}."
        )

    return {k: v for k, v in kwargs.items() if k in allowed_for_mode}


# ============================================================================
# Base Test Class
# ============================================================================


class RocprofsysTest:
    """Base class that auto-captures parametrized values and common fixtures onto self."""

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        run_test,
        assert_regex,
        assert_perfetto,
        assert_rocpd,
        assert_unified_memory_output,
        assert_causal_json,
        assert_file_exists,
        assert_timemory,
        assert_file_regex,
        get_test_num_threads,
        test_output_dir,
    ):

        self.run_test = run_test
        self.assert_regex = assert_regex
        self.assert_perfetto = assert_perfetto
        self.assert_rocpd = assert_rocpd
        self.assert_unified_memory_output = assert_unified_memory_output
        self.assert_causal_json = assert_causal_json
        self.assert_file_exists = assert_file_exists
        self.assert_timemory = assert_timemory
        self.assert_file_regex = assert_file_regex
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
        launcher: Launcher to use (mpi or shmem)

        num_procs: Number of processes (0 = disabled)
        working_directory: Custom working directory
        check_target_arch: If True, checks if the target supports the current system architectures (default: False)
                           Note: This requires @pytest.mark.gpu to be present
        skip_on_error: If True, pytest.skip on non-zero return code (default: False = fail)
        fail_on_pass: If True, pytest.fail on success and pytest.pass on failure (default: False)
        fail_on_not_found: If True, pytest.fail when binary not found (default: False = skip)
        fail_message: Custom failure message (default: "{runner_type} test failed: {output}")
        no_base_env: If true, don't use the base environment (default: False)
        **kwargs: Additional runner-specific arguments (see _FUNCTION_ALLOWED_KWARGS for valid kwargs)

    Returns:
        TestResult for further assertions
    """

    def _run_test(
        runner_type: str,
        target: str,
        env: Optional[dict[str, str]] = None,
        run_args: Optional[list[str]] = None,
        pre_run_args: Optional[list[str]] = None,
        launcher: Optional[BaselineRunner.Launcher | str] = None,
        num_procs: int = 0,
        working_directory: Optional[Path] = None,
        check_target_arch: bool = False,
        skip_on_error: bool = False,
        fail_on_pass: bool = False,
        fail_on_not_found: bool = False,
        fail_message: Optional[str] = None,
        no_base_env: bool = False,
        **kwargs,
    ) -> TestResult:
        filtered_kwargs = _filter_kwargs("run_test", runner_type, **kwargs)

        if num_procs > 0 and launcher is None:
            pytest.fail(
                f"num_procs={num_procs} was provided but no launcher was set. "
                f"Pass launcher='<launcher_name>' alongside num_procs."
            )

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

        env = env.copy() if env else {}

        # Apply --monochrome option if set
        if request.config.getoption("--monochrome", default=False):
            env["ROCPROFSYS_MONOCHROME"] = "ON"

        # Timeout: ROCPROFSYS_CI_TIMEOUT env, else @pytest.mark.timeout, else default
        ci_timeout_env = os.environ.get("ROCPROFSYS_CI_TIMEOUT")
        if ci_timeout_env is not None:
            timeout = int(ci_timeout_env)
        elif request.node.get_closest_marker("timeout"):
            timeout = request.node.get_closest_marker("timeout").args[0]
        else:
            timeout = 300

        env["ROCPROFSYS_CI_TIMEOUT"] = str(timeout)

        # Verify that MPI is available for "mpi_optional" tests
        if request.node.get_closest_marker("mpi_optional") and num_procs > 0:
            if not request.node.get_closest_marker("mpi"):
                num_procs = 0

        try:
            runner = runner_class(
                config=rocprof_config,
                target=target,
                output_dir=test_output_dir,
                run_args=run_args,
                pre_run_args=pre_run_args,
                env=env,
                timeout=timeout,
                launcher=launcher,
                num_procs=num_procs,
                working_directory=working_directory,
                no_base_env=no_base_env,
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
            short_msg = fail_message or f"{runner_type} test failed"
            ctest_mode = request.config.getoption("--ctest-mode", default="off")

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

            # TODO: This will be made the standard once TheRock switches to CTest
            if ctest_mode == "run":
                # Print details to stdout (captured by CTest) and fail with
                # a short message to avoid the same output appearing twice.
                print(f"\n{detail_text}", flush=True)
                msg = short_msg
            else:
                msg = f"{short_msg}\n\n{detail_text}"

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
              mode-specific regexes from kwargs (see _FUNCTION_ALLOWED_KWARGS for valid kwargs)
        subtest_name: Name shown in subtest output (defaults to "Regex validation")
        pass_regex: Explicit list of pass regex patterns (used if mode is None or no mode-specific found)
        fail_regex: Explicit list of fail regex patterns (used if mode is None or no mode-specific found)
        use_abort_fail_regex: Whether to validate against ROCPROFSYS_ABORT_FAIL_REGEX (default: True)
        skip_on_fail: If True, skip instead of fail when validation fails
        fail_message: Custom message for failure (defaults to validation message)
        **kwargs: Mode-specific regexes (see _FUNCTION_ALLOWED_KWARGS for valid kwargs)
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

        if mode is None and kwargs:
            pytest.fail(
                f"assert_regex received mode-specific kwargs {sorted(kwargs)} but no "
                f"'mode' was provided. Pass mode=... so they can be resolved, or use "
                f"pass_regex/fail_regex directly."
            )

        if mode is not None:
            filtered = _filter_kwargs("assert_regex", mode, **kwargs)
            mode_key = mode.replace("-", "_")
            mode_pass_regex = filtered.get(f"{mode_key}_pass_regex")
            if mode_pass_regex is not None:
                pass_regex = mode_pass_regex
            mode_fail_regex = filtered.get(f"{mode_key}_fail_regex")
            if mode_fail_regex is not None:
                fail_regex = mode_fail_regex

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

    Trace validation kwargs (``categories``, ``labels``, ``counts``, ``depths``,
    ``label_substrings``, etc.) are forwarded to
    ``validate_perfetto_trace``; see that function's docstring.

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
        check_counter_pairing: bool = False,
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
                check_counter_pairing=check_counter_pairing,
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
                        pytest.fail(
                            f"Pass regex not found: {pattern}\n{output}", pytrace=False
                        )
            if fail_regex:
                for pattern in fail_regex:
                    if re.search(pattern, validation.stdout):
                        record_subtest_failure(subtest_name)
                        pytest.fail(
                            f"Fail regex found: {pattern}\n{output}", pytrace=False
                        )
            _print_subtest_output(request, subtest_name, output)

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
        gpu_category_to_skip: GPU categories to skip tagged validation queries for
            (instinct, radeon, apu). Omit or pass empty to run all queries
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
        gpu_category_to_skip: Optional[list[str]] = None,
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
                gpu_category_to_skip=gpu_category_to_skip,
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
                        pytest.fail(
                            f"Pass regex not found: {pattern}\n{output}", pytrace=False
                        )
            if fail_regex:
                for pattern in fail_regex:
                    if re.search(pattern, validation.stdout):
                        record_subtest_failure(subtest_name)
                        pytest.fail(
                            f"Fail regex found: {pattern}\n{output}", pytrace=False
                        )
            _print_subtest_output(request, subtest_name, output)

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
                        pytest.fail(
                            f"Pass regex not found: {pattern}\n{output}", pytrace=False
                        )
            if fail_regex:
                for pattern in fail_regex:
                    if re.search(pattern, validation.stdout):
                        record_subtest_failure(subtest_name)
                        pytest.fail(
                            f"Fail regex found: {pattern}\n{output}", pytrace=False
                        )
            _print_subtest_output(request, subtest_name, output)

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
def assert_unified_memory_output(subtests, tests_dir, record_subtest_failure, request):
    """Fixture that returns an assert_unified_memory_output function."""
    if _is_assert_disabled(request, "assert_unified_memory_output"):
        return lambda *args, **kwargs: None

    def _assert_unified_memory_output(
        result: TestResult,
        subtest_name: str = "Unified-memory output validation",
        timeout: int = 60,
        pass_regex: Optional[list[str]] = None,
        fail_regex: Optional[list[str]] = None,
        skip_on_fail: bool = False,
        fail_message: Optional[str] = None,
    ) -> None:
        with subtests.test(subtest_name):
            validation = validate_unified_memory_outputs(
                result.output_dir,
                tests_dir=tests_dir,
                timeout=timeout,
            )
            output = f"Command: {validation.command}\n\n{validation.message}"
            if not validation.is_valid:
                msg = fail_message or f"Unified-memory validation failed:\n{output}"
                if skip_on_fail:
                    pytest.skip(msg)
                else:
                    record_subtest_failure(subtest_name)
                    pytest.fail(msg)
            if pass_regex:
                for pattern in pass_regex:
                    if not re.search(pattern, validation.stdout):
                        record_subtest_failure(subtest_name)
                        pytest.fail(
                            f"Pass regex not found: {pattern}\n{output}",
                            pytrace=False,
                        )
            if fail_regex:
                for pattern in fail_regex:
                    if re.search(pattern, validation.stdout):
                        record_subtest_failure(subtest_name)
                        pytest.fail(
                            f"Fail regex found: {pattern}\n{output}",
                            pytrace=False,
                        )
            _print_subtest_output(request, subtest_name, output)

    return _assert_unified_memory_output


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
                        pytest.fail(
                            f"Pass regex not found: {pattern}\n{output}", pytrace=False
                        )

            if fail_regex:
                for pattern in fail_regex:
                    if re.search(pattern, validation.stdout):
                        record_subtest_failure(subtest_name)
                        pytest.fail(
                            f"Fail regex found: {pattern}\n{output}", pytrace=False
                        )
            _print_subtest_output(request, subtest_name, output)

    return _assert_causal_json
