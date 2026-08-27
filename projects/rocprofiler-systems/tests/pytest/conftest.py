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

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

# Add the pytest directory to Python path for rocprofsys package
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from pytest import StashKey

from rocprofsys import environment
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
    ValidationResult,
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
from rocprofsys.markers import (
    CTestExport,
    MarkerCtx,
    MarkerRegistry,
    OnUnmet,
    caps,
    requires,
)

# Single registry owned by this conftest. Its register_* methods are bound to
# module-level names so the declarations in the "Marker declarations" section
# read naturally; the engine module itself holds no global state, so unit tests
# can use an independent MarkerRegistry (see rocprofsys/markers.py).
_MARKERS = MarkerRegistry()
register_marker = _MARKERS.register_marker
register_functional_marker = _MARKERS.register_functional_marker
min_version = _MARKERS.min_version
add_marker_dependency_if = _MARKERS.add_marker_dependency_if

# Key for storing the single test result on pytest items
# Item-level stash keys
_result_key: StashKey = StashKey()
_output_printed_key: StashKey[bool] = StashKey()
_original_nodeid_key: StashKey[str] = StashKey()

# GNU convention. Used for CTests
SKIP_RETURN_CODE = 77

# As test + subtests are collapsed into a single pytest, there is only one global timeout that the
# ctest will use as reference.
# The DEFAULT_TIMEOUT is used as the "run_test" timeout and
# the CTEST_TIMEOUT_BUFFER is a fixed amount added to handle subtests + flush + teardown
# CTests set their timeout to DEFAULT_TIMEOUT + CTEST_TIMEOUT_BUFFER
DEFAULT_TIMEOUT = 300
CTEST_TIMEOUT_BUFFER = 30  # Not overridable

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
        "--show-config-only",  # Only used by "rocprofiler-systems-pytest-config" test
        action="store_true",
        default=False,
        help="Show the test configuration and exit without running any tests",
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

    config.option.verbose = max(config.option.verbose, 1)  # -v
    config.option.tbstyle = "short"  # --tb=short
    config.option.no_header = True
    config.option.reportchars += "s"  # -rs

    if config.getoption("--ctest-mode", default="off") == "cleanup":
        _run_cleanup()
        pytest.exit("Cleanup complete", returncode=0)

    if config.getoption("--show-config-only", default=False):
        pytest._config_ref = config
        header = _generate_rocprofsys_config_header()
        for line in header:
            print(line)
        pytest.exit("Header generated", returncode=0)

    # Disable pytest-timeout plugin if detected
    # It will interfere with our timeout marker
    timeout_plugin = config.pluginmanager.get_plugin("timeout")
    if timeout_plugin:
        config.pluginmanager.unregister(timeout_plugin)

    # Register all markers (functional + non-functional) declared in this file.
    #   See the "Marker declarations" section and pytest_collection_modifyitems.
    _MARKERS.register_markers_with_pytest(config)

    # Markers must be registered to be used, enforced via strict_markers below.
    config.option.strict_markers = True

    # Keep a module-level ref for hooks that don't receive config directly
    pytest._config_ref = config


# ----------------------------------------------------------------------------
# Collection hooks
# ----------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests based on markers."""
    marker = metafunc.definition.get_closest_marker("python_versions")
    if marker is not None:
        rocprof_config = get_rocprof_config()
        supported = set(rocprof_config.capabilities.supported_python_versions or [])

        # When --python-versions is explicitly passed, always parametrize
        # with those exact versions so node IDs match.
        # Unsupported versions are marked as "skip"
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


def pytest_collection_modifyitems(config, items) -> None:
    """Modify items based on markers."""
    verbose = config.option.verbose > 0

    try:
        rocprof_config = get_rocprof_config()
    except Exception as e:
        pytest.exit(f"{e}")

    def _ctx_for(item: pytest.Item, args) -> MarkerCtx:
        return MarkerCtx(
            config=rocprof_config,
            gpu_info=get_gpu_info(),
            args=tuple(args),
        )

    # ----------------------------------------------------------------------------
    def base_modifications(item: pytest.Item) -> None:
        """This function should be called for every item."""
        _standardize_test_name(item, config, verbose=verbose)

        # Inject dependency markers declared via add_marker_dependency_if
        # (e.g. annotate -> papi, mpi_implementation -> mpi, multi_gpu -> gpu,
        # mpi_optional -> mpi when the target supports MPI).
        _MARKERS.resolve_dependencies(_ctx_for, item)

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

    # Marker checks. "Skip" markers are left for runtime evaluation.
    for item in items:
        base_modifications(item)
        _MARKERS.apply_skip_conditions(_ctx_for, item)


def pytest_collection_finish(session):
    if session.config.getoption("--ctest-mode", default="off") == "generate":
        raw_path = session.config.getoption("--ctest-output-path", default=None)
        output_path = Path(raw_path) if raw_path else None
        _ctest_generate_tests(session.items, output_path)


# ----------------------------------------------------------------------------
# Test execution hooks
# ----------------------------------------------------------------------------


@pytest.hookimpl(hookwrapper=True)  # Allows yield
def pytest_runtest_makereport(item, call):
    """
    Attaches a "Runner Output" section to the call-phase report of the form:

    =========================================
    Command: <command>
    Environment:
    <environment>
    =========================================
    Test Output:
    <test output>
    =========================================
    """
    outcome = yield
    rep = outcome.get_result()

    setattr(item, f"rep_{rep.when}", rep)

    if rep.when != "call" or item.stash.get(_output_printed_key, False):
        return

    item.stash[_output_printed_key] = True
    test_result = item.stash.get(_result_key, None)
    if test_result is None:
        return

    report_output = []
    cmd = " ".join(str(c) for c in getattr(test_result, "command", []))
    if cmd:
        report_output.append(f"{'='*70}")
        report_output.append(f"Command: {cmd}")
    test_env = getattr(test_result, "environment", None)
    if isinstance(test_env, environment.TestEnvironment):
        env_lines = test_env.format_layers()
        if env_lines:
            report_output.append("Environment:\n\n" + "\n".join(env_lines) + "\n")
            report_output.append(f"{'='*70}")
    test_output = getattr(test_result, "test_output", "")
    extra_output = getattr(test_result, "extra_output", None)
    if test_output or extra_output:
        report_output.append("Test Output:\n")
        if test_output:
            report_output.append(test_output)
        if extra_output:
            report_output.append(extra_output)
        report_output.append(f"{'='*70}")

    if not report_output:
        return

    rep.sections.append(("Runner Output", "\n".join(report_output) + "\n\n"))


def pytest_runtest_logreport(report):
    """Print the runner output inline for passing tests in CTest run mode.

    Failing tests already have their "Runner Output" section printed by pytest's
    failure summary.
    """
    config = getattr(pytest, "_config_ref", None)
    if config is None:
        return
    if config.getoption("--ctest-mode", default="off") != "run":
        return
    if report.when != "call" or not report.passed:
        return

    terminal = config.pluginmanager.get_plugin("terminalreporter")
    if terminal is None:
        return

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
# Marker declarations (single source of truth for all test markers)
# ----------------------------------------------------------------------------
# Every test marker (functional + non-functional) is declared here via the
# register_* helpers from rocprofsys/markers.py, which handles registration,
# dependency injection, capability-based skipping, and CTest label export.
#
# A condition (a marker ``skip_if`` or a dependency ``when``) receives a
# MarkerCtx and returns None when the requirement is met, or a skip-reason
# string otherwise.
# ----------------------------------------------------------------------------


def _overflow_reason(ctx: MarkerCtx) -> Optional[str]:
    if caps(ctx).perf_events_usable:
        return None
    return "Requires either perf_event_paranoid <= 2 or CAP_SYS_ADMIN to be available"


def _gpu_reason(ctx: MarkerCtx) -> Optional[str]:
    if ctx.gpu_info is not None and ctx.gpu_info.available:
        return None
    return "No valid GPU available"


def _annotate_reason(ctx: MarkerCtx) -> Optional[str]:
    msg = _overflow_reason(ctx)
    if msg is not None:
        return msg
    if not caps(ctx).papi_availability:
        return "PAPI not available"
    return None


def _mpi_reason(ctx: MarkerCtx) -> Optional[str]:
    if caps(ctx).mpiexec_exec is None:
        return "MPI not available"
    return None


def _mpi_implementation_reason(ctx: MarkerCtx) -> Optional[str]:
    req_impl = ctx.args[0]
    found = caps(ctx).mpi_implementation
    if req_impl != found:
        return f"Requires {req_impl}, but {found} found"
    return None


def _python_reason(ctx: MarkerCtx) -> Optional[str]:
    # Two independent checks; the first failing one is reported (a missing binary
    # takes precedence over missing supported versions).
    if ctx.config.rocprofsys_python is None:
        return "rocprof-sys-python binary not found"
    if not (
        caps(ctx).supported_python_versions is not None
        and os.environ.get("ROCPROFSYS_USE_PYTHON", "ON").upper() == "ON"
    ):
        return (
            "No supported Python versions. Each version needs a corresponding "
            "libpyrocprofsys.<IMPL>-<VERSION>-<ARCH>-<OS>-<ABI>.so in site-packages/rocprofsys."
        )
    return None


def _multi_gpu_reason(ctx: MarkerCtx) -> Optional[str]:
    num_gpu = ctx.args[0]
    device_count = ctx.gpu_info.device_count
    if device_count < num_gpu:
        return f"Test requires atleast {num_gpu} GPUs but system has {device_count}"
    return None


def _run_if_gpu_category_reason(ctx: MarkerCtx) -> Optional[str]:
    # The expression is always evaluated so an invalid expression aborts the
    # session (a developer error) regardless of GPU presence. A missing GPU takes
    # precedence over a false category when reporting the skip reason.
    gpu_reason = _gpu_reason(ctx)
    info = ctx.gpu_info
    category_context = {
        "instinct": info is not None and "instinct" in info.categories,
        "radeon": info is not None and "radeon" in info.categories,
        "apu": info is not None and "apu" in info.categories,
    }
    expr = ctx.args[0]
    try:
        result = eval(expr, {"__builtins__": {}}, category_context)
    except Exception as e:
        pytest.exit(f"Invalid run_if_gpu_category expression: {e}", returncode=1)
    if gpu_reason is not None:
        return gpu_reason
    if not result:
        return f"GPU category condition '{expr}' is False"
    return None


def _mpi_optional_supported(ctx: MarkerCtx) -> Optional[str]:
    """Return None to add the ``mpi`` marker for an ``mpi_optional`` target."""
    if _mpi_reason(ctx) is not None:
        return "MPI not available"
    target = ctx.args[0]
    try:
        target_path = ctx.config.get_target_executable(target)
    except FileNotFoundError:
        return f"target executable '{target}' not found"
    if not caps(ctx).target_support_mpi(target_path):
        return f"target '{target}' does not support MPI"
    return None


# Informational labels: purely for test categorization / CTest label filtering.
_INFORMATIONAL_LABELS = [
    "avail", "instrument", "baseline", "sampling", "binary_rewrite",
    "runtime_instrument", "sys_run", "decode", "videodecode", "jpegdecode",
    "rocprof_binary", "rocprof_config", "xgmi", "sdma", "group_by_queue",
    "group_by_stream", "openmp", "openmp_target", "fortran", "sampling_duration",
    "no_tmp_files", "rccl", "roctx", "time_window", "transpose", "network",
    "fork", "user_api", "thread_limit", "pthreads", "rewrite_caller", "locks",
    "caller_include", "causal", "causal_e2e", "code_coverage", "lulesh",
    "unit_tests", "hip_stream", "presets", "cli_help", "hpc", "hip",
    "scratch_memory", "rocm", "kfd", "unified_memory", "validation_usm",
    "selective_regions", "minimal", "rank_filter", "pytest_impl", "annotate",
]


# -- Functional skip markers ----------------------------------------------
register_functional_marker(
    "build_only",
    description="prevents the test from being run in install mode",
    skip_if=requires(
        lambda ctx: not ctx.config.is_installed,
        "Test only runs in build mode (build_only)",
    ),
)
register_functional_marker(
    "gpu",
    description="mark test as requiring a GPU",
    skip_if=_gpu_reason,
)
register_functional_marker(
    "ucx",
    skip_if=requires(lambda ctx: caps(ctx).ucx_availability, "UCX not available"),
)
register_functional_marker("mpi", skip_if=_mpi_reason)
register_functional_marker(
    "mpi_implementation",
    arg_hint="implementation",
    description="mark test as requiring specific MPI implementation",
    skip_if=_mpi_implementation_reason,
    ctest=CTestExport.ARGS,
)
register_functional_marker("overflow", skip_if=_overflow_reason)
register_functional_marker(
    "attach",
    skip_if=requires(
        lambda ctx: caps(ctx).ptrace_scope == 0,
        "Requires ptrace_scope to be 0. Run 'echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope' "
        "to enable attaching to process",
    ),
)
register_functional_marker("python", skip_if=_python_reason)
register_functional_marker(
    "julia",
    skip_if=requires(lambda ctx: caps(ctx).julia_exec, "Julia not available"),
)
register_functional_marker(
    "xnack",
    skip_if=requires(
        lambda ctx: get_xnack_support(ctx.config.rocm_path), "XNACK not supported"
    ),
)
register_functional_marker(
    "no_docker",
    skip_if=requires(
        lambda ctx: not caps(ctx).is_inside_docker,
        "Test cannot run inside a Docker container",
    ),
    ctest=CTestExport.NONE,
)
register_functional_marker(
    "shmem",
    skip_if=requires(lambda ctx: caps(ctx).oshrun_exec, "SHMEM not available"),
)
register_functional_marker(
    "nic",
    skip_if=requires(
        lambda ctx: caps(ctx).papi_nic_events is not None
        and caps(ctx).perf_events_usable,
        "Requires PAPI network events and perf_event_paranoid <= 2 (or CAP_SYS_ADMIN) to be available",
    ),
)
register_functional_marker(
    "ainic",
    skip_if=requires(
        lambda ctx: caps(ctx).ai_nic_devices,
        "No AI NIC devices found (amd-smi static reports no NETDEV entries)",
    ),
)
register_functional_marker(
    "run_if_gpu_category",
    arg_hint="expr",
    description=(
        "run test only if GPU category expression is true "
        "(e.g., 'apu and not instinct', 'instinct or radeon')"
    ),
    skip_if=_run_if_gpu_category_reason,
    ctest=CTestExport.NONE,
)
register_functional_marker(
    "multi_gpu",
    arg_hint="num",
    description="mark test as using requiring atleast num amount of GPUs",
    skip_if=_multi_gpu_reason,
    ctest=CTestExport.ALL,
)

# -- Minimum-version markers (built from the min_version template) ---------
min_version(
    "rocm_min_version",
    lambda ctx: ctx.config.rocm_version,
    parts=3,
    label="ROCm",
    description="mark test as requiring minimum ROCm version",
)
min_version(
    "oshrun_min_version",
    lambda ctx: caps(ctx).oshrun_version,
    parts=2,
    not_found_msg="OpenSHMEM version not found",
    too_old_msg="oshrun version {found} < required {req}",
    description="mark test as requiring minimum OpenSHMEM version",
)
min_version(
    "amdsmi_min_version",
    lambda ctx: caps(ctx).amdsmi_version,
    parts=2,
    label="AMD-SMI",
    description="mark test as requiring minimum AMD-SMI version",
)
min_version(
    "amdgpu_min_version",
    lambda ctx: caps(ctx).amdgpu_version,
    parts=3,
    label="amdgpu",
    description="mark test as requiring minimum amdgpu driver version",
)
min_version(
    "rocprofiler_sdk_min_version",
    lambda ctx: caps(ctx).rocprofiler_sdk_version,
    parts=3,
    not_found_msg=(
        "Requires rocprofiler-sdk minimum {req}, "
        "but system detected version {found}"
    ),
    too_old_msg=(
        "Requires rocprofiler-sdk minimum {req}, "
        "but system detected version {found}"
    ),
    description="mark test as requiring minimum rocprofiler-sdk version",
)

# -- Non-functional markers (labels / behavior; no skip condition) ---------
register_marker("papi")
register_marker(
    "mpi_optional",
    "If MPI is available and the target supports MPI, uses MPI to run the test",
    arg_hint="target",
    ctest=CTestExport.NONE,
)
register_marker(
    "preserve",
    "prevents the file from being deleted after the test, even if ROCPROFSYS_KEEP_TEST_OUTPUT is set to OFF",
    arg_hint="file",
    ctest=CTestExport.NONE,
)
register_marker(
    "rocpd",
    "mark test as using ROCpd and inject ROCpd env into given env",
    arg_hint="env",
    ctest=CTestExport.NAME,
)
register_marker(
    "python_versions",
    "Test will be parametrized by Python version",
    ctest=CTestExport.NONE,
)
register_marker(
    "timeout",
    "mark test as having a timeout of seconds (default: 300)",
    arg_hint="seconds",
    ctest=CTestExport.NONE,
)
register_marker(
    "depends_on",
    "declare CTest dependency on the named tests (used for CTest DEPENDS)",
    arg_hint="*names",
    ctest=CTestExport.NONE,
)
register_marker(
    "serialize",
    "mark test as serializable (used for CTest)",
    ctest=CTestExport.NONE,
)
register_marker(
    "class_name",
    (
        "hyphenated logical name for standardized / CTest test names "
        "(replaces the auto-derived class segment from TestCamelCase; "
        "e.g. 'rocprofiler-systems-instrument')"
    ),
    arg_hint="segment",
    ctest=CTestExport.NONE,
)
register_marker("rocprofiler", "mark test as using ROCProfiler counters")
register_marker("slow", "mark test as slow running")
register_marker("loops", "mark test as testing loop instrumentation")
for _label in _INFORMATIONAL_LABELS:
    register_marker(_label)

# -- Marker dependencies (add a marker when a trigger marker is present) ---
# Declarative successor to the old ``add_marker_if`` calls.
add_marker_dependency_if("gpu", when_present="multi_gpu")
add_marker_dependency_if("mpi", when_present="mpi_implementation")
add_marker_dependency_if("python", when_present="python_versions")
add_marker_dependency_if("papi", when_present="annotate", when=_annotate_reason)
add_marker_dependency_if(
    "mpi",
    when_present="mpi_optional",
    when=_mpi_optional_supported,
    on_unmet=OnUnmet.IGNORE,
)


# ----------------------------------------------------------------------------
# Test-category (tier) label injection from test_categories.yaml
# ----------------------------------------------------------------------------
# Single source of truth for tier policy is tests/test_categories.yaml.
# At CTest-generate time we read the YAML and append tier labels to each
# test's emitted LABELS set, so `ctest -L <tier>` Just Works from the
# installed share/rocprofiler-systems/tests directory.

TIER_ORDER = ["quick", "standard", "comprehensive", "full"]


@lru_cache(maxsize=1)
def _load_test_categories() -> Optional[dict]:
    """Load and compile test_categories.yaml from rocprofsys_tests_dir.

    Reads the YAML that CMake installs/configures into
    ``<build|install>/share/rocprofiler-systems/tests``

    Returns ``None`` (with a single STDERR warning) when the YAML is missing or
    PyYAML isn't importable
    """
    if yaml is None:
        print(
            "[test_categories] PyYAML not available - skipping tier label injection.",
            file=sys.stderr,
        )
        return None
    try:
        yaml_path = get_rocprof_config().rocprofsys_tests_dir / "test_categories.yaml"
    except Exception as exc:
        print(
            f"[test_categories] Could not resolve tests dir - skipping tier label injection: {exc}",
            file=sys.stderr,
        )
        return None
    if not yaml_path.exists():
        print(
            f"[test_categories] {yaml_path} not found - skipping tier label injection.",
            file=sys.stderr,
        )
        return None
    try:
        data = yaml.safe_load(yaml_path.read_text()) or {}
    except yaml.YAMLError as exc:
        print(
            f"[test_categories] Failed to load {yaml_path}: {exc}",
            file=sys.stderr,
        )
        return None

    def _compile_list(patterns):
        compiled = []
        for p in patterns or []:
            # Flatten one level: a YAML alias item (e.g. `- *common_excludes`)
            # expands to a nested list, so callers can mix a shared anchor with
            # per-tier additions.
            for pattern in p if isinstance(p, list) else [p]:
                try:
                    compiled.append(re.compile(pattern))
                except re.error as exc:
                    print(
                        f"[test_categories] Skipping invalid regex {pattern!r}: {exc}",
                        file=sys.stderr,
                    )
        return compiled

    def _flatten_labels(values):
        # One-level flatten (like _compile_list) so a YAML alias item expands
        # to a nested list without raising TypeError on the unhashable inner
        # list when set()-ed.
        flat = []
        for v in values or []:
            flat.extend(v if isinstance(v, list) else [v])
        return flat

    tier_cfg: dict = {}
    for tier in TIER_ORDER:
        cfg = (data.get("test_categories", {}) or {}).get(tier) or {}
        tier_cfg[tier] = {
            "include": _compile_list(cfg.get("regex_includes")),
            "exclude": _compile_list(cfg.get("regex_excludes")),
            "label_excludes": _compile_list(cfg.get("label_excludes")),
            "label_includes": _compile_list(cfg.get("label_includes")),
            "labels": _flatten_labels(cfg.get("added_supplementary_labels")),
        }

    return {"tiers": tier_cfg}


def _resolve_tier_labels(test_name: str, existing_labels: set[str]) -> set[str]:
    """Return tier labels (subset of TIER_ORDER) for *test_name*.

    Each tier is evaluated independently with the *exact* CTest filter model.
    The four YAML axes map to CTest options as (labels are pytest MARKERs):
      * ``regex_includes`` -> ``-R``   * ``regex_excludes`` -> ``-E``
      * ``label_includes``  -> ``-L``  * ``label_excludes`` -> ``-LE``

    The rocJenkins-style cascade ("matching quick also yields standard /
    comprehensive / full") is achieved by having those higher tiers use
    broad include patterns (typically ``regex_includes: [".*"]``). Per-tier
    ``regex_excludes`` punches a hole through the cascade for individual
    tests: listing ``testA`` under ``standard.regex_excludes`` drops
    ``standard`` from its label set even if ``quick`` / ``comprehensive`` /
    ``full`` match.

    In addition to the tier name, each matched tier contributes its
    ``added_supplementary_labels:`` to the test's labels.
    """
    categories = _load_test_categories()
    if not categories:
        return set()
    matched_indices: list[int] = []
    extra_labels: set[str] = set()
    for i, tier in enumerate(TIER_ORDER):
        cfg = categories["tiers"].get(tier) or {}
        include_regex = cfg.get("include", [])
        include_labels = cfg.get("label_includes", [])
        exclude_regex = cfg.get("exclude", [])
        exclude_labels = cfg.get("label_excludes", [])
        # -R: an empty include is a pass-through; otherwise the NAME must match.
        if include_regex and not any(p.search(test_name) for p in include_regex):
            continue
        # -L: an empty include is a pass-through; otherwise a marker label must
        # match. AND-combined with the -R axis above, exactly like CTest.
        if include_labels and not any(
            p.search(label) for p in include_labels for label in existing_labels
        ):
            continue
        # -E: NAME matching any exclude pattern vetoes the test.
        if any(p.search(test_name) for p in exclude_regex):
            continue
        # -LE: any marker label matching an exclude pattern vetoes the test.
        if any(p.search(label) for p in exclude_labels for label in existing_labels):
            continue
        matched_indices.append(i)
        extra_labels.update(cfg.get("labels", []))
    return {TIER_ORDER[i] for i in matched_indices} | extra_labels


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

        # Translate pytest markers to CTest labels per each marker's export policy

        for marker in item.iter_markers():
            labels |= _MARKERS.ctest_labels_for_marker(marker.name, marker.args)

        # Inject tier (quick/standard/comprehensive/full) labels from
        # test_categories.yaml
        labels |= _resolve_tier_labels(test_name, labels)

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


def _standardize_test_name(
    item: pytest.Item, config: pytest.Config, verbose: bool = False
) -> None:

    # Strip test prefix from the test method name
    test_name = item.name
    if test_name.startswith("test"):
        test_name = test_name[4:]
        if test_name.startswith(("_", "-")):
            test_name = test_name[1:]

    class_name = None
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
    formatted_name = formatted_name.replace("_", "-")
    while "--" in formatted_name:
        formatted_name = formatted_name.replace("--", "-")
    formatted_name = formatted_name.strip("-")
    formatted_name = formatted_name.lower()

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

    oshrun_strips_str = (
        "Yes (decoy '--' inserted)"
        if cap.oshrun_strips_double_dash
        else "No" if cap.oshrun_exec else "N/A"
    )

    if cap.amdsmi_version is not None:
        amdsmi_version_str = f"{cap.amdsmi_version[0]}.{cap.amdsmi_version[1]}"
    else:
        amdsmi_version_str = "Not found"

    if cap.amdgpu_version is not None:
        amdgpu_version_str = (
            f"{cap.amdgpu_version[0]}.{cap.amdgpu_version[1]}.{cap.amdgpu_version[2]}"
        )
    else:
        amdgpu_version_str = "Not found"

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
        _row("AMD-SMI version:", amdsmi_version_str),
        _row("amdgpu version:", amdgpu_version_str),
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
        _subrow("Strips '--':", oshrun_strips_str),
        _row("Offload tool:", offload_msg),
        _row("Rocminfo:", rocminfo_path if rocminfo_path else rocminfo_err_msg),
        "-" * 70,
        "System Capabilities:",
        _row("Detected num procs:", cap.num_procs),
        _row("UCX available:", cap.ucx_availability),
        _row("Perf event paranoid:", cap.perf_event_paranoid),
        _row("CAP_SYS_ADMIN:", cap.cap_sys_admin),
        _row("CAP_PERFMON:", cap.cap_perfmon),
        _row("Perf events usable:", cap.perf_events_usable),
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
    # Use fundamental system env to avoid verbose output
    header.extend(["-" * 70, "System Environment:"])
    for key, value in environment.fundamental_system_environment().items():
        header.append(_row(f"{key}:", value))
    header.extend(["=" * 70, ""])
    return header


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
        rocm_optional = False
        if pytest_config:
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


def _run_cleanup() -> None:
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
def library_path(rocprof_config) -> str:
    """Computed LD_LIBRARY_PATH (rocprofsys libs + user override + ROCm LLVM libs)."""
    return rocprof_config.get_library_path()


@pytest.fixture
def flat_env() -> dict[str, str]:
    """Environment variables for flat profile tests."""
    return environment.flat_environment()


@pytest.fixture
def lock_env() -> dict[str, str]:
    """Environment variables for thread lock tracing tests."""
    return environment.lock_environment()


@pytest.fixture
def perfetto_env() -> dict[str, str]:
    """Environment variables for perfetto-only tests."""
    return environment.perfetto_environment()


@pytest.fixture
def timemory_env() -> dict[str, str]:
    """Environment variables for timemory-only tests."""
    return environment.timemory_environment()


# ----------------------------------------------------------------------------
# Session-scoped Fixtures
# ----------------------------------------------------------------------------


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
def test_output_base(rocprof_config) -> Path:
    """Base directory for test outputs"""
    output_dir = rocprof_config.test_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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


def _print_subtest_output(request, subtest_name: str, output: str) -> None:
    """Print subtest validation output for important subtests when in CTest run mode."""
    if request.config.getoption("--ctest-mode", default="off") == "run" and output:
        print(f"\n--- {subtest_name} ---\n{output}\n", flush=True)


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
        library_path,
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
        self.library_path = library_path


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

        # Timeout: ROCPROFSYS_CI_TIMEOUT env, else @pytest.mark.timeout, else default
        ci_timeout_env = os.environ.get("ROCPROFSYS_CI_TIMEOUT")
        if ci_timeout_env is not None:
            # Shell-exported value: drives the subprocess timeout below and is
            # already carried by the user env layer. Do NOT echo it into the
            # test layer, or it would mask the real base default in dumps.
            timeout = int(ci_timeout_env)
        elif request.node.get_closest_marker("timeout"):
            timeout = request.node.get_closest_marker("timeout").args[0]
            env["ROCPROFSYS_CI_TIMEOUT"] = str(timeout)
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

        if not result.success and not fail_on_pass:
            short_msg = fail_message or f"{runner_type} test failed"
            if skip_on_error:
                pytest.skip(short_msg)
            else:
                pytest.fail(short_msg)

        if fail_on_pass and result.success:
            pytest.fail(f"{runner_type} test passed unexpectedly")

        return result

    return _run_test


@pytest.fixture
def assert_regex(subtests):
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
                    pytest.fail(msg)

    return _assert_regex


@pytest.fixture
def assert_file_regex(subtests):
    """Variant of assert_regex that validates against a file."""

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
                    pytest.fail(msg)

    return _assert_file_regex


@pytest.fixture
def assert_perfetto(subtests, tests_dir, request, test_output_dir):
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
                    pytest.fail(msg)
            if pass_regex:
                for pattern in pass_regex:
                    if not re.search(pattern, validation.stdout):
                        pytest.fail(
                            f"Pass regex not found: {pattern}\n{output}", pytrace=False
                        )
            if fail_regex:
                for pattern in fail_regex:
                    if re.search(pattern, validation.stdout):
                        pytest.fail(
                            f"Fail regex found: {pattern}\n{output}", pytrace=False
                        )
            _print_subtest_output(request, subtest_name, output)

    return _assert_perfetto


@pytest.fixture
def assert_rocpd(subtests, tests_dir, request):
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
            rocpd_files = result.rocpd_files
            if not rocpd_files:
                pytest.fail("ROCpd database not created")

            existing_rules = None
            if rules_files is not None:
                existing_rules = [r for r in rules_files if r.exists()]
                if not existing_rules:
                    pytest.fail("No validation rules found")

            def validate_candidate(rocpd_file: Path) -> ValidationResult:
                return validate_rocpd_database(
                    rocpd_file,
                    tests_dir=tests_dir,
                    rules_files=existing_rules,
                    timeout=timeout,
                    gpu_category_to_skip=gpu_category_to_skip,
                )

            passing_output, failures, global_failure = _validate_rocpd_candidates(
                rocpd_files,
                validate_candidate,
                pass_regex=pass_regex,
                fail_regex=fail_regex,
            )

            if global_failure is not None:
                msg = fail_message or f"ROCpd validation failed:\n{global_failure}"
                if skip_on_fail:
                    pytest.skip(msg)
                else:
                    pytest.fail(msg, pytrace=False)
            elif passing_output is None:
                output = "\n\n--- Next ROCpd candidate ---\n\n".join(failures)
                msg = fail_message or f"ROCpd validation failed:\n{output}"
                if skip_on_fail:
                    pytest.skip(msg)
                else:
                    pytest.fail(msg, pytrace=False)
            _print_subtest_output(request, subtest_name, passing_output)

    return _assert_rocpd


def _validate_rocpd_candidates(
    rocpd_files: list[Path],
    validate_candidate: Callable[[Path], ValidationResult],
    pass_regex: Optional[list[str]] = None,
    fail_regex: Optional[list[str]] = None,
) -> tuple[Optional[str], list[str], Optional[str]]:
    """Validate ROCpd candidates and return the first passing output.

    Multi-process runs can emit multiple ROCpd databases. Some rank-local
    databases may not contain the GPU rows required by a rule set, so the
    validation succeeds if any emitted candidate fully validates. A fail regex
    match is a global failure and stops validation immediately.
    """
    failures: list[str] = []

    for rocpd_file in rocpd_files:
        validation = validate_candidate(rocpd_file)
        output = f"Command: {validation.command}\n\n{validation.message}"
        if fail_regex:
            for pattern in fail_regex:
                if re.search(pattern, validation.stdout):
                    return None, failures, f"Fail regex found: {pattern}\n{output}"
        if not validation.is_valid:
            failures.append(output)
            continue

        regex_failure = None
        if pass_regex:
            for pattern in pass_regex:
                if not re.search(pattern, validation.stdout):
                    regex_failure = f"Pass regex not found: {pattern}"
                    break
        if regex_failure is not None:
            failures.append(f"{regex_failure}\n{output}")
            continue

        return output, failures, None

    return None, failures, None


@pytest.fixture
def assert_timemory(subtests, tests_dir, request):
    """Fixture that returns an assert_timemory function.

    Args not from validate_timemory_json:
        subtest_name: Name shown in subtest output (defaults to "Timemory validation")
        pass_regex: (Optional) Regex patterns that must be found in validation.stdout
        fail_regex: (Optional) Regex patterns that must NOT be found in validation.stdout
        skip_on_fail: If True, skip instead of fail when validation fails
        fail_message: Custom message for failure (defaults to validation message)
    """

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
                    pytest.fail(msg)
            if pass_regex:
                for pattern in pass_regex:
                    if not re.search(pattern, validation.stdout):
                        pytest.fail(
                            f"Pass regex not found: {pattern}\n{output}", pytrace=False
                        )
            if fail_regex:
                for pattern in fail_regex:
                    if re.search(pattern, validation.stdout):
                        pytest.fail(
                            f"Fail regex found: {pattern}\n{output}", pytrace=False
                        )
            _print_subtest_output(request, subtest_name, output)

    return _assert_timemory


@pytest.fixture
def assert_file_exists(subtests):
    """Fixture that returns an assert_file_exists function.

    Args not from validate_file_exists:
        subtest_name: Name shown in subtest output (defaults to "File existence validation")
        skip_on_fail: If True, skip instead of fail when validation fails
        fail_message: Custom message for failure (defaults to validation message)
    """

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
                        pytest.fail(msg)

    return _assert_file_exists


@pytest.fixture
def assert_unified_memory_output(subtests, tests_dir, request):
    """Fixture that returns an assert_unified_memory_output function."""

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
                    pytest.fail(msg)
            if pass_regex:
                for pattern in pass_regex:
                    if not re.search(pattern, validation.stdout):
                        pytest.fail(
                            f"Pass regex not found: {pattern}\n{output}",
                            pytrace=False,
                        )
            if fail_regex:
                for pattern in fail_regex:
                    if re.search(pattern, validation.stdout):
                        pytest.fail(
                            f"Fail regex found: {pattern}\n{output}",
                            pytrace=False,
                        )
            _print_subtest_output(request, subtest_name, output)

    return _assert_unified_memory_output


@pytest.fixture
def assert_causal_json(subtests, tests_dir, request):
    """Fixture that returns an assert_causal_json function.

    Args not from validate_causal_json:
        pass_regex: (Optional) Regex patterns that must be found in validation.stdout
        fail_regex: (Optional) Regex patterns that must NOT be found in validation.stdout
        skip_on_fail: If True, skip instead of fail when validation fails
        fail_message: Custom message for failure (defaults to validation message)
    """

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
                    pytest.fail(msg)

            if pass_regex:
                for pattern in pass_regex:
                    if not re.search(pattern, validation.stdout):
                        pytest.fail(
                            f"Pass regex not found: {pattern}\n{output}", pytrace=False
                        )

            if fail_regex:
                for pattern in fail_regex:
                    if re.search(pattern, validation.stdout):
                        pytest.fail(
                            f"Fail regex found: {pattern}\n{output}", pytrace=False
                        )
            _print_subtest_output(request, subtest_name, output)

    return _assert_causal_json
