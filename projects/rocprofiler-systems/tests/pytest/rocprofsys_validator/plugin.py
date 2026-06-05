# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""pytest plugin for rocprofsys-validator.

Registers hooks, CLI options, fixtures, and marks for rocprof-sys trace validation.
Auto-loaded via the pytest11 entry-point in pyproject.toml.

Hooks implemented:
- pytest_configure: registers requires_gpu and requires_counter marks
- pytest_addoption: registers --tp-bin, --pftrace-input, --rocpd-input, --timemory-input
- pytest_runtest_setup: runtime skip logic for @pytest.mark.requires_gpu

Fixtures provided (via pytest11 entry-point — available after pip install):
- gpu_profile (session): GPUProfile from TOML override or detect_gpu()
- perfetto_trace (function): TraceProcessor connected to a Perfetto trace file
- rocpd_db (function): sqlite3.Connection to a RocPD SQLite database (read-only)
- timemory_output (function): path to timemory output directory (containing .txt metric files)

Design decisions:
- D-13: Binary resolution: ROCPROFSYS_TRACE_PROCESSOR_SHELL → --tp-bin → hard error
- D-14: perfetto_trace fixture is function-scoped (thread-safe; session scope is Phase 3)
- D-15: run_sanity_checks runs before yield in perfetto_trace (not lazily)
- D-19: @pytest.mark.requires_gpu skips with "Requires GPU: {reason}. Detected: {name}"
- D-20: pytest --collect-only works with zero GPU hardware (setup deferred to execution)
- T-04-02: SQLite opened read-only via uri=True, mode=ro
- T-04-03: TraceProcessorConfig(load_timeout=30) caps subprocess startup
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from rocprofsys_validator.gpu import GPUProfile, detect_gpu
from rocprofsys_validator.sanity import run_sanity_checks

# ─────────────────────────────────────────────────────────────────────────────
# Hook: pytest_configure — register marks
# ─────────────────────────────────────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    """Register custom marks so pytest does not emit PytestUnknownMarkWarning.

    Marks registered:
    - requires_gpu: test requires GPU hardware; skips at runtime when arch == "unknown"
    - requires_counter(name): test requires a specific PMC counter name on the GPU
    """
    config.addinivalue_line(
        "markers",
        "requires_gpu: marks tests that require GPU hardware "
        "(deselect with '-m not requires_gpu')",
    )
    config.addinivalue_line(
        "markers",
        "requires_counter(name): marks tests that require a specific PMC counter name",
    )
    # Bridge --baseline-update to the env var that assert_baseline() honors, so
    # the CLI flag drives snapshot recapture without threading config through.
    if config.getoption("--baseline-update", default=False):
        os.environ["ROCPROFSYS_BASELINE_UPDATE"] = "1"

# ─────────────────────────────────────────────────────────────────────────────
# Hook: pytest_addoption — register CLI options
# ─────────────────────────────────────────────────────────────────────────────

def pytest_addoption(parser: pytest.Parser) -> None:
    """Register rocprofsys-specific pytest CLI options.

    Options:
    - --tp-bin: Path to trace_processor_shell binary
    - --pftrace-input: Path to Perfetto trace file
    - --rocpd-input: Path to RocPD SQLite database
    - --timemory-input: Path to timemory JSON output
    """
    parser.addoption(
        "--tp-bin",
        action="store",
        default=None,
        help=(
            "Path to trace_processor_shell binary. "
            "Alternative: set ROCPROFSYS_TRACE_PROCESSOR_SHELL env var."
        ),
    )
    parser.addoption(
        "--pftrace-input",
        action="store",
        default=None,
        help="Path to Perfetto trace file (.pftrace/.proto).",
    )
    parser.addoption(
        "--rocpd-input",
        action="store",
        default=None,
        help="Path to RocPD SQLite database.",
    )
    parser.addoption(
        "--timemory-input",
        action="store",
        default=None,
        help="Path to timemory output directory containing .txt metric files.",
    )
    parser.addoption(
        "--baseline-update",
        action="store_true",
        default=False,
        help=(
            "Recapture all assert_baseline() snapshots instead of comparing "
            "(like --snapshot-update). Sets ROCPROFSYS_BASELINE_UPDATE for the run."
        ),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Hook: pytest_runtest_setup — requires_gpu runtime skip (D-19, FOUND-05)
# ─────────────────────────────────────────────────────────────────────────────

def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip tests marked @pytest.mark.requires_gpu when no GPU is detected.

    Runs at SETUP time (not collection time) so the test appears as SKIPPED in
    the run report. The skip message format is exact per D-19:
    "Requires GPU: {reason}. Detected: {gpu.name}"

    """
    marker = item.get_closest_marker("requires_gpu")
    if marker is None:
        return
    gpu = detect_gpu()
    if gpu.arch == "unknown":
        reason = marker.args[0] if marker.args else "GPU hardware"
        pytest.skip(f"Requires GPU: {reason}. Detected: {gpu.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: resolve trace_processor_shell binary path
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_tp_bin(request: pytest.FixtureRequest) -> str:  # type: ignore[type-arg]
    """Resolve the trace_processor_shell binary path.

    Resolution order (D-13):
    1. ROCPROFSYS_TRACE_PROCESSOR_SHELL environment variable
    2. --tp-bin pytest CLI option
    3. Hard RuntimeError with actionable message

    Security (T-04-01): Path is passed as str to TraceProcessorConfig(bin_path=...);
    never shell-executed directly.

    Args:
        request: pytest.FixtureRequest (or compatible object with .config.getoption()).

    Returns:
        Absolute path string to the trace_processor_shell binary.

    Raises:
        RuntimeError: If neither the env var nor --tp-bin is configured.
    """
    bin_path: str | None = os.environ.get("ROCPROFSYS_TRACE_PROCESSOR_SHELL") or request.config.getoption(
        "--tp-bin", default=None
    )
    if not bin_path:
        raise RuntimeError(
            "trace_processor_shell binary not configured. "
            "Set ROCPROFSYS_TRACE_PROCESSOR_SHELL env var or "
            "pass --tp-bin /path/to/trace_processor_shell to pytest."
        )
    return bin_path

# ─────────────────────────────────────────────────────────────────────────────
# Fixture: gpu_profile — session-scoped GPU capability profile (FOUND-05, D-11)
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_profile_impl(request: pytest.FixtureRequest) -> GPUProfile:  # noqa: ARG001
    """Inner implementation of gpu_profile, exposed for direct testing."""
    profile_path = os.environ.get("ROCPROFSYS_GPU_PROFILE")
    if profile_path:
        return GPUProfile.from_toml(profile_path)
    return detect_gpu()

@pytest.fixture(scope="session")
def gpu_profile(request: pytest.FixtureRequest) -> GPUProfile:
    """Session-scoped fixture: GPU capability profile.

    Resolution order (D-11):
    1. ROCPROFSYS_GPU_PROFILE env var → GPUProfile.from_toml(path)
    2. detect_gpu() subprocess call (cached for process lifetime)

    Returns:
        GPUProfile with name, arch, counter_names, pmc_groups.
        Returns unknown profile when GPU detection fails (D-10, D-12).
    """
    return _gpu_profile_impl(request)

# Expose the implementation for direct testing without pytest fixture machinery
gpu_profile.__wrapped__ = _gpu_profile_impl  # type: ignore[attr-defined]

# ─────────────────────────────────────────────────────────────────────────────
# Fixture: perfetto_trace — function-scoped Perfetto TraceProcessor (FOUND-06, D-14)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def perfetto_trace(request: pytest.FixtureRequest):
    """Function-scoped fixture: connected Perfetto TraceProcessor.

    Lazy import of perfetto to avoid ImportError when perfetto is not installed.

    Resolution (D-13): ROCPROFSYS_TRACE_PROCESSOR_SHELL → --tp-bin → hard error.
    Sanity checks (D-15): run_sanity_checks before yielding to the test.
    Cleanup: tp.close() in finally block (T-04-03).
    Skips: when --pftrace-input is not provided.

    Yields:
        TraceProcessor instance connected to the trace file.

    Raises:
        RuntimeError: If binary is not configured (via _resolve_tp_bin).
        RuntimeError: If sanity checks find malformed trace data.
    """
    from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig  # noqa: PLC0415

    trace_path = request.config.getoption("--pftrace-input", default=None)
    if trace_path is None:
        pytest.skip("No --pftrace-input provided")

    bin_path = _resolve_tp_bin(request)
    # T-04-03: load_timeout=30 caps subprocess startup
    cfg = TraceProcessorConfig(bin_path=str(bin_path), load_timeout=30)
    tp = TraceProcessor(trace=str(trace_path), config=cfg)
    try:
        run_sanity_checks(tp)
        yield tp
    finally:
        tp.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fixture: rocpd_db — function-scoped RocPD SQLite connection (T-04-02)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rocpd_db(request: pytest.FixtureRequest):
    """Function-scoped fixture: read-only SQLite connection to a RocPD database.

    Security (T-04-02): sqlite3.connect("file:path?mode=ro", uri=True) opens
    read-only; write operations raise OperationalError.

    Skips: when --rocpd-input is not provided.

    Yields:
        sqlite3.Connection with row_factory=sqlite3.Row.
    """
    db_path = request.config.getoption("--rocpd-input", default=None)
    if db_path is None:
        pytest.skip("No --rocpd-input provided")

    # T-04-02: read-only URI mode prevents accidental or malicious modification
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fixture: timemory_output — function-scoped path to timemory output directory
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def timemory_output(request: pytest.FixtureRequest):
    """Function-scoped fixture: path to the timemory output directory.

    Skips: when --timemory-input is not provided.

    Yields:
        pathlib.Path: path to the timemory output directory containing .txt metric files.
    """
    from pathlib import Path  # noqa: PLC0415

    dir_path = request.config.getoption("--timemory-input", default=None)
    if dir_path is None:
        pytest.skip("No --timemory-input provided")

    yield Path(dir_path)
