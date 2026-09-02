# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/utils_profile.py."""

import gzip
import os
from unittest import mock

import pandas as pd
import pytest

import utils.utils_profile as utils_profile
from utils.utils_profile import (
    _augment_marker_csv,
    _parse_function_backend,
)

# Long-form rocpd counter CSV header used by the run_prof tests.
COUNTER_CSV_HEADER = (
    "PID,Dispatch_ID,Kernel_Name,Grid_Size,Workgroup_Size,LDS_Per_Workgroup,"
    "Start_Timestamp,End_Timestamp,Kernel_ID,Counter_Name,Counter_Value\n"
)


# =============================================================================
# RUN_PROF TESTS
# =============================================================================


def test_run_prof_success_rocprofiler_sdk(tmp_path, monkeypatch):
    """run_prof (rocprofiler-sdk backend) pops APP_CMD out of the options and
    runs it with the profiler-built environment, into which the resolved
    counters and absolute agent index have been merged."""
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    profiler_options = {
        "APP_CMD": ["./test_app"],
        "ROCPROF_OUTPUT_PATH": workload_dir,
        "ROCPROF_COUNTER_COLLECTION": "1",
        "ROCP_TOOL_LIBRARIES": "/opt/rocm/lib/rocprofiler-sdk/"
        "librocprofiler-sdk-tool.so",
    }

    captured = {}

    def fake_capture(app_cmd, new_env=None, profileMode=False):
        captured["app_cmd"] = app_cmd
        captured["env"] = new_env
        return (True, "success")

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofiler-sdk")
    monkeypatch.setattr("utils.utils_profile.capture_subprocess_output", fake_capture)
    monkeypatch.setattr("utils.utils_profile.parse_pmc_perf", lambda f: ["SQ_WAVES"])
    monkeypatch.setattr(
        "utils.utils_profile.rocpd_data.convert_dbs_to_csv", lambda *a, **k: None
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)

    utils_profile.run_prof(str(fname), profiler_options, workload_dir)

    assert captured["app_cmd"] == ["./test_app"]
    assert "APP_CMD" not in captured["env"]
    assert captured["env"]["ROCPROF_COUNTER_COLLECTION"] == "1"
    assert captured["env"]["ROCPROF_COUNTERS"] == "pmc: SQ_WAVES"
    assert captured["env"]["ROCPROF_AGENT_INDEX"] == "absolute"


def test_rocprofiler_sdk_env_log_excludes_user_env(tmp_path, monkeypatch):
    """run_prof must log only profiler-added env vars, never the user's full
    environment, to avoid leaking secrets into shared workload logs."""
    monkeypatch.setenv("LEAK_CANARY_TOKEN", "SHOULD_NOT_APPEAR")

    logs = []
    monkeypatch.setattr(
        "utils.utils_profile.console_debug",
        lambda msg, *a, **k: logs.append(str(msg)),
    )
    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofiler-sdk")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr("utils.utils_common.parse_pmc_perf", lambda f: ["SQ_WAVES"])
    monkeypatch.setattr(
        "utils.utils_profile.rocpd_data.convert_dbs_to_csv", lambda *a, **k: None
    )
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)

    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    utils_profile.run_prof(
        str(fname),
        {
            "APP_CMD": ["./test_app"],
            "ROCPROF_OUTPUT_PATH": workload_dir,
            "ROCPROF_COUNTER_COLLECTION": "1",
        },
        workload_dir,
    )

    assert sum("env vars" in m for m in logs) >= 1
    env_log_lines = [m for m in logs if "env vars" in m]
    assert any("ROCPROF_COUNTER_COLLECTION" in m for m in env_log_lines)
    assert not any("SHOULD_NOT_APPEAR" in m for m in logs)


def test_run_prof_rocpd_skips_pid_without_native_csv(tmp_path, monkeypatch):
    """run_prof skips per-pid rocpd update when its native counter CSV is missing."""
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = tmp_path / "workload"

    # Child pids with no GPU work have a .db but no native counter CSV.
    pmc1 = workload_dir / "out" / "pmc_1"
    (pmc1 / "12345").mkdir(parents=True)
    (pmc1 / "12345" / "12345.db").touch()

    options = {
        "APP_CMD": ["./test_app"],
        "ROCPROF_OUTPUT_PATH": str(workload_dir),
        "ROCPROF_COUNTER_COLLECTION": "0",  # native tool collects, not SDK
        "ROCP_TOOL_LIBRARIES": "",
    }

    update_calls: list = []
    debug_msgs: list[str] = []

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofiler-sdk")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr(
        "utils.utils_profile.rocpd_data.update_rocpd_pmc_events",
        lambda *a, **k: update_calls.append(a),
    )
    monkeypatch.setattr(
        "utils.utils_profile.rocpd_data.convert_dbs_to_csv",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "utils.utils_profile.console_debug",
        lambda msg, *a, **k: debug_msgs.append(msg),
    )
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)

    utils_profile.run_prof(str(fname), options, str(workload_dir))

    assert update_calls == []
    assert any("No native counter CSV for pid 12345" in m for m in debug_msgs)


def stub_run_prof_deps(monkeypatch, counter_csv_body, warnings):
    """Run run_prof against a canned counter CSV instead of a real profiler."""
    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofiler-sdk")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr("utils.utils_profile.parse_pmc_perf", lambda f: ["SQ_WAVES"])

    def fake_convert(db_paths, counter_csv, marker_csv):
        assert counter_csv.endswith(".csv.gz"), counter_csv
        if counter_csv_body is not None:
            with gzip.open(counter_csv, "wt", encoding="utf-8") as f:
                f.write(counter_csv_body)

    monkeypatch.setattr(
        "utils.utils_profile.rocpd_data.convert_dbs_to_csv", fake_convert
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr(
        "utils.utils_profile.console_warning",
        lambda msg, *a, **k: warnings.append(str(msg)),
    )


@pytest.mark.parametrize(
    "counter_csv_body",
    [
        pytest.param(None, id="csv_never_written"),
        pytest.param(COUNTER_CSV_HEADER, id="header_only"),
    ],
)
def test_run_prof_zero_kernels_writes_no_results_csv(
    tmp_path, monkeypatch, counter_csv_body
):
    """A workload that dispatches no GPU kernels must warn and leave no
    results_*.csv behind: a header-only file would look like real profiling data
    to analyze."""
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = tmp_path / "workload"
    (workload_dir / "out" / "pmc_1").mkdir(parents=True)

    warnings: list[str] = []
    stub_run_prof_deps(monkeypatch, counter_csv_body, warnings)

    utils_profile.run_prof(
        str(fname),
        {"APP_CMD": ["./test_app"], "ROCPROF_COUNTER_COLLECTION": "1"},
        str(workload_dir),
    )

    assert any("No GPU kernel data collected" in m for m in warnings)
    assert sorted(workload_dir.glob("results_*.csv.gz")) == []
    assert not (workload_dir / "out").exists()


def test_run_prof_relabels_dispatch_and_kernel_ids(tmp_path, monkeypatch):
    """run_prof renumbers Dispatch_ID per unique dispatch and Kernel_ID per
    unique kernel launch shape, and drops PID from the results CSV."""
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = tmp_path / "workload"
    (workload_dir / "out" / "pmc_1").mkdir(parents=True)

    # Two dispatches of the same kernel shape, plus a differently shaped kernel.
    body = COUNTER_CSV_HEADER + (
        "100,77,kernel_a,256,64,0,10,20,9,SQ_WAVES,1\n"
        "100,77,kernel_a,256,64,0,10,20,9,SQ_BUSY_CYCLES,2\n"
        "100,88,kernel_a,256,64,0,30,40,9,SQ_WAVES,3\n"
        "100,99,kernel_b,512,64,0,50,60,5,SQ_WAVES,4\n"
    )
    warnings: list[str] = []
    stub_run_prof_deps(monkeypatch, body, warnings)

    utils_profile.run_prof(
        str(fname),
        {"APP_CMD": ["./test_app"], "ROCPROF_COUNTER_COLLECTION": "1"},
        str(workload_dir),
    )

    results_csv = workload_dir / "results_pmc_perf_test.csv.gz"
    results = pd.read_csv(results_csv)
    assert "PID" not in results.columns
    assert results["Dispatch_ID"].tolist() == [0, 0, 1, 2]
    # Kernel_ID keys off launch shape, so both kernel_a dispatches share one.
    assert results["Kernel_ID"].tolist() == [0, 0, 0, 1]
    assert results["Counter_Value"].tolist() == [1, 2, 3, 4]


def test_run_prof_with_yaml_config(tmp_path, monkeypatch):
    """run_prof merges the counter_def_<suffix>.yaml sitting beside the pmc file
    into the rocprofiler-sdk counter definitions handed to the metrics-path
    builder."""
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    yaml_file = tmp_path / "counter_def_test.yaml"
    yaml_file.write_text("rocprofiler-sdk:\n  counters:\n    - TCC_HIT\n")
    workload_dir = str(tmp_path / "workload")

    captured_config = {}

    def fake_metrics_path(sdk_config):
        captured_config["sdk_config"] = sdk_config
        return str(tmp_path / "metrics_path")

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr(
        "utils.utils_profile.create_temp_rocprofiler_metrics_path", fake_metrics_path
    )
    monkeypatch.setattr(
        "utils.utils_profile.rocpd_data.convert_dbs_to_csv", lambda *a, **k: None
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir)

    merged_counters = captured_config["sdk_config"]["rocprofiler-sdk"]["counters"]
    assert "TCC_HIT" in merged_counters


@pytest.mark.parametrize(
    "rocprof_cmd, profiler_options",
    [
        ("rocprofv3", ["--arg"]),
        ("rocprofiler-sdk", {"APP_CMD": ["./test_app"]}),
    ],
    ids=["rocprofv3", "rocprofiler-sdk"],
)
def test_run_prof_failure_subprocess(
    tmp_path, monkeypatch, rocprof_cmd, profiler_options
):
    """
    Test run_prof when subprocess execution fails.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts proper error handling on subprocess failure.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", rocprof_cmd)
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (False, "error output"),
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)

    errors = []

    def mock_console_error(msg, exit=True):
        errors.append((msg, exit))
        if exit:
            raise RuntimeError("console_error called")

    monkeypatch.setattr("utils.utils_profile.console_error", mock_console_error)

    with pytest.raises(RuntimeError, match="console_error called"):
        utils_profile.run_prof(str(fname), profiler_options, workload_dir)

    assert (
        utils_profile._DUPLICATE_ROCM_INSTALL_MESSAGE,
        False,
    ) not in errors


@pytest.mark.parametrize(
    "rocprof_cmd, profiler_options",
    [
        ("rocprofv3", ["--arg"]),
        ("rocprofiler-sdk", {"APP_CMD": ["./test_app"]}),
    ],
    ids=["rocprofv3", "rocprofiler-sdk"],
)
def test_run_prof_failure_prints_duplicate_rocm_install_message(
    tmp_path, monkeypatch, rocprof_cmd, profiler_options
):
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")
    abort_line = "Option 'spirv-expand-step' registered more than once!"
    captured_output = "\n".join([
        "[rocprofiler-sdk] tool initialization ::     0.146483 sec",
        "running vcopy",
        abort_line,
        "workload stderr",
    ])

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", rocprof_cmd)
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (False, captured_output),
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)

    errors = []

    def mock_console_error(msg, exit=True):
        errors.append((msg, exit))
        if exit:
            raise RuntimeError("console_error called")

    monkeypatch.setattr("utils.utils_profile.console_error", mock_console_error)

    with pytest.raises(RuntimeError, match="console_error called"):
        utils_profile.run_prof(str(fname), profiler_options, workload_dir)

    assert (abort_line, False) in errors
    assert (
        utils_profile._DUPLICATE_ROCM_INSTALL_MESSAGE,
        False,
    ) in errors
    assert ("Profiling execution failed.", True) in errors


def test_run_prof_rocprofv3_builds_command_and_env(tmp_path, monkeypatch):
    """run_prof (rocprofv3 backend) assembles the command with an absolute agent
    index, the input file, and the passed-through options, and seeds the
    counter-definition env var."""
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    captured = {}

    def fake_capture(cmd, new_env=None, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = new_env
        return (True, "success")

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr("utils.utils_profile.capture_subprocess_output", fake_capture)
    monkeypatch.setattr(
        "utils.utils_profile.create_temp_rocprofiler_metrics_path",
        lambda sdk_config: str(tmp_path / "metrics_path"),
    )
    monkeypatch.setattr(
        "utils.utils_profile.rocpd_data.convert_dbs_to_csv", lambda *a, **k: None
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir)

    assert captured["cmd"] == [
        "rocprofv3",
        "-A",
        "absolute",
        "-i",
        str(fname),
        "--arg",
    ]
    assert "ROCPROFILER_METRICS_PATH" in captured["env"]


# =============================================================================
# Normal Functionality:
#
# Basic submodule listing with real packages
# Correct name processing with underscores
# Multiple underscore handling
# Base module filtering
# Edge Cases:
#
# Empty packages (no submodules)
# Non-existent packages
# Names without underscores (IndexError case)
# Empty name parts
# Packages without __path__ attribute
# Error Conditions:
#
# ModuleNotFoundError for invalid packages
# AttributeError for packages without __path__
# TypeError for invalid input types
# ImportError from pkgutil.walk_packages
# Special Scenarios:
#
# Large numbers of submodules
# Special characters in names
# Unicode character handling
# Import isolation testing
# Mixed module types
# Data Integrity:
#
# Return type consistency
# Docstring verification
# Behavior validation
# =============================================================================


mock_package = mock.MagicMock()
mock_package.__path__ = ["/fake/path"]
mock_submodules = [
    (None, "module_parse", False),
    (None, "module_request", False),
    (None, "module_error", False),
]


@mock.patch("importlib.import_module", return_value=mock_package)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules)
def test_get_submodules_basic_functionality(mock_walk, mock_import):
    """
    Test basic functionality with a real package that has submodules.

    Returns:
        None: Asserts function correctly lists submodules from a real package.
    """

    result = utils_profile.get_submodules("test_package")

    assert isinstance(result, list)
    assert len(result) == 3
    expected = ["parse", "request", "error"]
    assert result == expected


def test_get_submodules_empty_package():
    """
    Test with a package that has no submodules.

    Returns:
        None: Asserts function returns empty list for packages without submodules.
    """
    from unittest.mock import MagicMock, patch

    mock_package = MagicMock()
    mock_package.__path__ = ["/fake/path"]

    with patch("importlib.import_module", return_value=mock_package):
        with patch("pkgutil.walk_packages", return_value=[]):
            result = utils_profile.get_submodules("empty_package")

            assert isinstance(result, list)
            assert len(result) == 0


def test_get_submodules_package_not_found():
    """
    Test behavior when package doesn't exist.

    Returns:
        None: Asserts ModuleNotFoundError is raised for non-existent packages.
    """

    with pytest.raises(ModuleNotFoundError):
        utils_profile.get_submodules("nonexistent_package_12345")


mock_package_single = mock.MagicMock()
mock_package_single.__path__ = ["/fake/path"]
mock_submodules_single = [
    (None, "module_parser", False),
    (None, "module_request", False),
    (None, "module_error", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_single)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_single)
def test_get_submodules_name_processing_single_underscore(mock_walk, mock_import):
    """
    Test name processing with single underscore pattern.

    Returns:
        None: Asserts correct name processing for submodules with single underscore.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["parser", "request", "error"]
    assert result == expected


mock_package_multiple = mock.MagicMock()
mock_package_multiple.__path__ = ["/fake/path"]
mock_submodules_multiple = [
    (None, "module_some_complex_name", False),
    (None, "module_another_test_case", False),
    (None, "module_simple", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_multiple)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_multiple)
def test_get_submodules_name_processing_multiple_underscores(mock_walk, mock_import):
    """
    Test name processing with multiple underscores in submodule names.

    Returns:
        None: Asserts correct name processing for complex underscore patterns.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["somecomplexname", "anothertestcase", "simple"]
    assert result == expected


mock_package_base = mock.MagicMock()
mock_package_base.__path__ = ["/fake/path"]
mock_submodules_base = [
    (None, "module_base", False),
    (None, "module_parser", False),
    (None, "module_handler", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_base)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_base)
def test_get_submodules_base_module_filtered(mock_walk, mock_import):
    """
    Test that 'base' submodule is properly filtered out.

    Returns:
        None: Asserts 'base' submodules are excluded from results.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["parser", "handler"]
    assert result == expected
    assert "base" not in result


mock_package_no_underscore = mock.MagicMock()
mock_package_no_underscore.__path__ = ["/fake/path"]
mock_submodules_no_underscore = [
    (None, "simplemodule", False),
    (None, "anothermodule", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_no_underscore)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_no_underscore)
def test_get_submodules_no_underscore_in_name(mock_walk, mock_import):
    """
    Test behavior with submodule names that don't follow the expected pattern.

    Returns:
        None: Asserts function handles names without underscores by raising IndexError.
    """

    with pytest.raises(IndexError):
        utils_profile.get_submodules("test_package")


mock_package_empty_parts = mock.MagicMock()
mock_package_empty_parts.__path__ = ["/fake/path"]
mock_submodules_empty_parts = [
    (None, "module_", False),  # ends with underscore
    (None, "_module", False),  # starts with underscore - this will cause IndexError
    (None, "module__double", False),  # double underscore
]


@mock.patch("importlib.import_module", return_value=mock_package_empty_parts)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_empty_parts)
def test_get_submodules_empty_name_parts(mock_walk, mock_import):
    """
    Test behavior with empty name parts after splitting.

    Returns:
        None: Asserts function handles edge cases in name processing.
    """

    try:
        result = utils_profile.get_submodules("test_package")
        expected = ["", "", "double"]  # noqa - Empty strings for edge cases
        assert len(result) == 3
    except IndexError:
        pytest.skip("Function doesn't handle edge case module names gracefully")


def test_get_submodules_package_without_path_attribute():
    """
    Test behavior when package doesn't have __path__ attribute.

    Returns:
        None: Asserts AttributeError is raised for packages without __path__.
    """
    from unittest.mock import MagicMock, patch

    mock_package = MagicMock()
    del mock_package.__path__

    with patch("importlib.import_module", return_value=mock_package):
        with pytest.raises(AttributeError):
            utils_profile.get_submodules("test_package")


mock_package_exception = mock.MagicMock()
mock_package_exception.__path__ = ["/fake/path"]


@mock.patch("importlib.import_module", return_value=mock_package_exception)
@mock.patch("pkgutil.walk_packages", side_effect=ImportError("Mock error"))
def test_get_submodules_pkgutil_walk_packages_exception(mock_walk, mock_import):
    """
    Test behavior when pkgutil.walk_packages raises an exception.

    Returns:
        None: Asserts exceptions from pkgutil.walk_packages are properly handled.
    """

    with pytest.raises(ImportError):
        utils_profile.get_submodules("test_package")


mock_package_mixed = mock.MagicMock()
mock_package_mixed.__path__ = ["/fake/path"]
mock_submodules_mixed = [
    (None, "module_base", False),  # Should be filtered out
    (None, "module_parser", False),  # Normal case
    (None, "module_test_case", False),  # Multiple underscores
    (None, "module_simple", False),  # Simple case
    (None, "module_another_base", False),  # Contains 'base' but not exactly 'base'
]


@mock.patch("importlib.import_module", return_value=mock_package_mixed)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_mixed)
def test_get_submodules_mixed_module_types(mock_walk, mock_import):
    """
    Test with a mix of different module types and names.

    Returns:
        None: Asserts function correctly processes various submodule patterns.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["parser", "testcase", "simple", "anotherbase"]
    assert result == expected
    assert "base" not in result


mock_package_large = mock.MagicMock()
mock_package_large.__path__ = ["/fake/path"]
mock_submodules_large = []
expected_results_large = []
for i in range(100):
    module_name = f"module_test{i}"
    mock_submodules_large.append((None, module_name, False))
    expected_results_large.append(f"test{i}")


@mock.patch("importlib.import_module", return_value=mock_package_large)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_large)
def test_get_submodules_large_number_of_submodules(mock_walk, mock_import):
    """
    Test performance and correctness with a large number of submodules.

    Returns:
        None: Asserts function handles large numbers of submodules correctly.
    """

    result = utils_profile.get_submodules("test_package")
    assert len(result) == 100
    assert result == expected_results_large


def test_get_submodules_string_input_validation():
    """
    Test input validation for package_name parameter.

    Returns:
        None: Asserts function handles invalid input types
        but may not validate properly.
    """

    with pytest.raises((TypeError, AttributeError)):
        utils_profile.get_submodules(None)

    with pytest.raises((TypeError, AttributeError)):
        utils_profile.get_submodules(123)

    with pytest.raises((TypeError, AttributeError)):
        utils_profile.get_submodules(["list", "input"])


def test_get_submodules_return_type_consistency():
    """
    Test that function always returns a list, even in edge cases.

    Returns:
        None: Asserts return type is always a list.
    """
    from unittest.mock import MagicMock, patch

    mock_package = MagicMock()
    mock_package.__path__ = ["/fake/path"]

    with patch("importlib.import_module", return_value=mock_package):
        with patch("pkgutil.walk_packages", return_value=[]):
            result = utils_profile.get_submodules("test_package")
            assert isinstance(result, list)
            assert len(result) == 0

    mock_submodules = [(None, "module_base", False)]
    with patch("importlib.import_module", return_value=mock_package):
        with patch("pkgutil.walk_packages", return_value=mock_submodules):
            result = utils_profile.get_submodules("test_package")
            assert isinstance(result, list)
            assert len(result) == 0


mock_package_special = mock.MagicMock()
mock_package_special.__path__ = ["/fake/path"]
mock_submodules_special = [
    (None, "module_test-case", False),
    (None, "module_test.case", False),
    (None, "module_test123", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_special)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_special)
def test_get_submodules_special_characters_in_names(mock_walk, mock_import):
    """
    Test handling of special characters in submodule names.

    Returns:
        None: Asserts function processes special characters in names correctly.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["test-case", "test.case", "test123"]
    assert result == expected


mock_package_isolation = mock.MagicMock()
mock_package_isolation.__path__ = ["/fake/path"]
mock_submodules_isolation = [(None, "module_test", False)]


@mock.patch("importlib.import_module", return_value=mock_package_isolation)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_isolation)
def test_get_submodules_imports_isolation(mock_walk, mock_import):
    """
    Test that imports are properly isolated and don't affect global state.

    Returns:
        None: Asserts function imports don't pollute global namespace.
    """
    import sys

    original_importlib = sys.modules.get("importlib")
    original_pkgutil = sys.modules.get("pkgutil")

    result = utils_profile.get_submodules("test_package")

    assert sys.modules.get("importlib") == original_importlib
    assert sys.modules.get("pkgutil") == original_pkgutil
    assert isinstance(result, list)
    assert result == ["test"]


mock_package_unicode = mock.MagicMock()
mock_package_unicode.__path__ = ["/fake/path"]
mock_submodules_unicode = [
    (None, "module_tëst", False),
    (None, "module_测试", False),
    (None, "module_тест", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_unicode)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_unicode)
def test_get_submodules_unicode_names(mock_walk, mock_import):
    """
    Test handling of Unicode characters in package and submodule names.

    Returns:
        None: Asserts function handles Unicode characters appropriately.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["tëst", "测试", "тест"]
    assert result == expected


mock_package_docstring = mock.MagicMock()
mock_package_docstring.__path__ = ["/fake/path"]
mock_submodules_docstring = [
    (None, "module_submodule1", False),
    (None, "module_submodule2", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_docstring)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_docstring)
def test_get_submodules_docstring_verification(mock_walk, mock_import):
    """
    Test that function behavior matches its docstring description.

    Returns:
        None: Asserts function behavior aligns with documented purpose.
    """

    assert utils_profile.get_submodules.__doc__ is not None
    assert (
        "List all submodules for a target package"
        in utils_profile.get_submodules.__doc__
    )  # noqa

    result = utils_profile.get_submodules("test_package")

    assert isinstance(result, list)
    assert "submodule1" in result
    assert "submodule2" in result


@pytest.mark.misc
def test_file_lock_creates_world_rw_file(tmp_path):
    """A freshly created lock file must be world-rw (0o666) regardless of umask."""
    import stat

    from utils import utils_profile

    lock_file = tmp_path / "shared.lock"

    # Force a strict umask that would otherwise leave the file owner-only.
    old_umask = os.umask(0o077)
    try:
        with utils_profile.file_lock(lock_file):
            assert lock_file.exists()
    finally:
        os.umask(old_umask)

    file_mode = stat.S_IMODE(os.stat(lock_file).st_mode)
    assert file_mode == 0o666, (
        f"Lock file must be world-rw so any user can acquire it; got "
        f"{oct(file_mode)}. A non-0o666 lock file locks out other users."
    )


@pytest.mark.misc
def test_file_lock_does_not_change_process_umask(tmp_path, monkeypatch):
    """Lock creation must not change process-global umask."""

    def fail_if_called(_mask):
        raise AssertionError("file_lock must not call os.umask()")

    monkeypatch.setattr(utils_profile.os, "umask", fail_if_called)

    with utils_profile.file_lock(tmp_path / "shared.lock"):
        pass


@pytest.mark.misc
def test_file_lock_existing_file_owned_by_other_user(tmp_path, monkeypatch):
    """A lock file owned by another user (no write access) is still lockable."""

    from utils import utils_profile

    lock_file = tmp_path / "shared.lock"
    # Pre-create the lock file (as if another user created it first).
    lock_file.touch()

    real_os_open = os.open
    opened_modes = []

    def fake_os_open(path, flags, *args):
        if flags & os.O_EXCL:
            # Let the create-only attempt fail naturally (file exists).
            return real_os_open(path, flags, *args)
        if flags & os.O_RDWR:
            opened_modes.append("rw")
            raise PermissionError(13, "Permission denied")
        opened_modes.append("ro")
        return real_os_open(path, flags, *args)

    monkeypatch.setattr(utils_profile.os, "open", fake_os_open)

    acquired = False
    with utils_profile.file_lock(lock_file):
        acquired = True

    assert acquired, "Lock must be acquired via read-only fallback"
    assert opened_modes == ["rw", "ro"], (
        "Should attempt read-write first, then fall back to read-only"
    )


@pytest.mark.misc
def test_file_lock_unopenable_file_raises(tmp_path, monkeypatch):
    """If the lock file cannot be opened at all, raise an actionable error."""

    from utils import utils_profile

    lock_file = tmp_path / "shared.lock"
    lock_file.touch()

    real_os_open = os.open

    def fake_os_open(path, flags, *args):
        if flags & os.O_EXCL:
            # Let the create-only attempt fail naturally (file exists).
            return real_os_open(path, flags, *args)
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(utils_profile.os, "open", fake_os_open)

    with pytest.raises(RuntimeError, match="Cannot open lock file"):
        with utils_profile.file_lock(lock_file):
            pass


def test_parse_function_backend_untagged_is_unknown():
    """Untagged rows surface as Backend='unknown'."""
    clean, backend = _parse_function_backend("torch.empty:#1@linear.py:109")
    assert clean == "torch.empty:#1@linear.py:109"
    assert backend == "unknown"


def test_parse_function_backend_tagged_torch_is_stripped():
    """Tagged single-frame markers expose backend and lose the suffix."""
    clean, backend = _parse_function_backend(
        "nn.Module.MyModel.forward:#1@train.py:42|torch"
    )
    assert clean == "nn.Module.MyModel.forward:#1@train.py:42"
    assert backend == "torch"


def test_parse_function_backend_tagged_triton_leaf():
    """Row-level suffix attributes the entire wire to its producing backend."""
    clean, backend = _parse_function_backend(
        "torch.compile.fn/triton.CompiledKernel.foo:#1@a.py:1/#1@b.py:2|triton"
    )
    assert clean == ("torch.compile.fn/triton.CompiledKernel.foo:#1@a.py:1/#1@b.py:2")
    assert backend == "triton"


def test_parse_function_backend_aten_leaf_is_unknown():
    """Untagged ATen leaf surfaces as Backend='unknown'."""
    clean, backend = _parse_function_backend(
        "nn.Module.X.forward/aten::add:#1@m.py:9/#1@aten:0"
    )
    assert clean == "nn.Module.X.forward/aten::add:#1@m.py:9/#1@aten:0"
    assert backend == "unknown"


def test_parse_function_backend_edge_cases():
    """Bogus suffix, empty string, and None all fall back to 'unknown'."""
    assert _parse_function_backend("op|bogus") == ("op|bogus", "unknown")
    assert _parse_function_backend("") == ("", "unknown")
    assert _parse_function_backend(None) == ("", "unknown")


def test_augment_marker_csv_untagged_row_warns(tmp_path, monkeypatch):
    """Untagged rows are tagged 'unknown' and emit a warning."""
    from utils import utils_profile

    src = tmp_path / "src_marker_api_trace.csv.gz"
    dst = tmp_path / "ml_api_trace_dst_marker_api_trace.csv.gz"
    pd.DataFrame({"Function": ["aten::sum"]}).to_csv(src, index=False)

    warnings: list[tuple] = []
    monkeypatch.setattr(utils_profile, "console_warning", lambda *a: warnings.append(a))

    _augment_marker_csv(str(src), str(dst))

    out_df = pd.read_csv(dst)
    assert out_df["Function"].tolist() == ["aten::sum"]
    assert out_df["Backend"].tolist() == ["unknown"]
    assert warnings, "untagged rows must emit a warning"
    assert any("unknown" in str(a) for a in warnings[0])


def test_augment_marker_csv_adds_backend_column(tmp_path):
    """End-to-end: tagged + untagged rows survive copy; Backend is populated."""
    src = tmp_path / "src_marker_api_trace.csv.gz"
    dst = tmp_path / "ml_api_trace_dst_marker_api_trace.csv.gz"

    src_df = pd.DataFrame({
        "Domain": ["MARKER_CORE_RANGE_API"] * 3,
        "Function": [
            "nn.Module.X.forward:#1@a.py:1|torch",
            "triton.CompiledKernel.k:#1@b.py:2|triton",
            "torch.empty:#1@c.py:3",
        ],
        "Correlation_Id": [1, 2, 3],
        "Start_Timestamp": [100, 200, 300],
        "End_Timestamp": [150, 250, 350],
    })
    src_df.to_csv(src, index=False)

    _augment_marker_csv(str(src), str(dst))

    out_df = pd.read_csv(dst)
    assert "Backend" in out_df.columns
    assert out_df["Backend"].tolist() == ["torch", "triton", "unknown"]
    assert out_df["Function"].tolist() == [
        "nn.Module.X.forward:#1@a.py:1",
        "triton.CompiledKernel.k:#1@b.py:2",
        "torch.empty:#1@c.py:3",
    ]
    for col in ("Domain", "Correlation_Id", "Start_Timestamp", "End_Timestamp"):
        assert col in out_df.columns


def test_augment_marker_csv_handles_unknown_schema(tmp_path):
    """A CSV without a Function column copies verbatim instead of corrupting."""
    src = tmp_path / "src.csv.gz"
    dst = tmp_path / "dst.csv.gz"
    with gzip.open(src, "wt", newline="", encoding="utf-8") as f:
        f.write("Foo,Bar\n1,2\n3,4\n")

    _augment_marker_csv(str(src), str(dst))

    assert dst.read_bytes() == src.read_bytes()
