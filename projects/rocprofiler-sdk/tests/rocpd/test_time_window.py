#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
###############################################################################

import importlib.util
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest


def load_time_window_module():
    """Import rocpd.time_window without the compiled libpyrocpd dependency."""

    package_name = "_rocpd_time_window_test"
    package = types.ModuleType(package_name)
    package.__path__ = []

    importer = types.ModuleType(f"{package_name}.importer")

    class RocpdImportData:
        pass

    def execute_statement(connection, statement, is_script=False):
        database = (
            connection.connection if isinstance(connection, ImportData) else connection
        )
        assert isinstance(database, sqlite3.Connection)
        if is_script:
            return database.executescript(statement)
        return database.execute(statement)

    importer.RocpdImportData = RocpdImportData
    importer.execute_statement = execute_statement
    sys.modules[package_name] = package
    sys.modules[importer.__name__] = importer

    # configured into the build tree by CMake; falls back to the source tree
    test_dir = Path("@CMAKE_CURRENT_SOURCE_DIR@")
    if not test_dir.is_absolute():
        test_dir = Path(__file__).resolve().parent
    source_path = test_dir.parents[1] / "source/lib/python/rocpd/time_window.py"
    assert source_path.is_file(), f"cannot locate {source_path}"

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.time_window", source_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ImportData:
    """Minimal stand-in for RocpdImportData backed by an in-memory database."""

    def __init__(self):
        self.connection = sqlite3.connect(":memory:")
        self.table_info = {}

    def execute(self, *args):
        return self.connection.execute(*args)

    def commit(self):
        return self.connection.commit()

    def close(self):
        self.connection.close()

    def add_timed_view(self, name, columns, rows, column_types=None):
        source_name = f"source_{name}"
        column_types = column_types or {}
        definitions = ", ".join(
            f'"{column}" {column_types.get(column, "INTEGER")}' for column in columns
        )
        self.execute(f'CREATE TABLE "{source_name}" ({definitions})')
        placeholders = ", ".join("?" for _ in columns)
        self.connection.executemany(
            f'INSERT INTO "{source_name}" VALUES ({placeholders})', rows
        )
        self.execute(f'CREATE TEMPORARY VIEW "{name}" AS SELECT * FROM "{source_name}"')
        self.table_info[name] = [f'SELECT * FROM "{source_name}"']

    def add_markers(self, markers):
        rows = [
            (timestamp, timestamp, "MARKER_CORE_API", f'{{"message": "{name}"}}')
            for name, timestamp in markers
        ]
        self.add_timed_view(
            "regions_and_samples",
            ("start", "end", "category", "extdata"),
            rows,
            {"category": "TEXT", "extdata": "TEXT"},
        )


time_window = load_time_window_module()


@pytest.fixture
def import_data():
    data = ImportData()
    yield data
    data.close()


@pytest.fixture
def intervals(import_data):
    import_data.add_timed_view("intervals", ("start", "end"), ((100, 200),))
    return import_data


#
# Bounds must cover every table the window actually filters
#
def test_bounds_include_timestamp_only_tables(intervals):
    intervals.add_timed_view("samples", ("timestamp",), ((250,),))

    assert time_window.get_min_max_time(intervals) == (100, 250)


def test_native_sqlite_uses_event_bounds_not_process_envelope():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE intervals (start INTEGER, end INTEGER)")
    connection.execute("CREATE TABLE samples (timestamp INTEGER)")
    connection.execute(
        "CREATE TABLE rocpd_info_process (init INTEGER, fini INTEGER, "
        "start INTEGER, end INTEGER)"
    )
    connection.execute("INSERT INTO intervals VALUES (100, 200)")
    connection.execute("INSERT INTO samples VALUES (250)")
    connection.execute("INSERT INTO rocpd_info_process VALUES (0, 1000, 0, 1000)")
    connection.execute(
        "CREATE VIEW processes AS SELECT init, fini, start, end "
        "FROM rocpd_info_process"
    )
    try:
        assert time_window.percentages2timestamp(connection, "100", "250") == (
            100,
            250,
        )
    finally:
        connection.close()


def test_bounds_query_aggregates_each_table_before_union(intervals, monkeypatch):
    intervals.add_timed_view("samples", ("timestamp",), ((250,),))
    statements = []
    original_execute = time_window.execute_statement

    def record_statement(connection, statement, is_script=False):
        statements.append(statement)
        return original_execute(connection, statement, is_script)

    monkeypatch.setattr(time_window, "execute_statement", record_statement)

    assert time_window.get_min_max_time(intervals) == (100, 250)
    assert "SELECT MIN(start)" in statements[-1]
    assert "SELECT MIN(timestamp)" in statements[-1]


def test_timestamp_only_events_are_not_out_of_range(intervals):
    intervals.add_timed_view("samples", ("timestamp",), ((250,),))

    time_window.apply_time_window(intervals, start="225", end="250")

    assert intervals.execute("SELECT COUNT(*) FROM intervals").fetchone()[0] == 0
    assert intervals.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 1


def test_no_timed_events_warning_accounts_for_timestamp_tables(intervals, capsys):
    intervals.add_timed_view("samples", ("timestamp",), ((250,),))

    time_window.apply_time_window(intervals, start="225", end="250")

    captured = capsys.readouterr()
    assert "contains no timed events" not in (captured.out + captured.err)


def test_empty_window_reports_no_timed_events(import_data, capsys):
    import_data.add_timed_view("intervals", ("start", "end"), ((100, 110), (190, 200)))

    time_window.apply_time_window(import_data, start="120", end="180")

    assert "contains no timed events" in capsys.readouterr().err


def test_zero_length_trace_does_not_divide_by_zero(import_data, capsys):
    import_data.add_timed_view("intervals", ("start", "end"), ((100, 100),))
    import_data.add_markers((("begin", 100), ("finish", 100)))

    time_window.apply_time_window(import_data, start_marker="begin", end_marker="finish")

    assert "reduced the duration" not in capsys.readouterr().out


#
# Percentage parsing
#
@pytest.mark.parametrize("value", ("50%", " 50% ", "5e1%"))
def test_valid_percentage_is_converted(intervals, value):
    assert time_window.percentages2timestamp(intervals, value, None) == (150, 200)


@pytest.mark.parametrize("value", (150, 150.0))
def test_numeric_endpoints_are_accepted(intervals, value):
    assert time_window.percentages2timestamp(intervals, value, None) == (150, 200)


@pytest.mark.parametrize("value", ("50%%", "5%0", "%50", "%", "abc%", "5 0%"))
def test_malformed_percentages_are_rejected(intervals, value):
    with pytest.raises(time_window.TimeWindowError, match="Invalid --start percentage"):
        time_window.percentages2timestamp(intervals, value, None)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    (("-1%", None, (100, 200)), (None, "101%", (100, 200))),
)
def test_overhanging_percentages_are_clamped(intervals, capsys, start, end, expected):
    assert time_window.percentages2timestamp(intervals, start, end) == expected
    assert "using time window [100, 200] nsec instead" in capsys.readouterr().err


#
# Timestamp parsing, clamping and overlap handling
#
@pytest.mark.parametrize("value", ("nan", "inf", "-inf"))
def test_non_finite_timestamps_are_rejected(intervals, value):
    with pytest.raises(time_window.TimeWindowError, match="must be a finite"):
        time_window.percentages2timestamp(intervals, value, None)


def test_invalid_timestamp_identifies_the_argument(intervals):
    with pytest.raises(time_window.TimeWindowError, match="Invalid --end value"):
        time_window.percentages2timestamp(intervals, None, "later")


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    (
        ("0", None, (100, 200)),
        (None, "999999", (100, 200)),
        ("0", "999999", (100, 200)),
        ("150", "999999", (150, 200)),
    ),
)
def test_overhanging_bounds_are_clamped(intervals, capsys, start, end, expected):
    assert time_window.percentages2timestamp(intervals, start, end) == expected
    warning = capsys.readouterr().err
    assert "WARNING" in warning
    assert f"using time window [{expected[0]}, {expected[1]}] nsec instead" in warning


def test_bounds_inside_the_trace_are_not_clamped(intervals, capsys):
    assert time_window.percentages2timestamp(intervals, "120", "180") == (120, 180)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(("start", "end"), (("300", "400"), ("0", "50")))
def test_disjoint_window_is_rejected(intervals, start, end):
    with pytest.raises(time_window.TimeWindowError, match="does not overlap"):
        time_window.percentages2timestamp(intervals, start, end)


def test_missing_timing_data_is_reported(import_data):
    with pytest.raises(time_window.TimeWindowError, match="no timing data"):
        time_window.percentages2timestamp(import_data, "50%", None)


#
# Marker handling
#
def test_missing_marker_view_has_specific_error():
    connection = sqlite3.connect(":memory:")
    try:
        with pytest.raises(time_window.TimeWindowError, match="no marker data"):
            time_window.get_marker_timestamp(connection, "target")
    finally:
        connection.close()


def test_marker_schema_errors_are_not_reclassified():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE regions_and_samples (category TEXT)")
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such column"):
            time_window.get_marker_timestamp(connection, "target")
    finally:
        connection.close()


def test_unrelated_database_errors_are_not_reclassified():
    class LockedConnection:
        def execute(self, *_args):
            raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        time_window.get_marker_timestamp(LockedConnection(), "target")


def test_unknown_marker_is_reported(intervals):
    intervals.add_markers((("begin", 120),))

    with pytest.raises(time_window.TimeWindowError, match="not found"):
        time_window.get_marker_timestamp(intervals, "missing")


def test_ambiguous_marker_is_reported(intervals):
    intervals.add_markers((("begin", 120), ("begin", 130)))

    with pytest.raises(time_window.TimeWindowError, match="Ambiguous reference"):
        time_window.get_marker_timestamp(intervals, "begin")


def test_reversed_markers_report_marker_arguments(intervals):
    intervals.add_markers((("begin", 180), ("finish", 120)))

    with pytest.raises(time_window.TimeWindowError) as excinfo:
        time_window.apply_time_window(
            intervals, start_marker="begin", end_marker="finish"
        )

    message = str(excinfo.value)
    assert "--end-marker" in message and "--start-marker" in message


def test_reversed_timestamps_report_time_arguments(intervals):
    with pytest.raises(time_window.TimeWindowError) as excinfo:
        time_window.apply_time_window(intervals, start="180", end="120")

    message = str(excinfo.value)
    assert "--end (120)" in message and "--start (180)" in message


#
# Mixed marker and timestamp endpoints
#
@pytest.mark.parametrize(
    "kwargs",
    (
        {"start": "100", "end_marker": "finish"},
        {"start_marker": "begin", "end": "200"},
    ),
)
def test_mixed_marker_and_timestamp_endpoints_are_supported(intervals, kwargs):
    intervals.add_markers((("begin", 100), ("finish", 200)))

    time_window.apply_time_window(intervals, **kwargs)

    assert intervals.execute("SELECT COUNT(*) FROM intervals").fetchone()[0] == 1


def test_same_endpoint_cannot_use_time_and_marker(intervals):
    intervals.add_markers((("begin", 100),))

    with pytest.raises(
        time_window.TimeWindowError, match="both --start and --start-marker"
    ):
        time_window.apply_time_window(
            intervals, start="100", start_marker="begin", end="200"
        )


def test_no_window_arguments_is_a_no_op(intervals):
    assert time_window.apply_time_window(intervals, inclusive=True) is intervals
    assert intervals.execute("SELECT COUNT(*) FROM intervals").fetchone()[0] == 1


#
# Command-line error reporting
#
def test_process_args_or_exit_reports_time_window_errors():
    def fail(_input, _args):
        raise time_window.TimeWindowError("ERROR: bad window")

    with pytest.raises(SystemExit) as excinfo:
        time_window.process_args_or_exit(fail, None, None)

    assert str(excinfo.value) == "ERROR: bad window"


def test_process_args_or_exit_does_not_swallow_other_errors():
    def fail(_input, _args):
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        time_window.process_args_or_exit(fail, None, None)


def test_standalone_main_reports_invalid_window_without_traceback(intervals, monkeypatch):
    monkeypatch.setattr(time_window, "RocpdImportData", lambda _input: intervals)

    with pytest.raises(SystemExit) as excinfo:
        time_window.main(["--input", "trace.db", "--start", "invalid"])

    assert str(excinfo.value).startswith("ERROR: Invalid --start value")


@pytest.fixture(scope="module")
def real_rocpd_database():
    database = Path("@ROCPD_TIME_WINDOW_TEST_DB@")
    if not database.is_file():
        pytest.skip("configured ROCpd integration database is unavailable")
    return database


def run_rocpd_query(module_args, database, *time_window_args):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            *module_args,
            "--input",
            str(database),
            "--query",
            "SELECT COUNT(*) FROM regions",
            *time_window_args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("module_args", (("rocpd", "query"), ("rocpd.query",)))
def test_query_clis_report_disjoint_window_without_traceback(
    real_rocpd_database, module_args
):
    result = run_rocpd_query(
        module_args, real_rocpd_database, "--start", "0", "--end", "1"
    )

    assert result.returncode != 0
    assert "does not overlap trace time range" in result.stderr
    assert "Traceback" not in result.stderr


def test_query_cli_warns_and_continues_for_partial_overlap(real_rocpd_database):
    result = run_rocpd_query(
        ("rocpd.query",),
        real_rocpd_database,
        "--start",
        "0",
        "--end",
        "2000000000",
    )

    assert result.returncode == 0
    assert "# WARNING:" in result.stderr
    assert "using time window" in result.stderr
    assert "Traceback" not in result.stderr
