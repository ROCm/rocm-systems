# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the analyze run flow in src/rocprof_compute_base.py."""

import argparse
import functools
import re
import sqlite3

import common
import pytest
from sqlalchemy.exc import DBAPIError

from rocprof_compute_base import RocProfCompute

MODULE = "rocprof_compute_base"


class RecordingAnalyzer:
    """Analyzer stand-in that records each call into a shared list."""

    def __init__(self, name, calls, write_error=None, build_error=None):
        self.name = name
        self.calls = calls
        self.write_error = write_error
        self.build_error = build_error

    def __call__(self, args, supported_archs):
        return self

    def sanitize(self):
        self.calls.append((self.name, "sanitize"))

    def set_soc(self, soc):
        self.calls.append((self.name, "set_soc"))

    def pre_processing(self):
        self.calls.append((self.name, "pre_processing"))

    def build_database(self):
        self.calls.append((self.name, "build_database"))
        if self.build_error is not None:
            raise self.build_error

    def show_listing(self):
        self.calls.append((self.name, "show_listing"))

    def write_outputs(self, layout):
        folders_exist = layout.run_folder.is_dir() and all(
            folder.is_dir() for folder in layout.workload_folders.values()
        )
        self.calls.append((self.name, "write_outputs", folders_exist))
        if self.write_error is not None:
            raise self.write_error


def make_analyze_args(**overrides):
    """Build the analyze arguments run_analysis reads."""
    values = {
        "path": [["/workload"]],
        "output_directory": None,
        "output_name": None,
        "list_stats": False,
        "list_available_metrics": False,
        "list_torch_operators": False,
        "list_triton_operators": False,
        "torch_operator": None,
        "triton_operator": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def make_rocprof_compute(
    args, monkeypatch, calls, db_write_error=None, db_build_error=None
):
    """Return a RocProfCompute whose analyzers record into calls."""
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.cli_analysis",
        RecordingAnalyzer("cli", calls),
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.db_analysis",
        RecordingAnalyzer(
            "db",
            calls,
            write_error=db_write_error,
            build_error=db_build_error,
        ),
    )
    instance = RocProfCompute.__new__(RocProfCompute)
    instance._RocProfCompute__args = args
    instance._RocProfCompute__soc = {}
    instance._RocProfCompute__supported_archs = {}
    monkeypatch.setattr(instance, "print_graphic", lambda: None)
    monkeypatch.setattr(instance, "_load_workload_soc_specs", lambda: None)
    return instance


def record_error_and_exit(errors, *args, **kwargs):
    """Record a console_error call, then exit like the real one."""
    errors.append(args)
    raise SystemExit(1)


def test_run_analysis_builds_the_database_before_creating_the_run_folder(
    tmp_path, monkeypatch
):
    """The database is built in memory first; the folders come with the writes."""
    mocks = common.patch_console(monkeypatch, MODULE, "log", "warning")
    calls = []
    args = make_analyze_args(output_directory=str(tmp_path / "new"), output_name="r1")

    instance = make_rocprof_compute(args, monkeypatch, calls)
    prepared_run = instance.run_analysis()
    run_folder = tmp_path / "new" / "r1"
    assert prepared_run.layout.run_folder == run_folder
    assert not run_folder.exists()
    assert ("db", "build_database") in calls
    assert not any(call[1] == "write_outputs" for call in calls)

    instance.write_analysis_outputs(prepared_run)

    assert prepared_run.layout.workload_folders == {
        "/workload": run_folder / "workload-1"
    }
    assert (run_folder / "workload-1").is_dir()
    write_calls = [call for call in calls if call[1] == "write_outputs"]
    assert write_calls == [
        ("db", "write_outputs", True),
        ("cli", "write_outputs", True),
    ]
    assert mocks["log"].call_args.args[1] == f"Run folder: {run_folder.resolve()}"


def test_run_analysis_defaults_to_a_timestamped_folder_in_the_cwd(
    tmp_path, monkeypatch
):
    """Without -d or -o the run folder is a timestamped name in the cwd."""
    common.patch_console(monkeypatch, MODULE, "log", "warning")
    monkeypatch.chdir(tmp_path)

    instance = make_rocprof_compute(make_analyze_args(), monkeypatch, [])
    prepared_run = instance.run_analysis()
    assert not any(tmp_path.iterdir())

    instance.write_analysis_outputs(prepared_run)

    created = list(tmp_path.iterdir())
    assert len(created) == 1
    assert created[0].match("rocprof_compute_analyze_*")


@pytest.mark.parametrize(
    "listing_option",
    [
        "list_stats",
        "list_available_metrics",
        "list_torch_operators",
        "list_triton_operators",
    ],
)
def test_run_analysis_listing_creates_no_run_folder(
    listing_option, tmp_path, monkeypatch
):
    """A listing run prints through the CLI analyzer and writes nothing."""
    common.patch_console(monkeypatch, MODULE, "log", "warning")
    monkeypatch.chdir(tmp_path)
    calls = []
    instance = make_rocprof_compute(
        make_analyze_args(**{listing_option: True}), monkeypatch, calls
    )
    monkeypatch.setattr(instance, "list_metrics", lambda: None)

    instance.run_analysis()

    assert list(tmp_path.iterdir()) == []
    assert ("cli", "show_listing") in calls
    assert all(call[0] == "cli" for call in calls)


def test_run_analysis_existing_output_name_gets_collision_suffix(tmp_path, monkeypatch):
    """An existing -o folder is preserved and the rerun gets a unique name."""
    common.patch_console(monkeypatch, MODULE, "log", "warning")
    existing_run = tmp_path / "r1"
    existing_run.mkdir()
    calls = []
    args = make_analyze_args(output_directory=str(tmp_path), output_name="r1")

    instance = make_rocprof_compute(args, monkeypatch, calls)
    prepared_run = instance.run_analysis()
    assert list(tmp_path.iterdir()) == [existing_run]

    instance.write_analysis_outputs(prepared_run)

    created_runs = list(tmp_path.glob("r1_*"))
    assert len(created_runs) == 1
    assert re.fullmatch(r"r1_\d{8}_\d{6}(?:_\d+)?", created_runs[0].name)
    assert existing_run.is_dir()
    assert prepared_run.layout.run_folder == created_runs[0]


def test_run_analysis_invalid_output_name_errors_before_calculation(
    tmp_path, monkeypatch
):
    """An -o name outside [A-Za-z0-9_-] stops the run before any analyzer runs."""
    errors = []
    monkeypatch.setattr(
        f"{MODULE}.console_error", functools.partial(record_error_and_exit, errors)
    )
    calls = []
    args = make_analyze_args(output_directory=str(tmp_path), output_name="bad name")

    with pytest.raises(SystemExit):
        make_rocprof_compute(args, monkeypatch, calls).run_analysis()

    assert calls == []
    assert list(tmp_path.iterdir()) == []
    assert "only alphanumeric characters" in errors[-1][1]


@pytest.mark.parametrize(
    "write_error",
    [
        OSError("disk full"),
        sqlite3.OperationalError("disk I/O error"),
        DBAPIError("SELECT 1", None, sqlite3.OperationalError("disk I/O error")),
    ],
)
def test_output_failure_names_the_partial_run_folder(
    write_error, tmp_path, monkeypatch
):
    """Database and artifact write failures name the partial run folder."""
    common.patch_console(monkeypatch, MODULE, "log", "warning")
    errors = []
    monkeypatch.setattr(
        f"{MODULE}.console_error", functools.partial(record_error_and_exit, errors)
    )
    args = make_analyze_args(output_directory=str(tmp_path), output_name="r1")
    instance = make_rocprof_compute(args, monkeypatch, [], db_write_error=write_error)
    prepared_run = instance.run_analysis()

    with pytest.raises(SystemExit):
        instance.write_analysis_outputs(prepared_run)

    run_folder = tmp_path / "r1"
    assert run_folder.is_dir()
    assert str(run_folder.resolve()) in errors[-1][1]
    assert str(write_error) in errors[-1][1]


def test_database_build_failure_creates_no_run_folder(tmp_path, monkeypatch):
    """An input failure while building the database leaves nothing behind."""
    common.patch_console(monkeypatch, MODULE, "log", "warning")
    args = make_analyze_args(output_directory=str(tmp_path), output_name="r1")
    build_error = ValueError("unreadable PC sampling results")
    instance = make_rocprof_compute(args, monkeypatch, [], db_build_error=build_error)

    with pytest.raises(ValueError, match="unreadable PC sampling results"):
        instance.run_analysis()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("operator_option", ["torch_operator", "triton_operator"])
def test_run_analysis_operator_filter_skips_the_database(
    operator_option, tmp_path, monkeypatch
):
    """With an operator filter only the CLI analyzer runs, with a warning."""
    mocks = common.patch_console(monkeypatch, MODULE, "log", "warning")
    calls = []
    args = make_analyze_args(
        output_directory=str(tmp_path), output_name="r1", **{operator_option: ["*"]}
    )

    instance = make_rocprof_compute(args, monkeypatch, calls)
    prepared_run = instance.run_analysis()
    instance.write_analysis_outputs(prepared_run)

    assert all(call[0] == "cli" for call in calls)
    assert ("cli", "write_outputs", True) in calls
    assert "analysis.db" in mocks["warning"].call_args.args[1]
