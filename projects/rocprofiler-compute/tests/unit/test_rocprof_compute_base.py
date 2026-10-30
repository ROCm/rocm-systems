# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the analyze run flow in src/rocprof_compute_base.py."""

import argparse
import functools
import sqlite3

import common
import pytest

from rocprof_compute_base import RocProfCompute

MODULE = "rocprof_compute_base"


class RecordingAnalyzer:
    """Analyzer stand-in that records each call into a shared list."""

    def __init__(self, name, calls, write_error=None):
        self.name = name
        self.calls = calls
        self.write_error = write_error

    def __call__(self, args, supported_archs):
        return self

    def sanitize(self):
        self.calls.append((self.name, "sanitize"))

    def set_soc(self, soc):
        self.calls.append((self.name, "set_soc"))

    def pre_processing(self):
        self.calls.append((self.name, "pre_processing"))

    def build_database(self, db_path):
        self.calls.append((self.name, "build_database", db_path))

    def build_report_frames(self):
        self.calls.append((self.name, "build_report_frames"))
        self.report_runs = {}

    def kernel_rankings(self):
        return {}

    def set_report_runs(self, report_runs, kernel_rankings):
        self.calls.append((self.name, "set_report_runs"))

    def show_listing(self):
        self.calls.append((self.name, "show_listing"))

    def write_outputs(self, run_folder):
        self.calls.append((self.name, "write_outputs", run_folder.is_dir()))
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


def make_rocprof_compute(args, monkeypatch, calls, db_write_error=None):
    """Return a RocProfCompute whose analyzers record into calls."""
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.cli_analysis",
        RecordingAnalyzer("cli", calls),
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.db_analysis",
        RecordingAnalyzer("db", calls, write_error=db_write_error),
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


def test_run_analysis_writes_both_analyzers_into_a_new_run_folder(
    tmp_path, monkeypatch
):
    """The DB analyzer then the CLI analyzer write into the created folder."""
    mocks = common.patch_console(monkeypatch, MODULE, "log", "warning")
    calls = []
    args = make_analyze_args(output_directory=str(tmp_path / "new"), output_name="r1")

    make_rocprof_compute(args, monkeypatch, calls).run_analysis()

    run_folder = tmp_path / "new" / "r1"
    assert run_folder.is_dir()
    assert ("db", "build_database", run_folder / "analysis.db") in calls
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

    make_rocprof_compute(make_analyze_args(), monkeypatch, []).run_analysis()

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


def test_run_analysis_existing_output_name_errors_before_calculation(
    tmp_path, monkeypatch
):
    """An existing -o folder stops the run before any analyzer runs."""
    errors = []
    monkeypatch.setattr(
        f"{MODULE}.console_error", functools.partial(record_error_and_exit, errors)
    )
    (tmp_path / "r1").mkdir()
    calls = []
    args = make_analyze_args(output_directory=str(tmp_path), output_name="r1")

    with pytest.raises(SystemExit):
        make_rocprof_compute(args, monkeypatch, calls).run_analysis()

    assert calls == []
    assert "already exists" in errors[-1][1]


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
    "write_error", [OSError("disk full"), sqlite3.OperationalError("disk I/O error")]
)
def test_run_analysis_write_failure_names_the_partial_run_folder(
    write_error, tmp_path, monkeypatch
):
    """A write failure leaves the run folder and the error names it."""
    common.patch_console(monkeypatch, MODULE, "log", "warning")
    errors = []
    monkeypatch.setattr(
        f"{MODULE}.console_error", functools.partial(record_error_and_exit, errors)
    )
    args = make_analyze_args(output_directory=str(tmp_path), output_name="r1")
    instance = make_rocprof_compute(args, monkeypatch, [], db_write_error=write_error)

    with pytest.raises(SystemExit):
        instance.run_analysis()

    run_folder = tmp_path / "r1"
    assert run_folder.is_dir()
    assert str(run_folder.resolve()) in errors[-1][1]
    assert str(write_error) in errors[-1][1]


@pytest.mark.parametrize("operator_option", ["torch_operator", "triton_operator"])
def test_run_analysis_operator_filter_keeps_the_database(
    operator_option, tmp_path, monkeypatch
):
    """An operator filter changes no calculation, so the DB is still built."""
    common.patch_console(monkeypatch, MODULE, "log", "warning")
    calls = []
    args = make_analyze_args(
        output_directory=str(tmp_path), output_name="r1", **{operator_option: ["*"]}
    )

    make_rocprof_compute(args, monkeypatch, calls).run_analysis()

    assert ("db", "build_database", tmp_path / "r1" / "analysis.db") in calls
    assert ("db", "build_report_frames") in calls
    assert ("cli", "set_report_runs") in calls
    write_calls = [call for call in calls if call[1] == "write_outputs"]
    assert write_calls == [
        ("db", "write_outputs", True),
        ("cli", "write_outputs", True),
    ]
