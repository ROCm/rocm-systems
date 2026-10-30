# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Analyze writes every artifact into one new run folder and nowhere else."""

import re
import shutil
from pathlib import Path

import pytest

VCOPY_WORKLOAD = Path(__file__).resolve().parents[1] / "workloads/vcopy/MI300X_A1"

# The top-stats scratch tables are still written into the workload directory.
WORKLOAD_SCRATCH_FILES = {"pmc_dispatch_info.csv", "pmc_kernel_top.csv"}

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def copy_workload(tmp_path):
    """Return a private copy of the vcopy workload."""
    workload_path = tmp_path / "vcopy" / "MI300X_A1"
    shutil.copytree(VCOPY_WORKLOAD, workload_path)
    return workload_path


def snapshot_files(directory):
    """Return each file under directory mapped to its bytes."""
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def last_output_line(captured_output):
    """Return the last non-empty line printed, without colour codes."""
    lines = ANSI_ESCAPE.sub("", captured_output).strip().splitlines()
    return lines[-1].strip()


@pytest.mark.misc
@pytest.mark.parametrize("directory_flag", ["-d", "--output-directory"])
@pytest.mark.parametrize("name_flag", ["-o", "--output-name"])
def test_named_run_folder_is_created_with_missing_parents(
    binary_handler_analyze_rocprof_compute,
    tmp_path,
    capsys,
    directory_flag,
    name_flag,
):
    """-d names the parent, created if missing, and -o names the run folder."""
    workload_path = copy_workload(tmp_path)
    run_folder = tmp_path / "missing" / "parents" / "run1"

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(workload_path),
        directory_flag,
        str(run_folder.parent),
        name_flag,
        run_folder.name,
    ])

    assert code == 0
    assert (run_folder / "report.txt").is_file()
    assert (run_folder / "analysis.db").is_file()
    assert (run_folder / "kernel.csv").is_file()
    assert last_output_line(capsys.readouterr().out).endswith(
        f"Run folder: {run_folder.resolve()}"
    )


@pytest.mark.misc
def test_default_run_folders_are_timestamped_and_never_reused(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """Without -o every run gets its own timestamped folder."""
    workload_path = copy_workload(tmp_path)
    output_directory = tmp_path / "runs"

    for _ in range(2):
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            str(workload_path),
            "--output-directory",
            str(output_directory),
        ])
        assert code == 0

    run_folders = sorted(output_directory.iterdir())
    assert len(run_folders) == 2
    assert all(
        re.fullmatch(r"rocprof_compute_analyze_\d{8}_\d{6}(_\d+)?", folder.name)
        for folder in run_folders
    )


@pytest.mark.misc
def test_existing_output_name_errors_without_writing(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """A rerun with the same -o fails and leaves the earlier run untouched."""
    workload_path = copy_workload(tmp_path)
    run_folder = tmp_path / "runs" / "run1"
    arguments = [
        "analyze",
        "--path",
        str(workload_path),
        "--output-directory",
        str(run_folder.parent),
        "--output-name",
        run_folder.name,
    ]
    assert binary_handler_analyze_rocprof_compute(arguments) == 0
    first_run = snapshot_files(run_folder)

    assert binary_handler_analyze_rocprof_compute(arguments) == 1

    assert snapshot_files(run_folder) == first_run
    assert list(run_folder.parent.iterdir()) == [run_folder]


@pytest.mark.misc
def test_workload_directory_gains_only_the_scratch_tables(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """Analyze changes no existing workload file and adds only scratch tables."""
    workload_path = copy_workload(tmp_path)
    before = snapshot_files(workload_path)

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(workload_path),
        "--output-directory",
        str(tmp_path / "runs"),
    ])

    assert code == 0
    after = snapshot_files(workload_path)
    assert {path.name for path in after.keys() - before.keys()} <= (
        WORKLOAD_SCRATCH_FILES
    )
    assert {
        path
        for path in before
        if path.name not in WORKLOAD_SCRATCH_FILES and after[path] != before[path]
    } == set()


@pytest.mark.misc
def test_list_stats_creates_no_run_folder(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """A listing run prints and writes no run folder."""
    workload_path = copy_workload(tmp_path)
    output_directory = tmp_path / "runs"

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(workload_path),
        "--list-stats",
        "--output-directory",
        str(output_directory),
    ])

    assert code == 0
    assert not output_directory.exists()


@pytest.mark.misc
def test_invalid_dispatch_id_creates_no_run_folder(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """An unknown --kernel-iteration-range id fails before the run folder exists."""
    workload_path = copy_workload(tmp_path)
    output_directory = tmp_path / "runs"

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(workload_path),
        "--kernel-iteration-range",
        "999",
        "--output-directory",
        str(output_directory),
    ])

    assert code == 1
    assert not output_directory.exists()


@pytest.mark.misc
def test_removed_dispatch_option_is_rejected(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """The old --dispatch spelling is no longer accepted."""
    workload_path = copy_workload(tmp_path)

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(workload_path),
        "--dispatch",
        "1",
    ])

    assert code == 2
