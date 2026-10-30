# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/terminal_summary.py."""

import io
from pathlib import Path
from typing import List

from rocprof_compute_analyze.terminal_summary import (
    WorkloadArtifacts,
    artifact_group_lines,
    print_terminal_summary,
    read_artifact_groups,
    workload_folder_names,
    workload_mapping_lines,
)


def make_run_folder(run_folder: Path, folder_names: List[str]) -> None:
    """Create a run folder with the root DB and report and one workload folder
    holding two CSV files and one kernel folder for each name given."""
    run_folder.mkdir(parents=True)
    (run_folder / "analysis.db").write_text("", encoding="utf-8")
    (run_folder / "report.txt").write_text("", encoding="utf-8")
    for folder_name in folder_names:
        folder = run_folder / folder_name
        folder.mkdir()
        (folder / "kernels.csv").write_text("", encoding="utf-8")
        (folder / "dispatches.csv").write_text("", encoding="utf-8")
        (folder / "kernel-000-vecCopy").mkdir()


def test_workload_folder_names_follow_the_p_order() -> None:
    """Each -p path gets its 1-based position and its directory name."""
    assert workload_folder_names(["/runs/MI300X_A1", "/runs/MI200"]) == [
        "1-MI300X_A1",
        "2-MI200",
    ]


def test_read_artifact_groups_reads_root_and_workload_folders(tmp_path) -> None:
    """The reader returns the sorted root files and each workload folder's files
    and kernel folder count, in -p order."""
    run_folder = tmp_path / "run"
    make_run_folder(run_folder, ["1-vcopy", "2-MI200"])

    root_file_names, workloads = read_artifact_groups(
        run_folder, ["/work/vcopy", "/work/MI200"]
    )

    assert root_file_names == ["analysis.db", "report.txt"]
    assert workloads == [
        WorkloadArtifacts("1-vcopy", ["dispatches.csv", "kernels.csv"], 1),
        WorkloadArtifacts("2-MI200", ["dispatches.csv", "kernels.csv"], 1),
    ]


def test_artifact_groups_list_root_files_then_each_workload_folder() -> None:
    """The root lists its files; a workload folder lists its files and kernel count."""
    workloads = [WorkloadArtifacts("1-vcopy", ["dispatches.csv", "kernels.csv"], 1)]

    assert artifact_group_lines(["analysis.db", "report.txt"], workloads) == [
        "  analysis.db",
        "  report.txt",
        "  1-vcopy/",
        "    dispatches.csv, kernels.csv",
        "    1 kernel folder",
    ]


def test_artifact_groups_count_kernel_folders_in_plural() -> None:
    """A workload with several kernel folders says so in the plural."""
    workloads = [WorkloadArtifacts("1-vcopy", ["kernels.csv"], 2)]

    lines = artifact_group_lines([], workloads)

    assert lines[-1] == "    2 kernel folders"


def test_artifact_groups_follow_the_p_order_for_comparisons() -> None:
    """Workload folders are listed in -p order, not in name order."""
    workloads = [
        WorkloadArtifacts("1-vcopy", [], 0),
        WorkloadArtifacts("2-MI200", [], 0),
    ]

    lines = artifact_group_lines([], workloads)

    folder_lines = [line for line in lines if line.endswith("/")]
    assert folder_lines == ["  1-vcopy/", "  2-MI200/"]


def test_artifact_groups_wrap_a_long_file_list() -> None:
    """A file list wider than the terminal wraps onto indented lines, and no file
    name is split at its hyphens."""
    names = [f"roofline-gpu-{index:02d}-long-name.html" for index in range(8)]
    workloads = [WorkloadArtifacts("1-vcopy", names, 1)]

    lines = artifact_group_lines([], workloads)

    file_lines = lines[lines.index("  1-vcopy/") + 1 : -1]
    assert len(file_lines) > 1
    assert all(line.startswith("    ") and len(line) <= 80 for line in file_lines)
    assert all(any(name in line for line in file_lines) for name in names)


def test_workload_mapping_names_the_folder_of_each_p_path() -> None:
    """Every -p path is mapped to the folder that holds its workload."""
    assert workload_mapping_lines(["/work/MI300X_A1", "/work/MI200"]) == [
        "  -p /work/MI300X_A1 -> 1-MI300X_A1/",
        "  -p /work/MI200 -> 2-MI200/",
    ]


def test_single_workload_summary_has_no_mapping(tmp_path) -> None:
    """A single -p run prints the artifacts and no -p mapping."""
    run_folder = tmp_path / "run"
    make_run_folder(run_folder, ["1-vcopy"])
    output = io.StringIO()

    print_terminal_summary(run_folder, ["/work/vcopy"], output)

    text = output.getvalue()
    assert text.startswith("\n" + "-" * 80 + "\nArtifacts written\n")
    assert "-p " not in text


def test_comparison_summary_ends_with_the_p_mapping(tmp_path) -> None:
    """A comparison prints the artifacts, then the -p mapping, in that order."""
    run_folder = tmp_path / "run"
    make_run_folder(run_folder, ["1-vcopy", "2-MI200"])
    output = io.StringIO()

    print_terminal_summary(run_folder, ["/work/vcopy", "/work/MI200"], output)

    text = output.getvalue()
    assert text.index("Artifacts written") < text.index("Workload folders of the -p")
    assert text.rstrip().endswith("  -p /work/MI200 -> 2-MI200/")
