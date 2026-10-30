# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Analyze writes every artifact into one new run folder and nowhere else."""

import errno
import os
import re
import shutil
import stat
from pathlib import Path

import pandas as pd
import pytest

from tests.integration.common import RUN_FOLDER_PREFIX, last_output_line
from utils.utils_common import strip_ansi

VCOPY_WORKLOAD = Path(__file__).resolve().parents[1] / "workloads/vcopy/MI300X_A1"
MEM_LEVELS_WORKLOAD = (
    Path(__file__).resolve().parents[1] / "workloads/mem_levels_HBM/MI200"
)

PANEL_HEADING = re.compile(r"^(\d+)\. ", re.MULTILINE)
ARTIFACTS_HEADING = "Artifacts written"


def copy_workload(tmp_path):
    """Return a private copy of the vcopy workload."""
    workload_path = tmp_path / "vcopy" / "MI300X_A1"
    shutil.copytree(VCOPY_WORKLOAD, workload_path)
    return workload_path


def snapshot_files(directory):
    """Map every path under directory to its bytes, or None for a directory."""
    return {
        path.relative_to(directory): None if path.is_dir() else path.read_bytes()
        for path in directory.rglob("*")
    }


def panel_numbers(text):
    """Return the number of every panel heading, e.g. 2 for "2. System SOL"."""
    return {int(number) for number in PANEL_HEADING.findall(strip_ansi(text))}


def terminal_panels(captured_output):
    """Return the printed panels, from the first heading up to the artifact list."""
    text = strip_ansi(captured_output)
    start = text.index("0. Top Stats")
    return text[start : text.index(ARTIFACTS_HEADING)]


def artifacts_section(captured_output):
    """Return the terminal lines from the artifact list to the run folder line."""
    text = strip_ansi(captured_output)
    return text[text.index(ARTIFACTS_HEADING) : text.rindex(RUN_FOLDER_PREFIX)]


def run_into(analyze, workload_path, run_folder, *options):
    """Analyze workload_path into run_folder with any extra options."""
    return analyze([
        "analyze",
        "--path",
        str(workload_path),
        "--output-directory",
        str(run_folder.parent),
        "--output-name",
        run_folder.name,
        *options,
    ])


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
    assert (run_folder / "1-MI300X_A1" / "kernels.csv").is_file()
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
def test_run_leaves_the_workload_directory_byte_identical(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """A run adds no file or directory to the workload directory and changes none."""
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
    assert snapshot_files(workload_path) == before


@pytest.mark.misc
def test_run_folder_inside_the_workload_adds_only_that_folder(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """With -d inside the workload, existing files keep their bytes and the only
    new entries are under the run folder."""
    workload_path = copy_workload(tmp_path)
    before = snapshot_files(workload_path)
    run_folder = workload_path / "runs" / "run1"

    code = run_into(binary_handler_analyze_rocprof_compute, workload_path, run_folder)

    assert code == 0
    assert (run_folder / "report.txt").is_file()
    after = snapshot_files(workload_path)
    assert {path: after[path] for path in before} == before
    new_paths = after.keys() - before.keys()
    assert new_paths
    assert all(path.parts[0] == "runs" for path in new_paths)


def make_read_only(directory):
    """Remove write permission from every file and directory under directory."""
    for path in [directory, *directory.rglob("*")]:
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def restore_write(directory):
    """Give the owner write permission back so pytest can clean up tmp_path."""
    for path in [directory, *directory.rglob("*")]:
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


@pytest.mark.misc
def test_read_only_workload_directory_analyzes(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """A workload that cannot be written to still analyzes into a run folder."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory write permissions")
    workload_path = copy_workload(tmp_path)
    before = snapshot_files(workload_path)
    run_folder = tmp_path / "runs" / "run1"
    make_read_only(workload_path)
    try:
        code = run_into(
            binary_handler_analyze_rocprof_compute, workload_path, run_folder
        )
        assert code == 0
        assert (run_folder / "report.txt").is_file()
        assert snapshot_files(workload_path) == before
    finally:
        restore_write(workload_path)


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
@pytest.mark.parametrize(
    "filter_options",
    [
        ["--gpu-id", "9"],
        ["--gpu-id", "0", "9"],
        ["-k", "999"],
    ],
    ids=["unknown-gpu", "partly-invalid-gpu-list", "unknown-kernel"],
)
def test_unknown_filter_id_creates_no_run_folder(
    binary_handler_analyze_rocprof_compute, tmp_path, filter_options
):
    """An unknown --gpu-id or -k fails before the run folder exists."""
    workload_path = copy_workload(tmp_path)
    run_folder = tmp_path / "runs" / "run1"

    code = run_into(
        binary_handler_analyze_rocprof_compute,
        workload_path,
        run_folder,
        *filter_options,
    )

    assert code == 1
    assert not run_folder.parent.exists()


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


@pytest.mark.misc
def test_terminal_stops_at_speed_of_light_and_report_has_every_panel(
    binary_handler_analyze_rocprof_compute, tmp_path, capsys
):
    """stdout shows panels 0-2; report.txt holds the full report."""
    workload_path = copy_workload(tmp_path)
    run_folder = tmp_path / "runs" / "run1"

    code = run_into(binary_handler_analyze_rocprof_compute, workload_path, run_folder)

    assert code == 0
    assert panel_numbers(capsys.readouterr().out) == {0, 1, 2}
    report_panels = panel_numbers(
        (run_folder / "report.txt").read_text(encoding="utf-8")
    )
    assert {0, 1, 2}.issubset(report_panels)
    assert max(report_panels) > 2


@pytest.mark.misc
@pytest.mark.parametrize("block_flag", ["-b", "--block"])
def test_block_adds_its_section_to_the_terminal_only(
    binary_handler_analyze_rocprof_compute, tmp_path, capsys, block_flag
):
    """-b prints its block after speed-of-light; report.txt is unchanged by it."""
    workload_path = copy_workload(tmp_path)
    plain_folder = tmp_path / "runs" / "plain"
    block_folder = tmp_path / "runs" / "block"

    assert (
        run_into(binary_handler_analyze_rocprof_compute, workload_path, plain_folder)
        == 0
    )
    capsys.readouterr()
    code = run_into(
        binary_handler_analyze_rocprof_compute,
        workload_path,
        block_folder,
        block_flag,
        "7",
    )

    assert code == 0
    assert panel_numbers(capsys.readouterr().out) == {0, 1, 2, 7}
    block_report = (block_folder / "report.txt").read_text(encoding="utf-8")
    plain_report = (plain_folder / "report.txt").read_text(encoding="utf-8")
    assert block_report == plain_report


@pytest.mark.misc
def test_block_metric_id_prints_only_that_row(
    binary_handler_analyze_rocprof_compute, tmp_path, capsys
):
    """A -b metric id prints its row of the block and hides its siblings."""
    workload_path = copy_workload(tmp_path)
    run_folder = tmp_path / "runs" / "run1"

    code = run_into(
        binary_handler_analyze_rocprof_compute, workload_path, run_folder, "-b", "7.1.5"
    )

    assert code == 0
    terminal = terminal_panels(capsys.readouterr().out)
    assert "7.1.5" in terminal
    assert "7.1.4" not in terminal
    assert "7.1.4" in (run_folder / "report.txt").read_text(encoding="utf-8")


@pytest.mark.misc
def test_terminal_lists_artifacts_after_the_panels_and_ends_with_run_folder(
    binary_handler_analyze_rocprof_compute, tmp_path, capsys
):
    """The artifact groups follow speed-of-light; the run folder is the last line."""
    workload_path = copy_workload(tmp_path)
    run_folder = tmp_path / "runs" / "run1"

    code = run_into(binary_handler_analyze_rocprof_compute, workload_path, run_folder)

    assert code == 0
    output = capsys.readouterr().out
    text = strip_ansi(output)
    assert (
        text.index("2. System Speed-of-Light")
        < text.index(ARTIFACTS_HEADING)
        < text.rindex(RUN_FOLDER_PREFIX)
    )
    artifacts = artifacts_section(output)
    for name in ("analysis.db", "report.txt"):
        assert f"  {name}\n" in artifacts
    assert "  1-MI300X_A1/\n" in artifacts
    for name in ("kernels.csv", "dispatches.csv", "metrics-summary.csv"):
        assert name in artifacts
    assert "1 kernel folder" in artifacts
    assert "Workload folders of the -p paths" not in artifacts
    assert last_output_line(output).endswith(f"Run folder: {run_folder.resolve()}")


@pytest.mark.misc
def test_comparison_terminal_maps_each_p_path_to_its_workload_folder(
    binary_handler_analyze_rocprof_compute, tmp_path, capsys
):
    """A comparison maps each -p path to its workload folder before the run folder."""
    first_path = copy_workload(tmp_path)
    second_path = tmp_path / "vcopy" / "MI200"
    shutil.copytree(VCOPY_WORKLOAD.parent / "MI200", second_path)
    run_folder = tmp_path / "runs" / "compare"

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(first_path),
        "--path",
        str(second_path),
        "--output-directory",
        str(run_folder.parent),
        "--output-name",
        run_folder.name,
    ])

    assert code == 0
    output = capsys.readouterr().out
    text = strip_ansi(output)
    mapping = text.index("Workload folders of the -p paths")
    assert text.index(ARTIFACTS_HEADING) < mapping < text.rindex(RUN_FOLDER_PREFIX)
    assert f"  -p {first_path.resolve()} -> 1-MI300X_A1/\n" in text
    assert f"  -p {second_path.resolve()} -> 2-MI200/\n" in text
    assert last_output_line(output).endswith(f"Run folder: {run_folder.resolve()}")


@pytest.mark.misc
def test_block_run_still_ends_with_the_run_folder(
    binary_handler_analyze_rocprof_compute, tmp_path, capsys
):
    """With -b the artifact list still follows the selected panels and the run folder
    stays the last line."""
    workload_path = copy_workload(tmp_path)
    run_folder = tmp_path / "runs" / "run1"

    code = run_into(
        binary_handler_analyze_rocprof_compute, workload_path, run_folder, "-b", "7"
    )

    assert code == 0
    output = capsys.readouterr().out
    assert panel_numbers(output) == {0, 1, 2, 7}
    assert "  analysis.db\n" in artifacts_section(output)
    assert last_output_line(output).endswith(f"Run folder: {run_folder.resolve()}")


@pytest.mark.misc
def test_invalid_block_creates_no_run_folder(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """An unknown -b value fails before the run folder exists."""
    workload_path = copy_workload(tmp_path)
    run_folder = tmp_path / "runs" / "run1"

    code = run_into(
        binary_handler_analyze_rocprof_compute, workload_path, run_folder, "-b", "SQ"
    )

    assert code == 1
    assert not run_folder.parent.exists()


def workload_folder_names(run_folder):
    """Return the names of the subfolders of a run folder."""
    return sorted(path.name for path in run_folder.iterdir() if path.is_dir())


def file_names(folder):
    """Return the names of the files directly inside a folder."""
    return sorted(path.name for path in folder.iterdir() if path.is_file())


@pytest.mark.misc
def test_single_run_writes_the_fixed_layout(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """One workload gets the root report and DB, and one folder of files."""
    workload_path = copy_workload(tmp_path)
    run_folder = tmp_path / "runs" / "run1"

    code = run_into(binary_handler_analyze_rocprof_compute, workload_path, run_folder)

    assert code == 0
    assert sorted(path.name for path in run_folder.iterdir()) == [
        "1-MI300X_A1",
        "analysis.db",
        "report.txt",
    ]
    workload_folder = run_folder / "1-MI300X_A1"
    assert file_names(workload_folder) == [
        "dispatches.csv",
        "kernels.csv",
        "memory-chart-gpu-0.html",
        "metrics-summary.csv",
    ]
    kernels = pd.read_csv(workload_folder / "kernels.csv")
    assert workload_folder_names(workload_folder) == sorted(kernels["Folder"])
    for folder in kernels["Folder"]:
        assert (workload_folder / folder / f"{folder}-metrics.csv").is_file()
    # The fixture has no roofline.csv and no PC sampling, so neither is written.
    assert not list(run_folder.rglob("roofline*"))
    assert not list(run_folder.rglob("*pc-sampling*"))


@pytest.mark.misc
def test_comparison_run_writes_a_report_per_workload(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """Two -p workloads get a comparison report at the root and one report each."""
    first_path = copy_workload(tmp_path)
    second_path = tmp_path / "vcopy" / "MI200"
    shutil.copytree(VCOPY_WORKLOAD.parent / "MI200", second_path)
    run_folder = tmp_path / "runs" / "compare"
    single_folder = tmp_path / "runs" / "single"

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(first_path),
        "--path",
        str(second_path),
        "--output-directory",
        str(run_folder.parent),
        "--output-name",
        run_folder.name,
    ])
    assert code == 0
    assert (
        run_into(binary_handler_analyze_rocprof_compute, first_path, single_folder) == 0
    )

    assert sorted(path.name for path in run_folder.iterdir()) == [
        "1-MI300X_A1",
        "2-MI200",
        "analysis.db",
        "report.txt",
    ]
    for workload_folder in ("1-MI300X_A1", "2-MI200"):
        assert (run_folder / workload_folder / "report.txt").is_file()
        assert (run_folder / workload_folder / "memory-chart-gpu-0.html").is_file()
        assert (run_folder / workload_folder / "kernels.csv").is_file()
    # The report of a workload in a comparison is the report it gets alone.
    assert (run_folder / "1-MI300X_A1" / "report.txt").read_text(encoding="utf-8") == (
        single_folder / "report.txt"
    ).read_text(encoding="utf-8")


@pytest.mark.misc
def test_roofline_run_writes_the_roofline_html_in_the_workload_folder(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """A workload with roofline.csv gets its roofline HTML beside the CSV files."""
    workload_path = tmp_path / "mem_levels_HBM" / "MI200"
    shutil.copytree(MEM_LEVELS_WORKLOAD, workload_path)
    run_folder = tmp_path / "runs" / "roofline"

    code = run_into(binary_handler_analyze_rocprof_compute, workload_path, run_folder)

    assert code == 0
    workload_folder = run_folder / "1-MI200"
    assert (workload_folder / "roofline.csv").is_file()
    assert (workload_folder / "roofline-gpu-0.html").is_file()


@pytest.mark.misc
@pytest.mark.parametrize("output_format", ["stdout", "txt", "csv", "db"])
def test_output_format_warns_and_is_ignored(
    binary_handler_analyze_rocprof_compute, tmp_path, capsys, output_format
):
    """--output-format is a deprecated no-op: it warns and writes the usual folder."""
    workload_path = copy_workload(tmp_path)
    run_folder = tmp_path / "runs" / "run1"

    code = run_into(
        binary_handler_analyze_rocprof_compute,
        workload_path,
        run_folder,
        "--output-format",
        output_format,
    )

    assert code == 0
    captured = capsys.readouterr()
    output = strip_ansi(captured.out + captured.err)
    assert "--output-format is deprecated and has no effect" in output
    assert sorted(path.name for path in run_folder.iterdir()) == [
        "1-MI300X_A1",
        "analysis.db",
        "report.txt",
    ]


@pytest.mark.misc
def test_block_is_not_deprecated(
    binary_handler_analyze_rocprof_compute, tmp_path, capsys
):
    """-b selects terminal output without a deprecation warning."""
    workload_path = copy_workload(tmp_path)

    code = run_into(
        binary_handler_analyze_rocprof_compute,
        workload_path,
        tmp_path / "runs" / "run1",
        "-b",
        "7",
    )

    assert code == 0
    captured = capsys.readouterr()
    output = strip_ansi(captured.out + captured.err)
    assert "is deprecated" not in output


@pytest.mark.misc
def test_write_failure_leaves_the_partial_run_folder_and_names_it(
    binary_handler_analyze_rocprof_compute, tmp_path, capsys, monkeypatch
):
    """A write that fails part way keeps the partial run folder and names it."""
    workload_path = copy_workload(tmp_path)
    run_folder = tmp_path / "runs" / "run1"

    def fail_after_first_folder(folder, workload_paths, kernel_rankings):
        (folder / "1-MI300X_A1").mkdir()
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.write_workload_folders",
        fail_after_first_folder,
    )

    code = run_into(binary_handler_analyze_rocprof_compute, workload_path, run_folder)

    assert code == 1
    assert (run_folder / "1-MI300X_A1").is_dir()
    assert not (run_folder / "analysis.db").exists()
    captured = capsys.readouterr()
    output = strip_ansi(captured.out + captured.err)
    assert str(run_folder.resolve()) in output
    assert "No space left on device" in output


@pytest.mark.misc
def test_html_write_failure_leaves_the_partial_run_folder_and_names_it(
    binary_handler_analyze_rocprof_compute, tmp_path, capsys, monkeypatch
):
    """A memory chart HTML write that fails keeps the run folder and names it."""
    workload_path = copy_workload(tmp_path)
    run_folder = tmp_path / "runs" / "run1"
    real_write_text = Path.write_text

    def fail_on_memory_chart_html(self, *args, **kwargs):
        if self.name.startswith("memory-chart-gpu-"):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_on_memory_chart_html)

    code = run_into(binary_handler_analyze_rocprof_compute, workload_path, run_folder)

    assert code == 1
    assert not (run_folder / "analysis.db").exists()
    captured = capsys.readouterr()
    output = strip_ansi(captured.out + captured.err)
    assert str(run_folder.resolve()) in output
    assert "No space left on device" in output
