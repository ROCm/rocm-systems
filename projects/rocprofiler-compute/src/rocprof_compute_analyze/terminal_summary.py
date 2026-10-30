# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""The closing lines of the analyze terminal summary.

After the terminal sections, analyze lists the artifact groups written into the
run folder. A comparison also maps each -p path to its workload folder. The run
folder itself is printed last, by the caller.
"""

import textwrap
from pathlib import Path
from typing import List, NamedTuple, Sequence, TextIO, Tuple

from rocprof_compute_analyze.output_location import workload_folder_name

TERMINAL_WIDTH = 80


class WorkloadArtifacts(NamedTuple):
    """The artifacts of one workload folder: its file names and kernel folder count."""

    folder_name: str
    file_names: List[str]
    kernel_count: int


def workload_folder_names(workload_paths: Sequence[str]) -> List[str]:
    """Return the run-folder name of each -p workload, in -p order."""
    return [
        workload_folder_name(position, Path(workload_path))
        for position, workload_path in enumerate(workload_paths, start=1)
    ]


def _file_names(folder: Path) -> List[str]:
    """Return the sorted names of the files directly inside folder."""
    return sorted(child.name for child in folder.iterdir() if child.is_file())


def read_artifact_groups(
    run_folder: Path, workload_paths: Sequence[str]
) -> Tuple[List[str], List[WorkloadArtifacts]]:
    """Read the root file names and each workload folder's artifacts from disk."""
    workloads = []
    for folder_name in workload_folder_names(workload_paths):
        folder = run_folder / folder_name
        kernel_count = sum(1 for child in folder.iterdir() if child.is_dir())
        workloads.append(
            WorkloadArtifacts(folder_name, _file_names(folder), kernel_count)
        )
    return _file_names(run_folder), workloads


def artifact_group_lines(
    root_file_names: Sequence[str], workloads: Sequence[WorkloadArtifacts]
) -> List[str]:
    """Return the artifact groups as terminal lines.

    The run root lists its own files. Each workload folder lists its files, then
    how many kernel folders it holds.
    """
    lines = [f"  {name}" for name in root_file_names]
    for workload in workloads:
        lines.append(f"  {workload.folder_name}/")
        lines.extend(
            textwrap.wrap(
                ", ".join(workload.file_names),
                width=TERMINAL_WIDTH,
                initial_indent="    ",
                subsequent_indent="    ",
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
        if workload.kernel_count:
            noun = "folder" if workload.kernel_count == 1 else "folders"
            lines.append(f"    {workload.kernel_count} kernel {noun}")
    return lines


def workload_mapping_lines(workload_paths: Sequence[str]) -> List[str]:
    """Return one line per -p path, naming the workload folder that holds it."""
    return [
        f"  -p {workload_path} -> {folder_name}/"
        for workload_path, folder_name in zip(
            workload_paths, workload_folder_names(workload_paths)
        )
    ]


def print_terminal_summary(
    run_folder: Path, workload_paths: Sequence[str], output: TextIO
) -> None:
    """Print the artifact groups, then for a comparison the -p to folder mapping.

    The run folder must already hold every artifact. A single -p has one workload
    folder, so it gets no mapping.
    """
    print(f"\n{'-' * TERMINAL_WIDTH}", file=output)
    print("Artifacts written", file=output)
    root_file_names, workloads = read_artifact_groups(run_folder, workload_paths)
    for line in artifact_group_lines(root_file_names, workloads):
        print(line, file=output)
    if len(workload_paths) > 1:
        print("\nWorkload folders of the -p paths", file=output)
        for line in workload_mapping_lines(workload_paths):
            print(line, file=output)
