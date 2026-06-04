# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCPD profile artifact reader."""

import csv
from pathlib import Path

import pandas as pd

from utils.logger import console_debug, console_warning
from utils.profile_artifacts.interfaces import ArtifactReaderOptions
from utils.profile_artifacts.pmc_frame import load_pmc_frame_from_csv


class RocpdProfileArtifactReader:
    """Read current ROCPD profile artifacts."""

    def __init__(self, options: ArtifactReaderOptions) -> None:
        self._options = options

    def has_artifacts(self, workload_dir: Path) -> bool:
        return _pmc_perf_path(workload_dir).exists() or bool(
            _find_result_files(workload_dir)
        )

    def materialize_pmc_perf(self, workload_dir: Path, output_path: Path) -> Path:
        if output_path.exists():
            console_debug(f"Using existing {output_path}")
            return output_path

        console_warning(
            "Reading intermediate results_*.csv files is deprecated and "
            "will be removed in a future release."
        )
        _concat_results_to_pmc_perf(_find_result_files(workload_dir), output_path)
        console_debug(f"Created file: {output_path}")
        return output_path

    def read_pmc_frame(self, workload_dir: Path) -> pd.DataFrame:
        return load_pmc_frame_from_csv(
            workload_dir,
            is_rocpd=True,
            kernel_verbose=self._options.kernel_verbose,
            verbose=self._options.verbose,
        )


def _pmc_perf_path(workload_dir: Path) -> Path:
    return workload_dir / "pmc_perf.csv"


def _find_result_files(workload_dir: Path) -> list[Path]:
    return list(workload_dir.glob("results_*.csv"))


def _concat_results_to_pmc_perf(result_files: list[Path], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = None
        for result_file in result_files:
            with result_file.open(newline="", encoding="utf-8") as infile:
                reader = csv.reader(infile)
                header = next(reader)
                if writer is None:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                for row in reader:
                    writer.writerow(row)
