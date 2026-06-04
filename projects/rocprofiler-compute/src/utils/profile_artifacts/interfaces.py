# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Interfaces for profile artifact access."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class ArtifactReaderOptions:
    """Options needed to read and materialize profiling artifacts."""

    join_type: str = "grid"
    kokkos_trace: bool = False
    kernel_verbose: int = 5
    verbose: int = 0


class ProfileArtifactReader(Protocol):
    """Read profile artifacts without exposing their storage layout."""

    def has_artifacts(self, workload_dir: Path) -> bool:
        """Return True if this reader can find profile artifacts."""

    def materialize_pmc_perf(self, workload_dir: Path, output_path: Path) -> Path:
        """Ensure a pmc_perf.csv file exists and return its path."""

    def read_pmc_frame(self, workload_dir: Path) -> pd.DataFrame:
        """Return the canonical PMC DataFrame for analysis."""
