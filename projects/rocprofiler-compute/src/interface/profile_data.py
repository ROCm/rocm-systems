# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared contracts for profile data access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    import pandas as pd

ProfileDataFormat = Literal["csv", "rocpd"]


@dataclass(frozen=True)
class ProfileDataReaderOptions:
    """Options needed to read and materialize profile data."""

    join_type: str = "grid"
    kokkos_trace: bool = False
    kernel_verbose: int = 5
    verbose: int = 0


@dataclass(frozen=True)
class ProfilePassContext:
    """Context needed to finalize one profiling pass."""

    workload_dir: Path
    fbase: str
    profiler_command: str
    using_native_tool: bool
    torch_trace_enabled: bool
    retain_rocpd_output: bool = False
    kokkos_trace_enabled: bool = False


class ProfileDataReader(Protocol):
    """Read profile data without exposing its storage layout."""

    def has_profile_data(self, workload_dir: Path) -> bool:
        """Return True if this reader can find profile data."""

    def materialize_pmc_perf(self, workload_dir: Path, output_path: Path) -> Path:
        """Ensure a pmc_perf.csv file exists and return its path."""

    def read_pmc_frame(self, workload_dir: Path) -> pd.DataFrame:
        """Return the canonical PMC DataFrame for analysis."""


class ProfileDataWriter(Protocol):
    """Finalize profile data after a profiling pass."""

    def finalize_pass(self, context: ProfilePassContext) -> None:
        """Finalize profile data for one profiling pass."""
