# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared helpers for profile and analysis orchestrators."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import config
from interface.factory import create_profile_data_reader
from interface.profile_data import ProfileDataReaderOptions, ProfilePassContext
from utils.logger import console_debug, console_error
from utils.utils_common import create_temp_rocprofiler_metrics_path, parse_pmc_perf
from vendored import yaml

if TYPE_CHECKING:
    import pandas as pd

ProfilerOptions = list[str] | dict[str, str | list[str]]

_PROFILER_INTERNAL_RE = re.compile(
    r"^\[rocprofiler"  # rocprofiler-sdk and rocprofiler-compute tool messages
    r"|^[WI]\d{8}\s"  # glog-style timestamps (W/I followed by YYYYMMDD)
)


def is_live_attach(profiler_options: ProfilerOptions) -> bool:
    """Return True if the profiler options indicate live-attach mode."""
    return (isinstance(profiler_options, list) and "--pid" in profiler_options) or (
        isinstance(profiler_options, dict)
        and profiler_options.get("ROCPROF_ATTACH_PID") is not None
    )


def get_profile_input_info(fnames: list[str] | str) -> tuple[bool, str]:
    """Return whether the pass has multiple PMC files and the pass base name."""
    multiple_files = isinstance(fnames, list)
    fpath = Path(fnames[0]) if multiple_files else Path(fnames)
    return multiple_files, fpath.stem


def build_counter_collection_environment(
    fnames: list[str] | str,
) -> dict[str, str]:
    """Build environment updates needed for counter collection."""
    new_env = os.environ.copy()
    with open(
        config.rocprof_compute_home
        / "rocprof_compute_soc"
        / "profile_configs"
        / "sdk_config.yaml",
        encoding="utf-8",
    ) as filename:
        sdk_config = yaml.safe_load(filename)

    for fname in fnames if isinstance(fnames, list) else [fnames]:
        fname_path = Path(fname)
        counter_def_fname = fname_path.parent / (
            "counter_def_" + fname_path.name[len("pmc_perf_") :]
        )
        if counter_def_fname.exists():
            with open(counter_def_fname, encoding="utf-8") as file:
                sdk_config["rocprofiler-sdk"]["counters"].extend(
                    yaml.safe_load(file)["rocprofiler-sdk"]["counters"]
                )

    new_env["ROCPROFILER_METRICS_PATH"] = create_temp_rocprofiler_metrics_path(
        sdk_config
    )
    console_debug(
        "Adding env var for counter definitions: "
        f"ROCPROFILER_METRICS_PATH={new_env['ROCPROFILER_METRICS_PATH']}"
    )
    return new_env


def cleanup_counter_collection_environment(new_env: dict[str, str]) -> None:
    """Remove temporary counter definition files."""
    if new_env.get("ROCPROFILER_METRICS_PATH"):
        shutil.rmtree(new_env["ROCPROFILER_METRICS_PATH"], ignore_errors=True)


def classify_failed_profile_output(output: str) -> None:
    """Log subprocess output from a failed profile pass."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped:
            _classify_output_line(stripped)


def build_profile_pass_context(
    *,
    workload_dir: str,
    fbase: str,
    profiler_command: str,
    options: ProfilerOptions,
    torch_trace_enabled: bool,
    retain_rocpd_output: bool,
) -> ProfilePassContext:
    """Build writer context for one finalized profile pass."""
    return ProfilePassContext(
        workload_dir=Path(workload_dir),
        fbase=fbase,
        profiler_command=profiler_command,
        using_native_tool=_uses_native_counter_collection(profiler_command, options),
        torch_trace_enabled=torch_trace_enabled,
        retain_rocpd_output=retain_rocpd_output,
        kokkos_trace_enabled=_has_kokkos_trace(options),
    )


def build_rocpd_counter_options(
    fnames: list[str] | str,
    options: dict[str, str | list[str]],
) -> dict[str, str | list[str]]:
    """Add counter collection options for rocprofiler-sdk passes."""
    multiple_files = isinstance(fnames, list)
    updated_options = options.copy()
    if multiple_files:
        updated_options["ROCPROF_COUNTERS"] = ", ".join([
            f"pmc: {' '.join(parse_pmc_perf(fname))}" for fname in fnames
        ])
    else:
        updated_options["ROCPROF_COUNTERS"] = f"pmc: {' '.join(parse_pmc_perf(fnames))}"
    updated_options["ROCPROF_AGENT_INDEX"] = "absolute"
    return updated_options


class ProfileDataAnalysisOrchestrator:
    """Coordinate analysis reads through the profile data reader boundary."""

    def materialize_pmc_perf(
        self,
        workload_dir: Path,
        output_path: Path,
        profiling_config: dict[str, Any],
        options: ProfileDataReaderOptions,
    ) -> Path:
        """Ensure a compatibility pmc_perf.csv exists."""
        reader = create_profile_data_reader(profiling_config, options)
        return reader.materialize_pmc_perf(workload_dir, output_path)

    def has_profile_data(
        self,
        workload_dir: Path,
        profiling_config: dict[str, Any],
        options: ProfileDataReaderOptions,
    ) -> bool:
        """Return True when the configured reader can find profile data."""
        reader = create_profile_data_reader(profiling_config, options)
        return reader.has_profile_data(workload_dir)

    def read_pmc_frame(
        self,
        workload_dir: Path,
        profiling_config: dict[str, Any],
        options: ProfileDataReaderOptions,
    ) -> pd.DataFrame:
        """Read the canonical PMC DataFrame for analysis."""
        reader = create_profile_data_reader(profiling_config, options)
        return reader.read_pmc_frame(workload_dir)


def _classify_output_line(line: str) -> None:
    if _PROFILER_INTERNAL_RE.match(line):
        console_debug(line)
    else:
        console_error(line, exit=False)


def _uses_native_counter_collection(
    profiler_command: str,
    options: ProfilerOptions,
) -> bool:
    return (
        profiler_command == "rocprofiler-sdk"
        and isinstance(options, dict)
        and options.get("ROCPROF_COUNTER_COLLECTION") == "0"
    )


def _has_kokkos_trace(options: ProfilerOptions) -> bool:
    return isinstance(options, list) and "--kokkos-trace" in options
