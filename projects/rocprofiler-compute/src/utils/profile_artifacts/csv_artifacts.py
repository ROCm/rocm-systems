# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CSV profile artifact adapters."""

from pathlib import Path
from typing import Optional

import pandas as pd

from utils.profile_artifacts.interfaces import ArtifactReaderOptions, ProfilePassContext
from utils.profile_artifacts.pmc_frame import load_pmc_frame_from_csv
from utils.profile_data.csv_data import (
    CsvAnalysisData,
    CsvProfileData,
    concat_csv_files,
)
from utils.profile_data.csv_data import (
    join_csv_prof_files as join_csv_prof_files_impl,
)


class CsvProfileArtifactReader:
    """Read current CSV profile artifacts."""

    def __init__(self, options: ArtifactReaderOptions) -> None:
        self._options = options
        self._csv_data = CsvAnalysisData(
            join_type=options.join_type,
            kokkos_trace=options.kokkos_trace,
        )

    def has_artifacts(self, workload_dir: Path) -> bool:
        return self._csv_data.has_artifacts(workload_dir)

    def materialize_pmc_perf(self, workload_dir: Path, output_path: Path) -> Path:
        return self._csv_data.materialize_pmc_perf(workload_dir, output_path)

    def read_pmc_frame(self, workload_dir: Path) -> pd.DataFrame:
        return load_pmc_frame_from_csv(
            workload_dir,
            is_rocpd=False,
            kernel_verbose=self._options.kernel_verbose,
            verbose=self._options.verbose,
        )


class CsvProfileArtifactWriter:
    """Finalize current CSV profile artifacts."""

    def finalize_pass(self, context: ProfilePassContext) -> None:
        from utils import utils_profile as profile_ops

        result_files = _process_csv_outputs(context)
        if context.torch_trace_enabled:
            profile_ops.save_torch_trace_inputs(
                str(context.workload_dir),
                context.fbase,
                "csv",
            )
        if not result_files:
            profile_ops.console_warning(
                f"Cannot write results for {context.fbase}.csv due to no counter "
                "csv files generated."
            )
            return

        csv_profile_data = CsvProfileData()
        combined_results = concat_csv_files(result_files)
        csv_profile_data.normalize_counter_rows(combined_results)
        csv_profile_data.write_counter_results(
            combined_results,
            context.workload_dir,
            context.fbase,
        )
        csv_profile_data.standardize_headers(context.workload_dir, context.fbase)


def join_csv_prof_files(
    workload_dir: Path,
    *,
    join_type: str,
    kokkos_trace: bool,
) -> Optional[pd.DataFrame]:
    """Join CSV profiler outputs into the pmc_perf dataframe."""
    return join_csv_prof_files_impl(
        workload_dir,
        join_type=join_type,
        kokkos_trace=kokkos_trace,
    )


def _process_csv_outputs(context: ProfilePassContext) -> list[str]:
    from utils import utils_profile as profile_ops

    if context.profiler_command == "rocprofiler-sdk":
        return profile_ops.process_rocprofv3_output(
            str(context.workload_dir),
            using_native_tool=context.using_native_tool,
        )

    result_files = profile_ops.process_rocprofv3_output(
        str(context.workload_dir),
        using_native_tool=False,
    )
    if context.kokkos_trace_enabled:
        profile_ops.process_kokkos_trace_output(
            str(context.workload_dir),
            context.fbase,
        )
    return result_files
