# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCPD profile artifact adapters."""

from pathlib import Path

import pandas as pd

from interface.rocpd_data import RocpdAnalysisData, RocpdProfileData
from utils.profile_artifacts.interfaces import ArtifactReaderOptions, ProfilePassContext
from utils.profile_artifacts.pmc_frame import load_pmc_frame_from_csv


class RocpdProfileArtifactReader:
    """Read current ROCPD profile artifacts."""

    def __init__(self, options: ArtifactReaderOptions) -> None:
        self._options = options
        self._rocpd_data = RocpdAnalysisData()

    def has_artifacts(self, workload_dir: Path) -> bool:
        return self._rocpd_data.has_artifacts(workload_dir)

    def materialize_pmc_perf(self, workload_dir: Path, output_path: Path) -> Path:
        return self._rocpd_data.materialize_pmc_perf(workload_dir, output_path)

    def read_pmc_frame(self, workload_dir: Path) -> pd.DataFrame:
        return load_pmc_frame_from_csv(
            workload_dir,
            is_rocpd=True,
            kernel_verbose=self._options.kernel_verbose,
            verbose=self._options.verbose,
        )


class RocpdProfileArtifactWriter:
    """Finalize current ROCPD profile artifacts."""

    def finalize_pass(self, context: ProfilePassContext) -> None:
        from utils import utils_profile as profile_ops

        rocpd_profile_data = RocpdProfileData()
        db_paths = list((context.workload_dir / "out" / "pmc_1").glob("*/*.db"))
        if context.using_native_tool:
            rocpd_profile_data.update_native_counter_events(
                db_paths,
                context.workload_dir,
            )

        rocpd_profile_data.convert_dbs_to_csv(
            db_paths,
            _rocpd_counter_collection_path(context),
            _rocpd_marker_trace_path(context),
        )
        combined_rows = rocpd_profile_data.read_counter_rows(
            _rocpd_counter_collection_path(context)
        )
        if not combined_rows:
            profile_ops.console_warning(
                "No GPU kernel data collected. "
                "The workload may not have dispatched any GPU kernels."
            )
            profile_ops.shutil.rmtree(
                str(context.workload_dir / "out"),
                ignore_errors=True,
            )
            return

        rocpd_profile_data.normalize_counter_rows(combined_rows)
        rocpd_profile_data.write_counter_rows(
            _rocpd_counter_collection_path(context),
            context.workload_dir / f"results_{context.fbase}.csv",
            combined_rows,
        )
        profile_ops.console_warning(
            "Intermediate results_*.csv generation from rocpd databases is "
            "deprecated and will be replaced with automatic .db file "
            "retention in a future release."
        )
        if context.torch_trace_enabled:
            profile_ops.save_torch_trace_inputs(
                str(context.workload_dir),
                context.fbase,
                "rocpd",
            )
        if context.retain_rocpd_output:
            rocpd_profile_data.retain_databases(
                db_paths,
                context.workload_dir,
                context.fbase,
            )
        profile_ops.shutil.rmtree(str(context.workload_dir / "out"))


def _rocpd_counter_collection_path(context: ProfilePassContext) -> Path:
    return context.workload_dir / "out" / "pmc_1" / (
        f"{context.fbase}_counter_collection.csv"
    )


def _rocpd_marker_trace_path(context: ProfilePassContext) -> Path:
    return context.workload_dir / "out" / "pmc_1" / (
        f"{context.fbase}_marker_api_trace.csv"
    )
