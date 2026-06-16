# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""rocprofv3 profile and analysis orchestration."""

from __future__ import annotations

from pathlib import Path

from interface.factory import (
    create_profile_data_reader,
    create_profile_data_writer,
)
from interface.profile_data import ProfileDataReaderOptions, ProfilePassContext


class Rocprofv3ProfileOrchestrator:
    """Coordinate rocprofv3 profile data finalization."""

    def finalize_profile_pass(
        self,
        data_format: str,
        context: ProfilePassContext,
    ) -> None:
        create_profile_data_writer(data_format).finalize_pass(context)


class Rocprofv3AnalysisOrchestrator:
    """Coordinate rocprofv3 analysis profile data preparation."""

    def materialize_pmc_perf(
        self,
        workload_dir: Path,
        output_path: Path,
        profiling_config: dict,
        options: ProfileDataReaderOptions,
    ) -> Path:
        reader = create_profile_data_reader(profiling_config, options)
        return reader.materialize_pmc_perf(workload_dir, output_path)
