# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""rocprofv3 profile and analysis orchestration."""

from __future__ import annotations

from pathlib import Path

from interface.factory import (
    create_profile_artifact_reader,
    create_profile_artifact_writer,
)
from interface.profile_artifacts import ArtifactReaderOptions, ProfilePassContext


class Rocprofv3ProfileOrchestrator:
    """Coordinate rocprofv3 profile artifact finalization."""

    def finalize_profile_pass(
        self,
        artifact_format: str,
        context: ProfilePassContext,
    ) -> None:
        create_profile_artifact_writer(artifact_format).finalize_pass(context)


class Rocprofv3AnalysisOrchestrator:
    """Coordinate rocprofv3 analysis artifact preparation."""

    def materialize_pmc_perf(
        self,
        workload_dir: Path,
        output_path: Path,
        profiling_config: dict,
        options: ArtifactReaderOptions,
    ) -> Path:
        reader = create_profile_artifact_reader(profiling_config, options)
        return reader.materialize_pmc_perf(workload_dir, output_path)
