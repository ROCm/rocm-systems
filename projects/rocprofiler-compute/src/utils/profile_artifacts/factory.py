# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Factory helpers for profile artifact readers."""

from utils.profile_artifacts.csv_artifacts import CsvProfileArtifactReader
from utils.profile_artifacts.interfaces import (
    ArtifactReaderOptions,
    ProfileArtifactReader,
)
from utils.profile_artifacts.rocpd_artifacts import RocpdProfileArtifactReader


def create_profile_artifact_reader(
    profiling_config: dict,
    options: ArtifactReaderOptions,
) -> ProfileArtifactReader:
    """Create a reader for the profiling artifact format."""
    if profiling_config.get("format_rocprof_output", "rocpd") == "rocpd":
        return RocpdProfileArtifactReader(options)
    return CsvProfileArtifactReader(options)
