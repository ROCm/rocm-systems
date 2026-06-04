# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Factory helpers for profile artifact readers."""

from utils.profile_artifacts.csv_artifacts import (
    CsvProfileArtifactReader,
    CsvProfileArtifactWriter,
)
from utils.profile_artifacts.interfaces import (
    ArtifactReaderOptions,
    ProfileArtifactReader,
    ProfileArtifactWriter,
)
from utils.profile_artifacts.rocpd_artifacts import (
    RocpdProfileArtifactReader,
    RocpdProfileArtifactWriter,
)


def create_profile_artifact_reader(
    profiling_config: dict,
    options: ArtifactReaderOptions,
) -> ProfileArtifactReader:
    """Create a reader for the profiling artifact format."""
    if profiling_config.get("format_rocprof_output", "rocpd") == "rocpd":
        return RocpdProfileArtifactReader(options)
    return CsvProfileArtifactReader(options)


def create_profile_artifact_writer(format_rocprof_output: str) -> ProfileArtifactWriter:
    """Create a writer for the profiling artifact format."""
    if format_rocprof_output == "rocpd":
        return RocpdProfileArtifactWriter()
    return CsvProfileArtifactWriter()
