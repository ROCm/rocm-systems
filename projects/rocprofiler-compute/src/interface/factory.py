# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Factory helpers for profile artifact readers and writers."""

from interface.profile_artifacts import (
    ArtifactReaderOptions,
    ProfileArtifactFormat,
    ProfileArtifactReader,
    ProfileArtifactWriter,
)
from utils.logger import console_error


def create_profile_artifact_reader(
    profiling_config: dict,
    options: ArtifactReaderOptions,
) -> ProfileArtifactReader:
    """Create a reader for the profiling artifact format."""
    artifact_format = _artifact_format_from_config(profiling_config)
    if artifact_format == "rocpd":
        from interface.rocpd_data import RocpdProfileArtifactReader

        return RocpdProfileArtifactReader(options)

    from interface.csv_data import CsvProfileArtifactReader

    return CsvProfileArtifactReader(options)


def create_profile_artifact_writer(
    format_rocprof_output: str,
) -> ProfileArtifactWriter:
    """Create a writer for the profiling artifact format."""
    if format_rocprof_output == "rocpd":
        from interface.rocpd_data import RocpdProfileArtifactWriter

        return RocpdProfileArtifactWriter()

    if format_rocprof_output == "csv":
        from interface.csv_data import CsvProfileArtifactWriter

        return CsvProfileArtifactWriter()

    console_error(f"Unknown format_rocprof_output: {format_rocprof_output}")
    from interface.csv_data import CsvProfileArtifactWriter

    return CsvProfileArtifactWriter()


def _artifact_format_from_config(profiling_config: dict) -> ProfileArtifactFormat:
    configured_format = profiling_config.get("format_rocprof_output", "rocpd")
    if configured_format == "csv":
        return "csv"
    return "rocpd"
