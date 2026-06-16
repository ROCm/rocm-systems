# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Factory helpers for profile data readers and writers."""

from interface.profile_data import (
    ProfileDataFormat,
    ProfileDataReader,
    ProfileDataReaderOptions,
    ProfileDataWriter,
)
from utils.logger import console_error


def create_profile_data_reader(
    profiling_config: dict,
    options: ProfileDataReaderOptions,
) -> ProfileDataReader:
    """Create a reader for the profile data format."""
    data_format = _data_format_from_config(profiling_config)
    if data_format == "rocpd":
        from interface.rocpd_data import RocpdProfileDataReader

        return RocpdProfileDataReader(options)

    from interface.csv_data import CsvProfileDataReader

    return CsvProfileDataReader(options)


def create_profile_data_writer(
    format_rocprof_output: str,
) -> ProfileDataWriter:
    """Create a writer for the profile data format."""
    if format_rocprof_output == "rocpd":
        from interface.rocpd_data import RocpdProfileDataWriter

        return RocpdProfileDataWriter()

    if format_rocprof_output == "csv":
        from interface.csv_data import CsvProfileDataWriter

        return CsvProfileDataWriter()

    console_error(f"Unknown format_rocprof_output: {format_rocprof_output}")
    from interface.csv_data import CsvProfileDataWriter

    return CsvProfileDataWriter()


def _data_format_from_config(profiling_config: dict) -> ProfileDataFormat:
    configured_format = profiling_config.get("format_rocprof_output", "rocpd")
    if configured_format == "csv":
        return "csv"
    return "rocpd"
