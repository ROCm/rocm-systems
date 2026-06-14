# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Compatibility exports for profile artifact factories."""

from interface.factory import (
    create_profile_artifact_reader,
    create_profile_artifact_writer,
)

__all__ = [
    "create_profile_artifact_reader",
    "create_profile_artifact_writer",
]
