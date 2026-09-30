#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

from typing import Dict, NamedTuple, Optional

from . import libpyrocpd


class FeatureVersionRange(NamedTuple):
    """Version range over which a named feature is supported.

    Attributes:
        min_version: The earliest schema version that introduced this feature (inclusive).
        max_version: The latest schema version that supports this feature (inclusive),
                     or None if the feature has no known upper bound.
    """

    min_version: libpyrocpd.schema_version
    max_version: Optional[libpyrocpd.schema_version] = None


# Single source of truth for feature availability by schema version.
#
# Each entry maps a feature name to a FeatureVersionRange(min_version, max_version):
#   - min_version (inclusive): the schema version that introduced the feature.
#   - max_version (inclusive): the last schema version supporting the feature,
#                              or None if it remains supported in all later versions.
#
# All output backends (csv, otf2, perfetto) derive their active feature sets from this
# table via get_supported_features(). Adding or retiring a feature only requires editing
# this table — no changes are needed in any backend.
FEATURE_SCHEMA_VERSIONS: Dict[str, FeatureVersionRange] = {
    "graph_launch": FeatureVersionRange(
        min_version=libpyrocpd.schema_version(3, 0, 2),
        max_version=None,
    ),
    "spm_counters": FeatureVersionRange(
        min_version=libpyrocpd.schema_version(3, 0, 3),
        max_version=None,
    ),
    "hip_event": FeatureVersionRange(
        min_version=libpyrocpd.schema_version(3, 0, 4),
        max_version=None,
    ),
}


def get_supported_features_from_version(schema_version) -> frozenset:
    """Return the frozenset of feature names supported by the given schema version.

    A feature is included if:
        vrange.min_version <= schema_version
        and (vrange.max_version is None or schema_version <= vrange.max_version)
    """
    return frozenset(
        feature
        for feature, vrange in FEATURE_SCHEMA_VERSIONS.items()
        if schema_version >= vrange.min_version
        and (vrange.max_version is None or schema_version <= vrange.max_version)
    )


def get_supported_features(importData) -> frozenset:
    """Return the frozenset of feature names supported by importData's schema version."""
    return get_supported_features_from_version(importData.schema_version)
