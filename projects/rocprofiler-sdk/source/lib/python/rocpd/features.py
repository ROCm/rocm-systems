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

from . import libpyrocpd

# Single source of truth: feature name -> minimum schema version required.
# All output backends (csv, otf2, perfetto) derive their feature sets from this table.
FEATURE_SCHEMA_VERSIONS = {
    "graph_launch": libpyrocpd.schema_version(3, 0, 2),
}


def get_supported_features(importData) -> frozenset:
    """Return the frozenset of feature names supported by importData's schema version."""
    return frozenset(
        feature
        for feature, version in FEATURE_SCHEMA_VERSIONS.items()
        if importData.schema_version >= version
    )
