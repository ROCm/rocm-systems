# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Vendored dependencies for rocprofiler-compute.

This package contains external dependencies vendored (included directly) into
the source tree to eliminate external dependencies in profile mode.

Only the pure-Python portions should be installed/used for vendored
packages (no C extensions) for maximum portability.
For vendoring guidelines and adding new packages, see CONTRIBUTING.md.
"""

try:
    from .pyyaml.lib import yaml
except ImportError as e:
    raise ImportError("Vendored dependencies not found.") from e

__all__ = ["yaml"]
