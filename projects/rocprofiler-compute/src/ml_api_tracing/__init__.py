# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ML API tracing models and feature services.

The package initializer imports no submodules. models, constants, and
marker_format use only the standard library and are safe for profile mode
and injected workloads. analysis_models and errors are analysis-only modules
that require pandas.
"""
