# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ML API tracing models and feature services.

The package initializer imports no submodules. models, constants, marker_format,
profile, storage, and recording are safe for profile mode and injected workloads.
Recording loads optional Torch and Triton frameworks only at workload time,
when their selected backends are installed. analysis_models, errors, input,
processing, call_tree,
summary, display, selection, and analysis are analysis-only modules requiring pandas.
selection owns operator filtering and the shared kernel numbering policy.
"""
