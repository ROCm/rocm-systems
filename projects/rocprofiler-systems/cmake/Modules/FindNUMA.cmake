# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# FindNUMA.cmake
#
# Locates the NUMA (Non-Uniform Memory Access) library (libnuma) and its
# development headers. This module is needed because rocSHMEM's cmake config
# file calls find_dependency(NUMA), but neither cmake nor rocSHMEM ships a
# FindNUMA module.
#
# Result variables:
#   NUMA_FOUND        - True if libnuma and numa.h were found
#   NUMA_INCLUDE_DIR  - Directory containing numa.h
#   NUMA_LIBRARIES    - The libnuma library

find_path(NUMA_INCLUDE_DIR numa.h)
find_library(NUMA_LIBRARIES numa)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA DEFAULT_MSG NUMA_LIBRARIES NUMA_INCLUDE_DIR)

mark_as_advanced(NUMA_LIBRARIES NUMA_INCLUDE_DIR)
