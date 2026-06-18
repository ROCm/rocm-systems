# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

find_path(NUMA_INCLUDE_DIR numa.h)
find_library(NUMA_LIBRARIES numa)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA
  DEFAULT_MSG
  NUMA_LIBRARIES NUMA_INCLUDE_DIR)

mark_as_advanced(NUMA_LIBRARIES NUMA_INCLUDE_DIR)
