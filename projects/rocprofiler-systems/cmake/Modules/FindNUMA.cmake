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
#
# Imported target:
#   numa::numa        - rocSHMEM's cmake config references this target in the
#                       link interface of roc::rocshmem, so it must be defined
#                       here for find_dependency(NUMA) to satisfy the import.

find_path(
    NUMA_INCLUDE_DIR
    NAMES numa.h
    PATHS ${ROCM_PATH}/lib/rocm_sysdeps/include $ENV{ROCM_PATH}/lib/rocm_sysdeps/include
)
find_library(
    NUMA_LIBRARIES
    NAMES numa
    PATHS ${ROCM_PATH}/lib/rocm_sysdeps/lib $ENV{ROCM_PATH}/lib/rocm_sysdeps/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA DEFAULT_MSG NUMA_LIBRARIES NUMA_INCLUDE_DIR)

if(NUMA_FOUND AND NOT TARGET numa::numa)
    add_library(numa::numa UNKNOWN IMPORTED)
    set_target_properties(
        numa::numa
        PROPERTIES
            IMPORTED_LOCATION "${NUMA_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(NUMA_LIBRARIES NUMA_INCLUDE_DIR)
