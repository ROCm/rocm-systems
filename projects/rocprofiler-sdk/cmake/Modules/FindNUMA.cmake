# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# FindNUMA.cmake
#
# Locates the NUMA (Non-Uniform Memory Access) library (libnuma) and its
# development headers. Installed hsakmt and rocSHMEM config files may call
# find_dependency(NUMA), but cmake does not ship a FindNUMA module.
#
# Result variables:
#   NUMA_FOUND        - True if libnuma and numa.h were found
#   NUMA_INCLUDE_DIR  - Directory containing numa.h
#   NUMA_LIBRARIES    - The libnuma library
#
# Imported target:
#   numa::numa        - Imported target required by hsakmt and rocSHMEM.

include_guard(DIRECTORY)

# Prefer a packaged NUMA config, including the one provided by ROCm sysdeps.
find_package(NUMA CONFIG QUIET HINTS ${ROCM_PATH}/lib/rocm_sysdeps
             $ENV{ROCM_PATH}/lib/rocm_sysdeps)

if(TARGET numa::numa)
    if(NOT NUMA_INCLUDE_DIR)
        if(NUMA_INCLUDE_DIRS)
            set(NUMA_INCLUDE_DIR "${NUMA_INCLUDE_DIRS}")
        else()
            get_target_property(NUMA_INCLUDE_DIR numa::numa INTERFACE_INCLUDE_DIRECTORIES)
        endif()
    endif()
    if(NOT NUMA_LIBRARIES)
        set(NUMA_LIBRARIES numa::numa)
    endif()
else()
    find_path(
        NUMA_INCLUDE_DIR
        NAMES numa.h
        PATHS ${ROCM_PATH}/lib/rocm_sysdeps/include
              $ENV{ROCM_PATH}/lib/rocm_sysdeps/include)
    find_library(
        NUMA_LIBRARIES
        NAMES numa
        PATHS ${ROCM_PATH}/lib/rocm_sysdeps/lib $ENV{ROCM_PATH}/lib/rocm_sysdeps/lib)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA DEFAULT_MSG NUMA_LIBRARIES NUMA_INCLUDE_DIR)

if(NUMA_FOUND AND NOT TARGET numa::numa)
    add_library(numa::numa UNKNOWN IMPORTED)
    set_target_properties(
        numa::numa PROPERTIES IMPORTED_LOCATION "${NUMA_LIBRARIES}"
                              INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}")
endif()

mark_as_advanced(NUMA_LIBRARIES NUMA_INCLUDE_DIR)
