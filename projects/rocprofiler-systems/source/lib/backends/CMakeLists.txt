# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_subdirectory(concepts)
add_subdirectory(amd_smi)
add_subdirectory(procfs)

if(ROCPROFSYS_HIPFILE_SUPPORT)
    add_subdirectory(hipfile)
elseif(ROCPROFSYS_BUILD_TESTING)
    # hipFile backend tests drive a mock wrapper and have no hipFile dependency, so
    # they build even when the package was not found.
    add_subdirectory(hipfile/tests)
endif()

if(
    (ROCPROFSYS_ROCM_VERSION_MAJOR GREATER 6)
    OR (
        ROCPROFSYS_ROCM_VERSION_MAJOR EQUAL 6
        AND ROCPROFSYS_ROCM_VERSION_MINOR GREATER_EQUAL 4
    )
)
    add_subdirectory(rocprofiler_sdk)
endif()
