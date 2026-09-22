# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_subdirectory(concepts)
add_subdirectory(amd_smi)
add_subdirectory(procfs)

if(
    (ROCPROFSYS_ROCM_VERSION_MAJOR GREATER 6)
    OR (
        ROCPROFSYS_ROCM_VERSION_MAJOR EQUAL 6
        AND ROCPROFSYS_ROCM_VERSION_MINOR GREATER_EQUAL 4
    )
)
    add_subdirectory(rocprofiler_sdk)
endif()
