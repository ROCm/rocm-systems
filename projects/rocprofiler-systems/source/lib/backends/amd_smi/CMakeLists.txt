# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocprofiler-systems-amd-smi-backend INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-amd-smi-backend
    ALIAS rocprofiler-systems-amd-smi-backend
)

target_link_libraries(
    rocprofiler-systems-amd-smi-backend
    INTERFACE
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-compile-definitions
        rocprofiler-systems::rocprofiler-systems-rocm
)

if(ROCPROFSYS_BUILD_AINIC)
    target_compile_definitions(
        rocprofiler-systems-amd-smi-backend
        INTERFACE ROCPROFSYS_BUILD_AINIC=1
    )
endif()

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
