# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocprofiler-systems-rocprofiler-sdk-backend INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-rocprofiler-sdk-backend
    ALIAS rocprofiler-systems-rocprofiler-sdk-backend
)

target_link_libraries(
    rocprofiler-systems-rocprofiler-sdk-backend
    INTERFACE
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-rocm
)
