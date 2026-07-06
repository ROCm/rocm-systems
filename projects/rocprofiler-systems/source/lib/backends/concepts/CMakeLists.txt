# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocprofiler-systems-backend-concepts INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-backend-concepts
    ALIAS rocprofiler-systems-backend-concepts
)

target_link_libraries(
    rocprofiler-systems-backend-concepts
    INTERFACE
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-compile-definitions
)
