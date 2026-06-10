# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    rocprofiler-systems-rocprofiler-sdk-backend
    STATIC
    backend.cpp
    backend.hpp
    types.hpp
)
add_library(
    rocprofiler-systems::rocprofiler-systems-rocprofiler-sdk-backend
    ALIAS rocprofiler-systems-rocprofiler-sdk-backend
)

set_target_properties(
    rocprofiler-systems-rocprofiler-sdk-backend
    PROPERTIES LINKER_LANGUAGE CXX
)

target_link_libraries(
    rocprofiler-systems-rocprofiler-sdk-backend
    PUBLIC
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-rocm
)
