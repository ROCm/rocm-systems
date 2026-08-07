# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocprofiler-sdk-backend-tests OBJECT test_backend.cpp)

target_link_libraries(
    rocprofiler-sdk-backend-tests
    PUBLIC
        rocprofiler-systems-rocprofiler-sdk-backend
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-rocm
)
