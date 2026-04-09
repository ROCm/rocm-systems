# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

add_library(pmc-gpu-collector-tests OBJECT test_device.cpp)

target_link_libraries(
    pmc-gpu-collector-tests
    PUBLIC
        rocprofiler-systems-pmc-library
        rocprofiler-systems-googletest-library
        rocprofiler-systems-logger
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-rocm
        rocprofiler-systems::rocprofiler-systems-timemory
)
