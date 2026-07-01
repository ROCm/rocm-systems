# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

add_library(
    pmc-sdk-pmc-collector-tests
    OBJECT
    test_device.cpp
    test_sample.cpp
    test_collector_workflow.cpp
)

target_link_libraries(
    pmc-sdk-pmc-collector-tests
    PUBLIC
        rocprofiler-systems-pmc-library
        rocprofiler-systems-googletest-library
        rocprofiler-systems-logger
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-rocm
        rocprofiler-systems::rocprofiler-systems-timemory
)
