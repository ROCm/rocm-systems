# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(pmc-gpu-collector-tests OBJECT test_device.cpp test_power_metrics.cpp)

target_link_libraries(
    pmc-gpu-collector-tests
    PUBLIC
        rocprofiler-systems-googletest-library
        rocprofiler-systems-logger
        rocprofiler-systems::rocprofiler-systems-headers
)

if(ROCPROFSYS_BUILD_AINIC)
    target_compile_definitions(pmc-gpu-collector-tests PUBLIC ROCPROFSYS_BUILD_AINIC=1)
endif()
