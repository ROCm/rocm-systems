# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(trace-period-config-tests OBJECT test_trace_period_config.cpp)

target_link_libraries(
    trace-period-config-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-logger
        rocprofiler-systems-headers
        rocprofiler-systems-timemory
)

target_include_directories(
    trace-period-config-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${CMAKE_BINARY_DIR}/source/lib/
)
