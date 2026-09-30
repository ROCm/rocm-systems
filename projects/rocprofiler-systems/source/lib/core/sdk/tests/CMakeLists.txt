# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(tracing-config-tests OBJECT test_tracing_config.cpp)

target_link_libraries(
    tracing-config-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-logger
        rocprofiler-systems-headers
        rocprofiler-systems-rocm
        rocprofiler-systems-timemory
)

target_include_directories(tracing-config-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)
