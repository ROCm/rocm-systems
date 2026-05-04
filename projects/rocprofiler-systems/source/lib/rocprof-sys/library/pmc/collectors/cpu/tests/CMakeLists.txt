# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

add_library(pmc-cpu-collector-tests OBJECT test_device.cpp test_sample.cpp)

target_link_libraries(
    pmc-cpu-collector-tests
    PUBLIC
        rocprofiler-systems-pmc-library
        rocprofiler-systems-googletest-library
        rocprofiler-systems-logger
)

target_include_directories(
    pmc-cpu-collector-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
)
