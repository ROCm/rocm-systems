# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Mock backend and stub settings, so no hipFile package or AMD Infinity Storage hardware
# is required.
add_library(
    pmc-hipfile-collector-tests
    OBJECT
    test_device.cpp
    test_hipfile_traits.cpp
    test_metric_groups.cpp
    test_types.cpp
)

target_link_libraries(
    pmc-hipfile-collector-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-logger
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-rocm
        rocprofiler-systems::rocprofiler-systems-compile-definitions
)

target_include_directories(
    pmc-hipfile-collector-tests
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
        ${PROJECT_BINARY_DIR}/source/lib
)
