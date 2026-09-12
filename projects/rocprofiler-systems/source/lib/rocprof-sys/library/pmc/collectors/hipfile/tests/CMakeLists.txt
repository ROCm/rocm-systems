# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Drives the collector through a mock backend and stub settings/cache policies, so no
# hipFile package, no GPU-direct storage hardware, and no trace cache are required. This
# is what makes the fastpath and fallback attribution verifiable in CI.
add_library(
    pmc-hipfile-collector-tests
    OBJECT
    test_device.cpp
    test_hipfile_traits.cpp
    test_collector_workflow.cpp
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
