# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    buffered-domains-unit-test
    OBJECT
    test_kernel_dispatch.cpp
    test_memory_copy.cpp
    test_memory_allocation.cpp
    test_scratch_memory.cpp
    ${kfd_domains_test_sources}
)

target_link_libraries(
    buffered-domains-unit-test
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-common-library
        rocprofiler-systems-logger
)

target_include_directories(
    buffered-domains-unit-test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
        ${CMAKE_BINARY_DIR}/source/lib/
)
