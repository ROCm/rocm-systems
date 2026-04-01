# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

set(pmc_common_tests_sources test_collector_slice.cpp)

add_library(pmc-common-tests OBJECT ${pmc_common_tests_sources})

target_link_libraries(
    pmc-common-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-interface-library
)

target_include_directories(
    pmc-common-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib/core
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
)
