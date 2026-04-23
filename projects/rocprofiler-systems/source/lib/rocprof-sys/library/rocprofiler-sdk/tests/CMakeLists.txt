# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(rocprofiler_sdk_components_unit_tests_sources
    test_rccl_unit.cpp
    test_roctx_client.cpp
    ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys/library/rocprofiler-sdk/trace_control.cpp
)

add_library(
    rocprofiler-sdk-components-unit-tests
    OBJECT
    ${rocprofiler_sdk_components_unit_tests_sources}
)

target_link_libraries(
    rocprofiler-sdk-components-unit-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-core
        rocprofiler-systems::rocprofiler-systems-interface-library
        rocprofiler-systems::rocprofiler-systems-timemory
        rocprofiler-systems-common-library
        rocprofiler-systems-logger
)

target_include_directories(
    rocprofiler-sdk-components-unit-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
        ${CMAKE_BINARY_DIR}/source/lib/
)
