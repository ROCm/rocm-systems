# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocprofiler-sdk-kfd-events-unit-tests OBJECT test_kfd_queue.cpp)

target_link_libraries(
    rocprofiler-sdk-kfd-events-unit-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-core
        rocprofiler-systems::rocprofiler-systems-interface-library
        rocprofiler-systems::rocprofiler-systems-timemory
        rocprofiler-systems-common-library
        rocprofiler-systems-logger
)

target_include_directories(
    rocprofiler-sdk-kfd-events-unit-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
        ${CMAKE_BINARY_DIR}/source/lib/
)
