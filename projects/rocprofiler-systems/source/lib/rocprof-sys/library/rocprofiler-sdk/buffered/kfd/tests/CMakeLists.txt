# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    buffered-domains-unit-tests
    OBJECT
    test_queue.cpp
    test_page_fault.cpp
    test_page_migrate.cpp
    test_event_queue.cpp
    test_event_unmap_from_gpu.cpp
    test_event_dropped_events.cpp
    test_event_page_fault.cpp
    test_event_page_migrate.cpp
)

target_link_libraries(
    buffered-domains-unit-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-common-library
        rocprofiler-systems-logger
)

target_include_directories(
    buffered-domains-unit-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
        ${CMAKE_BINARY_DIR}/source/lib/
)
