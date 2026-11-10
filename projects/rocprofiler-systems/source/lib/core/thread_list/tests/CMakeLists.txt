# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(thread-list-tests OBJECT test_internal_thread_list.cpp)

target_link_libraries(
    thread-list-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(
    thread-list-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${CMAKE_BINARY_DIR}/source/lib
)
