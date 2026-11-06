# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(thread_list_sources ${CMAKE_CURRENT_LIST_DIR}/internal_thread_list.cpp)

set(thread_list_headers ${CMAKE_CURRENT_LIST_DIR}/internal_thread_list.hpp)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${thread_list_sources} ${thread_list_headers}
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
