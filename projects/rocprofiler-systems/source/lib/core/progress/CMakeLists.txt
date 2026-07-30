# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(progress_sources
    ${CMAKE_CURRENT_LIST_DIR}/bar.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker.cpp
)

set(progress_headers
    ${CMAKE_CURRENT_LIST_DIR}/bar.hpp
    ${CMAKE_CURRENT_LIST_DIR}/callback.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker.hpp
)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${progress_sources} ${progress_headers}
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
