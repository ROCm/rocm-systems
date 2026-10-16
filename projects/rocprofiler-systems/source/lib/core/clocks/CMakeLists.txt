# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(clocks_sources ${CMAKE_CURRENT_LIST_DIR}/waitable/steady.cpp)

set(clocks_headers
    ${CMAKE_CURRENT_LIST_DIR}/thread_cpu.hpp
    ${CMAKE_CURRENT_LIST_DIR}/timeline.hpp
    ${CMAKE_CURRENT_LIST_DIR}/waitable/contract.hpp
    ${CMAKE_CURRENT_LIST_DIR}/waitable/posix.hpp
    ${CMAKE_CURRENT_LIST_DIR}/waitable/steady.hpp
)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${clocks_sources} ${clocks_headers}
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
