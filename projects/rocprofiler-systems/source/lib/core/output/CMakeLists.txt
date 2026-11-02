# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(output_sources ${CMAKE_CURRENT_LIST_DIR}/output_summary.cpp)

set(output_headers ${CMAKE_CURRENT_LIST_DIR}/output_summary.hpp)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${output_sources} ${output_headers}
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
