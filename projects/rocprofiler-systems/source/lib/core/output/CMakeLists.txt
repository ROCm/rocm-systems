# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(output_sources
    ${CMAKE_CURRENT_LIST_DIR}/registry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/process_tree.cpp
    ${CMAKE_CURRENT_LIST_DIR}/summary_writer.cpp
)

set(output_headers
    ${CMAKE_CURRENT_LIST_DIR}/artifact.hpp
    ${CMAKE_CURRENT_LIST_DIR}/process_metadata.hpp
    ${CMAKE_CURRENT_LIST_DIR}/registry.hpp
    ${CMAKE_CURRENT_LIST_DIR}/process_tree.hpp
    ${CMAKE_CURRENT_LIST_DIR}/summary_writer.hpp
)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${output_sources} ${output_headers}
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
