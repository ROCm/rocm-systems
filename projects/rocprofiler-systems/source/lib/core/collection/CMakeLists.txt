# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(collection_sources
    ${CMAKE_CURRENT_LIST_DIR}/detail/source_runner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fork_gate.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sample_collector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/scheduler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tick_grid.cpp
)

set(collection_headers
    ${CMAKE_CURRENT_LIST_DIR}/detail/source_runner.hpp
    ${CMAKE_CURRENT_LIST_DIR}/fork_gate.hpp
    ${CMAKE_CURRENT_LIST_DIR}/guarded_read.hpp
    ${CMAKE_CURRENT_LIST_DIR}/metric_row.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sample_batch.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sample_collector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/scheduler.hpp
    ${CMAKE_CURRENT_LIST_DIR}/schema.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sink_interface.hpp
    ${CMAKE_CURRENT_LIST_DIR}/source_interface.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tick_grid.hpp
)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${collection_sources} ${collection_headers}
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
