# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(collection_sources ${CMAKE_CURRENT_LIST_DIR}/sample_collector.cpp)

set(collection_headers
    ${CMAKE_CURRENT_LIST_DIR}/record.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sample_collector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/scheduler.hpp
    ${CMAKE_CURRENT_LIST_DIR}/schema.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sink_interface.hpp
    ${CMAKE_CURRENT_LIST_DIR}/source_control.hpp
    ${CMAKE_CURRENT_LIST_DIR}/source_interface.hpp
)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${collection_sources} ${collection_headers}
)
