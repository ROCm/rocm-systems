# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(trace_cache_sources
    ${CMAKE_CURRENT_LIST_DIR}/cache_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/buffer_storage.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_types.cpp
    ${CMAKE_CURRENT_LIST_DIR}/discovery.cpp
    ${CMAKE_CURRENT_LIST_DIR}/metadata_registry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/post_processor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rocpd_processor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/perfetto_processor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unified_memory_processor.cpp
)

set(trace_cache_headers
    ${CMAKE_CURRENT_LIST_DIR}/cache_manager.hpp
    ${CMAKE_CURRENT_LIST_DIR}/data_types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/discovery.hpp
    ${CMAKE_CURRENT_LIST_DIR}/storage_parser.hpp
    ${CMAKE_CURRENT_LIST_DIR}/buffer_storage.hpp
    ${CMAKE_CURRENT_LIST_DIR}/cacheable.hpp
    ${CMAKE_CURRENT_LIST_DIR}/type_registry.hpp
    ${CMAKE_CURRENT_LIST_DIR}/cache_type_traits.hpp
    ${CMAKE_CURRENT_LIST_DIR}/metadata_registry.hpp
    ${CMAKE_CURRENT_LIST_DIR}/rocpd_processor.hpp
    ${CMAKE_CURRENT_LIST_DIR}/perfetto_processor.hpp
    ${CMAKE_CURRENT_LIST_DIR}/post_processor.hpp
    ${CMAKE_CURRENT_LIST_DIR}/unified_memory_processor.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sample_processor.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sample_type.hpp
    ${CMAKE_CURRENT_LIST_DIR}/storage_parser_alias.hpp
)

target_sources(rocprofiler-systems-core-library PRIVATE ${trace_cache_sources})

target_include_directories(
    rocprofiler-systems-core-library
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
