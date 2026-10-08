# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(profiler-hub_sources
    ${CMAKE_CURRENT_LIST_DIR}/writer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/writer_impl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/insert_validator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/json_serializers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reader_impl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reader_catalog.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reader_catalog.hpp
    ${CMAKE_CURRENT_LIST_DIR}/populate_reader_catalog.cpp
    ${CMAKE_CURRENT_LIST_DIR}/populate_reader_catalog.hpp
    ${CMAKE_CURRENT_LIST_DIR}/storage.cpp
    ${CMAKE_CURRENT_LIST_DIR}/storage_impl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ph_ctx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ph_ctx.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ph_future.hpp
    ${CMAKE_CURRENT_LIST_DIR}/profiler_hub_c_api.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cached_track_reader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cached_track_reader.hpp
    ${CMAKE_CURRENT_LIST_DIR}/connection_source.hpp
    ${CMAKE_CURRENT_LIST_DIR}/event_partitions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/event_partitions.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pooled_connection_source.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pooled_connection_source.hpp
    ${CMAKE_CURRENT_LIST_DIR}/track_read_options.hpp
    ${CMAKE_CURRENT_LIST_DIR}/track_row_reader.hpp
    ${CMAKE_CURRENT_LIST_DIR}/track_window.hpp
    ${CMAKE_CURRENT_LIST_DIR}/common/natural_merge_sort.hpp
    ${CMAKE_CURRENT_LIST_DIR}/common/thread_pool.cpp
    ${CMAKE_CURRENT_LIST_DIR}/common/thread_pool.hpp
    ${CMAKE_CURRENT_LIST_DIR}/common/connection.cpp
    ${CMAKE_CURRENT_LIST_DIR}/common/connection.hpp
    ${CMAKE_CURRENT_LIST_DIR}/common/connection_pool.cpp
    ${CMAKE_CURRENT_LIST_DIR}/common/connection_pool.hpp
)
target_sources(profiler-hub-objects PRIVATE ${profiler-hub_sources})

# ------------------------------------------------------------ #
# Common headers
# ------------------------------------------------------------ #
target_include_directories(
    profiler-hub-objects
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/common
)

find_package(Threads REQUIRED)
target_link_libraries(profiler-hub-objects PRIVATE Threads::Threads)

add_subdirectory(queries)
add_subdirectory(data_storage)

# Link queries library to profiler-hub-objects (for compile-time includes)
target_link_libraries(profiler-hub-objects PRIVATE profiler-hub_queries)
