# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(profiler-hub_sources
    ${CMAKE_CURRENT_LIST_DIR}/writer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/writer_impl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/insert_validator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/json_serializers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reader_impl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/storage.cpp
    ${CMAKE_CURRENT_LIST_DIR}/storage_impl.cpp
)
target_sources(profiler-hub-objects PRIVATE ${profiler-hub_sources})

# ------------------------------------------------------------ #
# Common headers
# ------------------------------------------------------------ #
target_include_directories(
    profiler-hub-objects
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/common
)

add_subdirectory(queries)
add_subdirectory(data_storage)

# Link queries library to profiler-hub-objects (for compile-time includes)
target_link_libraries(profiler-hub-objects PRIVATE profiler-hub_queries)
