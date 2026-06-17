# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------ #
# profiler-hub_queries static library
# ------------------------------------------------------------ #

add_library(profiler-hub_queries STATIC)

set(queries_sources
    ${CMAKE_CURRENT_LIST_DIR}/query_builder_base.cpp
    ${CMAKE_CURRENT_LIST_DIR}/select/select_query_builders.cpp
)

target_sources(profiler-hub_queries PRIVATE ${queries_sources})

target_include_directories(
    profiler-hub_queries
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
)

# Include common headers for traits
target_include_directories(
    profiler-hub_queries
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../common
)

target_compile_options(
    profiler-hub_queries
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall
        -Wextra
        -Wshadow
        -Wvla
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wnon-virtual-dtor
        -Woverloaded-virtual
        -Wnull-dereference
        >
)

set_target_properties(
    profiler-hub_queries
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
)
