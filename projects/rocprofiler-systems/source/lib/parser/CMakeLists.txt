# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems parser library
#
# ------------------------------------------------------------------------------#

add_library(rocprofiler-systems-parser STATIC)
add_library(
    rocprofiler-systems::rocprofiler-systems-parser
    ALIAS rocprofiler-systems-parser
)

target_sources(
    rocprofiler-systems-parser
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/schema.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/schema_validator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/value_text.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/config_store.cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/schema.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/schema_validator.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/value_text.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/config_store.hpp
)

get_filename_component(PARSER_SOURCE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)

target_include_directories(
    rocprofiler-systems-parser
    PUBLIC $<BUILD_INTERFACE:${PARSER_SOURCE_INCLUDE_DIR}>
)

target_link_libraries(
    rocprofiler-systems-parser
    PRIVATE
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-common-library>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-fmt>
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
