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
        ${CMAKE_CURRENT_SOURCE_DIR}/default_parser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/env_parser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cli_parser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/file_parser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/preset_parser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/precedence.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/value_provider.cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/schema.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/schema_validator.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/value_text.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/config_store.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/parse_result.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/default_parser.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/env_parser.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cli_parser.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/file_parser.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/preset_parser.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/precedence.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/value_provider.hpp
)

get_filename_component(PARSER_SOURCE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)

target_include_directories(
    rocprofiler-systems-parser
    PUBLIC $<BUILD_INTERFACE:${PARSER_SOURCE_INCLUDE_DIR}>
)

target_link_libraries(
    rocprofiler-systems-parser
    PUBLIC $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-json>
    PRIVATE
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-common-library>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-compile-options>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-fmt>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-sanitizer>
)

if(TARGET rocprofiler-systems::rocprofiler-systems-rocprofiler-sdk-backend)
    target_compile_definitions(
        rocprofiler-systems-parser
        PRIVATE ROCPROFSYS_PARSER_HAS_SDK_BACKEND
    )
    target_link_libraries(
        rocprofiler-systems-parser
        PRIVATE rocprofiler-systems::rocprofiler-systems-rocprofiler-sdk-backend
    )
endif()

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
