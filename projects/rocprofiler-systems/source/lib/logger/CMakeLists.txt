# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems logger library
#
# ------------------------------------------------------------------------------#

add_library(rocprofiler-systems-logger STATIC)
add_library(rocprofsys-logger ALIAS rocprofiler-systems-logger)
add_library(
    rocprofiler-systems::rocprofiler-systems-logger
    ALIAS rocprofiler-systems-logger
)

target_sources(
    rocprofiler-systems-logger
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/logger.cpp
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/logger.hpp ${CMAKE_CURRENT_SOURCE_DIR}/debug.hpp
)

get_filename_component(LOGGER_SOURCE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)

target_include_directories(
    rocprofiler-systems-logger
    PUBLIC $<BUILD_INTERFACE:${LOGGER_SOURCE_INCLUDE_DIR}>
)

target_link_libraries(
    rocprofiler-systems-logger
    PUBLIC $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-spdlog>
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
