# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(rocprofiler-systems-trace-time-window-example LANGUAGES CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

set(CMAKE_BUILD_TYPE "Debug")

add_executable(trace-time-window trace-time-window.cpp)
target_compile_options(trace-time-window PRIVATE ${_FLAGS})

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET trace-time-window)
    install(
        TARGETS trace-time-window
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()
