# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

if(ROCPROFSYS_DISABLE_EXAMPLES)
    if("openmp" IN_LIST ROCPROFSYS_DISABLE_EXAMPLES)
        return()
    endif()
endif()

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(rocprofiler-systems-openmp LANGUAGES CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

file(GLOB common_source ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp)
add_library(openmp-common OBJECT ${common_source})
target_include_directories(openmp-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)

include(${CMAKE_CURRENT_LIST_DIR}/openmp-helper.cmake)

target_compile_options(openmp-common PUBLIC -fopenmp)
target_link_options(openmp-common PUBLIC -fopenmp)
rocprofiler_systems_custom_compilation(
    COMPILER ${amdclangpp_EXECUTABLE}
    TARGET openmp-common
)

add_subdirectory(CG)
add_subdirectory(LU)
add_subdirectory(fortran)
add_subdirectory(target)
