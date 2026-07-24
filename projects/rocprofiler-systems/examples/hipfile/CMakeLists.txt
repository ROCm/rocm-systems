# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# Set AMD Clang as default compiler for standalone builds
if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(
        _AMDCLANG
        NAMES amdclang++
        HINTS ${ROCM_PATH} $ENV{ROCM_PATH} /opt/rocm
        PATH_SUFFIXES bin
    )
    if(_AMDCLANG)
        set(CMAKE_CXX_COMPILER ${_AMDCLANG})
    endif()
    unset(_AMDCLANG)
endif()

project(rocprofiler-systems-hipfile-example LANGUAGES CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

# This example requires the hipFILE runtime.
find_package(HIP QUIET)

if(NOT HIP_FOUND)
    message(WARNING "hip is not found. Skip hipfile example.")
    return()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCmVersion_DIR}/lib/cmake ${ROCmVersion_DIR})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()
string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")

# Locate hipFILE - prefer the exported CMake target, fall back to a manual search.
find_package(hipfile QUIET CONFIG)

set(hipfile_TARGET "")
if(TARGET hip::hipfile)
    set(hipfile_TARGET hip::hipfile)
elseif(TARGET hipfile)
    set(hipfile_TARGET hipfile)
else()
    find_path(
        hipfile_INCLUDE_DIR
        NAMES hipfile.h
        HINTS ${ROCmVersion_DIR} ${ROCM_PATH} $ENV{ROCM_PATH} /opt/rocm
        PATH_SUFFIXES include
    )
    find_library(
        hipfile_LIBRARY
        NAMES hipfile
        HINTS ${ROCmVersion_DIR} ${ROCM_PATH} $ENV{ROCM_PATH} /opt/rocm
        PATH_SUFFIXES lib lib64
    )
    if(hipfile_INCLUDE_DIR AND hipfile_LIBRARY)
        add_library(hipfile::hipfile UNKNOWN IMPORTED)
        set_target_properties(
            hipfile::hipfile
            PROPERTIES
                IMPORTED_LOCATION "${hipfile_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${hipfile_INCLUDE_DIR}"
        )
        set(hipfile_TARGET hipfile::hipfile)
    endif()
endif()

if(NOT hipfile_TARGET)
    message(
        WARNING
        "-- hipFILE runtime not found! Skipping hipfile example (requires the hipfile library and hipfile.h)."
    )
    return()
endif()

find_package(Threads REQUIRED)

add_executable(hipfile-trace hipfile_trace.cpp)
target_compile_definitions(hipfile-trace PRIVATE __HIP_PLATFORM_AMD__)
target_link_libraries(hipfile-trace PRIVATE hip::host ${hipfile_TARGET} Threads::Threads)
target_compile_options(hipfile-trace PRIVATE ${_FLAGS})

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET hipfile-trace)
    install(
        TARGETS hipfile-trace
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()
