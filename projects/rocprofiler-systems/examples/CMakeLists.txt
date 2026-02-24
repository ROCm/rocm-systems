# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(rocprofiler-systems-examples LANGUAGES C CXX)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)

set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
set(CMAKE_CXX_VISIBILITY_PRESET "default")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_CLANG_TIDY)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME examples)

if(ROCPROFSYS_BUILD_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -fno-omit-frame-pointer")
endif()

option(BUILD_SHARED_LIBS "Build dynamic libraries" ON)

if(CMAKE_PROJECT_NAME STREQUAL "rocprofiler-systems")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
    rocprofiler_systems_add_option(ROCPROFSYS_INSTALL_EXAMPLES
                                   "Install rocprofiler-systems examples" OFF
    )
else()
    option(ROCPROFSYS_INSTALL_EXAMPLES "Install rocprofiler-systems examples" ON)
endif()

if(ROCPROFSYS_INSTALL_EXAMPLES)
    include(GNUInstallDirs)
endif()

# GPU architectures to compile for
# If not set, auto-detects from system GPU (or empty if no GPU present)
if(
    NOT DEFINED ROCPROFSYS_GFX_TARGETS
    OR ROCPROFSYS_GFX_TARGETS STREQUAL ""
    OR ROCPROFSYS_GFX_TARGETS STREQUAL "default"
)
    rocprofiler_systems_get_gfx_archs(ROCPROFSYS_GFX_TARGETS)
endif()

set(ROCPROFSYS_GFX_TARGETS
    "${ROCPROFSYS_GFX_TARGETS}"
    CACHE STRING
    "GPU architectures to compile for (semicolon-separated)"
    FORCE
)

if(ROCPROFSYS_GFX_TARGETS)
    message(STATUS "")
    rocprofiler_systems_message(STATUS "Detected targets:")
    foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
        rocprofiler_systems_lookup_gfx(${arch} _GFX_TYPE)
        message(STATUS "    ${arch} (categories: ${_GFX_TYPE})")
    endforeach()
    message(STATUS "")
else()
    rocprofiler_systems_message(STATUS "No GPU targets detected/set")
endif()

set(ROCPROFSYS_EXAMPLE_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "")
# defines function for creating causal profiling exes
include(${CMAKE_CURRENT_LIST_DIR}/causal-helpers.cmake)

add_subdirectory(transpose)
add_subdirectory(parallel-overhead)
add_subdirectory(code-coverage)
add_subdirectory(user-api)
add_subdirectory(openmp)
add_subdirectory(mpi)
add_subdirectory(python)
add_subdirectory(lulesh)
add_subdirectory(rccl)
add_subdirectory(rewrite-caller)
add_subdirectory(causal)
add_subdirectory(trace-time-window)
add_subdirectory(fork)
add_subdirectory(videodecode)
add_subdirectory(jpegdecode)
add_subdirectory(roctx)
add_subdirectory(thread-limit)
add_subdirectory(transferBench)
add_subdirectory(scratch-memory)
