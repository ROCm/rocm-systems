# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-systems-scratch-memory-example LANGUAGES CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

find_program(
    HIPCC_EXECUTABLE
    NAMES hipcc
    HINTS ${ROCmVersion_DIR} ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATHS ${ROCmVersion_DIR} ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    NO_CACHE
)
mark_as_advanced(HIPCC_EXECUTABLE)

if(NOT HIPCC_EXECUTABLE)
    message(AUTHOR_WARNING "hipcc not found. Cannot build scratch-memory target.")
    return()
endif()

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC)
    if(
        CMAKE_CXX_COMPILER STREQUAL HIPCC_EXECUTABLE
        OR "${CMAKE_CXX_COMPILER}" MATCHES "hipcc"
    )
        set(CMAKE_CXX_COMPILER_IS_HIPCC 1 CACHE BOOL "HIP compiler")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(
    HSA_RUNTIME_LIBRARY
    NAMES hsa-runtime64
    HINTS ${ROCmVersion_DIR} ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATHS ${ROCmVersion_DIR} ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES lib lib64
)

if(NOT HSA_RUNTIME_LIBRARY)
    message(
        AUTHOR_WARNING
        "HSA runtime library not found. Cannot build scratch-memory target"
    )
    return()
endif()

find_package(Threads REQUIRED)

add_executable(scratch-memory scratch-memory.cpp)
target_compile_options(
    scratch-memory
    PRIVATE -W -Wall -Wextra -Wpedantic -Wshadow -Werror
)
target_link_libraries(scratch-memory PRIVATE Threads::Threads ${HSA_RUNTIME_LIBRARY})

foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
    target_compile_options(scratch-memory PRIVATE --offload-arch=${arch})
    target_link_options(scratch-memory PRIVATE --offload-arch=${arch})
endforeach()

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
    rocprofiler_systems_custom_compilation(COMPILER ${HIPCC_EXECUTABLE} TARGET scratch-memory)
endif()

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET scratch-memory)
    install(
        TARGETS scratch-memory
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()
