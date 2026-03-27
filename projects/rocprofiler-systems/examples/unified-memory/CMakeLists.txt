# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(rocprofiler-systems-unified-memory-example LANGUAGES CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

find_package(hip QUIET HINTS ${ROCmVersion_DIR} PATHS ${ROCmVersion_DIR})

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
    message(AUTHOR_WARNING "hipcc not found. Cannot build unified-memory target.")
    return()
endif()

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
    if(
        CMAKE_CXX_COMPILER STREQUAL HIPCC_EXECUTABLE
        OR "${CMAKE_CXX_COMPILER}" MATCHES "hipcc"
    )
        set(CMAKE_CXX_COMPILER_IS_HIPCC 1 CACHE BOOL "HIP compiler")
    endif()
endif()

if(
    (
        NOT CMAKE_CXX_COMPILER_IS_HIPCC
        OR (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT hip_FOUND)
    )
    AND (NOT COMMAND rocprofiler_systems_custom_compilation AND NOT HIPCC_EXECUTABLE)
)
    message(AUTHOR_WARNING "unified-memory target could not be built")
    return()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unified-memory unified-memory.cpp)

foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
    target_compile_options(unified-memory PRIVATE --offload-arch=${arch})
    target_link_options(unified-memory PRIVATE --offload-arch=${arch})
endforeach()

if(
    CMAKE_CXX_COMPILER_ID MATCHES "Clang"
    AND NOT CMAKE_CXX_COMPILER_IS_HIPCC
    AND NOT HIPCC_EXECUTABLE
)
    target_link_libraries(
        unified-memory
        PRIVATE
            $<TARGET_NAME_IF_EXISTS:rocprofiler-systems::rocprofiler-systems-compile-options>
            $<TARGET_NAME_IF_EXISTS:hip::host>
            $<TARGET_NAME_IF_EXISTS:hip::device>
    )
else()
    target_compile_options(unified-memory PRIVATE -W -Wall)
endif()

if("${CMAKE_BUILD_TYPE}" MATCHES "Release")
    target_compile_options(unified-memory PRIVATE -g1)
endif()

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
    rocprofiler_systems_custom_compilation(COMPILER ${HIPCC_EXECUTABLE} TARGET unified-memory)
endif()

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET unified-memory)
    install(
        TARGETS unified-memory
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()
