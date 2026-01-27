# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(rocprofiler-systems-transferBench-example LANGUAGES CXX)

if(ROCPROFSYS_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(
        ${PROJECT_NAME} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
        OR ${_DIR} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
    )
        return()
    endif()
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
    message(AUTHOR_WARNING "hipcc could not be found. Cannot build transferBench target")
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
    message(AUTHOR_WARNING "transferBench target could not be built")
    return()
endif()

find_package(Threads REQUIRED)

# Find HSA runtime library
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

find_path(
    HSA_RUNTIME_INCLUDE_DIR
    NAMES hsa/hsa.h
    HINTS ${ROCmVersion_DIR} ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATHS ${ROCmVersion_DIR} ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES include
)

if(NOT HSA_RUNTIME_LIBRARY OR NOT HSA_RUNTIME_INCLUDE_DIR)
    message(
        AUTHOR_WARNING
        "HSA runtime library not found. Cannot build transferBench target"
    )
    return()
endif()

add_executable(transferBench AllToAll.cpp)
target_link_libraries(transferBench PRIVATE Threads::Threads ${HSA_RUNTIME_LIBRARY})
target_include_directories(
    transferBench
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${HSA_RUNTIME_INCLUDE_DIR}
)

if(
    CMAKE_CXX_COMPILER_ID MATCHES "Clang"
    AND NOT CMAKE_CXX_COMPILER_IS_HIPCC
    AND NOT HIPCC_EXECUTABLE
)
    target_link_libraries(
        transferBench
        PRIVATE
            $<TARGET_NAME_IF_EXISTS:rocprofiler-systems::rocprofiler-systems-compile-options>
            $<TARGET_NAME_IF_EXISTS:hip::host>
            $<TARGET_NAME_IF_EXISTS:hip::device>
    )
else()
    target_compile_options(transferBench PRIVATE -W -Wall)
endif()

if("${CMAKE_BUILD_TYPE}" MATCHES "Release")
    target_compile_options(transferBench PRIVATE -g1)
endif()

# Specify GPU architectures to compile for
foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
    rocprofiler_systems_lookup_gfx(${arch} GPU_CATEGORIES)
    if("apu" IN_LIST GPU_CATEGORIES AND NOT "instinct" IN_LIST GPU_CATEGORIES)
        continue()
    endif()
    target_compile_options(transferBench PRIVATE --offload-arch=${arch})
    target_link_options(transferBench PUBLIC --offload-arch=${arch})
endforeach()

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
    # defined in MacroUtilities.cmake
    rocprofiler_systems_custom_compilation(COMPILER ${HIPCC_EXECUTABLE} TARGET transferBench)
endif()

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET transferBench)
    install(
        TARGETS transferBench
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()
