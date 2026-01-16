# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(rocprofiler-systems-fork LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
list(APPEND _FLAGS -fno-inline)

find_package(Threads REQUIRED)
find_package(rocprofiler-systems REQUIRED COMPONENTS user)

# Basic fork example
add_executable(fork-example fork.cpp)
target_link_libraries(
    fork-example
    PRIVATE Threads::Threads rocprofiler-systems::rocprofiler-systems
)
target_compile_options(fork-example PRIVATE ${_FLAGS})

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET fork-example)
    install(
        TARGETS fork-example
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()

# HIP fork example (multi-process concurrency test)
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

if(HIPCC_EXECUTABLE)
    if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
        if(
            CMAKE_CXX_COMPILER STREQUAL HIPCC_EXECUTABLE
            OR "${CMAKE_CXX_COMPILER}" MATCHES "hipcc"
        )
            set(CMAKE_CXX_COMPILER_IS_HIPCC 1 CACHE BOOL "HIP compiler")
        endif()
    endif()

    if(
        CMAKE_CXX_COMPILER_IS_HIPCC
        OR hip_FOUND
        OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND HIPCC_EXECUTABLE)
        OR COMMAND rocprofiler_systems_custom_compilation
    )
        add_executable(hipMallocConcurrencyMproc hipMallocConcurrencyMproc.cpp)
        target_link_libraries(hipMallocConcurrencyMproc PRIVATE Threads::Threads)
        set_target_properties(
            hipMallocConcurrencyMproc
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/fork"
        )
        # Specify GPU architectures to compile for
        foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
            target_compile_options(
                hipMallocConcurrencyMproc
                PRIVATE --offload-arch=${arch}
            )
            target_link_options(hipMallocConcurrencyMproc PUBLIC --offload-arch=${arch})
        endforeach()

        if(
            CMAKE_CXX_COMPILER_ID MATCHES "Clang"
            AND NOT CMAKE_CXX_COMPILER_IS_HIPCC
            AND NOT HIPCC_EXECUTABLE
        )
            target_link_libraries(
                hipMallocConcurrencyMproc
                PRIVATE
                    $<TARGET_NAME_IF_EXISTS:rocprofiler-systems::rocprofiler-systems-compile-options>
                    $<TARGET_NAME_IF_EXISTS:hip::host>
                    $<TARGET_NAME_IF_EXISTS:hip::device>
            )
        else()
            target_compile_options(hipMallocConcurrencyMproc PRIVATE -W -Wall)
        endif()

        if("${CMAKE_BUILD_TYPE}" MATCHES "Release")
            target_compile_options(hipMallocConcurrencyMproc PRIVATE -g1)
        endif()

        if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
            # defined in MacroUtilities.cmake
            rocprofiler_systems_custom_compilation(COMPILER ${HIPCC_EXECUTABLE} TARGET hipMallocConcurrencyMproc)
        endif()

        if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET hipMallocConcurrencyMproc)
            install(
                TARGETS hipMallocConcurrencyMproc
                DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
                COMPONENT rocprofiler-systems-examples
            )
        endif()
    else()
        message(
            AUTHOR_WARNING
            "hipMallocConcurrencyMproc target could not be built (missing HIP support)"
        )
    endif()
else()
    message(
        AUTHOR_WARNING
        "hipcc could not be found. Cannot build hipMallocConcurrencyMproc target"
    )
endif()
