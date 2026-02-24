# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-systems-scratch-memory-example LANGUAGES CXX)

if(ROCPROFSYS_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(
        PROJECT_NAME IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
        OR _DIR IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
    )
        return()
    endif()
endif()

find_program(
    hipcc_EXECUTABLE
    NAMES hipcc
    HINTS ${ROCmVersion_DIR} ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATHS ${ROCmVersion_DIR} ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    NO_CACHE
)
mark_as_advanced(hipcc_EXECUTABLE)

if(NOT hipcc_EXECUTABLE)
    message(AUTHOR_WARNING "hipcc not found. Cannot build scratch-memory target.")
    return()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(scratch-memory scratch-memory.cpp)
target_compile_options(
    scratch-memory
    PRIVATE -xhip -W -Wall -Wextra -Wpedantic -Wshadow -Werror
)

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
target_link_libraries(scratch-memory PRIVATE Threads::Threads ${HSA_RUNTIME_LIBRARY})

rocprofiler_systems_custom_compilation(COMPILER ${hipcc_EXECUTABLE} TARGET scratch-memory)

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET scratch-memory)
    install(
        TARGETS scratch-memory
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()
