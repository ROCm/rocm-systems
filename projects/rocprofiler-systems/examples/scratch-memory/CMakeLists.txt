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

if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH}
        ENV ROCM_PATH
        /opt/rocm
        PATHS ${ROCM_PATH}
        ENV ROCM_PATH
        /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE
    )
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
else()
    message(WARNING "HIP compiler not found. Cannot build scratch-memory target.")
    return()
endif()

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set_source_files_properties(scratch-memory.cpp PROPERTIES LANGUAGE HIP)
add_executable(scratch-memory)
target_sources(scratch-memory PRIVATE scratch-memory.cpp)
target_compile_options(
    scratch-memory
    PRIVATE -W -Wall -Wextra -Wpedantic -Wshadow -Werror
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

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET scratch-memory)
    install(
        TARGETS scratch-memory
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()
