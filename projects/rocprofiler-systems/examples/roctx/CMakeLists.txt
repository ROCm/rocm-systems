# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(rocprofiler-systems-roctx-example LANGUAGES CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

find_package(hip QUIET HINTS ${ROCmVersion_DIR} PATHS ${ROCmVersion_DIR})
find_package(Threads REQUIRED)

find_program(
    HIPCC_EXECUTABLE
    NAMES hipcc
    HINTS ${ROCmVersion_DIR} ${ROCM_PATH}
    PATHS /opt/rocm
    PATH_SUFFIXES bin
    ENV ROCM_PATH
    NO_CACHE
)
mark_as_advanced(HIPCC_EXECUTABLE)

if(NOT HIPCC_EXECUTABLE)
    message(AUTHOR_WARNING "hipcc could not be found. Cannot build roctx target")
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
    message(AUTHOR_WARNING "roctx target could not be built")
    return()
endif()

# Find the roctx library
find_library(
    ROCTX_LIBRARY
    NAMES rocprofiler-sdk-roctx
    PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64 /opt/rocm/lib
    REQUIRED
)

set(ROCTX_EXAMPLE_TARGETS
    roctx
    pause_resume
    selective_region
    selective_region_pause_1
    selective_region_pause_2
    selective_region_pause_3
)

set(ROCTX_KERNEL_HEADER_TARGETS
    pause_resume
    selective_region
    selective_region_pause_1
    selective_region_pause_2
    selective_region_pause_3
)

foreach(_target IN LISTS ROCTX_EXAMPLE_TARGETS)
    set(_sources ${_target}.cpp)
    if(${_target} IN_LIST ROCTX_KERNEL_HEADER_TARGETS)
        list(APPEND _sources roctx_example_kernels.hpp)
    endif()
    add_executable(${_target} ${_sources})
    target_link_libraries(${_target} PRIVATE Threads::Threads)

    if(
        CMAKE_CXX_COMPILER_ID MATCHES "Clang"
        AND NOT CMAKE_CXX_COMPILER_IS_HIPCC
        AND NOT HIPCC_EXECUTABLE
    )
        target_link_libraries(
            ${_target}
            PRIVATE
                $<TARGET_NAME_IF_EXISTS:rocprofiler-systems::rocprofiler-systems-compile-options>
                $<TARGET_NAME_IF_EXISTS:hip::host>
                $<TARGET_NAME_IF_EXISTS:hip::device>
        )
    else()
        target_compile_options(${_target} PRIVATE -W -Wall)
    endif()

    target_link_libraries(${_target} PRIVATE ${ROCTX_LIBRARY})

    if("${CMAKE_BUILD_TYPE}" MATCHES "Release")
        target_compile_options(${_target} PRIVATE -g1)
    endif()

    foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
        target_compile_options(${_target} PRIVATE --offload-arch=${arch})
        target_link_options(${_target} PUBLIC --offload-arch=${arch})
    endforeach()

    if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
        rocprofiler_systems_custom_compilation(COMPILER ${HIPCC_EXECUTABLE} TARGET ${_target})
    endif()

    if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET ${_target})
        install(
            TARGETS ${_target}
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
            COMPONENT rocprofiler-systems-examples
        )
    endif()
endforeach()
