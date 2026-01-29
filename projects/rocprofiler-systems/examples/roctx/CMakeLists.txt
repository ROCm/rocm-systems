# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(rocprofiler-systems-roctx-example LANGUAGES CXX)

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

add_executable(roctx roctx.cpp)
target_link_libraries(roctx PRIVATE Threads::Threads)

if(
    CMAKE_CXX_COMPILER_ID MATCHES "Clang"
    AND NOT CMAKE_CXX_COMPILER_IS_HIPCC
    AND NOT HIPCC_EXECUTABLE
)
    target_link_libraries(
        roctx
        PRIVATE
            $<TARGET_NAME_IF_EXISTS:rocprofiler-systems::rocprofiler-systems-compile-options>
            $<TARGET_NAME_IF_EXISTS:hip::host>
            $<TARGET_NAME_IF_EXISTS:hip::device>
    )
else()
    target_compile_options(roctx PRIVATE -W -Wall)
endif()

# Find the library
find_library(
    ROCTX_LIBRARY
    NAMES rocprofiler-sdk-roctx
    PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64
    REQUIRED
)
# Link it to target
target_link_libraries(roctx PRIVATE ${ROCTX_LIBRARY})

if("${CMAKE_BUILD_TYPE}" MATCHES "Release")
    target_compile_options(roctx PRIVATE -g1)
endif()

# Specify GPU architectures to compile for
foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
    target_compile_options(roctx PRIVATE --offload-arch=${arch})
    target_link_options(roctx PUBLIC --offload-arch=${arch})
endforeach()

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
    # defined in MacroUtilities.cmake
    rocprofiler_systems_custom_compilation(COMPILER ${HIPCC_EXECUTABLE} TARGET roctx)
endif()

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET roctx)
    install(
        TARGETS roctx
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()
