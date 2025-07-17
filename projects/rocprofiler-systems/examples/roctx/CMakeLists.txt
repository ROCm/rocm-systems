# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

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

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
    # defined in MacroUtilities.cmake
    rocprofiler_systems_custom_compilation(COMPILER ${HIPCC_EXECUTABLE} TARGET roctx)
endif()

if(ROCPROFSYS_INSTALL_EXAMPLES)
    install(TARGETS roctx DESTINATION bin COMPONENT rocprofiler-systems-examples)
endif()
