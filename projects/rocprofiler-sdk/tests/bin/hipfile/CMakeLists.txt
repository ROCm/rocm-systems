# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-bin-hipfile
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(hipfile QUIET CONFIG)
find_package(TBB QUIET)

set(ROCPROFILER_SDK_HIPFILE_TRACE_CLIENT_TARGET
    hipfile-trace-client
    PARENT_SCOPE)

set(HIPFILE_TRACE_TARGET "")
set(HIPFILE_TRACE_TBB_TARGET "")
set(HIPFILE_TRACE_DISABLED OFF)
unset(HIPFILE_TRACE_INCLUDE_DIR)

if(TARGET hip::hipfile)
    set(HIPFILE_TRACE_TARGET hip::hipfile)
elseif(TARGET hipfile)
    set(HIPFILE_TRACE_TARGET hipfile)
endif()

set(HIPFILE_TRACE_INCLUDE_HINTS ${ROCM_PATH})
if(HIPFILE_TRACE_TARGET)
    get_target_property(HIPFILE_TRACE_TARGET_INCLUDES ${HIPFILE_TRACE_TARGET}
                        INTERFACE_INCLUDE_DIRECTORIES)
    if(HIPFILE_TRACE_TARGET_INCLUDES)
        list(APPEND HIPFILE_TRACE_INCLUDE_HINTS ${HIPFILE_TRACE_TARGET_INCLUDES})
    endif()
endif()

find_path(
    HIPFILE_TRACE_INCLUDE_DIR
    NAMES hipfile-api-trace.h hipfile.h
    HINTS ${HIPFILE_TRACE_INCLUDE_HINTS}
    PATHS ${ROCM_PATH}
    PATH_SUFFIXES include)

if(NOT HIPFILE_TRACE_TARGET)
    find_library(
        HIPFILE_TRACE_LIBRARY
        NAMES hipfile
        HINTS ${ROCM_PATH}
        PATHS ${ROCM_PATH}
        PATH_SUFFIXES lib lib64)
    if(HIPFILE_TRACE_LIBRARY AND HIPFILE_TRACE_INCLUDE_DIR)
        add_library(rocprofiler-test-hipfile-lib UNKNOWN IMPORTED)
        set_target_properties(
            rocprofiler-test-hipfile-lib
            PROPERTIES IMPORTED_LOCATION "${HIPFILE_TRACE_LIBRARY}"
                       INTERFACE_INCLUDE_DIRECTORIES "${HIPFILE_TRACE_INCLUDE_DIR}")
        set(HIPFILE_TRACE_TARGET rocprofiler-test-hipfile-lib)
    endif()
endif()

if(TARGET TBB::tbb)
    set(HIPFILE_TRACE_TBB_TARGET TBB::tbb)
else()
    find_library(
        HIPFILE_TRACE_TBB_LIBRARY
        NAMES tbb
        HINTS ${ROCM_PATH}
        PATHS ${ROCM_PATH}
        PATH_SUFFIXES lib lib64)
    if(HIPFILE_TRACE_TBB_LIBRARY)
        add_library(rocprofiler-test-hipfile-tbb UNKNOWN IMPORTED)
        set_target_properties(rocprofiler-test-hipfile-tbb
                              PROPERTIES IMPORTED_LOCATION "${HIPFILE_TRACE_TBB_LIBRARY}")
        set(HIPFILE_TRACE_TBB_TARGET rocprofiler-test-hipfile-tbb)
    endif()
endif()

if(NOT HIPFILE_TRACE_TARGET OR NOT HIPFILE_TRACE_INCLUDE_DIR)
    set(HIPFILE_TRACE_DISABLED ON)
endif()

if(NOT EXISTS "${HIPFILE_TRACE_INCLUDE_DIR}/hipfile-api-trace.h")
    set(HIPFILE_TRACE_DISABLED ON)
endif()

if(NOT HIPFILE_TRACE_TBB_TARGET)
    set(HIPFILE_TRACE_DISABLED ON)
endif()

set(ROCPROFILER_SDK_HIPFILE_TRACE_CLIENT_DISABLED
    ${HIPFILE_TRACE_DISABLED}
    PARENT_SCOPE)

if(HIPFILE_TRACE_DISABLED)
    return()
endif()

add_executable(hipfile-trace-client hipfile_trace.cpp)
target_link_libraries(hipfile-trace-client PRIVATE ${HIPFILE_TRACE_TARGET}
                                                   ${HIPFILE_TRACE_TBB_TARGET})
target_compile_definitions(hipfile-trace-client PRIVATE __HIP_PLATFORM_AMD__)
target_compile_features(hipfile-trace-client PRIVATE cxx_std_17)
set_target_properties(hipfile-trace-client PROPERTIES CXX_EXTENSIONS OFF)
