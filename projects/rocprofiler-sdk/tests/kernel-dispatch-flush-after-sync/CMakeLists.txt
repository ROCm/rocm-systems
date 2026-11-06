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

#
# Regression test for ROCm/rocm-systems#14378: a rocprofiler_flush_buffer() issued right
# after a synchronize must include the kernel-dispatch record of the kernel that just
# completed. The app is its own tool and exits non-zero if any flush lacks the record.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

project(rocprofiler-sdk-tests-kernel-dispatch-flush-after-sync LANGUAGES CXX HIP)

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

find_package(rocprofiler-sdk REQUIRED)
find_package(Threads REQUIRED)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(kernel-dispatch-flush-after-sync)
target_sources(kernel-dispatch-flush-after-sync PRIVATE main.cpp)
target_link_libraries(
    kernel-dispatch-flush-after-sync
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            Threads::Threads)

rocprofiler_add_integration_execute_test(
    kernel-dispatch-flush-after-sync
    TARGET kernel-dispatch-flush-after-sync
    TIMEOUT 60
    LABELS "integration-tests"
    # The record reaches the flush through the inline queue-interposition path. Pin it on
    # so a change to its default cannot silently move the test onto the legacy path, which
    # has its own older version of this race.
    ENVIRONMENT "ROCPROFILER_QUEUE_INTERPOSITION=1")

# Streams on one GPU, each flushing from its own thread: the flushes overlap, so a later
# one must not return before the records an earlier one waits for.
rocprofiler_add_integration_execute_test(
    kernel-dispatch-flush-after-sync-multi-stream
    TARGET kernel-dispatch-flush-after-sync
    ARGS 8
    TIMEOUT 60
    LABELS "integration-tests"
    ENVIRONMENT "ROCPROFILER_QUEUE_INTERPOSITION=1")
