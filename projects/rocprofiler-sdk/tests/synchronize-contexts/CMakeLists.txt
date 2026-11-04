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
# A tool that synchronizes on a GPU operation and immediately reads its records. The tool
# and the workload live in one binary because the assertion is about the ordering between
# them, so the executable's own exit status is the verdict and no separate validation step
# is needed.
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

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

project(
    rocprofiler-sdk-tests-synchronize-contexts
    LANGUAGES CXX HIP
    VERSION 0.0.0)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)
find_package(Threads REQUIRED)

add_executable(synchronize-contexts)
target_sources(synchronize-contexts PRIVATE main.cpp)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(
    synchronize-contexts PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                 rocprofiler-sdk::tests-build-flags Threads::Threads)

# rocprofiler_configure is resolved by dlsym on the executable, so it has to be in the
# dynamic symbol table rather than merely defined.
set_target_properties(synchronize-contexts PROPERTIES ENABLE_EXPORTS ON)

# Queue interposition is pinned rather than left to runtime qualification, so the code
# path under test is the same on every run.
rocprofiler_add_integration_execute_test(
    synchronize-contexts
    COMMAND $<TARGET_FILE:synchronize-contexts>
    DEPENDS synchronize-contexts
    TIMEOUT 120
    LABELS "integration-tests"
    ENVIRONMENT
        "ROCPROFILER_QUEUE_INTERPOSITION=1"
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

#
# Counter records for a dispatch are written on a consumer thread, so a window that
# synchronizes and flushes can be short. Asserted per window, never in aggregate.
#
add_executable(synchronize-contexts-counter-window)
target_sources(synchronize-contexts-counter-window PRIVATE counter_window.cpp)
set_source_files_properties(counter_window.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(
    synchronize-contexts-counter-window
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            Threads::Threads)

set_target_properties(synchronize-contexts-counter-window PROPERTIES ENABLE_EXPORTS ON)

rocprofiler_add_integration_execute_test(
    synchronize-contexts-counter-window
    COMMAND $<TARGET_FILE:synchronize-contexts-counter-window>
    DEPENDS synchronize-contexts-counter-window
    TIMEOUT 300
    LABELS "integration-tests"
    ENVIRONMENT
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )
