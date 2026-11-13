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
# Tests for rocprofiler_synchronize_context. One executable per route records take out of
# the runtime -- tracing_window.cpp, counter_window.cpp, signal_less_window.cpp -- plus
# no_late_records.cpp, which enables all of them at once to assert that nothing arrives
# after a window closes. Each tool and its workload share a binary because the assertion
# is about the ordering between them, so the exit status is the whole verdict.
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

#
# Buffer tracing records, plus the flag and argument validation common to the API.
#
add_executable(synchronize-contexts-tracing-window)
target_sources(synchronize-contexts-tracing-window PRIVATE tracing_window.cpp)
set_source_files_properties(tracing_window.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(
    synchronize-contexts-tracing-window
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            Threads::Threads)

# rocprofiler_configure is resolved by dlsym on the executable, so it has to be in the
# dynamic symbol table rather than merely defined.
set_target_properties(synchronize-contexts-tracing-window PROPERTIES ENABLE_EXPORTS ON)

# Queue interposition is pinned rather than left to runtime qualification, so the code
# path under test is the same on every run.
rocprofiler_add_integration_execute_test(
    synchronize-contexts-tracing-window
    COMMAND $<TARGET_FILE:synchronize-contexts-tracing-window>
    DEPENDS synchronize-contexts-tracing-window
    TIMEOUT 120
    LABELS "integration-tests"
    ENVIRONMENT
        "ROCPROFILER_QUEUE_INTERPOSITION=1"
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

#
# Counter records are written on a consumer thread, later than the dispatch record tracing
# produces, so this is a separate route the wait has to reach.
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

#
# Signal-less kernel-dispatch completion: a third route, where a dispatch completes from a
# firmware record instead of an SDK completion signal. The feature is read once per
# process and changes completion semantics process-wide, so it cannot share a binary.
#
add_executable(synchronize-contexts-signal-less-window)
target_sources(synchronize-contexts-signal-less-window PRIVATE signal_less_window.cpp)
set_source_files_properties(signal_less_window.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(
    synchronize-contexts-signal-less-window
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            Threads::Threads)

set_target_properties(synchronize-contexts-signal-less-window PROPERTIES ENABLE_EXPORTS
                                                                         ON)

# Signal-less eligibility is decided on the interposed enqueue path, so interposition is
# pinned too -- without it the feature switch is on and unreachable.
#
# The skip expression reads the SDK's teardown summary: a run that registered no batch
# never reached the path this test exists for, and would otherwise pass by re-proving what
# the tracing-window test proves. The cost is that it also swallows a failing run on such
# a machine -- but a failure there is a failure of the signal path, which that test
# covers.
rocprofiler_add_integration_execute_test(
    synchronize-contexts-signal-less-window
    COMMAND $<TARGET_FILE:synchronize-contexts-signal-less-window>
    DEPENDS synchronize-contexts-signal-less-window
    TIMEOUT 120
    LABELS "integration-tests"
    ENVIRONMENT
        "ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS=true"
        "ROCPROFILER_QUEUE_INTERPOSITION=1"
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    SKIP_REGULAR_EXPRESSION "signal-less summary: registered=0 ")

#
# The complementary assertion: that no record arrives after the synchronize returns. Every
# route is enabled on one context, because a drain reaching two of them and missing the
# third is the defect being looked for. Most of the runtime is deliberate idling.
#
add_executable(synchronize-contexts-no-late-records)
target_sources(synchronize-contexts-no-late-records PRIVATE no_late_records.cpp)
set_source_files_properties(no_late_records.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(
    synchronize-contexts-no-late-records
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            Threads::Threads)

set_target_properties(synchronize-contexts-no-late-records PROPERTIES ENABLE_EXPORTS ON)

# Interposition is left to runtime qualification rather than pinned: the property under
# test is that the drain reaches every route, whichever one the runtime picked.
rocprofiler_add_integration_execute_test(
    synchronize-contexts-no-late-records
    COMMAND $<TARGET_FILE:synchronize-contexts-no-late-records>
    DEPENDS synchronize-contexts-no-late-records
    TIMEOUT 120
    LABELS "integration-tests"
    ENVIRONMENT
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )
