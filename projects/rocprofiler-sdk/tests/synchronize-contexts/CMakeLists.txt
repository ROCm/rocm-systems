#
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
# Tests for rocprofiler_synchronize_context. Each test is a tool preloaded into an
# application: the tool configures the services, drives the windows and holds every
# assertion; the application supplies the GPU work. They meet at sync_test_workload.h, so
# no tool here includes HIP and no application here includes rocprofiler.
#
# Five tools, one per record route or asserted property. Two applications, differing only
# in the GPU work they offer -- one lane on the default stream, or many lanes each with
# its own stream -- so the four single-lane tools share one binary.
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
# Applications. HIP only: they are profiled, they do not profile. Each resolves its tool
# out of the process with dlsym, hence the link against the dl library.
#
function(synchronize_contexts_add_app NAME SOURCE)
    set(_TARGET synchronize-contexts-${NAME}-testapp)

    set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE HIP)
    add_executable(${_TARGET})
    target_sources(${_TARGET} PRIVATE ${SOURCE})
    target_link_libraries(${_TARGET} PRIVATE rocprofiler-sdk::tests-build-flags
                                             Threads::Threads ${CMAKE_DL_LIBS})
endfunction()

# One lane on the default stream, with a kernel and a host-to-device copy. The workload
# behind every tool that does not need concurrency.
synchronize_contexts_add_app(serial apps/serial_app.cpp)

# Many lanes, each with its own stream and scratch, so dispatches from different threads
# are genuinely in flight at once.
synchronize_contexts_add_app(concurrent apps/concurrent_app.cpp)

#
# Tools. Preloaded into an application, so rocprofiler_configure is found in the library
# rather than needing the executable's dynamic symbol table exported.
#
function(synchronize_contexts_add_client NAME SOURCE)
    set(_TARGET synchronize-contexts-${NAME}-client)

    add_library(${_TARGET} SHARED)
    target_sources(${_TARGET} PRIVATE ${SOURCE})
    target_link_libraries(
        ${_TARGET} PRIVATE rocprofiler-sdk::rocprofiler-sdk
                           rocprofiler-sdk::tests-build-flags Threads::Threads)
endfunction()

synchronize_contexts_add_client(tracing-window tracing_window_client.cpp)
synchronize_contexts_add_client(counter-window counter_window_client.cpp)
synchronize_contexts_add_client(signal-less-window signal_less_window_client.cpp)
synchronize_contexts_add_client(no-late-records no_late_records_client.cpp)
synchronize_contexts_add_client(concurrent-windows concurrent_windows_client.cpp)

# Every tool prints its verdict under the same tag, so one expression covers all six
# tests. No pass expression: a matched one makes the exit status advisory, and these tests
# assert on teardown, which runs after the verdict is printed.
set(SYNC_CTX_FAIL_REGEX
    "\\[synchronize-contexts\\] FAILED|${ROCPROFILER_DEFAULT_FAIL_REGEX}")

#
# Buffer tracing records, plus the flag and argument validation common to the API.
# Interposition is pinned so the path under test is the same on every run.
#
rocprofiler_add_integration_execute_test(
    synchronize-contexts-tracing-window
    COMMAND $<TARGET_FILE:synchronize-contexts-serial-testapp>
    DEPENDS synchronize-contexts-tracing-window-client
    TIMEOUT 120
    LABELS "integration-tests;synchronize-contexts"
    PRELOAD "$<TARGET_FILE:synchronize-contexts-tracing-window-client>"
    ENVIRONMENT "ROCPROFILER_QUEUE_INTERPOSITION=1"
    FAIL_REGULAR_EXPRESSION "${SYNC_CTX_FAIL_REGEX}")

#
# Counter records are written on a consumer thread, later than the dispatch record tracing
# produces: a separate route the wait has to reach.
#
rocprofiler_add_integration_execute_test(
    synchronize-contexts-counter-window
    COMMAND $<TARGET_FILE:synchronize-contexts-serial-testapp>
    DEPENDS synchronize-contexts-counter-window-client
    TIMEOUT 300
    LABELS "integration-tests;synchronize-contexts"
    PRELOAD "$<TARGET_FILE:synchronize-contexts-counter-window-client>"
    FAIL_REGULAR_EXPRESSION "${SYNC_CTX_FAIL_REGEX}")

#
# Signal-less kernel-dispatch completion: a third route, where a dispatch completes from a
# firmware record instead of an SDK completion signal. Eligibility is decided on the
# interposed enqueue path, so interposition is pinned too.
#
# The skip expression reads the SDK's teardown summary. A run that registered no batch
# never reached this route and would otherwise pass by re-proving what tracing-window
# proves; the cost is that it also swallows a failing run on such a machine.
#
rocprofiler_add_integration_execute_test(
    synchronize-contexts-signal-less-window
    COMMAND $<TARGET_FILE:synchronize-contexts-serial-testapp>
    DEPENDS synchronize-contexts-signal-less-window-client
    TIMEOUT 120
    LABELS "integration-tests;synchronize-contexts"
    PRELOAD "$<TARGET_FILE:synchronize-contexts-signal-less-window-client>"
    ENVIRONMENT "ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS=true"
                "ROCPROFILER_QUEUE_INTERPOSITION=1"
    FAIL_REGULAR_EXPRESSION "${SYNC_CTX_FAIL_REGEX}"
    SKIP_REGULAR_EXPRESSION "signal-less summary: registered=0 ")

#
# The complementary assertion: that no record arrives after the synchronize returns. All
# three routes are enabled on one context, since a drain reaching two and missing the
# third is the defect being looked for. Most of the runtime is deliberate idling.
#
# Interposition is left to runtime qualification: the drain must reach every route,
# whichever one the runtime picked.
#
rocprofiler_add_integration_execute_test(
    synchronize-contexts-no-late-records
    COMMAND $<TARGET_FILE:synchronize-contexts-serial-testapp>
    DEPENDS synchronize-contexts-no-late-records-client
    TIMEOUT 120
    LABELS "integration-tests;synchronize-contexts"
    PRELOAD "$<TARGET_FILE:synchronize-contexts-no-late-records-client>"
    FAIL_REGULAR_EXPRESSION "${SYNC_CTX_FAIL_REGEX}")

#
# The windows above, but with dispatches in flight on other threads -- the only coverage
# of the drain's cross-thread behaviour. Interposition is pinned so the enqueue path the
# synchronize contends with is the same on every run.
#
rocprofiler_add_integration_execute_test(
    synchronize-contexts-concurrent-windows
    COMMAND $<TARGET_FILE:synchronize-contexts-concurrent-testapp>
    DEPENDS synchronize-contexts-concurrent-windows-client
    TIMEOUT 120
    LABELS "integration-tests;synchronize-contexts"
    PRELOAD "$<TARGET_FILE:synchronize-contexts-concurrent-windows-client>"
    ENVIRONMENT "ROCPROFILER_QUEUE_INTERPOSITION=1"
    FAIL_REGULAR_EXPRESSION "${SYNC_CTX_FAIL_REGEX}")

#
# The same pair with signal-less completion requested: the enqueue path differs between
# the two routes. Skipped when no batch registered, since the run never reached the route.
#
rocprofiler_add_integration_execute_test(
    synchronize-contexts-concurrent-windows-signal-less
    COMMAND $<TARGET_FILE:synchronize-contexts-concurrent-testapp>
    DEPENDS synchronize-contexts-concurrent-windows-client
    TIMEOUT 120
    LABELS "integration-tests;synchronize-contexts"
    PRELOAD "$<TARGET_FILE:synchronize-contexts-concurrent-windows-client>"
    ENVIRONMENT "ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS=true"
                "ROCPROFILER_QUEUE_INTERPOSITION=1"
    FAIL_REGULAR_EXPRESSION "${SYNC_CTX_FAIL_REGEX}"
    SKIP_REGULAR_EXPRESSION "signal-less summary: registered=0 ")
