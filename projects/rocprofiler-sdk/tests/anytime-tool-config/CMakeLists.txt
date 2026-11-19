# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#
# Anytime tool configuration integration tests
#
# Each subdirectory is a self-contained execute + validate test pair that exercises
# registering a rocprofiler-sdk tool at a controlled point during a running process via
# rocprofiler_force_configure (the "anytime" initialization path). The top-level tests
# below cover client finalization: concurrent finalization and reattaching after a detach.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

find_package(Threads REQUIRED)

add_executable(concurrent-finalization)
target_sources(concurrent-finalization PRIVATE concurrent-finalization.cpp)
target_link_libraries(
    concurrent-finalization PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                    rocprofiler-sdk::tests-build-flags Threads::Threads)

rocprofiler_add_integration_execute_test(
    concurrent-finalization
    TARGET concurrent-finalization
    TIMEOUT 15
    LABELS "integration-tests;anytime-tool-config"
    ENVIRONMENT "ROCPROFILER_LIBRARY_CTOR=0"
    PASS_REGULAR_EXPRESSION "Concurrent finalization waits for the active client"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_execute_test(
    concurrent-finalization-at-exit
    TARGET concurrent-finalization
    ARGS --exit-while-finalizing
    TIMEOUT 15
    LABELS "integration-tests;anytime-tool-config"
    ENVIRONMENT "ROCPROFILER_LIBRARY_CTOR=0"
    PASS_REGULAR_EXPRESSION "Process exit waited for the active client"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Attach, detach and attach again with tools (tests/tools/reattach-tool.cpp) that finalize
# themselves from tool_detach, as rocprofiler-systems does. The tools are loaded through
# ROCP_TOOL_LIBRARIES, like a real attach tool, and print "reattach-tool[<name>]: FAIL:
# ..." on a lifecycle violation.
add_executable(reattach-after-detach)
target_sources(reattach-after-detach PRIVATE reattach-after-detach.cpp)
target_link_libraries(
    reattach-after-detach
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            ${CMAKE_DL_LIBS} Threads::Threads)
add_dependencies(reattach-after-detach rocprofiler-sdk-reattach-tool
                 rocprofiler-sdk-reattach-tool-2)

set(_REATTACH_TOOL_1 $<TARGET_FILE:rocprofiler-sdk-reattach-tool>)
set(_REATTACH_TOOL_2 $<TARGET_FILE:rocprofiler-sdk-reattach-tool-2>)
# printed by the tools when they are finalized
set(_REATTACH_FINI_1 "reattach-tool\\[rocprofiler-sdk-reattach-tool\\]: tool_fini after")
set(_REATTACH_FINI_2
    "reattach-tool\\[rocprofiler-sdk-reattach-tool-2\\]: tool_fini after")

# reattach_tool_test(<name> <pass regex> <tool library>... [MODE <driver mode>])
function(reattach_tool_test _NAME _PASS_REGEX)
    cmake_parse_arguments(ARG "" "MODE" "" ${ARGN})
    list(JOIN ARG_UNPARSED_ARGUMENTS ":" _TOOL_LIBRARIES)
    rocprofiler_add_integration_execute_test(
        ${_NAME}
        TARGET reattach-after-detach
        ARGS ${ARG_MODE} ${ARG_UNPARSED_ARGUMENTS}
        TIMEOUT 15
        LABELS "integration-tests;anytime-tool-config;attachment"
        ENVIRONMENT "ROCPROFILER_LIBRARY_CTOR=1" "ROCP_TOOL_LIBRARIES=${_TOOL_LIBRARIES}"
        PASS_REGULAR_EXPRESSION "${_PASS_REGEX}"
        FAIL_REGULAR_EXPRESSION
            "${ROCPROFILER_DEFAULT_FAIL_REGEX}|reattach-tool[^:]*: FAIL")
endfunction()

# one client: the finalize request from tool_detach goes through finalize(); the request
# the driver makes afterwards finalizes the tool
reattach_tool_test(
    reattach-after-detach
    "${_REATTACH_FINI_1} 3 session\\(s\\).*Reattach after detach succeeded with 1 client\\(s\\)"
    ${_REATTACH_TOOL_1})

# two clients: the finalize requests go through the per-client finalizer; the second tool
# is finalized at process exit
reattach_tool_test(
    reattach-after-detach-two-clients
    "Reattach after detach succeeded with 2 client\\(s\\).*${_REATTACH_FINI_2} 3 session\\(s\\)"
    ${_REATTACH_TOOL_1}
    ${_REATTACH_TOOL_2})

# the finalize request comes from a helper thread that tool_detach joins
reattach_tool_test(
    reattach-after-detach-helper-thread
    "${_REATTACH_FINI_1} 3 session\\(s\\).*Reattach after detach succeeded with 1 client\\(s\\)"
    ${_REATTACH_TOOL_1}
    MODE
    --finalize-on-helper-thread)

# the process exits while tool_detach runs on another thread: the exit waits for it (the
# tool's static objects must survive the detach), then finalizes the tool
reattach_tool_test(
    reattach-after-detach-exit-during-detach "${_REATTACH_FINI_1} 1 session\\(s\\)"
    ${_REATTACH_TOOL_1} MODE --exit-during-detach)

# the tool calls exit() from tool_detach: rocprofiler-sdk still finalizes it at exit
reattach_tool_test(
    reattach-after-detach-exit-from-detach
    "calling exit\\(\\) from tool_detach.*${_REATTACH_FINI_1} 1 session\\(s\\)"
    ${_REATTACH_TOOL_1} MODE --exit-from-detach)

# a tool calls exit() from its finalize: with one client the re-entrant finalization must
# return (the process exits instead of deadlocking); with two clients the other client
# must still be finalized by the exit
reattach_tool_test(
    reattach-after-detach-exit-from-finalize "calling exit\\(\\) from tool_fini"
    ${_REATTACH_TOOL_1} MODE --exit-from-finalize)
reattach_tool_test(
    reattach-after-detach-exit-from-finalize-two-clients
    "calling exit\\(\\) from tool_fini.*${_REATTACH_FINI_2} 3 session\\(s\\)"
    ${_REATTACH_TOOL_1} ${_REATTACH_TOOL_2} MODE --exit-from-finalize)

# two threads request the same client's finalization at once (two clients, so the
# per-client path is taken): the second request returns only once the finalize callback
# has completed
reattach_tool_test(
    reattach-after-detach-concurrent-finalize
    "Concurrent finalization request waited for the finalize callback.*${_REATTACH_FINI_2} 3 session\\(s\\)"
    ${_REATTACH_TOOL_1}
    ${_REATTACH_TOOL_2}
    MODE
    --concurrent-finalize)

add_subdirectory(early-hip-late-dispatch)
add_subdirectory(early-hip-late-memcpy)
add_subdirectory(early-hip-late-memalloc)
add_subdirectory(early-multi-late-overlap)
add_subdirectory(early-hsa-late-hsa)
add_subdirectory(early-hip-two-late-distinct)
add_subdirectory(early-hip-late-marker)
add_subdirectory(early-callback-late-buffered)
add_subdirectory(early-hip-late-stopstart)
add_subdirectory(early-hip-late-double-configure)

# RCCL variations are registered unconditionally so they remain visible (as DISABLED) in
# CI when RCCL is absent. Each test's CMakeLists sets DISABLED based on rccl availability;
# the python-rccl workload target only exists when RCCL is installed (see
# tests/lib/python/CMakeLists.txt).
add_subdirectory(early-hiprccl-late-hipmemcpy)
add_subdirectory(early-hiprccl-late-hiprcclmemcpy)

# Codec variations (rocDecode / rocJPEG), same unconditional-registration + DISABLED
# pattern as the RCCL and the existing tests/rocdecode + tests/rocjpeg tests. The
# python-rocdecode / python-rocjpeg workload targets only exist when the codec packages
# are installed; the tests are additionally skipped at runtime if the codec VA-API backend
# is unavailable.
add_subdirectory(early-hip-late-rocdecode)
add_subdirectory(early-hip-late-rocjpeg)
