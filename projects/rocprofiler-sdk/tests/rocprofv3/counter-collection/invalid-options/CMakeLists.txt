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
# rocprofv3 counter collection option validation tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-invalid-options
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(_NOOP_APPLICATION ${CMAKE_COMMAND} -E true)
# CTest's PASS_REGULAR_EXPRESSION ignores exit status, so check both in a wrapper.
set(_CHECK_COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/check_command.py)

# Conflict: SPM + pmc counter collection
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-spm-with-pmc-will-fail
    COMMAND
        ${_CHECK_COMMAND} --expected-exit-code 1 --expected-output
        "SPM feature cannot be enabled along with pc sampling or pmc counter collection"
        -- $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --spm-beta-enabled --spm SQ_WAVES
        --pmc SQ_WAVES -- ${_NOOP_APPLICATION}
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "counter-collection")

# Invalid: a --pmc flag with no counters
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-empty-pmc-group-will-fail
    COMMAND
        ${_CHECK_COMMAND} --expected-exit-code 1 --expected-output
        "Each --pmc must specify at least one counter" --
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --pmc --
        ${_NOOP_APPLICATION}
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "counter-collection")

# Tolerated: a device qualifier naming a GPU index that does not exist
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-out-of-range-device-qualifier
    COMMAND
        ${_CHECK_COMMAND} --expected-exit-code 0 --expected-output
        "Device ID not found for PMC counter 'SQ_CYCLES:device=16384'" --
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES SQ_CYCLES:device=16384 --
        ${_NOOP_APPLICATION}
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "counter-collection"
    ENVIRONMENT "ROCPROF_LOG_LEVEL=warning")
