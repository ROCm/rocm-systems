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

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pmc_groups.json
               ${CMAKE_CURRENT_BINARY_DIR}/pmc_groups.json COPYONLY)

# The counter-collection option conflicts below are rejected while parsing arguments, so
# the application never launches. It exits zero to keep WILL_FAIL attributable to
# rocprofv3 rejecting the options rather than to the application itself failing.
set(_NOOP_APPLICATION ${CMAKE_COMMAND} -E true)

# Conflict: --pmc on the command line + pmc_groups from an input file
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-cli-pmc-with-input-pmc-groups-will-fail
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --input
        ${CMAKE_CURRENT_BINARY_DIR}/pmc_groups.json -- ${_NOOP_APPLICATION}
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests;counter-collection"
    WILL_FAIL
    FAIL_REGULAR_EXPRESSION "Cannot specify both --pmc and")

# Conflict: SPM + pmc counter collection
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-spm-with-pmc-will-fail
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --spm-beta-enabled --spm SQ_WAVES --pmc
        SQ_WAVES -- ${_NOOP_APPLICATION}
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests;counter-collection"
    WILL_FAIL
    FAIL_REGULAR_EXPRESSION
        "SPM feature cannot be enabled along with pc sampling or pmc counter collection")

# Conflict: multi-pass counter collection + --collection-period
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multipass-with-collection-period-will-fail
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --pmc GRBM_COUNT
        --collection-period 0:1:1 -- ${_NOOP_APPLICATION}
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests;counter-collection"
    WILL_FAIL
    FAIL_REGULAR_EXPRESSION "is not compatible with --collection-period")

# Conflict: multi-pass counter collection + attach mode
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multipass-with-pid-will-fail
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --pmc GRBM_COUNT --pid
        99999999 -- ${_NOOP_APPLICATION}
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests;counter-collection"
    WILL_FAIL
    FAIL_REGULAR_EXPRESSION "is not compatible with attach mode")

# Invalid: a --pmc flag with no counters
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-empty-pmc-group-will-fail
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --pmc --
            ${_NOOP_APPLICATION}
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests;counter-collection"
    WILL_FAIL
    FAIL_REGULAR_EXPRESSION "Each --pmc must specify at least one counter")
