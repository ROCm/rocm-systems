# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-roctx-pause-resume-nested
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(roctx-pause-resume-nested-env "${PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-roctx-pause-resume-nested-tracing
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --marker-trace --kernel-trace
        --selected-regions-ref-count --selected-regions -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-nested-trace -o out --output-format json
        --log-level info -- $<TARGET_FILE:roctx-pause-resume>
    DEPENDS roctx-pause-resume
    TIMEOUT 45
    LABELS "integration-tests;pause-resume"
    ENVIRONMENT "${roctx-pause-resume-nested-env}"
    FIXTURES_SETUP rocprofv3-test-roctx-pause-resume-nested-tracing)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-roctx-pause-resume-nested-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/roctx-pause-resume-nested-trace/out_results.json
    TIMEOUT 45
    LABELS "integration-tests;pause-resume"
    FIXTURES_REQUIRED rocprofv3-test-roctx-pause-resume-nested-tracing
    FAIL_REGULAR_EXPRESSION "AssertionError")

# Test --selected-regions with --kernel-trace WITHOUT --marker-trace This verifies that
# marker control callbacks are registered even without --marker-trace
rocprofiler_add_integration_execute_test(
    rocprofv3-test-selected-regions-kernel-only
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --selected-regions --kernel-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-selected-kernel -o out --output-format json
        --log-level info -- $<TARGET_FILE:roctx-pause-resume>
    DEPENDS roctx-pause-resume
    TIMEOUT 45
    LABELS "integration-tests;pause-resume;selected-regions"
    ENVIRONMENT "${roctx-pause-resume-nested-env}"
    FIXTURES_SETUP rocprofv3-test-selected-regions-kernel-only)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-selected-regions-kernel-only
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/roctx-pause-resume-nested-trace/out_results.json
    TIMEOUT 45
    LABELS "integration-tests;pause-resume;selected-regions"
    FIXTURES_REQUIRED rocprofv3-test-selected-regions-kernel-only
    FAIL_REGULAR_EXPRESSION "AssertionError")
