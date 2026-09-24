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
# Early HIP tool + TWO late tools with distinct service subsets.
#
# Exercises multiple rocprofiler_force_configure() calls in one process. Each json-tool
# .so is single-instance (state in globals), so the two late tools use two distinct
# libraries (json-tool-late and json-tool-late2). Proves each late context captures only
# its own service subset of the shared workload.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-anytime-early-hip-two-late-distinct
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

set(early-hip-two-late-distinct-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$<TARGET_FILE_DIR:rocprofiler-sdk-json-tool-late>:$ENV{LD_LIBRARY_PATH}"
    "ROCPROFILER_SDK_TEST_LIB_DIR=$<TARGET_FILE_DIR:rocprofiler-sdk-json-tool-late>"
    "ROCPROFILER_TOOL_CONTEXTS=HIP_API_BUFFERED"
    "ROCPROFILER_TOOL_OUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/early-hip-two-late-distinct-tool-a.json"
    "ROCPROFILER_TOOL_LATE_OUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/early-hip-two-late-distinct-tool-b1.json"
    "ROCPROFILER_TOOL_LATE2_OUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/early-hip-two-late-distinct-tool-b2.json"
    "ROCPROFILER_TOOL_DISABLE_PERFETTO=1")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY conftest.py)

rocprofiler_add_integration_execute_test(
    anytime-early-hip-two-late-distinct
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_early_hip_two_late_distinct.py
    DEPENDS rocprofiler-sdk-json-tool rocprofiler-sdk-json-tool-late
            rocprofiler-sdk-json-tool-late2 python-hip-kernels
    TIMEOUT 120
    LABELS "integration-tests;anytime-tool-config"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    ENVIRONMENT "${early-hip-two-late-distinct-env}"
    SKIP_REGULAR_EXPRESSION "test skipped"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP anytime-early-hip-two-late-distinct)

rocprofiler_add_integration_validate_test(
    anytime-early-hip-two-late-distinct
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests;anytime-tool-config"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED anytime-early-hip-two-late-distinct
    ARGS --tool-a-input
         ${CMAKE_CURRENT_BINARY_DIR}/early-hip-two-late-distinct-tool-a.json
         --tool-b1-input
         ${CMAKE_CURRENT_BINARY_DIR}/early-hip-two-late-distinct-tool-b1.json
         --tool-b2-input
         ${CMAKE_CURRENT_BINARY_DIR}/early-hip-two-late-distinct-tool-b2.json
         --phase2-dispatches
         20
         --phase2-copies
         40)
