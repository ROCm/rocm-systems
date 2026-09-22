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
# Early (preloaded, HIP-only) tool + late (memory-copy) tool.
#
# Same anytime-initialization shape as early-hip-late-dispatch, but the late tool enables
# the MEMORY_COPY service. Proves a late tool captures exactly its own service subset
# (memory copies) of a shared workload, disjoint from the early tool's HIP-API subset.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-anytime-early-hip-late-memcpy
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

set(early-hip-late-memcpy-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$<TARGET_FILE_DIR:rocprofiler-sdk-json-tool-late>:$ENV{LD_LIBRARY_PATH}"
    "ROCPROFILER_SDK_TEST_LIB_DIR=$<TARGET_FILE_DIR:rocprofiler-sdk-json-tool-late>"
    "ROCPROFILER_TOOL_CONTEXTS=HIP_API_BUFFERED"
    "ROCPROFILER_TOOL_OUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/early-hip-late-memcpy-tool-a.json"
    "ROCPROFILER_TOOL_LATE_OUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/early-hip-late-memcpy-tool-b.json"
    "ROCPROFILER_TOOL_DISABLE_PERFETTO=1")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY conftest.py)

rocprofiler_add_integration_execute_test(
    anytime-early-hip-late-memcpy
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_early_hip_late_memcpy.py
    DEPENDS rocprofiler-sdk-json-tool rocprofiler-sdk-json-tool-late python-hip-kernels
    TIMEOUT 120
    LABELS "integration-tests;anytime-tool-config"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    ENVIRONMENT "${early-hip-late-memcpy-env}"
    SKIP_REGULAR_EXPRESSION "test skipped"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP anytime-early-hip-late-memcpy)

rocprofiler_add_integration_validate_test(
    anytime-early-hip-late-memcpy
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests;anytime-tool-config"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED anytime-early-hip-late-memcpy
    ARGS --tool-a-input ${CMAKE_CURRENT_BINARY_DIR}/early-hip-late-memcpy-tool-a.json
         --tool-b-input ${CMAKE_CURRENT_BINARY_DIR}/early-hip-late-memcpy-tool-b.json
         --phase1-copies 24 --phase2-copies 40)
