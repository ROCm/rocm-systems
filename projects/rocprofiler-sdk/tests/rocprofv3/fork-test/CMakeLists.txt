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
# rocprofv3 fork test - validates multi-process profiling with -d and -o flags
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-fork-test
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# Test with -o and -d flags to ensure multiple PIDs create separate database files
rocprofiler_add_integration_execute_test(
    rocprofv3-test-fork
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -r -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-output -o fork_out --output-format rocpd csv --
        $<TARGET_FILE:fork-test>
    DEPENDS fork-test
    TIMEOUT 45
    LABELS "integration-tests;fork-test"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-fork
    DISABLED $<NOT:$<TARGET_EXISTS:fork-test>>)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-fork
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests;fork-test"
    FIXTURES_REQUIRED rocprofv3-test-fork
    ARGS --output-dir ${CMAKE_CURRENT_BINARY_DIR}/fork-test-output --kernel-trace-pattern
         ${CMAKE_CURRENT_BINARY_DIR}/fork-test-output/fork_out*_kernel_trace.csv)
