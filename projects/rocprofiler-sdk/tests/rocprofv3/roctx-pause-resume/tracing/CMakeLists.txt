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
    rocprofiler-sdk-tests-rocprofv3-roctx-pause-resume
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(roctx-pause-resume-env "${PRELOAD_ENV}")

# Test matrix: (num_blocks, iterations, timeout)
# Workload is kept roughly constant by reducing iterations as grid size grows.
set(ROCTX_TRACING_TEST_MATRIX
    "256;8192;60"
    "512;8192;60"
    "1024;4096;60"
    "2048;2048;90"
    "4096;1024;120")

foreach(_CONFIG ${ROCTX_TRACING_TEST_MATRIX})
    list(GET _CONFIG 0 _NUM_BLOCKS)
    list(GET _CONFIG 1 _ITERATIONS)
    list(GET _CONFIG 2 _TIMEOUT)

    rocprofiler_add_integration_execute_test(
        rocprofv3-test-roctx-pause-resume-tracing-blocks-${_NUM_BLOCKS}
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --marker-trace --kernel-trace --pmc
            SQ_WAVES -d ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace-${_NUM_BLOCKS} -o out
            --output-format json --log-level info --
            $<TARGET_FILE:roctx-pause-resume> --num-blocks ${_NUM_BLOCKS}
            --iterations ${_ITERATIONS}
        DEPENDS roctx-pause-resume
        TIMEOUT ${_TIMEOUT}
        LABELS "integration-tests;pause-resume"
        ENVIRONMENT "${roctx-pause-resume-env}"
        FIXTURES_SETUP rocprofv3-test-roctx-pause-resume-tracing-blocks-${_NUM_BLOCKS})

    rocprofiler_add_integration_validate_test(
        rocprofv3-test-roctx-pause-resume-tracing-blocks-${_NUM_BLOCKS}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        ARGS --json-input
             ${CMAKE_CURRENT_BINARY_DIR}/roctx-pause-resume-trace-${_NUM_BLOCKS}/out_results.json
        TIMEOUT 45
        LABELS "integration-tests;pause-resume"
        FIXTURES_REQUIRED rocprofv3-test-roctx-pause-resume-tracing-blocks-${_NUM_BLOCKS}
        FAIL_REGULAR_EXPRESSION "AssertionError")
endforeach()
