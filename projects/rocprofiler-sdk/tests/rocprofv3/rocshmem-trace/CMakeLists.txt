# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
    rocprofiler-sdk-tests-rocprofv3-rocshmem-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocSHMEM)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(rocshmem-tracing-env "${PRELOAD_ENV}")

# rocSHMEM real applications require MPI + multiple GPUs to be useful, so this
# integration test is only enabled when a rocshmem-demo / rocshmem-example target
# is available in the build (mirrors how rocjpeg-trace gates on rocjpeg-demo).
if(NOT TARGET rocshmem-demo)
    set(IS_DISABLED ON)
endif()

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocshmem-tracing
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --rocshmem-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format json csv
        --log-level env --
        $<IF:$<TARGET_EXISTS:rocshmem-demo>,$<TARGET_FILE:rocshmem-demo>,rocshmem-demo>
    DEPENDS rocshmem-demo
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-rocshmem-tracing
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-rocshmem-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocshmem-demo-trace/out_results.json
         --otf2-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocshmem-demo-trace/out_results.otf2
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocshmem-demo-trace/out_results.pftrace
         --csv-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocshmem-demo-trace/out_rocshmem_api_trace.csv
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-rocshmem-tracing
    DISABLED "${IS_DISABLED}")
