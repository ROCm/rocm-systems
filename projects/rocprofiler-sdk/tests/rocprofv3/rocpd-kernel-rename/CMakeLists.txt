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

cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-rocpd-rename
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(rocprofv3-rocpd-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )

find_package(Python3 REQUIRED)

#########################################################################################
#
# generate rocpd database
#
#########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-no-rename
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-no-kernel-rename -o out --output-format rocpd
        json csv --kernel-trace -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-rename)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-rename
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-kernel-rename -o out --output-format csv
        --kernel-trace --kernel-rename -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-rename)

#########################################################################################
#
# CSV generate
#
#########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-rename-csv-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f csv --kernel-rename -d
        ${CMAKE_CURRENT_BINARY_DIR}/generated-rename-csv/ -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-no-kernel-rename/out_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-rename-csv-generation
    FIXTURES_REQUIRED rocprofv3-test-rocpd-rename)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-no-rename-csv-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f csv -d
        ${CMAKE_CURRENT_BINARY_DIR}/generated-no-rename-csv/ -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-no-kernel-rename/out_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-rename-csv-generation
    FIXTURES_REQUIRED rocprofv3-test-rocpd-rename)

#########################################################################################
#
# Validation
#
#########################################################################################

rocprofiler_add_integration_validate_test(
    rocprofv3-test-rocpd-rename
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --rename-csv-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-kernel-rename/out_kernel_trace.csv
         --no-rename-csv-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-no-kernel-rename/out_kernel_trace.csv
         --generated-rename-csv-input
         ${CMAKE_CURRENT_BINARY_DIR}/generated-rename-csv/out_kernel_trace.csv
         --generated-no-rename-csv-input
         ${CMAKE_CURRENT_BINARY_DIR}/generated-no-rename-csv/out_kernel_trace.csv
    TIMEOUT 120
    LABELS "integration-tests;rocpd" PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FIXTURES_REQUIRED rocprofv3-test-rocpd-rename-csv-generation)
