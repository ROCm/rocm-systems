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

cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-pc-sampling-selected-regions-stochastic
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_sdk_pc_sampling_stochastic_disabled(IS_PC_SAMPLING_STOCHASTIC_DISABLED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# stochastic, selected-regions
rocprofiler_add_integration_execute_test(
    rocprofv3-test-pc-sampling-selected-regions-stochastic
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pc-sampling-method stochastic
        --pc-sampling-interval 1048576 --pc-sampling-unit cycles --selected-regions
        --kernel-trace -d ${CMAKE_CURRENT_BINARY_DIR}/selected-regions -o out
        --output-format json csv -- $<TARGET_FILE:roctx-pause-resume>
    DEPENDS roctx-pause-resume
    TIMEOUT 90
    LABELS "integration-tests;pc-sampling;pause-resume;selected-regions;stochastic"
    ENVIRONMENT "${PRELOAD_ENV}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-pc-sampling-selected-regions-stochastic
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-pc-sampling-selected-regions-stochastic
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --input-csv
         ${CMAKE_CURRENT_BINARY_DIR}/selected-regions/out_pc_sampling_stochastic.csv
         --input-json ${CMAKE_CURRENT_BINARY_DIR}/selected-regions/out_results.json
    TIMEOUT 60
    LABELS "integration-tests;pc-sampling;pause-resume;selected-regions;stochastic"
    FIXTURES_REQUIRED rocprofv3-test-pc-sampling-selected-regions-stochastic
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

# stochastic, selected-regions with reference counting
rocprofiler_add_integration_execute_test(
    rocprofv3-test-pc-sampling-selected-regions-ref-count-stochastic
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pc-sampling-method stochastic
        --pc-sampling-interval 1048576 --pc-sampling-unit cycles --selected-regions
        --selected-regions-ref-count --kernel-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/selected-regions-ref-count -o out --output-format json
        csv -- $<TARGET_FILE:roctx-pause-resume>
    DEPENDS roctx-pause-resume
    TIMEOUT 90
    LABELS "integration-tests;pc-sampling;pause-resume;selected-regions;stochastic"
    ENVIRONMENT "${PRELOAD_ENV}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-pc-sampling-selected-regions-ref-count-stochastic
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-pc-sampling-selected-regions-ref-count-stochastic
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --input-csv
         ${CMAKE_CURRENT_BINARY_DIR}/selected-regions-ref-count/out_pc_sampling_stochastic.csv
         --input-json
         ${CMAKE_CURRENT_BINARY_DIR}/selected-regions-ref-count/out_results.json
    TIMEOUT 60
    LABELS "integration-tests;pc-sampling;pause-resume;selected-regions;stochastic"
    FIXTURES_REQUIRED rocprofv3-test-pc-sampling-selected-regions-ref-count-stochastic
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")
