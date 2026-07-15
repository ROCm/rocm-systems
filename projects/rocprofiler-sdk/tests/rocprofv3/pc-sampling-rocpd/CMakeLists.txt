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
    rocprofiler-sdk-tests-rocprofv3-pc-sampling-rocpd
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

# This suite exercises the stochastic PC-sampling ROCPD path (the supported method).
rocprofiler_sdk_pc_sampling_stochastic_disabled(IS_PC_SAMPLING_STOCHASTIC_DISABLED)

set(rocprofv3-pc-sampling-rocpd-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    "OMPI_ALLOW_RUN_AS_ROOT=1"
    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# Produce a stochastic PC-sampling ROCPD database (+ JSON oracle).  No
# --pc-sampling-decode-instructions here, so instruction text is disassembled on demand
# during post-processing (the first disassembly path).
rocprofiler_add_integration_execute_test(
    rocprofv3-test-pc-sampling-rocpd
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pc-sampling-beta-enabled 1
        --kernel-trace --pc-sampling-unit cycles --pc-sampling-method stochastic
        --pc-sampling-interval 1048576 -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc-sampling-rocpd-input -o out --output-format rocpd
        json -- $<TARGET_FILE:exec-mask-manipulation>
    DEPENDS exec-mask-manipulation
    TIMEOUT 120
    LABELS "integration-tests;pc-sampling;stochastic;rocpd"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${rocprofv3-pc-sampling-rocpd-env}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-pc-sampling-rocpd
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

# Use rocpd convert to generate CSV from the rocpd DB.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-pc-sampling-rocpd-csv-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f csv -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc-sampling-rocpd-csv -o out -i
        ${CMAKE_CURRENT_BINARY_DIR}/pc-sampling-rocpd-input/out_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;pc-sampling;rocpd"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${rocprofv3-pc-sampling-rocpd-env}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-pc-sampling-rocpd-csv-generation
    FIXTURES_REQUIRED rocprofv3-test-pc-sampling-rocpd
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-pc-sampling-rocpd
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini LIBRARY_PATH "${rocprofiler-sdk_LIB_DIR}"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/pc-sampling-rocpd-input/out_results.json
         --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/pc-sampling-rocpd-input/out_results.db
         --rocpd2csv-kernel-input
         ${CMAKE_CURRENT_BINARY_DIR}/pc-sampling-rocpd-csv/out_kernel_trace.csv
         --rocpd2csv-agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/pc-sampling-rocpd-csv/out_agent_info.csv
         --rocpd2csv-pc-sampling-input
         ${CMAKE_CURRENT_BINARY_DIR}/pc-sampling-rocpd-csv/out_pc_sampling_stochastic.csv
    TIMEOUT 120
    LABELS "integration-tests;pc-sampling;stochastic;rocpd"
    PRELOAD "${PRELOAD_ENV}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_REQUIRED rocprofv3-test-pc-sampling-rocpd-csv-generation
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

# Opt-in disassembly coverage: --pc-sampling-decode-instructions persists the disassembled
# instruction text at finalization into rocpd_disassembly_data (the second disassembly
# path); the decoded view then serves the stored text instead of decoding on demand.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-pc-sampling-rocpd-disasm
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pc-sampling-beta-enabled 1
        --kernel-trace --pc-sampling-unit cycles --pc-sampling-method stochastic
        --pc-sampling-interval 1048576 --pc-sampling-decode-instructions -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc-sampling-rocpd-disasm-input -o out --output-format
        rocpd -- $<TARGET_FILE:exec-mask-manipulation>
    DEPENDS exec-mask-manipulation
    TIMEOUT 120
    LABELS "integration-tests;pc-sampling;stochastic;rocpd"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${rocprofv3-pc-sampling-rocpd-env}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-pc-sampling-rocpd-disasm
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-pc-sampling-rocpd-disasm
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini LIBRARY_PATH "${rocprofiler-sdk_LIB_DIR}"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    ARGS --rocpd-disasm-input
         ${CMAKE_CURRENT_BINARY_DIR}/pc-sampling-rocpd-disasm-input/out_results.db
    TIMEOUT 120
    LABELS "integration-tests;pc-sampling;stochastic;rocpd"
    PRELOAD "${PRELOAD_ENV}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_REQUIRED rocprofv3-test-pc-sampling-rocpd-disasm
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")
