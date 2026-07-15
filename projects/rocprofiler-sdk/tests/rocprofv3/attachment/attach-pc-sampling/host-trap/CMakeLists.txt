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
    rocprofiler-sdk-tests-rocprofv3-attachment-attach-pc-sampling-host-trap
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_sdk_pc_sampling_disabled(IS_PC_SAMPLING_DISABLED)

set(ROCPROFILER_MEMCHECK_TYPES "AddressSanitizer" "UndefinedBehaviorSanitizer"
                               "ThreadSanitizer")

if(IS_PC_SAMPLING_DISABLED OR (ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST
                                                        ROCPROFILER_MEMCHECK_TYPES))
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

set(attach-pc-sampling-host-trap-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

set(ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX
    "PC sampling unavailable|PC sampling configuration is not supported")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-attachment-attach-pc-sampling-host-trap
    COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/../run_attach_pc_sampling_test.sh
        $<TARGET_FILE:exec-mask-manipulation> $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        ${CMAKE_CURRENT_BINARY_DIR} warning out host_trap time 1000 5000 1048576
    DEPENDS rocprofiler-sdk::rocprofv3 exec-mask-manipulation
    TIMEOUT 120
    LABELS "integration-tests;attachment;pc-sampling"
    ENVIRONMENT "${attach-pc-sampling-host-trap-env}"
    SKIP_REGULAR_EXPRESSION
        "This test is skipped.|${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-attachment-attach-pc-sampling-host-trap
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-attachment-attach-pc-sampling-host-trap
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --input-csv
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-pc-sampling-output/out_pc_sampling_host_trap.csv
         --input-json
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-pc-sampling-output/out_results.json
         --agent-info-csv
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-pc-sampling-output/out_agent_info.csv
         --skip-if
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-pc-sampling-output/skipped
    TIMEOUT 60
    LABELS "integration-tests;attachment;pc-sampling"
    FIXTURES_REQUIRED rocprofv3-test-attachment-attach-pc-sampling-host-trap
    SKIP_REGULAR_EXPRESSION "SKIPPED|${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    DISABLED "${IS_DISABLED}")
