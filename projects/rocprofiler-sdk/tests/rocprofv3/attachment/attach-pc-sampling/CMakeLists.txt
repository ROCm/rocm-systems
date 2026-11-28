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

cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-attachment-pc-sampling
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_sdk_pc_sampling_disabled(IS_HOST_TRAP_DISABLED)
rocprofiler_sdk_pc_sampling_stochastic_disabled(IS_STOCHASTIC_DISABLED)

set(_attachment_memchecks AddressSanitizer UndefinedBehaviorSanitizer)
list(APPEND _attachment_memchecks ThreadSanitizer LeakSanitizer)
if(ROCPROFILER_MEMCHECK IN_LIST _attachment_memchecks)
    set(IS_HOST_TRAP_DISABLED TRUE)
    set(IS_STOCHASTIC_DISABLED TRUE)
endif()

set(attachment-pc-sampling-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

function(add_attach_pc_sampling_test METHOD UNIT INTERVAL DISABLED)
    set(_test_name rocprofv3-test-attachment-pc-sampling-${METHOD})
    set(_output_dir ${CMAKE_CURRENT_BINARY_DIR}/${METHOD}-output)
    set(_skip_regex "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}")
    set(_execute_skip_regex "This test is skipped.|${_skip_regex}")

    rocprofiler_add_integration_execute_test(
        ${_test_name}
        COMMAND
            ${CMAKE_CURRENT_SOURCE_DIR}/run_attach_pc_sampling_test.sh
            $<TARGET_FILE:attachment-test> $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
            ${_output_dir} ${METHOD} ${UNIT} ${INTERVAL} "${_skip_regex}"
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 90
        LABELS "attachment;pc-sampling;${METHOD}"
        DISABLED ${DISABLED}
        PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
        ENVIRONMENT "${attachment-pc-sampling-env}"
        FAIL_REGULAR_EXPRESSION
            "ERROR|FATAL|reference count is zero|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION "${_execute_skip_regex}"
        FIXTURES_SETUP ${_test_name})

    rocprofiler_add_integration_validate_test(
        ${_test_name}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        TIMEOUT 30
        LABELS "attachment;pc-sampling;${METHOD}"
        DISABLED ${DISABLED}
        SKIP_REGULAR_EXPRESSION "SKIPPED|${_skip_regex}"
        FIXTURES_REQUIRED ${_test_name}
        ARGS --json-input ${_output_dir}/out_results.json --method ${METHOD} --skip-if
             ${_output_dir}/skipped)
endfunction()

add_attach_pc_sampling_test(host_trap time 1 "${IS_HOST_TRAP_DISABLED}")
add_attach_pc_sampling_test(stochastic cycles 1048576 "${IS_STOCHASTIC_DISABLED}")
