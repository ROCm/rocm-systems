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
# rocprofv3 attachment test for ROCTx pause/resume
#

cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-attachment-roctx-pause-resume
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(ROCPROFILER_MEMCHECK_TYPES "AddressSanitizer" "UndefinedBehaviorSanitizer"
                               "ThreadSanitizer" "LeakSanitizer")

if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning")
else()
    set(LOG_LEVEL "info")
endif()

set(attachment-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

foreach(_MODE normal selected)
    rocprofiler_add_integration_execute_test(
        rocprofv3-test-attachment-roctx-pause-resume-${_MODE}
        COMMAND
            ${CMAKE_CURRENT_SOURCE_DIR}/run_attachment_roctx_pause_resume_test.sh
            $<TARGET_FILE:roctx-attach-pause-resume>
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${CMAKE_CURRENT_BINARY_DIR}
            ${LOG_LEVEL} out ${_MODE}
        DEPENDS rocprofiler-sdk::rocprofv3 roctx-attach-pause-resume
        TIMEOUT 90
        LABELS "attachment;pause-resume"
        DISABLED ${IS_DISABLED}
        PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
        ENVIRONMENT "${attachment-env}"
        FAIL_REGULAR_EXPRESSION
            "ERROR|FATAL|Skipping output generation|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION "This test is skipped."
        FIXTURES_SETUP rocprofv3-test-attachment-roctx-pause-resume-${_MODE})

    rocprofiler_add_integration_validate_test(
        rocprofv3-test-attachment-roctx-pause-resume-${_MODE}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        TIMEOUT 30
        LABELS "attachment;pause-resume"
        DISABLED ${IS_DISABLED}
        SKIP_REGULAR_EXPRESSION "SKIPPED"
        FIXTURES_REQUIRED rocprofv3-test-attachment-roctx-pause-resume-${_MODE}
        ARGS --json-input
             ${CMAKE_CURRENT_BINARY_DIR}/attachment-roctx-${_MODE}-output/out_results.json
             --skip-if
             ${CMAKE_CURRENT_BINARY_DIR}/attachment-roctx-${_MODE}-output/skipped
             --mode
             ${_MODE})
endforeach()
