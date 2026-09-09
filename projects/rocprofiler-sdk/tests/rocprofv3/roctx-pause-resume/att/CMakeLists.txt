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

include(FindPackageHandleStandardArgs)

project(
    rocprofiler-sdk-tests-rocprofv3-roctx-pause-resume-att
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(roctx-pause-resume-att-env "${PRELOAD_ENV}")

find_library(
    attdecoder_LIBRARY
    NAMES rocprof-trace-decoder
    HINTS ${ROCM_PATH}
    PATHS ${ROCM_PATH}
    PATH_SUFFIXES lib)

if(attdecoder_LIBRARY)
    cmake_path(GET attdecoder_LIBRARY PARENT_PATH attdecoder_LIB_DIR)
endif()

find_package_handle_standard_args(attdecoder REQUIRED_VARS attdecoder_LIB_DIR
                                                           attdecoder_LIBRARY)

set(IS_DISABLED ON)
if(attdecoder_FOUND)
    set(IS_DISABLED OFF)
    set(ATT_LIB --att-library-path ${attdecoder_LIB_DIR})
endif()

rocprofiler_add_integration_execute_test(
    rocprofv3-test-roctx-pause-resume-att
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${ATT_LIB} --att --marker-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-att -o out --output-format json --log-level env
        -- $<TARGET_FILE:roctx-pause-resume> 8 1
    DEPENDS roctx-pause-resume
    TIMEOUT 45
    LABELS "integration-tests;thread-trace;pause-resume"
    DISABLED ${IS_DISABLED}
    ENVIRONMENT "${roctx-pause-resume-att-env}"
    FIXTURES_SETUP rocprofv3-test-roctx-pause-resume-att)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-roctx-pause-resume-att
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input ${CMAKE_CURRENT_BINARY_DIR}/roctx-pause-resume-att/out_results.json
         --output-dir ${CMAKE_CURRENT_BINARY_DIR}/roctx-pause-resume-att
    TIMEOUT 45
    LABELS "integration-tests;thread-trace;pause-resume"
    DISABLED ${IS_DISABLED}
    FIXTURES_REQUIRED rocprofv3-test-roctx-pause-resume-att
    FAIL_REGULAR_EXPRESSION "AssertionError")
