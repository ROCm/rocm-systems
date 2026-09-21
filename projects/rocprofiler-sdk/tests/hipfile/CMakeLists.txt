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
    rocprofiler-sdk-tests-hipfile-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(HIPFILE_TRACE_ENVIRONMENT)
if(DEFINED ROCM_PATH AND EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
    set(HIPFILE_SYSDEPS_LIBRARY_DIR "${ROCM_PATH}/lib/rocm_sysdeps/lib")
elseif(DEFINED ENV{ROCM_PATH} AND EXISTS "$ENV{ROCM_PATH}/lib/rocm_sysdeps/lib")
    set(HIPFILE_SYSDEPS_LIBRARY_DIR "$ENV{ROCM_PATH}/lib/rocm_sysdeps/lib")
elseif(EXISTS "/opt/rocm/lib/rocm_sysdeps/lib")
    set(HIPFILE_SYSDEPS_LIBRARY_DIR "/opt/rocm/lib/rocm_sysdeps/lib")
endif()

if(HIPFILE_SYSDEPS_LIBRARY_DIR)
    list(
        APPEND
        HIPFILE_TRACE_ENVIRONMENT
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:${HIPFILE_SYSDEPS_LIBRARY_DIR}:$ENV{LD_LIBRARY_PATH}"
        )
endif()

set(IS_DISABLED ON)
if(TARGET hipfile-trace-client)
    set(IS_DISABLED OFF)
    set(HIPFILE_TRACE_DEPENDS hipfile-trace-client)
    set(HIPFILE_TRACE_COMMAND $<TARGET_FILE:hipfile-trace-client>)
else()
    set(HIPFILE_TRACE_DEPENDS rocprofiler-sdk-json-tool)
    set(HIPFILE_TRACE_COMMAND ${CMAKE_COMMAND})
endif()

rocprofiler_add_integration_execute_test(
    hipfile-tracing
    COMMAND ${HIPFILE_TRACE_COMMAND}
    DEPENDS ${HIPFILE_TRACE_DEPENDS}
    TIMEOUT 45
    LABELS "integration-tests"
    DISABLED ${IS_DISABLED}
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>" "${PRELOAD_ENV}"
    ENVIRONMENT
        "ROCPROFILER_TOOL_OUTPUT_FILE=hipfile-tracing-test.json"
        "ROCPROFILER_TOOL_CONTEXTS=HIPFILE_API_CALLBACK HIPFILE_API_BUFFERED HIPFILE_API_EXT_BUFFERED"
        "ROCPROFILER_TOOL_DISABLE_PERFETTO=1"
        ${HIPFILE_TRACE_ENVIRONMENT}
    FIXTURES_SETUP hipfile-tracing)

rocprofiler_add_integration_validate_test(
    hipfile-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    DISABLED ${IS_DISABLED}
    FIXTURES_REQUIRED hipfile-tracing
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/hipfile-tracing-test.json)
