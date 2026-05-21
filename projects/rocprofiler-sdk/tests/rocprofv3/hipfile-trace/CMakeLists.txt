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
    rocprofiler-sdk-tests-rocprofv3-hipfile-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(hipfile QUIET CONFIG)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(IS_DISABLED OFF)
set(HIPFILE_TRACE_TARGET "")
unset(HIPFILE_TRACE_INCLUDE_DIR)

if(TARGET hip::hipfile)
    set(HIPFILE_TRACE_TARGET hip::hipfile)
elseif(TARGET hipfile)
    set(HIPFILE_TRACE_TARGET hipfile)
endif()

set(HIPFILE_TRACE_INCLUDE_HINTS ${ROCM_PATH})
if(HIPFILE_TRACE_TARGET)
    get_target_property(HIPFILE_TRACE_TARGET_INCLUDES ${HIPFILE_TRACE_TARGET}
                        INTERFACE_INCLUDE_DIRECTORIES)
    if(HIPFILE_TRACE_TARGET_INCLUDES)
        list(APPEND HIPFILE_TRACE_INCLUDE_HINTS ${HIPFILE_TRACE_TARGET_INCLUDES})
    endif()
endif()

find_path(
    HIPFILE_TRACE_INCLUDE_DIR
    NAMES hipfile-api-trace.h hipfile.h
    HINTS ${HIPFILE_TRACE_INCLUDE_HINTS}
    PATHS ${ROCM_PATH}
    PATH_SUFFIXES include)

if(NOT HIPFILE_TRACE_TARGET)
    find_library(
        HIPFILE_TRACE_LIBRARY
        NAMES hipfile
        HINTS ${ROCM_PATH}
        PATHS ${ROCM_PATH}
        PATH_SUFFIXES lib lib64)
    if(HIPFILE_TRACE_LIBRARY AND HIPFILE_TRACE_INCLUDE_DIR)
        add_library(rocprofv3-test-hipfile-lib UNKNOWN IMPORTED)
        set_target_properties(
            rocprofv3-test-hipfile-lib
            PROPERTIES IMPORTED_LOCATION "${HIPFILE_TRACE_LIBRARY}"
                       INTERFACE_INCLUDE_DIRECTORIES "${HIPFILE_TRACE_INCLUDE_DIR}")
        set(HIPFILE_TRACE_TARGET rocprofv3-test-hipfile-lib)
    endif()
endif()

if(NOT HIPFILE_TRACE_TARGET OR NOT HIPFILE_TRACE_INCLUDE_DIR)
    set(IS_DISABLED ON)
endif()

if(NOT EXISTS "${HIPFILE_TRACE_INCLUDE_DIR}/hipfile-api-trace.h")
    set(IS_DISABLED ON)
endif()

if(NOT IS_DISABLED)
    add_executable(hipfile-trace-client hipfile_trace.cpp)
    target_link_libraries(hipfile-trace-client PRIVATE ${HIPFILE_TRACE_TARGET})
    target_compile_definitions(hipfile-trace-client PRIVATE __HIP_PLATFORM_AMD__)
    target_compile_features(hipfile-trace-client PRIVATE cxx_std_17)
    set_target_properties(hipfile-trace-client PROPERTIES CXX_EXTENSIONS OFF)
endif()

rocprofiler_add_integration_execute_test(
    rocprofv3-test-hipfile-tracing
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hipfile-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format json csv rocpd
        --log-level env --
        $<IF:$<TARGET_EXISTS:hipfile-trace-client>,$<TARGET_FILE:hipfile-trace-client>,hipfile-trace-client>
    DEPENDS hipfile-trace-client
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-hipfile-tracing
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-hipfile-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/hipfile-trace-client-trace/out_results.json
         --otf2-input
         ${CMAKE_CURRENT_BINARY_DIR}/hipfile-trace-client-trace/out_results.otf2
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/hipfile-trace-client-trace/out_results.pftrace
         --csv-input
         ${CMAKE_CURRENT_BINARY_DIR}/hipfile-trace-client-trace/out_hipfile_api_trace.csv
         --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/hipfile-trace-client-trace/out_results.db
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-hipfile-tracing
    DISABLED "${IS_DISABLED}")
