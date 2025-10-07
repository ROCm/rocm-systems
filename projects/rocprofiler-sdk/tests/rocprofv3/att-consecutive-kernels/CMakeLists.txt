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

#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

include(FindPackageHandleStandardArgs)

project(
    rocprofiler-sdk-tests-rocprofv3-att
    LANGUAGES CXX
    VERSION 0.0.0)

set(CMAKE_MESSAGE_INDENT "[${PROJECT_NAME}] ")

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

find_package(rocprofiler-sdk REQUIRED)

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

if(ROCPROFILER_DISABLE_UNSTABLE_CTESTS)
    set(IS_DISABLED ON)
endif()

# consecutive kernel test
add_test(
    NAME rocprofv3-test-att-consecutive-kernels-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${ATT_LIB} --att
        --att-consecutive-kernels 8 -d ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out
        --log-level env -- $<TARGET_FILE:vector-ops>)

set_tests_properties(
    rocprofv3-test-att-consecutive-kernels-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DISABLED ${IS_DISABLED})

add_test(NAME rocprofv3-test-att-consecutive-kernels-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py
                 --csv-directory-input ${CMAKE_CURRENT_BINARY_DIR}/vector-ops-trace/)

set_tests_properties(
    rocprofv3-test-att-consecutive-kernels-validate
    PROPERTIES TIMEOUT
               60
               LABELS
               "integration-tests"
               DEPENDS
               rocprofv3-test-att-consecutive-kernels-execute
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               DISABLED
               ${IS_DISABLED})
