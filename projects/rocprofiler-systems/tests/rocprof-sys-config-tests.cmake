# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

# -------------------------------------------------------------------------------------- #
#
# general config file tests
#
# -------------------------------------------------------------------------------------- #

file(
    WRITE
    ${CMAKE_CURRENT_BINARY_DIR}/invalid.cfg
    "
ROCPROFSYS_CONFIG_FILE =
FOOBAR = ON
"
)

if(TARGET parallel-overhead)
    set(_CONFIG_TEST_EXE $<TARGET_FILE:parallel-overhead>)
else()
    set(_CONFIG_TEST_EXE ls)
endif()

add_test(
    NAME rocprofiler-systems-invalid-config
    COMMAND $<TARGET_FILE:rocprofiler-systems-instrument> -- ${_CONFIG_TEST_EXE}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

set_tests_properties(
    rocprofiler-systems-invalid-config
    PROPERTIES
        ENVIRONMENT
            "ROCPROFSYS_CONFIG_FILE=${CMAKE_CURRENT_BINARY_DIR}/invalid.cfg;ROCPROFSYS_CI=ON;ROCPROFSYS_CI_TIMEOUT=120"
        TIMEOUT 120
        LABELS "config"
        WILL_FAIL ON
)

add_test(
    NAME rocprofiler-systems-missing-config
    COMMAND $<TARGET_FILE:rocprofiler-systems-instrument> -- ${_CONFIG_TEST_EXE}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

set_tests_properties(
    rocprofiler-systems-missing-config
    PROPERTIES
        ENVIRONMENT
            "ROCPROFSYS_CONFIG_FILE=${CMAKE_CURRENT_BINARY_DIR}/missing.cfg;ROCPROFSYS_CI=ON;ROCPROFSYS_CI_TIMEOUT=120"
        TIMEOUT 120
        LABELS "config"
        WILL_FAIL ON
)
