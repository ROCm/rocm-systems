# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

set(UNIT_TEST_OBJECTS
    $<TARGET_OBJECTS:lib-common-tests>
    $<TARGET_OBJECTS:trace-cache-tests>
    $<TARGET_OBJECTS:demangler-tests>
    $<TARGET_OBJECTS:logger-tests>
)

if(ROCPROFSYS_USE_ROCM)
    list(
        APPEND UNIT_TEST_OBJECTS
        $<TARGET_OBJECTS:pmc-common-tests>
        $<TARGET_OBJECTS:pmc-gpu-collector-tests>
    )
endif()

add_executable(rocprof-sys-unit-tests ${UNIT_TEST_OBJECTS})

target_link_libraries(
    rocprof-sys-unit-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-common-library
        rocprofiler-systems-core-library
)
