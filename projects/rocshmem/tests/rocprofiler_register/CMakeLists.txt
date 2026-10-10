###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

# Regression test for rocprofiler-register integration (AIPROFSYST-787).
#
# Verifies that ROCSHMEM_ROCPROFILER_REGISTER=1 was set at compile time by:
#   (a) Link-time: fails if rocprofiler_register_import_rocshmem is missing
#       from librocshmem.a.
#   (b) Runtime:   calls the function and asserts the returned version equals
#       the version in rocshmem_config.h.
#
# No MPI or GPU required.

add_executable(rocshmem-rocprofiler-register-import-test
    test_import.cpp
)

target_link_libraries(rocshmem-rocprofiler-register-import-test
    PRIVATE
        ${PROJECT_NAME}
        rocprofiler-register::rocprofiler-register
)

add_test(
    NAME    rocshmem-rocprofiler-register-import
    COMMAND rocshmem-rocprofiler-register-import-test
)

set_tests_properties(rocshmem-rocprofiler-register-import
    PROPERTIES
        LABELS "rocprofiler-register;quick"
)
