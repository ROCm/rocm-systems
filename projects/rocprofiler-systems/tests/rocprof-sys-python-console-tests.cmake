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
# python console script tests
#   * These tests do not use the code present in examples/python
#
# -------------------------------------------------------------------------------------- #

if(NOT ROCPROFSYS_USE_PYTHON)
    return()
endif()

set(_INDEX 0)
foreach(_VERSION ${ROCPROFSYS_PYTHON_VERSIONS})
    list(GET ROCPROFSYS_PYTHON_ROOT_DIRS ${_INDEX} Python3_ROOT_DIR)
    rocprofiler_systems_python_console_script(
        "${BINARY_NAME_PREFIX}-python" "rocprofsys"
        VERSION ${_VERSION}
        ROOT_DIR "${Python3_ROOT_DIR}"
    )
    math(EXPR _INDEX "${_INDEX} + 1")
endforeach()
