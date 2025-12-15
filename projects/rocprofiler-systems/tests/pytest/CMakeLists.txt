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

#
# rocprofiler-systems pytest install
#
include_guard(GLOBAL)

# ------------------------------------------------------------------------------#
# Pytest specific
# ------------------------------------------------------------------------------#

install(
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/test_binaries.py
        ${CMAKE_CURRENT_LIST_DIR}/test_hip_stream.py
        ${CMAKE_CURRENT_LIST_DIR}/test_jpegdecode.py
        ${CMAKE_CURRENT_LIST_DIR}/test_openmp.py
        ${CMAKE_CURRENT_LIST_DIR}/test_openmp.py
        ${CMAKE_CURRENT_LIST_DIR}/test_roctx.py
        ${CMAKE_CURRENT_LIST_DIR}/test_time_window.py
        ${CMAKE_CURRENT_LIST_DIR}/test_transpose.py
        ${CMAKE_CURRENT_LIST_DIR}/test_videodecode.py
    DESTINATION share/rocprofiler-systems/tests/pytest
)

# Pytest configuration files
install(
    FILES ${CMAKE_CURRENT_LIST_DIR}/conftest.py ${CMAKE_CURRENT_LIST_DIR}/requirements.txt
    DESTINATION share/rocprofiler-systems/tests/pytest
)

# Rocprofsys pytest module
install(
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/rocprofsys/__init__.py
        ${CMAKE_CURRENT_LIST_DIR}/rocprofsys/config.py
        ${CMAKE_CURRENT_LIST_DIR}/rocprofsys/gpu.py
        ${CMAKE_CURRENT_LIST_DIR}/rocprofsys/runners.py
        ${CMAKE_CURRENT_LIST_DIR}/rocprofsys/validators.py
    DESTINATION share/rocprofiler-systems/tests/pytest/rocprofsys
)
