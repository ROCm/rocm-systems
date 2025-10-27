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

set(rocprofv3_PYTHON_SOURCES __init__.py avail.py)

# rocprofv3 is pure Python (uses ctypes, no compilation needed)
set(ROCPROFV3_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python/site-packages/rocprofv3"
    CACHE STRING "Installation directory for rocprofv3 Python package")

set(ROCPROFV3_OUTPUT_DIR
    "${PROJECT_BINARY_DIR}/${ROCPROFV3_INSTALL_DIR}")

# Copy Python files to build directory
foreach(_SOURCE ${rocprofv3_PYTHON_SOURCES})
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${_SOURCE}
        ${ROCPROFV3_OUTPUT_DIR}/${_SOURCE}
        COPYONLY)
endforeach()

# Install Python files
install(
    FILES ${rocprofv3_PYTHON_SOURCES}
    DESTINATION ${ROCPROFV3_INSTALL_DIR}
    COMPONENT tools)
