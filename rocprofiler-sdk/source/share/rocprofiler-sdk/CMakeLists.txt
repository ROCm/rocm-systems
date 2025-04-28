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

configure_file(
    convert-counters-collection-format.py
    ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk/convert-counters-collection-format.py
    COPYONLY)

set(CONVERT_CC_FORMAT_PATH
    ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk/convert-counters-collection-format.py)

add_executable(rocprofiler-sdk::convert-counters-collection-format IMPORTED)
set_property(TARGET rocprofiler-sdk::convert-counters-collection-format
             PROPERTY IMPORTED_LOCATION ${CONVERT_CC_FORMAT_PATH})

install(
    FILES
        ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk/convert-counters-collection-format.py
    DESTINATION share/rocprofiler-sdk
    COMPONENT core)
