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

set(PACKAGE_OUTPUT_DIR
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofv3_avail_module)

set(ROCPROFV3_AVAIL_SOURCES __init__.py avail.py)

foreach(_FILE ${ROCPROFV3_AVAIL_SOURCES})
    configure_file(${CMAKE_CURRENT_LIST_DIR}/${_FILE} ${PACKAGE_OUTPUT_DIR}/${_FILE}
                   COPYONLY)
    install(
        FILES ${PACKAGE_OUTPUT_DIR}/${_FILE}
        DESTINATION ${CMAKE_INSTALL_BINDIR}/rocprofv3_avail_module
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        COMPONENT tools)
endforeach()
