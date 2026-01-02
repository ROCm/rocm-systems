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

# Under install mode, run build_standalone.sh to build the standalone binary

execute_process(
    COMMAND
        ${CMAKE_CURRENT_LIST_DIR}/build_standalone.sh --output-dir
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    RESULT_VARIABLE BUILD_STANDALONE_RESULT
    OUTPUT_VARIABLE BUILD_STANDALONE_OUTPUT
    ERROR_VARIABLE BUILD_STANDALONE_ERROR
)

if(NOT BUILD_STANDALONE_RESULT EQUAL 0)
    rocprofiler_systems_message(FATAL_ERROR "Build standalone binary failed (${BUILD_STANDALONE_RESULT}):\n${BUILD_STANDALONE_OUTPUT}\n${BUILD_STANDALONE_ERROR}")
endif()

if(BUILD_STANDALONE_RESULT EQUAL 0)
    install(
        FILES ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/rocprofsys-tests.pyz
        DESTINATION share/rocprofiler-systems
    )
endif()
