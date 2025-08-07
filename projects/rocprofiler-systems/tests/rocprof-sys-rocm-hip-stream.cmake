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
# ROCm tests
#
# -------------------------------------------------------------------------------------- #

find_package(ROCmVersion)

if(NOT ROCmVersion_FOUND)
    message(
        WARNING
        "ROCmVersion_FOUND not found, skipping tests in ${CMAKE_CURRENT_LIST_FILE}"
    )
    return()
endif()

if(${ROCmVersion_FULL_VERSION} VERSION_GREATER_EQUAL "7.0")
    message(STATUS "Adding Group-By Tests")

    rocprofiler_systems_add_test(
        SKIP_REWRITE SKIP_RUNTIME SKIP_BASELINE
        NAME transpose-group-by-queue
        TARGET transpose
        MPI ${TRANSPOSE_USE_MPI}
        GPU ON
        NUM_PROCS ${NUM_PROCS}
        ENVIRONMENT "${_base_environment};ROCPROFSYS_ROCM_GROUP_BY_QUEUE=YES"
        LABEL "group-by-queue"
        RUNTIME_TIMEOUT 480
    )

    rocprofiler_systems_add_test(
        SKIP_REWRITE SKIP_RUNTIME SKIP_BASELINE
        NAME transpose-group-by-stream
        TARGET transpose
        MPI ${TRANSPOSE_USE_MPI}
        GPU ON
        NUM_PROCS ${NUM_PROCS}
        ENVIRONMENT "${_base_environment};ROCPROFSYS_ROCM_GROUP_BY_QUEUE=NO"
        LABEL "group-by-queue"
        RUNTIME_TIMEOUT 480
    )
endif()
