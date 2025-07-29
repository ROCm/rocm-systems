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
# hipEvent* flow tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME
    NAME transpose-hipEvent-flow
    TARGET transpose
    LABELS "hipEvent"
    MPI ${TRANSPOSE_USE_MPI}
    GPU ON
    NUM_PROCS ${NUM_PROCS}
    REWRITE_ARGS -e -v 2 -E uniform_int_distribution
    ENVIRONMENT
        "${_base_environment};ROCPROFSYS_ROCM_EVENTS=${ROCPROFSYS_ROCM_EVENTS_TEST}"
    REWRITE_RUN_PASS_REGEX "${_ROCP_PASS_REGEX}"
    SAMPLING_PASS_REGEX "${_ROCP_PASS_REGEX}"
)

# Validate the generated perfetto file.
add_test(
    NAME validate-transpose-hipEvent-flow-perfetto
    COMMAND
        ${ROCPROFSYS_VALIDATION_PYTHON}
        ${CMAKE_CURRENT_LIST_DIR}/validate-perfetto-proto.py -i
        ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/transpose-hipEvent-flow-sampling/perfetto-trace.proto
        --flow-events hipEventRecord -t /opt/trace_processor/bin/trace_processor_shell -p
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)
