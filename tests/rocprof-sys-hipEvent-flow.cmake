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

set(_hipEvent_environment
    "${_base_environment}"
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_USE_SAMPLING=ON"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=OFF"
    "ROCPROFSYS_SAMPLING_FREQ=10"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_FILE_OUTPUT=ON"
    "ROCPROFSYS_TIMEMORY_COMPONENTS=wall_clock papi_array network_stats"
    "ROCPROFSYS_USE_PID=OFF"
    "ROCPROFSYS_SAMPLING_DELAY=0.05"
)

# Run the hipEvent test
add_test(
    NAME streams-hipEvent-flow
    COMMAND $<TARGET_FILE:rocprofiler-systems-sample> -- ${PROJECT_BINARY_DIR}/streams 2
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

set_tests_properties(
    streams-hipEvent-flow
    PROPERTIES ENVIRONMENT "${_hipEvent_environment}" TIMEOUT 120 LABEL "hipEvent"
)

if(_VALID_GPU)
    # Validate the generated perfetto file.
    add_test(
        NAME validate-streams-hipEvent-flow-perfetto
        COMMAND
            ${ROCPROFSYS_VALIDATION_PYTHON}
            ${CMAKE_CURRENT_LIST_DIR}/validate-perfetto-proto.py -i
            ${PROJECT_BINARY_DIR}/rocprofsys-streams-output/perfetto-trace.proto
            --flow-events hipEventRecord -t /opt/trace_processor/bin/trace_processor_shell
            -p
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    )
else()
    rocprofiler_systems_message(
        STATUS "validate-streams-hipEvent-flow-perfetto requires a GPU and no valid GPUs were found"
    )
endif()
