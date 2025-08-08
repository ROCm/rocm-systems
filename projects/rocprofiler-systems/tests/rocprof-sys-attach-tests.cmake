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
# attach tests
#
# -------------------------------------------------------------------------------------- #

set(_VALID_PTRACE_SCOPE OFF)

if(EXISTS "/proc/sys/kernel/yama/ptrace_scope")
    file(READ "/proc/sys/kernel/yama/ptrace_scope" _PTRACE_SCOPE LIMIT 1)

    if("${_PTRACE_SCOPE}" EQUAL 0)
        set(_VALID_PTRACE_SCOPE ON)
    endif()
else()
    rocprofiler_systems_message(
        AUTHOR_WARNING
        "Disabling attach tests. Run 'echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope' to enable attaching to process"
    )
endif()

if(NOT _VALID_PTRACE_SCOPE)
    return()
endif()

if(NOT TARGET parallel-overhead)
    return()
endif()

add_test(
    NAME parallel-overhead-attach
    COMMAND
        ${CMAKE_CURRENT_LIST_DIR}/run-rocprof-sys-pid.sh
        $<TARGET_FILE:rocprofiler-systems-instrument> -ME "\.c$" -E fib -e -v 1 --label
        return args file -l -- $<TARGET_FILE:parallel-overhead> 30 8 1000
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

set(_parallel_overhead_attach_environ
    "${_attach_environment}"
    "ROCPROFSYS_OUTPUT_PATH=rocprof-sys-tests-output"
    "ROCPROFSYS_OUTPUT_PREFIX=parallel-overhead-attach/"
)

set_tests_properties(
    parallel-overhead-attach
    PROPERTIES
        ENVIRONMENT "${_parallel_overhead_attach_environ}"
        TIMEOUT 300
        LABELS "parallel-overhead;attach"
        PASS_REGULAR_EXPRESSION
            "Outputting.*(perfetto-trace.proto).*Outputting.*(wall_clock.txt)"
        FAIL_REGULAR_EXPRESSION "Dyninst was unable to attach to the specified process"
)
