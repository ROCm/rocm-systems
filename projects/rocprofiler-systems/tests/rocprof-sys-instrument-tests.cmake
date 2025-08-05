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
# binary-rewrite and runtime-instrumentation tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    SKIP_SAMPLING SKIP_RUNTIME
    NAME rewrite-caller
    TARGET rewrite-caller
    LABELS "caller-include"
    REWRITE_ARGS
        -e
        -i
        256
        --caller-include
        "^inner"
        -v
        2
        --print-instrumented
        functions
    RUN_ARGS 17
    ENVIRONMENT "${_base_environment};ROCPROFSYS_COUT_OUTPUT=ON"
    BASELINE_PASS_REGEX "number of calls made = 17"
    REWRITE_PASS_REGEX "\\[function\\]\\[Forcing\\] caller-include-regex :: 'outer'"
    REWRITE_RUN_PASS_REGEX ">>> ._outer ([ \\|]+) 17"
)

rocprofiler_systems_add_test(
    NAME parallel-overhead
    TARGET parallel-overhead
    REWRITE_ARGS -e -v 2 --min-instructions=8
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=8
        --label
        file
        line
        return
        args
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
)

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME
    NAME parallel-overhead-locks-perfetto
    TARGET parallel-overhead-locks
    LABELS "locks"
    REWRITE_ARGS -e -v 2 --min-instructions=8
    RUN_ARGS 10 4 1000
    ENVIRONMENT
        "${_lock_environment};ROCPROFSYS_FLAT_PROFILE=ON;ROCPROFSYS_PROFILE=OFF;ROCPROFSYS_TRACE=ON;ROCPROFSYS_SAMPLING_KEEP_INTERNAL=OFF"
)
