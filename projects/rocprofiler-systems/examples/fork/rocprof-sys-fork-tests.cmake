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
# fork tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    NAME fork
    TARGET fork-example
    REWRITE_ARGS -e -v 2 --print-instrumented modules -i 16
    RUNTIME_ARGS -e -v 1 --label file -i 16
    ENVIRONMENT
        "${_base_environment};ROCPROFSYS_SAMPLING_FREQ=250;ROCPROFSYS_SAMPLING_REALTIME=ON"
    SAMPLING_PASS_REGEX "fork.. called on PID"
    RUNTIME_PASS_REGEX "fork.. called on PID"
    REWRITE_RUN_PASS_REGEX "fork.. called on PID"
    SAMPLING_FAIL_REGEX "(${ROCPROFSYS_ABORT_FAIL_REGEX})"
    RUNTIME_FAIL_REGEX "(${ROCPROFSYS_ABORT_FAIL_REGEX})"
    REWRITE_RUN_FAIL_REGEX "(${ROCPROFSYS_ABORT_FAIL_REGEX})"
)
