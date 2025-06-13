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
# rccl tests
#
# -------------------------------------------------------------------------------------- #

foreach(_TARGET ${RCCL_TEST_TARGETS})
    string(REPLACE "rccl-tests::" "" _NAME "${_TARGET}")
    string(REPLACE "_" "-" _NAME "${_NAME}")
    rocprofiler_systems_add_test(
        SKIP_RUNTIME
        NAME rccl-test-${_NAME}
        TARGET ${_TARGET}
        LABELS "rccl-tests;rcclp"
        MPI ON
        GPU ON
        NUM_PROCS 1
        SAMPLING_TIMEOUT 300
        REWRITE_TIMEOUT 300
        REWRITE_ARGS
            -e
            -v
            2
            -i
            8
            --label
            file
            line
            return
            args
        RUNTIME_ARGS
            -e
            -v
            1
            -i
            8
            --label
            file
            line
            return
            args
            -ME
            sysdeps
            --log-file
            rccl-test-${_NAME}.log
        RUN_ARGS -t
                 1
                 -g
                 1
                 -i
                 10
                 -w
                 2
                 -m
                 2
                 -p
                 -c
                 1
                 -z
                 -s
                 1
        ENVIRONMENT "${_rccl_environment}")

    rocprofiler_systems_add_validation_test(
        NAME rccl-test-${_NAME}-sampling
        PERFETTO_METRIC "rocm_rccl_api"
        PERFETTO_FILE "perfetto-trace.proto"
        LABELS "rccl-tests;rcclp"
        ARGS --counter-names "RCCL Comm" -p)
endforeach()
