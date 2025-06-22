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
    NAME parallel-overhead-locks
    TARGET parallel-overhead-locks
    LABELS "locks"
    REWRITE_ARGS -e -i 256
    RUNTIME_ARGS -e -i 256
    RUN_ARGS 30 4 1000
    ENVIRONMENT
        "${_lock_environment};ROCPROFSYS_PROFILE=ON;ROCPROFSYS_TRACE=ON;ROCPROFSYS_COLLAPSE_THREADS=OFF;ROCPROFSYS_SAMPLING_REALTIME=ON;ROCPROFSYS_SAMPLING_REALTIME_FREQ=10;ROCPROFSYS_SAMPLING_REALTIME_TIDS=0;ROCPROFSYS_SAMPLING_KEEP_INTERNAL=OFF"
    REWRITE_RUN_PASS_REGEX
        "wall_clock .*\\|_pthread_create .* 4 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000"
    RUNTIME_PASS_REGEX
        "wall_clock .*\\|_pthread_create .* 4 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000"
)

rocprofiler_systems_add_test(
    SKIP_RUNTIME
    NAME parallel-overhead-locks-timemory
    TARGET parallel-overhead-locks
    LABELS "locks"
    REWRITE_ARGS -e -v 2 --min-instructions=32 --dyninst-options InstrStackFrames SaveFPR
                 TrampRecursive
    RUN_ARGS 10 4 1000
    ENVIRONMENT
        "${_lock_environment};ROCPROFSYS_FLAT_PROFILE=ON;ROCPROFSYS_PROFILE=ON;ROCPROFSYS_TRACE=OFF;ROCPROFSYS_SAMPLING_KEEP_INTERNAL=OFF"
    REWRITE_RUN_PASS_REGEX
        "start_thread (.*) 4 (.*) pthread_mutex_lock (.*) 4000 (.*) pthread_mutex_unlock (.*) 4000"
)
