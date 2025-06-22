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
# time-window tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_SAMPLING ${_TRACE_WINDOW_SKIP}
    NAME trace-time-window
    TARGET trace-time-window
    REWRITE_ARGS -e -v 2 --caller-include inner -i 4096
    RUNTIME_ARGS -e -v 1 --caller-include inner -i 4096
    LABELS "time-window"
    ENVIRONMENT "${_window_environment};ROCPROFSYS_TRACE_DURATION=1.25"
)

rocprofiler_systems_add_validation_test(
    NAME trace-time-window-binary-rewrite
    TIMEMORY_METRIC "wall_clock"
    TIMEMORY_FILE "wall_clock.json"
    PERFETTO_METRIC "host"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "time-window"
    FAIL_REGEX "outer_d|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS -l
         trace-time-window.inst
         outer_a
         outer_b
         outer_c
         -c
         1
         1
         1
         1
         -d
         0
         1
         1
         1
         -p
)

rocprofiler_systems_add_validation_test(
    NAME trace-time-window-runtime-instrument
    TIMEMORY_METRIC "wall_clock"
    TIMEMORY_FILE "wall_clock.json"
    PERFETTO_METRIC "host"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "time-window"
    FAIL_REGEX "outer_d|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS -l
         trace-time-window
         outer_a
         outer_b
         outer_c
         -c
         1
         1
         1
         1
         -d
         0
         1
         1
         1
         -p
)

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_SAMPLING ${_TRACE_WINDOW_SKIP}
    NAME trace-time-window-delay
    TARGET trace-time-window
    REWRITE_ARGS -e -v 2 --caller-include inner -i 4096
    RUNTIME_ARGS -e -v 1 --caller-include inner -i 4096
    LABELS "time-window"
    ENVIRONMENT
        "${_window_environment};ROCPROFSYS_TRACE_DELAY=0.75;ROCPROFSYS_TRACE_DURATION=0.75"
)

rocprofiler_systems_add_validation_test(
    NAME trace-time-window-delay-binary-rewrite
    TIMEMORY_METRIC "wall_clock"
    TIMEMORY_FILE "wall_clock.json"
    PERFETTO_METRIC "host"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "time-window"
    ARGS -l
         outer_c
         outer_d
         -c
         1
         1
         -d
         0
         0
         -p
)

rocprofiler_systems_add_validation_test(
    NAME trace-time-window-delay-runtime-instrument
    TIMEMORY_METRIC "wall_clock"
    TIMEMORY_FILE "wall_clock.json"
    PERFETTO_METRIC "host"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "time-window"
    ARGS -l
         outer_c
         outer_d
         -c
         1
         1
         -d
         0
         0
         -p
)
