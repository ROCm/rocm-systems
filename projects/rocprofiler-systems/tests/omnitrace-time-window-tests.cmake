# -------------------------------------------------------------------------------------- #
#
# time-window tests
#
# -------------------------------------------------------------------------------------- #

if(_OS_RELEASE STREQUAL "ubuntu-18.04")
    set(_TRACE_WINDOW_SKIP SKIP_RUNTIME)
endif()

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING ${_TRACE_WINDOW_SKIP}
    NAME trace-time-window
    TARGET trace-time-window
    REWRITE_ARGS -e -v 2 --caller-include inner -i 4096
    RUNTIME_ARGS -e -v 1 --caller-include inner -i 4096
    LABELS "time-window"
    ENVIRONMENT "${_window_environment};OMNITRACE_TRACE_DURATION=1.25")

omnitrace_add_validation_test(
    NAME trace-time-window-binary-rewrite
    TIMEMORY_METRIC "wall_clock"
    TIMEMORY_FILE "wall_clock.json"
    PERFETTO_METRIC "host"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "time-window"
    FAIL_REGEX "outer_d|OMNITRACE_ABORT_FAIL_REGEX"
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
         -p)

omnitrace_add_validation_test(
    NAME trace-time-window-runtime-instrument
    TIMEMORY_METRIC "wall_clock"
    TIMEMORY_FILE "wall_clock.json"
    PERFETTO_METRIC "host"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "time-window"
    FAIL_REGEX "outer_d|OMNITRACE_ABORT_FAIL_REGEX"
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
         -p)

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING ${_TRACE_WINDOW_SKIP}
    NAME trace-time-window-delay
    TARGET trace-time-window
    REWRITE_ARGS -e -v 2 --caller-include inner -i 4096
    RUNTIME_ARGS -e -v 1 --caller-include inner -i 4096
    LABELS "time-window"
    ENVIRONMENT
        "${_window_environment};OMNITRACE_TRACE_DELAY=0.75;OMNITRACE_TRACE_DURATION=0.75")

omnitrace_add_validation_test(
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
         -p)

omnitrace_add_validation_test(
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
         -p)
