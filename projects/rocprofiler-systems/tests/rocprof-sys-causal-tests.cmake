# -------------------------------------------------------------------------------------- #
#
# causal profiling tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_causal_test(
    NAME cpu-omni-func
    TARGET causal-cpu-omni
    RUN_ARGS 70 10 432525 1000000000
    CAUSAL_MODE "function"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-omni-func-ndebug
    TARGET causal-cpu-omni-ndebug
    RUN_ARGS 70 10 432525 1000000000
    CAUSAL_MODE "function"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-omni-line
    TARGET causal-cpu-omni
    RUN_ARGS 70 10 432525 1000000000
    CAUSAL_MODE "line"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

rocprofiler_systems_add_causal_test(
    NAME both-omni-func
    TARGET causal-both-omni
    RUN_ARGS 70 10 432525 400000000
    CAUSAL_MODE "function"
    CAUSAL_ARGS
        -n
        2
        -w
        1
        -d
        3
        --monochrome
        -g
        ${CMAKE_BINARY_DIR}/rocprof-sys-tests-config/causal-both-omni-func
        -l
        causal-both-omni
        -v
        3
        -b
        timer
    ENVIRONMENT "ROCPROFSYS_STRICT_CONFIG=OFF"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

rocprofiler_systems_add_causal_test(
    NAME lulesh-func
    TARGET lulesh-omni
    RUN_ARGS -i 35 -s 50 -p
    CAUSAL_MODE "function"
    CAUSAL_ARGS -s 0,10,25,50,75
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME lulesh-func-ndebug
    TARGET lulesh-omni-ndebug
    RUN_ARGS -i 35 -s 50 -p
    CAUSAL_MODE "function"
    CAUSAL_ARGS -s 0,10,25,50,75
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME lulesh-line
    TARGET lulesh-omni
    RUN_ARGS -i 35 -s 50 -p
    CAUSAL_MODE "line"
    CAUSAL_ARGS -s 0,10,25,50,75 -S lulesh.cc
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

# set(_causal_e2e_exe_args 80 100 432525 100000000) set(_causal_e2e_exe_args 80 12 432525
# 500000000)
set(_causal_e2e_exe_args 80 50 432525 100000000)
set(_causal_common_args
    "-n 5 -e -s 0 10 20 30 -B $<TARGET_FILE_BASE_NAME:causal-cpu-omni>")

macro(
    causal_e2e_args_and_validation
    _NAME
    _TEST
    _MODE
    _EXPER
    _V10 # expected value for virtual speedup of 15
    _V20
    _V30
    _TOL # tolerance for virtual speedup
    )
    # arguments to rocprofiler-systems-causal
    set(${_NAME}_args "${_causal_common_args} ${_MODE} ${_EXPER}")

    # arguments to validate-causal-json.py
    set(${_NAME}_valid
        "-n 0 -i rocprof-sys-tests-output/causal-cpu-omni-${_TEST}-e2e/causal/experiments.json -v ${_EXPER} $<TARGET_FILE_BASE_NAME:causal-cpu-omni> 10 ${_V10} ${_TOL} ${_EXPER} $<TARGET_FILE_BASE_NAME:causal-cpu-omni> 20 ${_V20} ${_TOL} ${_EXPER} $<TARGET_FILE_BASE_NAME:causal-cpu-omni> 30 ${_V30} ${_TOL}"
        )

    # patch string for command-line
    string(REPLACE " " ";" ${_NAME}_args "${${_NAME}_args}")
    string(REPLACE " " ";" ${_NAME}_valid "${${_NAME}_valid}")
endmacro()

causal_e2e_args_and_validation(_causal_slow_func slow-func "-F" "cpu_slow_func" 10 20 20
                               5)
causal_e2e_args_and_validation(_causal_fast_func fast-func "-F" "cpu_fast_func" 0 0 0 5)
causal_e2e_args_and_validation(_causal_line_100 line-100 "-S" "causal.cpp:100" 10 20 20 5)
causal_e2e_args_and_validation(_causal_line_110 line-110 "-S" "causal.cpp:110" 0 0 0 5)

if(ROCPROFSYS_BUILD_NUMBER GREATER 1)
    set(_causal_e2e_environment)
else()
    set(_causal_e2e_environment "ROCPROFSYS_VERBOSE=0")
endif()

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-omni-slow-func-e2e
    TARGET causal-cpu-omni
    LABELS "causal-e2e"
    RUN_ARGS ${_causal_e2e_exe_args}
    CAUSAL_MODE "func"
    CAUSAL_ARGS ${_causal_slow_func_args}
    CAUSAL_VALIDATE_ARGS ${_causal_slow_func_valid}
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    ENVIRONMENT "${_causal_e2e_environment}"
    PROPERTIES PROCESSORS 2 PROCESSOR_AFFINITY OFF)

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-omni-fast-func-e2e
    TARGET causal-cpu-omni
    LABELS "causal-e2e"
    RUN_ARGS ${_causal_e2e_exe_args}
    CAUSAL_MODE "func"
    CAUSAL_ARGS ${_causal_fast_func_args}
    CAUSAL_VALIDATE_ARGS ${_causal_fast_func_valid}
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    ENVIRONMENT "${_causal_e2e_environment}"
    PROPERTIES PROCESSORS 2 PROCESSOR_AFFINITY OFF)

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-omni-line-100-e2e
    TARGET causal-cpu-omni
    LABELS "causal-e2e"
    RUN_ARGS ${_causal_e2e_exe_args}
    CAUSAL_MODE "line"
    CAUSAL_ARGS ${_causal_line_100_args}
    CAUSAL_VALIDATE_ARGS ${_causal_line_100_valid}
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    ENVIRONMENT "${_causal_e2e_environment}"
    PROPERTIES PROCESSORS 2 PROCESSOR_AFFINITY OFF)

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-omni-line-110-e2e
    TARGET causal-cpu-omni
    LABELS "causal-e2e"
    RUN_ARGS ${_causal_e2e_exe_args}
    CAUSAL_MODE "line"
    CAUSAL_ARGS ${_causal_line_110_args}
    CAUSAL_VALIDATE_ARGS ${_causal_line_110_valid}
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    ENVIRONMENT "${_causal_e2e_environment}"
    PROPERTIES PROCESSORS 2 PROCESSOR_AFFINITY OFF)
