# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -------------------------------------------------------------------------------------- #
#
# causal profiling tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_causal_test(
    NAME cpu-rocprofsys-func
    TARGET causal-cpu-rocprofsys
    RUN_ARGS 70 10 432525 1000000000
    CAUSAL_MODE "function"
    LABELS "theRock"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
)

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-rocprofsys-func-ndebug
    TARGET causal-cpu-rocprofsys-ndebug
    RUN_ARGS 70 10 432525 1000000000
    CAUSAL_MODE "function"
    LABELS "theRock"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
)

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-rocprofsys-line
    TARGET causal-cpu-rocprofsys
    RUN_ARGS 70 10 432525 1000000000
    CAUSAL_MODE "line"
    LABELS "theRock"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
)

rocprofiler_systems_add_causal_test(
    NAME both-rocprofsys-func
    TARGET causal-both-rocprofsys
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
        ${CMAKE_BINARY_DIR}/rocprof-sys-tests-config/causal-both-rocprofsys-func
        -l
        causal-both-rocprofsys
        -v
        3
        -b
        timer
    ENVIRONMENT "ROCPROFSYS_STRICT_CONFIG=OFF"
    LABELS "theRock"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
)

rocprofiler_systems_add_causal_test(
    NAME lulesh-func
    TARGET lulesh-rocprofsys
    RUN_ARGS -i 35 -s 50 -p
    CAUSAL_MODE "function"
    CAUSAL_ARGS -s 0,10,25,50,75
    LABELS "theRock"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
)

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME lulesh-func-ndebug
    TARGET lulesh-rocprofsys-ndebug
    RUN_ARGS -i 35 -s 50 -p
    CAUSAL_MODE "function"
    CAUSAL_ARGS -s 0,10,25,50,75
    LABELS "theRock"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
)

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME lulesh-line
    TARGET lulesh-rocprofsys
    RUN_ARGS -i 35 -s 50 -p
    CAUSAL_MODE "line"
    CAUSAL_ARGS -s 0,10,25,50,75 -S lulesh.cc
    LABELS "theRock"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
)

# set(_causal_e2e_exe_args 80 103 432525 100000000) set(_causal_e2e_exe_args 80 12 432525
# 500000000)
set(_causal_e2e_exe_args 80 50 432525 100000000)
set(_causal_common_args
    "-n 5 -e -s 0 10 20 30 -B $<TARGET_FILE_BASE_NAME:causal-cpu-rocprofsys>"
)

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

    # Determine the output path and binary name based on whether using installed tests
    if(ROCPROFSYS_INSTALL_TESTS)
        set(_causal_output_base "@ROCPROFSYS_TEST_OUTPUT@/rocprof-sys-tests-output")
        # For install tests, get the installed binary path and extract the basename
        rocprofiler_systems_get_installed_sample_path(_binary_path "causal-cpu-rocprofsys")
        get_filename_component(_binary_name "${_binary_path}" NAME)
    else()
        set(_causal_output_base "rocprof-sys-tests-output")
        set(_binary_name "$<TARGET_FILE_BASE_NAME:causal-cpu-rocprofsys>")
    endif()

    # arguments to validate-causal-json.py
    set(${_NAME}_valid
        "-n 0 -i ${_causal_output_base}/causal-cpu-rocprofsys-${_TEST}-e2e/causal/experiments.json -v ${_EXPER} ${_binary_name} 10 ${_V10} ${_TOL} ${_EXPER} ${_binary_name} 20 ${_V20} ${_TOL} ${_EXPER} ${_binary_name} 30 ${_V30} ${_TOL}"
    )

    # patch string for command-line
    string(REPLACE " " ";" ${_NAME}_args "${${_NAME}_args}")
    string(REPLACE " " ";" ${_NAME}_valid "${${_NAME}_valid}")
endmacro()

causal_e2e_args_and_validation(_causal_slow_func slow-func "-F" "cpu_slow_func" 10 20 20
                               5
)
causal_e2e_args_and_validation(_causal_fast_func fast-func "-F" "cpu_fast_func" 0 0 0 5)
causal_e2e_args_and_validation(_causal_line_103 line-103 "-S" "causal.cpp:103" 10 20 20 5)
causal_e2e_args_and_validation(_causal_line_113 line-113 "-S" "causal.cpp:113" 0 0 0 5)

if(ROCPROFSYS_BUILD_NUMBER GREATER 1)
    set(_causal_e2e_environment)
else()
    set(_causal_e2e_environment "ROCPROFSYS_VERBOSE=0")
endif()

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-rocprofsys-slow-func-e2e
    TARGET causal-cpu-rocprofsys
    LABELS "causal-e2e"
    RUN_ARGS ${_causal_e2e_exe_args}
    CAUSAL_MODE "func"
    CAUSAL_ARGS ${_causal_slow_func_args}
    CAUSAL_VALIDATE_ARGS ${_causal_slow_func_valid}
    LABELS "theRock"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    ENVIRONMENT "${_causal_e2e_environment}"
    PROPERTIES PROCESSORS 2 PROCESSOR_AFFINITY OFF
)

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-rocprofsys-fast-func-e2e
    TARGET causal-cpu-rocprofsys
    LABELS "causal-e2e"
    RUN_ARGS ${_causal_e2e_exe_args}
    CAUSAL_MODE "func"
    CAUSAL_ARGS ${_causal_fast_func_args}
    CAUSAL_VALIDATE_ARGS ${_causal_fast_func_valid}
    LABELS "theRock"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    ENVIRONMENT "${_causal_e2e_environment}"
    PROPERTIES PROCESSORS 2 PROCESSOR_AFFINITY OFF
)

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-rocprofsys-line-103-e2e
    TARGET causal-cpu-rocprofsys
    LABELS "causal-e2e"
    RUN_ARGS ${_causal_e2e_exe_args}
    CAUSAL_MODE "line"
    CAUSAL_ARGS ${_causal_line_103_args}
    CAUSAL_VALIDATE_ARGS ${_causal_line_103_valid}
    LABELS "theRock"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    ENVIRONMENT "${_causal_e2e_environment}"
    PROPERTIES PROCESSORS 2 PROCESSOR_AFFINITY OFF
)

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-rocprofsys-line-113-e2e
    TARGET causal-cpu-rocprofsys
    LABELS "causal-e2e"
    RUN_ARGS ${_causal_e2e_exe_args}
    CAUSAL_MODE "line"
    CAUSAL_ARGS ${_causal_line_113_args}
    CAUSAL_VALIDATE_ARGS ${_causal_line_113_valid}
    LABELS "theRock"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    ENVIRONMENT "${_causal_e2e_environment}"
    PROPERTIES PROCESSORS 2 PROCESSOR_AFFINITY OFF
)
