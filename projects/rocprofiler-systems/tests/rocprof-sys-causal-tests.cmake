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
# causal profiling tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_causal_test(
    NAME cpu-rocprofsys-func
    TARGET causal-cpu-rocprofsys
    RUN_ARGS 70 10 432525 1000000000
    CAUSAL_MODE "function"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
)

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-rocprofsys-func-ndebug
    TARGET causal-cpu-rocprofsys-ndebug
    RUN_ARGS 70 10 432525 1000000000
    CAUSAL_MODE "function"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
)

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-rocprofsys-line
    TARGET causal-cpu-rocprofsys
    RUN_ARGS 70 10 432525 1000000000
    CAUSAL_MODE "line"
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
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
)

rocprofiler_systems_add_causal_test(
    NAME lulesh-func
    TARGET lulesh-rocprofsys
    RUN_ARGS -i 35 -s 50 -p
    CAUSAL_MODE "function"
    CAUSAL_ARGS -s 0,10,25,50,75
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
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
)

# set(_causal_e2e_exe_args 80 100 432525 100000000) set(_causal_e2e_exe_args 80 12 432525
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

    # arguments to validate-causal-json.py
    set(${_NAME}_valid
        "-n 0 -i rocprof-sys-tests-output/causal-cpu-rocprofsys-${_TEST}-e2e/causal/experiments.json -v ${_EXPER} $<TARGET_FILE_BASE_NAME:causal-cpu-rocprofsys> 10 ${_V10} ${_TOL} ${_EXPER} $<TARGET_FILE_BASE_NAME:causal-cpu-rocprofsys> 20 ${_V20} ${_TOL} ${_EXPER} $<TARGET_FILE_BASE_NAME:causal-cpu-rocprofsys> 30 ${_V30} ${_TOL}"
    )

    # patch string for command-line
    string(REPLACE " " ";" ${_NAME}_args "${${_NAME}_args}")
    string(REPLACE " " ";" ${_NAME}_valid "${${_NAME}_valid}")
endmacro()

causal_e2e_args_and_validation(_causal_slow_func slow-func "-F" "cpu_slow_func" 10 20 20
                               5
)
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
    NAME cpu-rocprofsys-slow-func-e2e
    TARGET causal-cpu-rocprofsys
    LABELS "causal-e2e"
    RUN_ARGS ${_causal_e2e_exe_args}
    CAUSAL_MODE "func"
    CAUSAL_ARGS ${_causal_slow_func_args}
    CAUSAL_VALIDATE_ARGS ${_causal_slow_func_valid}
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
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    ENVIRONMENT "${_causal_e2e_environment}"
    PROPERTIES PROCESSORS 2 PROCESSOR_AFFINITY OFF
)

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-rocprofsys-line-100-e2e
    TARGET causal-cpu-rocprofsys
    LABELS "causal-e2e"
    RUN_ARGS ${_causal_e2e_exe_args}
    CAUSAL_MODE "line"
    CAUSAL_ARGS ${_causal_line_100_args}
    CAUSAL_VALIDATE_ARGS ${_causal_line_100_valid}
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    ENVIRONMENT "${_causal_e2e_environment}"
    PROPERTIES PROCESSORS 2 PROCESSOR_AFFINITY OFF
)

rocprofiler_systems_add_causal_test(
    SKIP_BASELINE
    NAME cpu-rocprofsys-line-110-e2e
    TARGET causal-cpu-rocprofsys
    LABELS "causal-e2e"
    RUN_ARGS ${_causal_e2e_exe_args}
    CAUSAL_MODE "line"
    CAUSAL_ARGS ${_causal_line_110_args}
    CAUSAL_VALIDATE_ARGS ${_causal_line_110_valid}
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    ENVIRONMENT "${_causal_e2e_environment}"
    PROPERTIES PROCESSORS 2 PROCESSOR_AFFINITY OFF
)
