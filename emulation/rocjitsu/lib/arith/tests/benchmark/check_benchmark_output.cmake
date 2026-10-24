# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Smoke-test benchmark CSV and command-line rejection without collecting performance results.

cmake_minimum_required(VERSION 3.20)

# No timing assertions: validate CSV and exhaustive API coverage.
execute_process(
    COMMAND "${BENCHMARK}" --min-ms 0 --repetitions 3
    RESULT_VARIABLE status
    OUTPUT_VARIABLE csv
    ERROR_VARIABLE error
)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "Benchmark failed: ${error}")
endif()
string(REPLACE "\n" ";" lines "${csv}")
list(POP_FRONT lines header)
if(
    NOT header
        STREQUAL
        "Category,Instruction,Wave,Semantics,CPU path,Variant,Non-finite inputs,ns/wave,Speedup"
)
    message(FATAL_ERROR "Invalid CSV header: ${header}")
endif()
set(actual)
foreach(line IN LISTS lines)
    if(line STREQUAL "")
        continue()
    endif()
    # Only Variant may contain quoted commas. Replace quoted fields before counting.
    string(REGEX REPLACE "\"([^\"]|\"\")*\"" "quoted-field" unquoted "${line}")
    string(REPLACE "," ";" fields "${unquoted}")
    list(LENGTH fields count)
    if(NOT count EQUAL 9)
        message(FATAL_ERROR "Invalid field count (${count}): ${line}")
    endif()
    list(GET fields 0 category)
    list(GET fields 1 instruction)
    list(GET fields 4 cpu)
    list(GET fields 8 speedup)
    if(cpu STREQUAL "scalar" AND NOT speedup STREQUAL "")
        message(FATAL_ERROR "Scalar rows must leave speedup empty: ${line}")
    endif()
    list(GET fields 6 non_finite)
    if(non_finite STREQUAL "one NaN lane")
        list(APPEND nan_rows "${instruction}")
    elseif(NOT non_finite STREQUAL "")
        message(FATAL_ERROR "Unexpected non-finite inputs: ${line}")
    else()
        list(APPEND finite_rows "${instruction}")
    endif()
    list(APPEND actual "rj_arith_${instruction}")
    if(NOT category MATCHES "^(wmma|dot|fp|math)$")
        message(FATAL_ERROR "Unexpected category: ${category}")
    endif()
    list(GET fields 7 median)
    if(NOT median MATCHES "^[0-9]+\\.[0-9]$")
        message(FATAL_ERROR "CSV must contain only measured results: ${line}")
    endif()
endforeach()
foreach(symbol IN LISTS actual)
    string(REGEX REPLACE "^rj_arith_" "" instruction "${symbol}")
    if(NOT instruction IN_LIST finite_rows)
        message(FATAL_ERROR "Missing finite inputs: ${instruction}")
    endif()
    if(instruction MATCHES "_(f16|bf16|f32|f64|fp8|bf8)_")
        if(NOT instruction IN_LIST nan_rows)
            message(FATAL_ERROR "Missing NaN inputs: ${instruction}")
        endif()
    elseif(instruction IN_LIST nan_rows)
        message(
            FATAL_ERROR
            "Integer instruction has NaN inputs: ${instruction}"
        )
    endif()
endforeach()
set(expected)
foreach(category wmma dot fp math)
    file(
        STRINGS "${RJ_ARITH_INCLUDE}/rj_arith_${category}.h"
        declarations
        REGEX "^int rj_arith_"
    )
    foreach(declaration IN LISTS declarations)
        string(REGEX MATCH "rj_arith_[a-zA-Z0-9_]+" symbol "${declaration}")
        list(APPEND expected "${symbol}")
    endforeach()
endforeach()
foreach(list_name expected actual)
    list(REMOVE_DUPLICATES ${list_name})
    list(SORT ${list_name})
endforeach()
if(NOT actual STREQUAL expected)
    message(
        FATAL_ERROR
        "Benchmark must measure every WMMA, DOT, FP and math entry point and no others"
    )
endif()
foreach(
    arguments
    IN
    ITEMS
        "--csv"
        "1"
        "--unknown;1"
        "--min-ms"
        "--min-ms;-1"
        "--min-ms;nan"
        "--min-ms;inf"
        "--noise-threshold;0.01"
        "--min-repetitions;5"
        "--max-repetitions;100"
        "--repetitions;0"
        "--repetitions;-1"
        "--repetitions;1.5"
        "--repetitions;2147483648"
        "--min-ms;0;--min-ms;0"
)
    execute_process(
        COMMAND "${BENCHMARK}" ${arguments}
        RESULT_VARIABLE status
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT status EQUAL 2 OR NOT output STREQUAL "")
        message(
            FATAL_ERROR
            "Invalid arguments were not rejected cleanly: ${arguments}"
        )
    endif()
endforeach()
