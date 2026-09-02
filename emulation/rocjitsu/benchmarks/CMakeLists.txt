# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# BUILD_TESTING enters this directory only to register the stdlib-only runner
# tests. Keep every heavyweight benchmark dependency behind
# RJ_BUILD_BENCHMARKS so a normal rocjitsu build remains unchanged.

if(BUILD_TESTING)
    find_package(Python3 3.11 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME RocjitsuBenchmarkRunnerTest
            COMMAND
                ${Python3_EXECUTABLE} -m unittest discover -s
                ${CMAKE_CURRENT_SOURCE_DIR}/tests -v
        )
        set_tests_properties(
            RocjitsuBenchmarkRunnerTest
            PROPERTIES
                LABELS "benchmark;unit"
                TIMEOUT 30
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
        )
    else()
        message(
            STATUS
            "Skipping benchmark runner tests (Python 3.11 or newer not found)"
        )
    endif()
endif()

if(NOT RJ_BUILD_BENCHMARKS)
    return()
endif()

foreach(_package IN ITEMS hip hipblas-common hipblaslt)
    # ROCM_PATH is the authoritative SDK for benchmark builds. Discard stale
    # package-cache entries so a reused build tree cannot mix ROCm installs.
    unset("${_package}_DIR")
    unset("${_package}_DIR" CACHE)
    find_package(
        ${_package}
        CONFIG
        REQUIRED
        PATHS "${ROCM_PATH}"
        NO_DEFAULT_PATH
    )
endforeach()
unset(_package)

add_executable(
    rocjitsu_benchmark_hipblaslt
    src/hipblaslt_workloads.cpp
)
target_compile_features(rocjitsu_benchmark_hipblaslt PRIVATE cxx_std_20)
target_compile_options(
    rocjitsu_benchmark_hipblaslt
    PRIVATE -Wall -Wextra -Wpedantic
)
target_link_libraries(
    rocjitsu_benchmark_hipblaslt
    PRIVATE hip::host roc::hipblaslt
)
set_target_properties(
    rocjitsu_benchmark_hipblaslt
    PROPERTIES
        OUTPUT_NAME rocjitsu-benchmark-hipblaslt
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

add_custom_target(rocjitsu_benchmark_workloads ALL)
add_dependencies(
    rocjitsu_benchmark_workloads
    rocjitsu_benchmark_hipblaslt
    rocjitsu_bin
    rocjitsu_shared
)
