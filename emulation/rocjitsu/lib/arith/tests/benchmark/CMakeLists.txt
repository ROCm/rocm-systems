# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build the CSV microbenchmark and its sampling/output regression checks.

add_executable(
    rj_arith_benchmark
    benchmark.cpp
    sampling.cpp
    workloads.cpp
    wmma.cpp
    dot.cpp
    fp.cpp
    math.cpp
)
target_include_directories(
    rj_arith_benchmark
    PRIVATE ${rj_arith_source_dir}/tests
)
target_link_libraries(rj_arith_benchmark PRIVATE rj_arith)
rj_configure_target(rj_arith_benchmark WARNINGS)

# Smoke checks have no minimum duration and use fixed repetitions, never asserting speed.
if(RJ_INSTALL_TESTS)
    rj_install_test_target(rj_arith_benchmark)
endif()
rj_add_test(
    NAME "rj_arith.benchmark"
    COMMAND rj_arith_benchmark --min-ms 0 --repetitions 5
    INSTALL_COMMAND bin/rj_arith_benchmark --min-ms 0 --repetitions 5
)
rj_add_test(
    NAME "rj_arith.benchmark_output"
    COMMAND
        ${CMAKE_COMMAND}
        -DBENCHMARK=$<TARGET_FILE:rj_arith_benchmark>
        -DRJ_ARITH_INCLUDE=${rj_arith_source_dir}/include/rj_arith/detail
        -P
        ${CMAKE_CURRENT_SOURCE_DIR}/check_benchmark_output.cmake
    INSTALL_COMMAND
        ${CMAKE_COMMAND}
        -DBENCHMARK=bin/rj_arith_benchmark
        -DRJ_ARITH_INCLUDE=arith/include
        -P
        arith/check_benchmark_output.cmake
)

add_executable(
    rj_arith_benchmark_sampling_test
    sampling_test.cpp
    sampling.cpp
    workloads.cpp
)
target_include_directories(
    rj_arith_benchmark_sampling_test
    PRIVATE ${rj_arith_source_dir}/tests
)
target_link_libraries(
    rj_arith_benchmark_sampling_test
    PRIVATE GTest::gtest_main rj_arith
)
rj_add_gtest_tests(
    rj_arith_benchmark_sampling_test
    TEST_FILTER "*"
    TEST_PREFIX "rj_arith."
)

if(RJ_INSTALL_TESTS)
    file(
        APPEND "${RJ_INSTALL_CTEST_FILE}"
        "include(\"rj_arith_benchmark_sampling_test.cmake\")\n"
    )
    install(
        FILES check_benchmark_output.cmake
        DESTINATION ${RJ_INSTALL_TEST_DIR}/arith
    )
    foreach(category IN ITEMS wmma dot fp math)
        install(
            FILES
                ${rj_arith_source_dir}/include/rj_arith/detail/rj_arith_${category}.h
            DESTINATION ${RJ_INSTALL_TEST_DIR}/arith/include
        )
    endforeach()
endif()
