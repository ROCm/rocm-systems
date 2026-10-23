# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build instruction tests and standalone public-header checks.

include(GoogleTest)

add_executable(
    rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12_test
    wmma/v_wmma_f32_16x16x16_f16_w32_gfx12_test.cpp
)
target_link_libraries(
    rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12_test
    PRIVATE rj_arith GTest::gtest_main
)
target_include_directories(
    rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${rj_arith_source_dir}/src
)
rj_configure_target(rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12_test WARNINGS)
rj_add_gtest_tests(
    rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12_test
    TEST_PREFIX "rj_arith."
)

add_executable(
    rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12_test
    wmma/v_wmma_f32_16x16x16_bf16_w32_gfx12_test.cpp
)
target_link_libraries(
    rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12_test
    PRIVATE rj_arith GTest::gtest_main
)
target_include_directories(
    rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${rj_arith_source_dir}/src
)
rj_configure_target(rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12_test WARNINGS)
rj_add_gtest_tests(
    rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12_test
    TEST_PREFIX "rj_arith."
)

add_executable(rj_arith_v_fma_test fp/v_fma_test.cpp)
target_link_libraries(rj_arith_v_fma_test PRIVATE rj_arith GTest::gtest_main)
target_include_directories(
    rj_arith_v_fma_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${rj_arith_source_dir}/src
)
rj_configure_target(rj_arith_v_fma_test WARNINGS)
rj_add_gtest_tests(rj_arith_v_fma_test TEST_PREFIX "rj_arith.")

add_executable(rj_arith_cpu_test common/cpu_test.cpp)
target_link_libraries(rj_arith_cpu_test PRIVATE rj_arith GTest::gtest_main)
target_include_directories(
    rj_arith_cpu_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${rj_arith_source_dir}/src
)
rj_configure_target(rj_arith_cpu_test WARNINGS)
rj_add_gtest_tests(rj_arith_cpu_test TEST_PREFIX "rj_arith.")

add_subdirectory(headers)
add_subdirectory(cpuinfo)

if(RJ_INSTALL_TESTS)
    foreach(
        target
        IN
        ITEMS
            rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12_test
            rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12_test
            rj_arith_v_fma_test
            rj_arith_cpu_test
    )
        file(APPEND "${RJ_INSTALL_CTEST_FILE}" "include(\"${target}.cmake\")\n")
    endforeach()
endif()
