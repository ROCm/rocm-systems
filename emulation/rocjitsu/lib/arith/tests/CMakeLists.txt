# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build instruction tests and standalone public-header checks.

include(GoogleTest)

function(rj_arith_add_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE rj_arith GTest::gtest_main)
    target_include_directories(
        ${target}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${rj_arith_source_dir}/src
    )
    rj_configure_target(${target} WARNINGS)
    rj_add_gtest_tests(${target} TEST_PREFIX "rj_arith.")
    if(RJ_INSTALL_TESTS)
        file(APPEND "${RJ_INSTALL_CTEST_FILE}" "include(\"${target}.cmake\")\n")
    endif()
endfunction()

rj_arith_add_test(rj_arith_v_wmma_f32_16x16x16_f16_w32_test wmma/v_wmma_f32_16x16x16_f16_w32_test.cpp)
rj_arith_add_test(rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12_test wmma/v_wmma_f32_16x16x16_f16_w32_gfx12_test.cpp)
rj_arith_add_test(rj_arith_v_wmma_f32_16x16x16_f16_w64_test wmma/v_wmma_f32_16x16x16_f16_w64_test.cpp)
rj_arith_add_test(rj_arith_v_wmma_f32_16x16x16_f16_w64_gfx12_test wmma/v_wmma_f32_16x16x16_f16_w64_gfx12_test.cpp)
rj_arith_add_test(rj_arith_v_wmma_f32_16x16x16_bf16_w32_test wmma/v_wmma_f32_16x16x16_bf16_w32_test.cpp)
rj_arith_add_test(rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12_test wmma/v_wmma_f32_16x16x16_bf16_w32_gfx12_test.cpp)
rj_arith_add_test(rj_arith_v_wmma_f32_16x16x16_bf16_w64_test wmma/v_wmma_f32_16x16x16_bf16_w64_test.cpp)
rj_arith_add_test(rj_arith_v_wmma_f32_16x16x16_bf16_w64_gfx12_test wmma/v_wmma_f32_16x16x16_bf16_w64_gfx12_test.cpp)
rj_arith_add_test(rj_arith_v_fma_test fp/v_fma_test.cpp)
rj_arith_add_test(rj_arith_cpu_test common/cpu_test.cpp)
add_subdirectory(headers)
add_subdirectory(cpuinfo)
