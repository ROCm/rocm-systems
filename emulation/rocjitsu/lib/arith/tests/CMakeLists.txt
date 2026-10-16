# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Define instruction-family test binaries and discover their individual GoogleTest cases.
# Shared test utilities, standalone header checks, and cpuinfo are wired here.

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

# One explicit source, executable and arithmetic test per instruction.
foreach(
    instruction
    IN
    ITEMS
        v_swmmac_bf16_16x16x32_bf16_w32
        v_swmmac_bf16_16x16x32_bf16_w64
        v_swmmac_f16_16x16x32_f16_w32
        v_swmmac_f16_16x16x32_f16_w64
        v_swmmac_f32_16x16x32_bf16_w32
        v_swmmac_f32_16x16x32_bf16_w64
        v_swmmac_f32_16x16x32_bf8_bf8_w32
        v_swmmac_f32_16x16x32_bf8_bf8_w64
        v_swmmac_f32_16x16x32_bf8_fp8_w32
        v_swmmac_f32_16x16x32_bf8_fp8_w64
        v_swmmac_f32_16x16x32_f16_w32
        v_swmmac_f32_16x16x32_f16_w64
        v_swmmac_f32_16x16x32_fp8_bf8_w32
        v_swmmac_f32_16x16x32_fp8_bf8_w64
        v_swmmac_f32_16x16x32_fp8_fp8_w32
        v_swmmac_f32_16x16x32_fp8_fp8_w64
        v_swmmac_i32_16x16x32_iu4_w32
        v_swmmac_i32_16x16x32_iu4_w64
        v_swmmac_i32_16x16x32_iu8_w32
        v_swmmac_i32_16x16x32_iu8_w64
        v_swmmac_i32_16x16x64_iu4_w32
        v_swmmac_i32_16x16x64_iu4_w64
        v_wmma_bf16_16x16x16_bf16_w32
        v_wmma_bf16_16x16x16_bf16_w32_gfx12
        v_wmma_bf16_16x16x16_bf16_w64
        v_wmma_bf16_16x16x16_bf16_w64_gfx12
        v_wmma_f16_16x16x16_f16_w32
        v_wmma_f16_16x16x16_f16_w32_gfx12
        v_wmma_f16_16x16x16_f16_w64
        v_wmma_f16_16x16x16_f16_w64_gfx12
        v_wmma_f32_16x16x16_bf16_w32
        v_wmma_f32_16x16x16_bf16_w32_gfx12
        v_wmma_f32_16x16x16_bf16_w64
        v_wmma_f32_16x16x16_bf16_w64_gfx12
        v_wmma_f32_16x16x16_bf8_bf8_w32_gfx12
        v_wmma_f32_16x16x16_bf8_bf8_w64_gfx12
        v_wmma_f32_16x16x16_bf8_fp8_w32_gfx12
        v_wmma_f32_16x16x16_bf8_fp8_w64_gfx12
        v_wmma_f32_16x16x16_f16_w32
        v_wmma_f32_16x16x16_f16_w32_gfx12
        v_wmma_f32_16x16x16_f16_w64
        v_wmma_f32_16x16x16_f16_w64_gfx12
        v_wmma_f32_16x16x16_fp8_bf8_w32_gfx12
        v_wmma_f32_16x16x16_fp8_bf8_w64_gfx12
        v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12
        v_wmma_f32_16x16x16_fp8_fp8_w64_gfx12
        v_wmma_i32_16x16x16_iu4_w32
        v_wmma_i32_16x16x16_iu4_w32_gfx12
        v_wmma_i32_16x16x16_iu4_w64
        v_wmma_i32_16x16x16_iu4_w64_gfx12
        v_wmma_i32_16x16x16_iu8_w32
        v_wmma_i32_16x16x16_iu8_w32_gfx12
        v_wmma_i32_16x16x16_iu8_w64
        v_wmma_i32_16x16x16_iu8_w64_gfx12
        v_wmma_i32_16x16x32_iu4_w32_gfx12
        v_wmma_i32_16x16x32_iu4_w64_gfx12
)
    string(REGEX REPLACE "^v_" "" test_name "${instruction}")
    set(target "rj_arith_${test_name}_test")
    rj_arith_add_test(${target} wmma/${instruction}_test.cpp)
endforeach()

# Group related operations, element types, wave widths and architectures; retain one test per entry point.
foreach(
    mnemonic
    IN
    ITEMS
        v_round_f64
        v_fma_dx9_zero
        v_fma
        v_fma_mix
        v_fmaak
        v_fmac
        v_fmamk
        v_pk_fma_f16
        v_pk_fmac_f16
        v_rcp_rsq_sqrt_f64
)
    string(REGEX REPLACE "^v_" "" test_name "${mnemonic}")
    set(target "rj_arith_${test_name}_test")
    rj_arith_add_test(${target} fp/${mnemonic}_test.cpp)
endforeach()

# Infrastructure has no all-entry-point registry; contracts live in family helpers.
rj_arith_add_test(rj_arith_cpu_test common/cpu_test.cpp)
add_subdirectory(headers)

# Standalone utilities.
add_subdirectory(cpuinfo)

if(UNIX AND CMAKE_NM AND RJ_ARITH_ENABLE_X86_64)
    add_test(
        NAME rj_arith.baseline_linkage
        COMMAND
            ${CMAKE_COMMAND} -DNM=${CMAKE_NM} -DARCHIVE=$<TARGET_FILE:rj_arith>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/common/baseline_linkage.cmake
    )
endif()
