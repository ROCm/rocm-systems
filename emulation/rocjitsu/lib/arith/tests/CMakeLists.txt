# Define instruction-family test binaries and discover their individual GoogleTest cases.
# Shared test utilities, standalone header checks, cpuinfo, and benchmarks are wired here.

include(GoogleTest)

# One explicit source, executable and arithmetic test per instruction.
foreach(instruction IN ITEMS
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
    v_wmma_f32_16x16x16_bf8_bf8_w32
    v_wmma_f32_16x16x16_bf8_bf8_w64
    v_wmma_f32_16x16x16_bf8_fp8_w32
    v_wmma_f32_16x16x16_bf8_fp8_w64
    v_wmma_f32_16x16x16_f16_w32
    v_wmma_f32_16x16x16_f16_w32_gfx12
    v_wmma_f32_16x16x16_f16_w64
    v_wmma_f32_16x16x16_f16_w64_gfx12
    v_wmma_f32_16x16x16_fp8_bf8_w32
    v_wmma_f32_16x16x16_fp8_bf8_w64
    v_wmma_f32_16x16x16_fp8_fp8_w32
    v_wmma_f32_16x16x16_fp8_fp8_w64
    v_wmma_i32_16x16x16_iu4_w32
    v_wmma_i32_16x16x16_iu4_w32_gfx12
    v_wmma_i32_16x16x16_iu4_w64
    v_wmma_i32_16x16x16_iu4_w64_gfx12
    v_wmma_i32_16x16x16_iu8_w32
    v_wmma_i32_16x16x16_iu8_w32_gfx12
    v_wmma_i32_16x16x16_iu8_w64
    v_wmma_i32_16x16x16_iu8_w64_gfx12
    v_wmma_i32_16x16x32_iu4_w32
    v_wmma_i32_16x16x32_iu4_w64
)
  string(REGEX REPLACE "^v_" "" test_name "${instruction}")
  set(target "rj_arith_${test_name}_test")
  add_executable(${target} wmma/${instruction}_test.cpp)
  target_link_libraries(${target} PRIVATE rj_arith GTest::gtest_main)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
  endif()
  gtest_discover_tests(${target} TEST_PREFIX "rj_arith.")
endforeach()

# Group related operations, element types, wave widths and architectures; retain one test per entry point.
foreach(mnemonic IN ITEMS
    v_add_sub
    v_round_f64
    v_fma_dx9_zero
    v_fma
    v_fma_mix
    v_fmaak
    v_fmac
    v_fmamk
    v_exponent
    v_mul_dx9_zero_f32
    v_mul
    v_pk_add_mul_f16
    v_pk_fma_f16
    v_pk_fmac_f16
    v_rcp_rsq_sqrt_f64
)
  string(REGEX REPLACE "^v_" "" test_name "${mnemonic}")
  set(target "rj_arith_${test_name}_test")
  add_executable(${target} fp/${mnemonic}_test.cpp)
  target_link_libraries(${target} PRIVATE rj_arith GTest::gtest_main)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
  endif()
  gtest_discover_tests(${target} TEST_PREFIX "rj_arith.")
endforeach()

# Related comparison predicates share a binary; every entry point has its own test.


# Integer families share references and contract checks, with explicit API tests.


# Conversion layouts share contract checks; family references remain independent.


# Dot-product formats retain family-specific arithmetic and status expectations.


# Register-only operations share routing and preservation checks.


# Graphics families retain their independent interpolation and selection references.


# VOPD wrappers retain the ordinary instructions' arithmetic and status contracts.


# Scalar families keep required SCC/EXEC results distinct from FP status.


# Math families share unary layouts, with independent loose and exact references.


# Infrastructure has no all-entry-point registry; contracts live in family helpers.
add_executable(rj_arith_cpu_test common/cpu_test.cpp)
target_link_libraries(rj_arith_cpu_test PRIVATE rj_arith GTest::gtest_main)
target_include_directories(rj_arith_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${rj_arith_source_dir}/src)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(rj_arith_cpu_test PRIVATE -Wall -Wextra -Werror -pedantic)
endif()
gtest_discover_tests(rj_arith_cpu_test TEST_PREFIX "rj_arith.")
add_subdirectory(headers)

# Standalone utilities.
add_subdirectory(benchmark)
add_subdirectory(cpuinfo)
