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
    v_div_fixup
    v_div_fmas
    v_div_scale
    v_fma_dx9_zero
    v_fma
    v_fma_mix
    v_fmaak
    v_fmac
    v_fmamk
    v_exponent
    v_min_max_med3
    v_min_max_med3_num
    v_min_max
    v_min_max_num
    v_minimum_maximum3
    v_minimum_maximum
    v_minimummaximum_maximumminimum
    v_minmax_maxmin
    v_minmax_maxmin_num
    v_mul_dx9_zero_f32
    v_mul
    v_pk_add_mul_f16
    v_pk_fma_f16
    v_pk_fmac_f16
    v_pk_selection_f16
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
foreach(family IN ITEMS
    constant
    integer_equality
    integer_ordering
    class
    float_equality
    float_ordered
    float_unordered
)
  set(target "rj_arith_compare_${family}_test")
  add_executable(${target} compare/${family}_test.cpp)
  target_link_libraries(${target} PRIVATE rj_arith GTest::gtest_main)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
  endif()
  gtest_discover_tests(${target} TEST_PREFIX "rj_arith.")
endforeach()

# Integer families share references and contract checks, with explicit API tests.
foreach(family IN ITEMS
    half_binary
    half_ternary
    add
    multiply
    select
    combined
    bitfield
    bit_count
    boolean
    mad
    packed_binary
    packed_mad
    shift
    sad
    carry
    mad64
)
  set(target "rj_arith_integer_${family}_test")
  add_executable(${target} integer/${family}_test.cpp)
  target_link_libraries(${target} PRIVATE rj_arith GTest::gtest_main)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
  endif()
  gtest_discover_tests(${target} TEST_PREFIX "rj_arith.")
endforeach()

# Conversion layouts share contract checks; family references remain independent.
foreach(family IN ITEMS
    integer_pack
    byte
    float16
    float32
    float64
    normalized
    packed
    fp8_widen
    fp8_narrow
)
  set(target "rj_arith_conversion_${family}_test")
  add_executable(${target} conversion/${family}_test.cpp)
  target_link_libraries(${target} PRIVATE rj_arith GTest::gtest_main)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
  endif()
  gtest_discover_tests(${target} TEST_PREFIX "rj_arith.")
endforeach()

# Dot-product formats retain family-specific arithmetic and status expectations.
foreach(family IN ITEMS
    integer
    fp8
    half
    dot2
)
  set(target "rj_arith_dot_${family}_test")
  add_executable(${target} dot/${family}_test.cpp)
  target_link_libraries(${target} PRIVATE rj_arith GTest::gtest_main)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
  endif()
  gtest_discover_tests(${target} TEST_PREFIX "rj_arith.")
endforeach()

# Register-only operations share routing and preservation checks.
foreach(family IN ITEMS
    move
    swap
    ds
    permlane
    relative
    lane
)
  set(target "rj_arith_register_${family}_test")
  add_executable(${target} register/${family}_test.cpp)
  target_link_libraries(${target} PRIVATE rj_arith GTest::gtest_main)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
  endif()
  gtest_discover_tests(${target} TEST_PREFIX "rj_arith.")
endforeach()

# Graphics families retain their independent interpolation and selection references.
foreach(family IN ITEMS
    cube
    interp
    mullit
)
  set(target "rj_arith_graphics_${family}_test")
  add_executable(${target} graphics/${family}_test.cpp)
  target_link_libraries(${target} PRIVATE rj_arith GTest::gtest_main)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
  endif()
  gtest_discover_tests(${target} TEST_PREFIX "rj_arith.")
endforeach()

# VOPD wrappers retain the ordinary instructions' arithmetic and status contracts.
foreach(family IN ITEMS
    alu
    fused
    dot
)
  set(target "rj_arith_dual_${family}_test")
  add_executable(${target} dual/${family}_test.cpp)
  target_link_libraries(${target} PRIVATE rj_arith GTest::gtest_main)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
  endif()
  gtest_discover_tests(${target} TEST_PREFIX "rj_arith.")
endforeach()

# Scalar families keep required SCC/EXEC results distinct from FP status.
foreach(family IN ITEMS
    boolean
    shift
    move
    field
    count
    pack
    arithmetic
    compare_integer
    compare_float
    exec
    fp
    round
    conversion
    state
    relative
    pseudo
)
  set(target "rj_arith_scalar_${family}_test")
  add_executable(${target} scalar/${family}_test.cpp)
  target_link_libraries(${target} PRIVATE rj_arith GTest::gtest_main)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
  endif()
  gtest_discover_tests(${target} TEST_PREFIX "rj_arith.")
endforeach()

# Math families share unary layouts, with independent loose and exact references.
foreach(family IN ITEMS
    round
    root_reciprocal
    log_exp
    decompose
    trig
    rcp_iflag
    preop
)
  set(target "rj_arith_math_${family}_test")
  add_executable(${target} math/${family}_test.cpp)
  target_link_libraries(${target} PRIVATE rj_arith GTest::gtest_main)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
  endif()
  gtest_discover_tests(${target} TEST_PREFIX "rj_arith.")
endforeach()

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
