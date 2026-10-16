# Check x86 compiler support and assign ISA flags to the matching backend sources.
# Keep the baseline dispatch code independent of these optional instruction sets.

include(CheckCXXCompilerFlag)

set(rj_arith_x86_64_v3_flags "-march=x86-64-v3")
set(rj_arith_x86_64_v4_flags "-march=x86-64-v4")
set(rj_arith_avx512vnni_flags ${rj_arith_x86_64_v4_flags} "-mavx512vnni")
set(rj_arith_avx512bf16_flags ${rj_arith_x86_64_v4_flags} "-mavx512bf16")

# Format the feature flag lists as command-line strings for the compiler checks.
list(JOIN rj_arith_avx512vnni_flags " " rj_arith_avx512vnni_check_flags)
list(JOIN rj_arith_avx512bf16_flags " " rj_arith_avx512bf16_check_flags)

check_cxx_compiler_flag("${rj_arith_x86_64_v3_flags}" RJ_ARITH_COMPILER_X86_64_V3)
check_cxx_compiler_flag("${rj_arith_x86_64_v4_flags}" RJ_ARITH_COMPILER_X86_64_V4)
check_cxx_compiler_flag("${rj_arith_avx512bf16_check_flags}" RJ_ARITH_COMPILER_AVX512BF16)
check_cxx_compiler_flag("${rj_arith_avx512vnni_check_flags}" RJ_ARITH_COMPILER_AVX512VNNI)

set(rj_arith_x86_64_v3_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/fp/x86_64/half_fma_exact_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp/x86_64/binary_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp/x86_64/fma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp/x86_64/fma_exact_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp/x86_64/fp64_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp/x86_64/frexp_exp_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp/x86_64/half_binary_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp/x86_64/half_exponent_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp/x86_64/half_fma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp/x86_64/ldexp_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp/x86_64/mixed_fma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp/x86_64/packed_fma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/register/x86_64/dpp_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/wmma/x86_64/fp8_wmma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/wmma/x86_64/integer_wmma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/wmma/x86_64/swmmac_float_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/wmma/x86_64/swmmac_integer_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/wmma/x86_64/wmma_exact_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/wmma/x86_64/wmma_x86_64_v3.cpp
)

set(rj_arith_x86_64_v4_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/wmma/x86_64/wmma_exact_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common/x86_64/exact_arithmetic_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp/x86_64/fma_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp/x86_64/ldexp_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/register/x86_64/dpp_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/wmma/x86_64/swmmac_float_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/wmma/x86_64/swmmac_integer_x86_64_v4.cpp
)

set(rj_arith_avx512bf16_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/wmma/x86_64/wmma_avx512bf16.cpp
)

set(rj_arith_avx512vnni_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/wmma/x86_64/integer_wmma_avx512vnni.cpp
)

# Keep source selection, compile flags and feature definitions together for each ISA.
function(rj_arith_add_x86_64_backend feature)
  if(NOT RJ_ARITH_COMPILER_${feature})
    return()
  endif()
  string(TOLOWER "${feature}" backend)
  target_sources(rj_arith PRIVATE ${rj_arith_${backend}_sources})
  set_source_files_properties(${rj_arith_${backend}_sources}
    TARGET_DIRECTORY rj_arith
    PROPERTIES COMPILE_OPTIONS "${rj_arith_${backend}_flags}"
  )
  target_compile_definitions(rj_arith PRIVATE RJ_ARITH_HAVE_${feature}=1)
endfunction()

foreach(feature IN ITEMS X86_64_V3 X86_64_V4 AVX512BF16 AVX512VNNI)
  rj_arith_add_x86_64_backend(${feature})
endforeach()
