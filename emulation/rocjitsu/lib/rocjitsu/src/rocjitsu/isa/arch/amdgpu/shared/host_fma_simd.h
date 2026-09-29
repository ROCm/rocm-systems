// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/simd.h"

#include <bit>
#include <type_traits>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace rocjitsu::amdgpu::host_fma {

/// Raw host arithmetic only. The caller establishes rounding/flush controls
/// and retains the architectural NaN, denormal and boundary repairs.
inline bool available() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  // The FMA feature bit alone does not prove that the OS saves AVX state.
  return __builtin_cpu_supports("avx") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
// Fixed 128-bit arguments retain the baseline x86-64 ABI across the target
// boundary. No AVX instruction executes before the runtime capability check.
[[gnu::target("fma"), gnu::noinline]] inline __m128 f32x4(__m128 a, __m128 b, __m128 c) {
  return _mm_fmadd_ps(a, b, c);
}

[[gnu::target("fma"), gnu::noinline]] inline __m128d f64x2(__m128d a, __m128d b, __m128d c) {
  return _mm_fmadd_pd(a, b, c);
}
#endif

template <typename Simd> inline Simd fused(Simd a, Simd b, Simd c) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if constexpr (sizeof(Simd) == 16 && std::is_trivially_copyable_v<Simd>) {
    if (available()) {
      if constexpr (std::is_same_v<typename Simd::value_type, float>)
        return std::bit_cast<Simd>(
            f32x4(std::bit_cast<__m128>(a), std::bit_cast<__m128>(b), std::bit_cast<__m128>(c)));
      else if constexpr (std::is_same_v<typename Simd::value_type, double>)
        return std::bit_cast<Simd>(
            f64x2(std::bit_cast<__m128d>(a), std::bit_cast<__m128d>(b), std::bit_cast<__m128d>(c)));
    }
  }
#endif
  return util::stdx::fma(a, b, c);
}

} // namespace rocjitsu::amdgpu::host_fma
