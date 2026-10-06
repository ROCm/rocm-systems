// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_SIMD_X86_AVX512_H_
#define UTIL_DETAIL_SIMD_X86_AVX512_H_

#include "util/detail/simd/types.h"

#include <bit>
#include <cstdint>

#include <immintrin.h>

namespace util {

/// Fused multiply-add for native f32 SIMD. GCC's experimental::simd FMA
/// customization point does not inline on AVX-512 and otherwise emits a call
/// plus a ZMM spill/reload. Keep that toolchain workaround in the shared SIMD
/// layer; other targets use the portable TS operation.
[[gnu::always_inline]] inline native<float> native_fma(native<float> a, native<float> b,
                                                       native<float> c) {
  static_assert(native<float>::size() == 16);
  return std::bit_cast<native<float>>(_mm512_fmadd_ps(
      std::bit_cast<__m512>(a), std::bit_cast<__m512>(b), std::bit_cast<__m512>(c)));
}

/// Expand the low native<uint32_t>::size() bits into 0/1 uint32_t lanes. This
/// is the numeric form needed by carry-in arithmetic; selection paths should
/// use simd_mask_from_bits directly. AVX-512 materializes the lanes with a
/// k-mask zero-masked broadcast; narrower targets use vector bit tests.
[[gnu::always_inline]] inline native<uint32_t> simd_u32_lanes_from_bits(uint64_t bits) {
  using U = native<uint32_t>;
  static_assert(U::size() <= 32);
  static_assert(U::size() == 16);
  return std::bit_cast<U>(_mm512_maskz_set1_epi32(static_cast<__mmask16>(bits), 1));
}

} // namespace util

#endif // UTIL_DETAIL_SIMD_X86_AVX512_H_
