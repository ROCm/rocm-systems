// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_SIMD_PORTABLE_H_
#define UTIL_DETAIL_SIMD_PORTABLE_H_

#include "util/detail/simd/types.h"

#include <bit>
#include <cstdint>

namespace util {

/// Fused multiply-add for native f32 SIMD. GCC's experimental::simd FMA
/// customization point does not inline on AVX-512 and otherwise emits a call
/// plus a ZMM spill/reload. Keep that toolchain workaround in the shared SIMD
/// layer; other targets use the portable TS operation.
[[gnu::always_inline]] inline native<float> native_fma(native<float> a, native<float> b,
                                                       native<float> c) {
  return stdx::fma(a, b, c);
}

/// Expand the low native<uint32_t>::size() bits into 0/1 uint32_t lanes. This
/// is the numeric form needed by carry-in arithmetic; selection paths should
/// use simd_mask_from_bits directly. AVX-512 materializes the lanes with a
/// k-mask zero-masked broadcast; narrower targets use vector bit tests.
[[gnu::always_inline]] inline native<uint32_t> simd_u32_lanes_from_bits(uint64_t bits) {
  using U = native<uint32_t>;
  static_assert(U::size() <= 32);
  const U lane_bits([](auto i) { return uint32_t{1} << i; });
  const U lane_indices([](auto i) { return static_cast<uint32_t>(i); });
  return (U(static_cast<uint32_t>(bits)) & lane_bits) >> lane_indices;
}

} // namespace util

#endif // UTIL_DETAIL_SIMD_PORTABLE_H_
