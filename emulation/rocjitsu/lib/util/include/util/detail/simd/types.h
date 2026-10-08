// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_SIMD_TYPES_H_
#define UTIL_DETAIL_SIMD_TYPES_H_

#include "util/detail/simd/config.h"

#include <cstddef>
#include <cstdint>

namespace util {

#if __has_include(<experimental/simd>)
namespace stdx = std::experimental;

template <class T> using native = stdx::native_simd<T>;

/// Native-SIMD width measured in 32-bit lanes. Convenience constant.
template <class T> constexpr std::size_t native_width_v = native<T>::size();
#else
template <class T> struct native {
  static constexpr std::size_t size() { return 1; }
  constexpr T operator[](std::size_t) const { return T{}; }
};
template <class T> constexpr std::size_t native_width_v = native<T>::size();
#endif

/// Native-SIMD width for 64-bit lane types (e.g. native<uint64_t>/<double>),
/// measured in 64-bit lanes — half of native_width_v<uint32_t> on the same
/// vector width (8 vs 16 on AVX-512). Defined in both branches: in the
/// no-`<experimental/simd>` fallback native<uint64_t>::size() is 1.
inline constexpr std::size_t native_width64 = native<uint64_t>::size();

#if __has_include(<experimental/simd>)
/// A `native_width64`-wide SIMD of a 32-bit lane type (e.g. narrow32<float>,
/// narrow32<int32_t>). Used by the mixed-width f64<->32-bit conversion glue: a
/// chunk of `native_width64` (8 on AVX-512) f64 lanes pairs with the same number
/// of 32-bit lanes, so the 32-bit side is a `fixed_size_simd` of that width — a
/// direct `static_simd_cast` bridges it to/from `native<double>` (also
/// `native_width64`-wide) with no bit_cast.
template <class T> using narrow32 = stdx::fixed_size_simd<T, native_width64>;

#else
template <class T> struct narrow32 {
  static constexpr std::size_t size() { return 1; }
  constexpr T operator[](std::size_t) const { return T{}; }
};
#endif

} // namespace util

#endif // UTIL_DETAIL_SIMD_TYPES_H_
