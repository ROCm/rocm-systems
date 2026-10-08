// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_SIMD_NATIVE64_MATH_H_
#define UTIL_DETAIL_SIMD_NATIVE64_MATH_H_

#include "util/detail/simd/common.h"

namespace util {

inline native<double> trunc_simd(native<double> a) {
  return round_fixup_simd<double, false>(a, [](native<double> x) { return stdx::trunc(x); });
}

inline native<double> ceil_simd(native<double> a) {
  return round_fixup_simd<double, false>(a, [](native<double> x) { return stdx::ceil(x); });
}

inline native<double> floor_simd(native<double> a) {
  return round_fixup_simd<double, false>(a, [](native<double> x) { return stdx::floor(x); });
}

inline native<double> rndne_simd(native<double> a) { return rndne_bits_simd(a); }

template <typename V> V ieee_maximum_simd(V a, V b) { return ieee_maximum_masked_simd(a, b); }

template <typename V> V ieee_minimum_simd(V a, V b) { return ieee_minimum_masked_simd(a, b); }

inline native<double> frexp_mant_f64_simd(native<double> x) {
  using U = native<uint64_t>;
  U v = std::bit_cast<U>(x);
  U sign = v & U(0x8000000000000000ull);
  U E = (v >> 52) & U(0x7FFull);
  U M = v & U(0xFFFFFFFFFFFFFull); // 52 mantissa bits
  U normal = sign | (U(1022ull) << 52) | M;
  const native<double> mf = stdx::static_simd_cast<native<double>>(M);
  const U p = (std::bit_cast<U>(mf) >> 52) - U(1023ull);
  U dn = sign | (U(1022ull) << 52) | ((M << (U(52ull) - p)) & U(0xFFFFFFFFFFFFFull));
  U out = normal;
  stdx::where(E == 0ull, out) = v;                   // ±0 (M==0); overwritten if denormal
  stdx::where((E == 0ull) && (M != 0ull), out) = dn; // denormal -> renormalized
  stdx::where(E == 2047ull, out) = v;                // Inf passes through unchanged
  stdx::where((E == 2047ull) && (M != 0ull), out) = v | U(0x0008000000000000ull); // quiet NaN
  return std::bit_cast<native<double>>(out);
}

inline native<uint64_t> frexp_exp_f64_simd(native<double> x) {
  using U = native<uint64_t>;
  U v = std::bit_cast<U>(x);
  U E = (v >> 52) & U(0x7FFull);
  U M = v & U(0xFFFFFFFFFFFFFull);
  U normal = E - U(1022ull);
  const native<double> mf = stdx::static_simd_cast<native<double>>(M);
  const U p = (std::bit_cast<U>(mf) >> 52) - U(1023ull);
  U dn = p - U(1073ull);
  U out = normal;
  stdx::where(E == 0ull, out) = U(0ull);
  stdx::where((E == 0ull) && (M != 0ull), out) = dn;
  stdx::where(E == 2047ull, out) = U(0ull);
  return out;
}

} // namespace util

#endif // UTIL_DETAIL_SIMD_NATIVE64_MATH_H_
