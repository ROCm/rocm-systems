// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_SIMD_COMPAT64_MATH_H_
#define UTIL_DETAIL_SIMD_COMPAT64_MATH_H_

#include "util/detail/simd/common.h"

namespace util {

inline native<double> trunc_simd(native<double> a) {
  return map_native64_scalar<double>(a, [](double x) { return std::trunc(x); });
}

inline native<double> ceil_simd(native<double> a) {
  return map_native64_scalar<double>(a, [](double x) { return std::ceil(x); });
}

inline native<double> floor_simd(native<double> a) {
  return map_native64_scalar<double>(a, [](double x) { return std::floor(x); });
}

inline native<double> rndne_simd(native<double> a) {
  return map_native64_scalar<double>(a, [](double x) { return rndne_scalar(x); });
}

template <typename V> V ieee_maximum_simd(V a, V b) {
  if constexpr (std::is_same_v<typename V::value_type, double>) {
    return map_native64_scalar<double>(a, b, [](double lhs, double rhs) {
      if (std::isnan(lhs) || std::isnan(rhs))
        return std::numeric_limits<double>::quiet_NaN();
      if (lhs == rhs)
        return std::signbit(lhs) ? rhs : lhs;
      return lhs > rhs ? lhs : rhs;
    });
  } else {
    return ieee_maximum_masked_simd(a, b);
  }
}

template <typename V> V ieee_minimum_simd(V a, V b) {
  if constexpr (std::is_same_v<typename V::value_type, double>) {
    return map_native64_scalar<double>(a, b, [](double lhs, double rhs) {
      if (std::isnan(lhs) || std::isnan(rhs))
        return std::numeric_limits<double>::quiet_NaN();
      if (lhs == rhs)
        return std::signbit(lhs) ? lhs : rhs;
      return lhs < rhs ? lhs : rhs;
    });
  } else {
    return ieee_minimum_masked_simd(a, b);
  }
}

inline native<double> frexp_mant_f64_simd(native<double> x) {
  using U = native<uint64_t>;
  U v = std::bit_cast<U>(x);
  U out = map_native64_scalar<uint64_t>(v, [](uint64_t bits) -> uint64_t {
    constexpr uint64_t kSign = 0x8000000000000000ull;
    constexpr uint64_t kMant = 0x000FFFFFFFFFFFFFull;
    constexpr uint64_t kQuiet = 0x0008000000000000ull;
    uint64_t sign = bits & kSign;
    uint64_t E = (bits >> 52) & 0x7FFull;
    uint64_t M = bits & kMant;
    if (E == 0ull) {
      if (M == 0ull)
        return bits;
      uint64_t p = 63u - static_cast<uint64_t>(std::countl_zero(M));
      return sign | (1022ull << 52) | ((M << (52ull - p)) & kMant);
    }
    if (E == 2047ull)
      return M == 0ull ? bits : bits | kQuiet;
    return sign | (1022ull << 52) | M;
  });
  return std::bit_cast<native<double>>(out);
}

inline native<uint64_t> frexp_exp_f64_simd(native<double> x) {
  using U = native<uint64_t>;
  U v = std::bit_cast<U>(x);
  return map_native64_scalar<uint64_t>(v, [](uint64_t bits) -> uint64_t {
    uint64_t E = (bits >> 52) & 0x7FFull;
    uint64_t M = bits & 0x000FFFFFFFFFFFFFull;
    if (E == 0ull) {
      if (M == 0ull)
        return 0ull;
      uint64_t p = 63u - static_cast<uint64_t>(std::countl_zero(M));
      return static_cast<uint64_t>(static_cast<int64_t>(p) - 1073);
    }
    if (E == 2047ull)
      return 0ull;
    return static_cast<uint64_t>(static_cast<int64_t>(E) - 1022);
  });
}

} // namespace util

#endif // UTIL_DETAIL_SIMD_COMPAT64_MATH_H_
