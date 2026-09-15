// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_UTIL_BIG_INT_H_
#define ROCJITSU_UTIL_BIG_INT_H_

#include <cstdint>

namespace util {
namespace detail {

class fallback_int128_t {
public:
  constexpr fallback_int128_t() = default;
  explicit constexpr fallback_int128_t(uint64_t value) : low_(value) {}

  explicit constexpr operator uint64_t() const { return low_; }

  friend constexpr fallback_int128_t operator-(fallback_int128_t value) {
    fallback_int128_t result;
    result.low_ = ~value.low_ + 1;
    result.high_ = ~value.high_ + (result.low_ == 0 ? 1 : 0);
    return result;
  }

  friend constexpr fallback_int128_t operator+(fallback_int128_t lhs, fallback_int128_t rhs) {
    fallback_int128_t result;
    result.low_ = lhs.low_ + rhs.low_;
    result.high_ = lhs.high_ + rhs.high_ + (result.low_ < lhs.low_ ? 1 : 0);
    return result;
  }

  friend constexpr fallback_int128_t operator-(fallback_int128_t lhs, fallback_int128_t rhs) {
    return lhs + -rhs;
  }

  friend constexpr bool operator<(fallback_int128_t lhs, fallback_int128_t rhs) {
    const bool lhs_negative = (lhs.high_ >> 63) != 0;
    const bool rhs_negative = (rhs.high_ >> 63) != 0;
    if (lhs_negative != rhs_negative)
      return lhs_negative;
    if (lhs.high_ != rhs.high_)
      return lhs.high_ < rhs.high_;
    return lhs.low_ < rhs.low_;
  }

  friend constexpr bool operator>(fallback_int128_t lhs, fallback_int128_t rhs) {
    return rhs < lhs;
  }

private:
  uint64_t low_ = 0;
  uint64_t high_ = 0;
};

} // namespace detail

/// Signed 128-bit integer for widened addition, subtraction, and range checks.
///
/// Portable code may rely on construction from uint64_t, explicit conversion
/// to uint64_t, unary negation, binary addition and subtraction, and less-than
/// and greater-than comparisons. Native implementations may expose additional
/// operators, but those are outside this interface so the fallback remains
/// usable on toolchains without 128-bit integer support.
#if defined(__SIZEOF_INT128__)
using int128_t = __int128_t;
#else
using int128_t = detail::fallback_int128_t;
#endif

} // namespace util

#endif // ROCJITSU_UTIL_BIG_INT_H_
