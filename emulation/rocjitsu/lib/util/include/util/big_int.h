// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_UTIL_BIG_INT_H_
#define ROCJITSU_UTIL_BIG_INT_H_

#include <compare>
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

class fallback_uint128_t {
public:
  constexpr fallback_uint128_t(uint64_t low = 0, uint64_t high = 0) : low_(low), high_(high) {}

  explicit constexpr operator uint32_t() const { return static_cast<uint32_t>(low_); }
  explicit constexpr operator uint64_t() const { return low_; }
  explicit constexpr operator bool() const { return low_ != 0 || high_ != 0; }

  friend constexpr bool operator==(fallback_uint128_t, fallback_uint128_t) = default;
  friend constexpr std::strong_ordering operator<=>(fallback_uint128_t lhs,
                                                    fallback_uint128_t rhs) {
    return lhs.high_ == rhs.high_ ? lhs.low_ <=> rhs.low_ : lhs.high_ <=> rhs.high_;
  }

  friend constexpr fallback_uint128_t operator+(fallback_uint128_t lhs, fallback_uint128_t rhs) {
    const uint64_t low = lhs.low_ + rhs.low_;
    return {low, lhs.high_ + rhs.high_ + (low < lhs.low_)};
  }

  friend constexpr fallback_uint128_t operator-(fallback_uint128_t lhs, fallback_uint128_t rhs) {
    return {lhs.low_ - rhs.low_, lhs.high_ - rhs.high_ - (lhs.low_ < rhs.low_)};
  }

  friend constexpr fallback_uint128_t operator*(fallback_uint128_t lhs, fallback_uint128_t rhs) {
    // Split the low words into 32-bit limbs so every partial product and carry
    // fits in uint64_t. Terms above bit 127 are discarded, as for unsigned ints.
    const uint64_t a0 = static_cast<uint32_t>(lhs.low_), a1 = lhs.low_ >> 32;
    const uint64_t b0 = static_cast<uint32_t>(rhs.low_), b1 = rhs.low_ >> 32;
    const uint64_t low = a0 * b0;
    const uint64_t cross = a1 * b0 + (low >> 32);
    const uint64_t middle = a0 * b1 + static_cast<uint32_t>(cross);
    const uint64_t high =
        a1 * b1 + (cross >> 32) + (middle >> 32) + lhs.high_ * rhs.low_ + lhs.low_ * rhs.high_;
    return {(middle << 32) | static_cast<uint32_t>(low), high};
  }

  friend constexpr fallback_uint128_t operator&(fallback_uint128_t lhs, fallback_uint128_t rhs) {
    return {lhs.low_ & rhs.low_, lhs.high_ & rhs.high_};
  }

  friend constexpr fallback_uint128_t operator|(fallback_uint128_t lhs, fallback_uint128_t rhs) {
    return {lhs.low_ | rhs.low_, lhs.high_ | rhs.high_};
  }

  friend constexpr fallback_uint128_t operator<<(fallback_uint128_t value, int distance) {
    if (distance == 0)
      return value;
    if (distance >= 64)
      return {0, value.low_ << (distance - 64)};
    return {value.low_ << distance, (value.high_ << distance) | (value.low_ >> (64 - distance))};
  }

  friend constexpr fallback_uint128_t operator>>(fallback_uint128_t value, int distance) {
    if (distance == 0)
      return value;
    if (distance >= 64)
      return {value.high_ >> (distance - 64), 0};
    return {(value.low_ >> distance) | (value.high_ << (64 - distance)), value.high_ >> distance};
  }

  constexpr fallback_uint128_t &operator++() { return *this = *this + 1; }
  constexpr fallback_uint128_t &operator>>=(int distance) { return *this = *this >> distance; }

private:
  uint64_t low_;
  uint64_t high_;
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

/// Unsigned 128-bit integer for exact significands and bitwise arithmetic.
///
/// Supports construction from uint64_t, explicit conversion to uint32_t,
/// uint64_t or bool, comparisons, modular +, - and *, bitwise & and |, shifts,
/// prefix increment and right-shift assignment. Shift counts must be in [0, 127].
#if defined(__SIZEOF_INT128__)
using uint128_t = __uint128_t;
#else
using uint128_t = detail::fallback_uint128_t;
#endif

} // namespace util

#endif // ROCJITSU_UTIL_BIG_INT_H_
