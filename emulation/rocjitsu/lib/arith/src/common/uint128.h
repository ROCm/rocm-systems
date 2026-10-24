// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Portable unsigned modulo-2^128 arithmetic used by exact numerical models.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_UINT128_H_
#define RJ_ARITH_UINT128_H_

#include <stdint.h>

namespace rocjitsu::arith {

// Unsigned modulo-2^128 arithmetic adapted from rocjitsu util/big_int.h.
// Shift counts must be in [0, 127]. Explicit conversions retain the low bits.
class Uint128 {
public:
  // Construct from low and high 64-bit words; omitted words are zero.
  constexpr Uint128(uint64_t low = 0, uint64_t high = 0) : low_(low), high_(high) {}

  // Return the low 32 bits.
  explicit constexpr operator uint32_t() const { return static_cast<uint32_t>(low_); }

  // Return the low 64 bits.
  explicit constexpr operator uint64_t() const { return low_; }

  // Return whether any bit is set.
  explicit constexpr operator bool() const { return low_ != 0 || high_ != 0; }

  // Return whether both 128-bit values are equal.
  friend constexpr bool operator==(Uint128 lhs, Uint128 rhs) {
    return lhs.low_ == rhs.low_ && lhs.high_ == rhs.high_;
  }

  // Return whether the 128-bit values differ.
  friend constexpr bool operator!=(Uint128 lhs, Uint128 rhs) { return !(lhs == rhs); }

  // Compare unsigned 128-bit values.
  friend constexpr bool operator>(Uint128 lhs, Uint128 rhs) {
    return lhs.high_ == rhs.high_ ? lhs.low_ > rhs.low_ : lhs.high_ > rhs.high_;
  }

  // Add modulo 2^128.
  friend constexpr Uint128 operator+(Uint128 lhs, Uint128 rhs) {
    const uint64_t low = lhs.low_ + rhs.low_;
    return {low, lhs.high_ + rhs.high_ + (low < lhs.low_)};
  }

  // Subtract modulo 2^128.
  friend constexpr Uint128 operator-(Uint128 lhs, Uint128 rhs) {
    return {lhs.low_ - rhs.low_, lhs.high_ - rhs.high_ - (lhs.low_ < rhs.low_)};
  }

  // Multiply modulo 2^128.
  friend constexpr Uint128 operator*(Uint128 lhs, Uint128 rhs) {
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

  // Return the bitwise intersection.
  friend constexpr Uint128 operator&(Uint128 lhs, Uint128 rhs) {
    return {lhs.low_ & rhs.low_, lhs.high_ & rhs.high_};
  }

  // Return the bitwise union.
  friend constexpr Uint128 operator|(Uint128 lhs, Uint128 rhs) {
    return {lhs.low_ | rhs.low_, lhs.high_ | rhs.high_};
  }

  // Shift left modulo 2^128; distance must be in [0, 127].
  friend constexpr Uint128 operator<<(Uint128 value, int distance) {
    if (distance == 0) {
      return value;
    }
    if (distance >= 64) {
      return {0, value.low_ << (distance - 64)};
    }
    return {value.low_ << distance, (value.high_ << distance) | (value.low_ >> (64 - distance))};
  }

  // Shift right with zero fill; distance must be in [0, 127].
  friend constexpr Uint128 operator>>(Uint128 value, int distance) {
    if (distance == 0) {
      return value;
    }
    if (distance >= 64) {
      return {value.high_ >> (distance - 64), 0};
    }
    return {(value.low_ >> distance) | (value.high_ << (64 - distance)), value.high_ >> distance};
  }

  // Increment modulo 2^128 and return the updated value by reference.
  constexpr Uint128 &operator++() { return *this = *this + 1; }

  // Shift in place with zero fill; distance must be in [0, 127].
  constexpr Uint128 &operator>>=(int distance) { return *this = *this >> distance; }

private:
  uint64_t low_;
  uint64_t high_;
};

} // namespace rocjitsu::arith

#endif
