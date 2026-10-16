// SPDX-License-Identifier: MIT

// Type traits and alias-safe bit casts for FP32/FP64 representations.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include <cstring>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith {

// Raw IEEE binary32/binary64 storage and classification masks.
template <typename Float> struct FloatBits {
  static_assert(std::is_same_v<Float, float> || std::is_same_v<Float, double>);
  using UInt = std::conditional_t<std::is_same_v<Float, float>, uint32_t, uint64_t>;
  static constexpr unsigned fraction_bits = std::is_same_v<Float, float> ? 23 : 52;
  static constexpr UInt sign = UInt(1) << (sizeof(UInt) * 8 - 1);
  static constexpr UInt magnitude = sign - 1;
  static constexpr UInt infinity = magnitude & ~((UInt(1) << fraction_bits) - 1);
  static constexpr UInt quiet = UInt(1) << (fraction_bits - 1);

  // Copy a floating-point representation into its matching unsigned integer type.
  static UInt bits(Float value) {
    UInt result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  }

  // Copy an unsigned bit pattern into its matching floating-point type.
  static Float value(UInt bits) {
    Float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
  }
};

} // namespace rocjitsu::arith
