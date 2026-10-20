// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file fp_format.h
/// @brief Floating-point encoding layouts shared by comparisons, arithmetic and modifiers.
/// @details Describes the bits only; rounding, NaN selection and denormal handling
/// belong to the operation using the format.

#include <bit>
#include <cstdint>
#include <type_traits>

namespace rocjitsu::amdgpu::fp_format {

/// @brief Binary interchange format carried in an unsigned lane type.
template <typename LaneType, unsigned ExponentBits, unsigned MantissaBits> struct Format {
  using Lane = LaneType;
  static constexpr unsigned kExponentBits = ExponentBits;
  static constexpr unsigned kMantissaBits = MantissaBits;
  static constexpr unsigned kWidth = 1 + ExponentBits + MantissaBits;
  static constexpr Lane kSign = Lane{1} << (kWidth - 1);
  static constexpr Lane kMagnitude = kSign - 1;
  static constexpr Lane kBits = kSign | kMagnitude;
  static constexpr Lane kExponentMax = (Lane{1} << ExponentBits) - 1;
  static constexpr Lane kInfinity = kExponentMax << MantissaBits;
  static constexpr Lane kQuiet = Lane{1} << (MantissaBits - 1);
  static constexpr Lane kMinNormal = Lane{1} << MantissaBits;
  static constexpr Lane kFraction = kMinNormal - 1;
  static constexpr Lane kBias = kExponentMax >> 1;

  static_assert(std::is_unsigned_v<Lane> && kWidth <= 8 * sizeof(Lane));
};

// F16 uses the low 16 bits of a 32-bit register or SIMD lane.
using F16 = Format<uint32_t, 5, 10>;
using F32 = Format<uint32_t, 8, 23>;
using F64 = Format<uint64_t, 11, 52>;

/// @brief Whether V is the format's lane type or a SIMD vector of it.
template <typename Fmt, typename V>
inline constexpr bool is_lane_v = std::is_same_v<V, typename Fmt::Lane> || requires {
  requires std::is_same_v<typename V::value_type, typename Fmt::Lane>;
};

/// @brief std::bit_width of an unsigned scalar or of each SIMD lane.
/// @details Used to normalize a subnormal significand: the result is the
/// position of the leading one plus one, or zero for a zero lane.
template <typename V> constexpr V bit_width(V value) {
  if constexpr (std::is_unsigned_v<V>) {
    return static_cast<V>(std::bit_width(value));
  } else {
    using Lane = typename V::value_type;
    static_assert(std::is_unsigned_v<Lane>);
    constexpr unsigned kLaneBits = 8 * sizeof(Lane);
    // Set every bit below the leading one; the width is then the number of ones.
    for (unsigned shift = 1; shift < kLaneBits; shift <<= 1)
      value |= value >> shift;
    value = value - ((value >> 1) & V(static_cast<Lane>(0x5555555555555555ull)));
    value = (value & V(static_cast<Lane>(0x3333333333333333ull))) +
            ((value >> 2) & V(static_cast<Lane>(0x3333333333333333ull)));
    value = (value + (value >> 4)) & V(static_cast<Lane>(0x0f0f0f0f0f0f0f0full));
    for (unsigned shift = 8; shift < kLaneBits; shift <<= 1)
      value = value + (value >> shift);
    return value & V(Lane{0x7f});
  }
}

} // namespace rocjitsu::amdgpu::fp_format
