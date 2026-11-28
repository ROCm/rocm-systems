/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <limits>

enum class HalfRoundingMode { Downward, Upward, TowardZero };

// Round the binary32 significand directly; neither the host floating-point
// environment nor a host half conversion participates in the reference.
inline uint16_t RoundFloatToHalfBits(float value, HalfRoundingMode mode) {
  static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559 &&
                std::numeric_limits<float>::digits == 24 &&
                std::numeric_limits<float>::max_exponent == 128);
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  const uint32_t magnitude = bits & 0x7fffffffu;
  const uint32_t exponent = magnitude >> 23;
  const uint32_t fraction = magnitude & 0x7fffffu;

  if (exponent == 0xffu) {
    // Preserve infinity; a NaN payload is not part of the casting contract.
    return sign | (fraction == 0 ? 0x7c00u : 0x7e00u);
  }

  const bool round_away = (mode == HalfRoundingMode::Downward && sign != 0) ||
                          (mode == HalfRoundingMode::Upward && sign == 0);
  if (magnitude >= 0x47800000u) {  // 2^16, beyond the binary16 finite range.
    return sign | (round_away ? 0x7c00u : 0x7bffu);
  }

  uint32_t truncated;
  bool discarded;
  if (magnitude < 0x33800000u) {  // Smaller than the minimum binary16 subnormal, 2^-24.
    truncated = 0;
    discarded = magnitude != 0;
  } else if (magnitude < 0x38800000u) {  // Binary16 subnormal: units of 2^-24.
    const uint32_t significand = 0x800000u | fraction;
    const uint32_t shift = 126u - exponent;  // Between 14 and 23 in this range.
    truncated = significand >> shift;
    discarded = (significand & ((1u << shift) - 1u)) != 0;
  } else {
    truncated = ((exponent - 112u) << 10) | (fraction >> 13);
    discarded = (fraction & 0x1fffu) != 0;
  }

  // Incrementing also handles a carry from subnormal to normal, or from the
  // largest finite value to infinity. Signed zero is preserved when rounding in.
  return static_cast<uint16_t>(sign | (truncated + (round_away && discarded)));
}
