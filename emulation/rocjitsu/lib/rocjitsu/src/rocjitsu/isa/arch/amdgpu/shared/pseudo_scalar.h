// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// FP16 result rounding for gfx10 interpolation. Ordinary arithmetic and SDWA
// numerical behavior are implemented by rj-arith.

#include "util/data_types.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rocjitsu::amdgpu::pseudo_scalar {

namespace detail {

inline uint16_t next_up_f16(uint16_t value) {
  if ((value & 0x7fffu) > 0x7c00u || value == 0x7c00u) {
    return value;
  }
  if (value == 0xfc00u) {
    return 0xfbffu;
  }
  if ((value & 0x7fffu) == 0) {
    return 0x0001u;
  }
  return static_cast<uint16_t>((value & 0x8000u) != 0 ? value - 1u : value + 1u);
}

inline uint16_t next_down_f16(uint16_t value) {
  if ((value & 0x7fffu) > 0x7c00u || value == 0xfc00u) {
    return value;
  }
  if (value == 0x7c00u) {
    return 0x7bffu;
  }
  if ((value & 0x7fffu) == 0) {
    return 0x8001u;
  }
  return static_cast<uint16_t>((value & 0x8000u) != 0 ? value + 1u : value - 1u);
}

// Round an interpolation result to F16 with guest rounding and overflow policy.
inline uint16_t round_f64_to_f16(double value, uint32_t round_mode, bool fp16_ovfl) {
  if (std::isnan(value) || std::isinf(value)) {
    return util::f32_to_f16(static_cast<float>(value));
  }

  constexpr double MAX_F16 = 65504.0;
  constexpr double RNE_OVERFLOW_THRESHOLD = 65520.0;
  if (std::fabs(value) > MAX_F16) {
    const bool negative = std::signbit(value);
    bool to_infinity = false;
    switch (round_mode & 3u) {
    case 0:
      to_infinity = std::fabs(value) >= RNE_OVERFLOW_THRESHOLD;
      break;
    case 1:
      to_infinity = !negative;
      break;
    case 2:
      to_infinity = negative;
      break;
    default:
      break;
    }
    if (to_infinity && !fp16_ovfl) {
      return negative ? 0xfc00u : 0x7c00u;
    }
    return negative ? 0xfbffu : 0x7bffu;
  }

  const uint16_t candidate = util::f32_to_f16(static_cast<float>(value));
  const double candidate_value = util::f16_to_f32(candidate);
  if (candidate_value == value) {
    return candidate;
  }

  const uint16_t lower = candidate_value < value ? candidate : next_down_f16(candidate);
  const uint16_t upper = candidate_value > value ? candidate : next_up_f16(candidate);

  switch (round_mode & 3u) {
  case 0: {
    const double lower_distance = value - util::f16_to_f32(lower);
    const double upper_distance = util::f16_to_f32(upper) - value;
    if (lower_distance < upper_distance) {
      return lower;
    }
    if (upper_distance < lower_distance) {
      return upper;
    }
    return (lower & 1u) == 0 ? lower : upper;
  }
  case 1:
    return upper;
  case 2:
    return lower;
  case 3:
    return value < 0.0 ? upper : lower;
  default:
    return candidate;
  }
}

} // namespace detail

// Clamp the interpolation result, then round directly from FP64 to F16.
inline uint16_t round_f16_result(double value, uint32_t round_mode, bool clamp, bool fp16_ovfl,
                                 bool clamp_nan_to_zero) {
  if (clamp && (clamp_nan_to_zero || !std::isnan(value))) {
    if (std::isnan(value) || value <= 0.0) {
      value = 0.0;
    } else if (value > 1.0) {
      value = 1.0;
    }
  }
  return detail::round_f64_to_f16(value, round_mode, fp16_ovfl);
}

} // namespace rocjitsu::amdgpu::pseudo_scalar
