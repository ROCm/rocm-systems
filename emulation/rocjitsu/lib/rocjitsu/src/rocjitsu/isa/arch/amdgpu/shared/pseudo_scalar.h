// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Legacy FP16 result rounding retained for VM output staging.
// New instruction implementations use rj-arith; this helper has remaining
// fp_mode/SDWA callers that determine its eventual retirement.

#include "util/data_types.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rocjitsu::amdgpu::pseudo_scalar {

namespace detail {

enum class ResultProvenance : uint8_t { VALUE, FINITE_OVERFLOW, FINITE_UNDERFLOW, INVALID_DOMAIN };

struct EvaluationResult {
  double value;
  ResultProvenance provenance;
};

inline EvaluationResult apply_output_modifiers(EvaluationResult result, uint32_t omod, bool clamp) {
  const double unmodified_value = result.value;
  if (omod == 1) {
    result.value *= 2.0;
  } else if (omod == 2) {
    result.value *= 4.0;
  } else if (omod == 3) {
    result.value *= 0.5;
  }

  if (result.provenance == ResultProvenance::VALUE && std::isfinite(unmodified_value) &&
      unmodified_value != 0.0) {
    if (std::isinf(result.value)) {
      result.provenance = ResultProvenance::FINITE_OVERFLOW;
    } else if (result.value == 0.0) {
      result.provenance = ResultProvenance::FINITE_UNDERFLOW;
    }
  }

  if (omod != 0 && result.value == 0.0 && result.provenance == ResultProvenance::VALUE) {
    result.value = 0.0;
  }

  if (clamp) {
    const bool negative_underflow =
        result.provenance == ResultProvenance::FINITE_UNDERFLOW && std::signbit(result.value);
    if (std::isnan(result.value) || result.value < 0.0 || negative_underflow) {
      result = {0.0, ResultProvenance::VALUE};
    } else if (result.value > 1.0 || result.provenance == ResultProvenance::FINITE_OVERFLOW) {
      result = {1.0, ResultProvenance::VALUE};
    } else if (result.value == 0.0 && result.provenance == ResultProvenance::VALUE) {
      result.value = 0.0;
    }
  }
  return result;
}

inline uint32_t saturated_f32(EvaluationResult result, uint32_t round_mode) {
  const bool negative = std::signbit(result.value);
  const uint32_t sign = negative ? 0x80000000u : 0;
  if (result.provenance == ResultProvenance::FINITE_UNDERFLOW) {
    const bool rounds_to_minimum =
        (!negative && (round_mode & 3u) == 1) || (negative && (round_mode & 3u) == 2);
    return sign | (rounds_to_minimum ? 1u : 0u);
  }

  const bool rounds_to_infinity = (round_mode & 3u) == 0 || (!negative && (round_mode & 3u) == 1) ||
                                  (negative && (round_mode & 3u) == 2);
  return sign | (rounds_to_infinity ? 0x7f800000u : 0x7f7fffffu);
}

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

inline uint16_t saturated_f16(EvaluationResult result, uint32_t round_mode, bool fp16_ovfl) {
  const bool negative = std::signbit(result.value);
  const uint16_t sign = negative ? 0x8000u : 0;
  if (result.provenance == ResultProvenance::FINITE_UNDERFLOW) {
    const bool rounds_to_minimum =
        (!negative && (round_mode & 3u) == 1) || (negative && (round_mode & 3u) == 2);
    return static_cast<uint16_t>(sign | (rounds_to_minimum ? 1u : 0u));
  }

  const bool rounds_to_infinity = (round_mode & 3u) == 0 || (!negative && (round_mode & 3u) == 1) ||
                                  (negative && (round_mode & 3u) == 2);
  if (fp16_ovfl || !rounds_to_infinity) {
    return static_cast<uint16_t>(sign | 0x7bffu);
  }
  return static_cast<uint16_t>(sign | 0x7c00u);
}

inline uint16_t round_f64_to_f16(EvaluationResult result, uint32_t round_mode, bool fp16_ovfl) {
  if (result.provenance == ResultProvenance::INVALID_DOMAIN) {
    return 0xfe00u;
  }
  if (result.provenance != ResultProvenance::VALUE) {
    return saturated_f16(result, round_mode, fp16_ovfl);
  }

  const double value = result.value;
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

/// @brief Apply optional numeric modifiers to a host value, then narrow it to F16.
/// @details This helper does not choose an instruction's output ordering. A caller
/// may pass a wide arithmetic result to apply modifiers before narrowing, pass an
/// already-rounded F16 value represented in F64, or disable modifiers and apply
/// the shared raw-bit stage afterward. The caller owns MODE input/output-denormal
/// policy; clamp_nan_to_zero selects CLAMP's NaN conversion separately.
inline uint16_t round_f16_result(double value, uint32_t round_mode, uint32_t omod, bool clamp,
                                 bool fp16_ovfl, bool clamp_nan_to_zero) {
  const bool effective_clamp = clamp && (clamp_nan_to_zero || !std::isnan(value));
  const detail::EvaluationResult modified = detail::apply_output_modifiers(
      {value, detail::ResultProvenance::VALUE}, omod, effective_clamp);
  return detail::round_f64_to_f16(modified, round_mode, fp16_ovfl);
}

} // namespace rocjitsu::amdgpu::pseudo_scalar
