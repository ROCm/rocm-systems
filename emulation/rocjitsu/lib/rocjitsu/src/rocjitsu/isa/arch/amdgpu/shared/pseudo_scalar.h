// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file Shared pseudo-scalar transcendental implementations.
///
/// @details Physical gfx1201 F32 operations match their vector equivalents: fixed rounding
/// and unconditional denormal flushing. Half execution shares the vector mapping in
/// transcendental.h. Section 7.2.3.1 requires nonzero OMOD to flush
/// output denormals and map negative zero to positive zero. Vector LOG, RSQ and SQRT
/// functional examples specify negative quiet NaNs for invalid domains.

#include "util/amdgpu_exp.h"
#include "util/amdgpu_log.h"
#include "util/amdgpu_rcp.h"
#include "util/amdgpu_rsq.h"
#include "util/amdgpu_sqrt.h"
#include "util/data_types.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rocjitsu::amdgpu::pseudo_scalar {

/// @brief Pseudo-scalar transcendental operation implemented by the shared helper.
enum class Operation : uint8_t { EXP2, LOG2, RCP, RSQ, SQRT };

namespace detail {

enum class ResultProvenance : uint8_t { VALUE, FINITE_OVERFLOW, FINITE_UNDERFLOW, INVALID_DOMAIN };

struct EvaluationResult {
  double value;
  ResultProvenance provenance;
};

inline float quiet_nan(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  // Test signaling NaNs specifically: a general NaN check can become a host
  // floating-point comparison that raises FE_INVALID before the quieting step.
  if ((bits & 0x7fc00000u) == 0x7f800000u && (bits & 0x003fffffu) != 0)
    bits |= 0x00400000u;
  return std::bit_cast<float>(bits);
}

inline EvaluationResult apply_output_modifiers(EvaluationResult result, uint32_t omod, bool clamp) {
  const double unmodified_value = result.value;
  if (omod == 1)
    result.value *= 2.0;
  else if (omod == 2)
    result.value *= 4.0;
  else if (omod == 3)
    result.value *= 0.5;

  if (result.provenance == ResultProvenance::VALUE && std::isfinite(unmodified_value) &&
      unmodified_value != 0.0) {
    if (std::isinf(result.value))
      result.provenance = ResultProvenance::FINITE_OVERFLOW;
    else if (result.value == 0.0)
      result.provenance = ResultProvenance::FINITE_UNDERFLOW;
  }

  if (omod != 0 && result.value == 0.0 && result.provenance == ResultProvenance::VALUE)
    result.value = 0.0;

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

inline uint32_t round_f64_to_f32(EvaluationResult result, uint32_t round_mode) {
  if (result.provenance == ResultProvenance::INVALID_DOMAIN)
    return 0xffc00000u;
  if (result.provenance != ResultProvenance::VALUE)
    return saturated_f32(result, round_mode);

  float rounded = static_cast<float>(result.value);
  if ((round_mode & 3u) == 0 || std::isnan(result.value) ||
      static_cast<double>(rounded) == result.value)
    return std::bit_cast<uint32_t>(rounded);

  switch (round_mode & 3u) {
  case 1:
    if (static_cast<double>(rounded) < result.value)
      rounded = std::nextafter(rounded, std::numeric_limits<float>::infinity());
    break;
  case 2:
    if (static_cast<double>(rounded) > result.value)
      rounded = std::nextafter(rounded, -std::numeric_limits<float>::infinity());
    break;
  case 3:
    if ((result.value > 0.0 && static_cast<double>(rounded) > result.value) ||
        (result.value < 0.0 && static_cast<double>(rounded) < result.value))
      rounded = std::nextafter(rounded, 0.0f);
    break;
  default:
    break;
  }
  return std::bit_cast<uint32_t>(rounded);
}

inline uint16_t next_up_f16(uint16_t value) {
  if ((value & 0x7fffu) > 0x7c00u || value == 0x7c00u)
    return value;
  if (value == 0xfc00u)
    return 0xfbffu;
  if ((value & 0x7fffu) == 0)
    return 0x0001u;
  return static_cast<uint16_t>((value & 0x8000u) != 0 ? value - 1u : value + 1u);
}

inline uint16_t next_down_f16(uint16_t value) {
  if ((value & 0x7fffu) > 0x7c00u || value == 0xfc00u)
    return value;
  if (value == 0x7c00u)
    return 0x7bffu;
  if ((value & 0x7fffu) == 0)
    return 0x8001u;
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
  if (fp16_ovfl || !rounds_to_infinity)
    return static_cast<uint16_t>(sign | 0x7bffu);
  return static_cast<uint16_t>(sign | 0x7c00u);
}

inline uint16_t round_f64_to_f16(EvaluationResult result, uint32_t round_mode, bool fp16_ovfl) {
  if (result.provenance == ResultProvenance::INVALID_DOMAIN)
    return 0xfe00u;
  if (result.provenance != ResultProvenance::VALUE)
    return saturated_f16(result, round_mode, fp16_ovfl);

  const double value = result.value;
  if (std::isnan(value) || std::isinf(value))
    return util::f32_to_f16(static_cast<float>(value));

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
    if (to_infinity && !fp16_ovfl)
      return negative ? 0xfc00u : 0x7c00u;
    return negative ? 0xfbffu : 0x7bffu;
  }

  const uint16_t candidate = util::f32_to_f16(static_cast<float>(value));
  const double candidate_value = util::f16_to_f32(candidate);
  if (candidate_value == value)
    return candidate;

  const uint16_t lower = candidate_value < value ? candidate : next_down_f16(candidate);
  const uint16_t upper = candidate_value > value ? candidate : next_up_f16(candidate);

  switch (round_mode & 3u) {
  case 0: {
    const double lower_distance = value - util::f16_to_f32(lower);
    const double upper_distance = util::f16_to_f32(upper) - value;
    if (lower_distance < upper_distance)
      return lower;
    if (upper_distance < lower_distance)
      return upper;
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

inline float apply_source_modifiers(float value, bool absolute, bool negate) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if (absolute)
    bits &= 0x7fffffffu;
  if (negate)
    bits ^= 0x80000000u;
  return std::bit_cast<float>(bits);
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

/// @brief Execute a pseudo-scalar F32 transcendental operation.
/// @details Apply source modifiers and quiet NaNs before the hardware mapping, then apply
/// OMOD and CLAMP before nearest-even narrowing and unconditional output-denormal flushing.
/// All F32 operations ignore MODE rounding and denormal fields, matching their vector
/// equivalents. Output-modifier overflow rounds to infinity.
/// @param operation Transcendental operation to execute.
/// @param source Raw F32 source value.
/// @param absolute Whether to clear the source sign bit before evaluation.
/// @param negate Whether to toggle the source sign bit after applying absolute value.
/// @param round_mode Ignored F32 rounding mode; retained for the shared instruction interface.
/// @param denorm_mode Ignored F32 denormal mode; retained for the shared instruction interface.
/// @param omod Numeric VOP3 OMOD encoding: 0 unchanged, 1 multiply by 2, 2 multiply by 4, and 3
/// multiply by 0.5.
/// @param clamp Whether to clamp NaN and negative results to zero and results above one to one.
/// @returns Raw 32-bit F32 result encoding.
inline uint32_t execute_f32(Operation operation, float source, bool absolute, bool negate,
                            [[maybe_unused]] uint32_t round_mode,
                            [[maybe_unused]] uint32_t denorm_mode, uint32_t omod, bool clamp) {
  source = detail::apply_source_modifiers(source, absolute, negate);
  source = detail::quiet_nan(source);
  // Physical GFX12 scalar and vector mappings agree in every FP MODE.
  float mapped = 0.0f;
  switch (operation) {
  case Operation::EXP2:
    mapped = util::amdgpu_exp_f32(source);
    break;
  case Operation::LOG2:
    mapped = util::amdgpu_log_f32(source);
    break;
  case Operation::RCP:
    mapped = util::amdgpu_rcp_f32(source);
    break;
  case Operation::RSQ:
    mapped = util::amdgpu_rsq_f32(source);
    break;
  case Operation::SQRT:
    mapped = util::amdgpu_sqrt_f32(source);
    break;
  }
  const detail::EvaluationResult evaluated{mapped, detail::ResultProvenance::VALUE};
  detail::EvaluationResult value = detail::apply_output_modifiers(evaluated, omod, clamp);
  // Output scaling is exact in F64. Detect F32 overflow before narrowing,
  // so directed host rounding cannot replace the required infinity with a finite value.
  if (std::isfinite(value.value) && std::abs(value.value) > std::numeric_limits<float>::max())
    value.provenance = detail::ResultProvenance::FINITE_OVERFLOW;
  uint32_t result = detail::round_f64_to_f32(value, 0);
  if ((result & 0x7f800000u) == 0 && (result & 0x007fffffu) != 0)
    result &= 0x80000000u;
  return result;
}

} // namespace rocjitsu::amdgpu::pseudo_scalar
