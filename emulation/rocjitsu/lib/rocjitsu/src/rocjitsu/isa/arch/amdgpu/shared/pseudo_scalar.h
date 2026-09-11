// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file Shared pseudo-scalar transcendental implementations.
///
/// @details Architectural numeric behavior follows the AMD RDNA4 Instruction Set Architecture
/// Reference Guide: section 7.10 requires the usual DENORMAL and ROUND mode bits, and section
/// 7.2.3.1 requires nonzero OMOD to flush output denormals and map negative zero to positive zero.
/// The functional examples for the vector V_LOG_F32, V_RSQ_F32, and V_SQRT_F32 equivalents specify
/// negative quiet NaNs for invalid domains. The host standard-library functions below provide the
/// approximate transcendental values; the surrounding logic applies these architectural rules.

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

inline float flush_input_f32(float value, uint32_t denorm_mode) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((denorm_mode & 1u) == 0 && (bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0)
    return std::copysign(0.0f, value);
  return value;
}

inline float flush_input_f16(float value, uint32_t denorm_mode) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if ((denorm_mode & 1u) == 0 && magnitude != 0 && magnitude < 0x38800000u)
    return std::bit_cast<float>(bits & 0x80000000u);
  return value;
}

inline float quiet_nan(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0)
    bits |= 0x00400000u;
  return std::bit_cast<float>(bits);
}

inline EvaluationResult evaluate(Operation operation, double value) {
  if (std::isnan(value))
    return {value, ResultProvenance::VALUE};

  switch (operation) {
  case Operation::EXP2: {
    if (std::isinf(value))
      return {value < 0.0 ? 0.0 : value, ResultProvenance::VALUE};
    const double result = std::exp2(value);
    if (std::isinf(result))
      return {result, ResultProvenance::FINITE_OVERFLOW};
    if (result == 0.0)
      return {result, ResultProvenance::FINITE_UNDERFLOW};
    return {result, ResultProvenance::VALUE};
  }
  case Operation::LOG2:
    if (value == 0.0)
      return {-std::numeric_limits<double>::infinity(), ResultProvenance::VALUE};
    if (value < 0.0)
      return {std::numeric_limits<double>::quiet_NaN(), ResultProvenance::INVALID_DOMAIN};
    if (std::isinf(value))
      return {value, ResultProvenance::VALUE};
    return {std::log2(value), ResultProvenance::VALUE};
  case Operation::RCP:
    if (value == 0.0)
      return {std::copysign(std::numeric_limits<double>::infinity(), value),
              ResultProvenance::VALUE};
    if (std::isinf(value))
      return {std::copysign(0.0, value), ResultProvenance::VALUE};
    return {1.0 / value, ResultProvenance::VALUE};
  case Operation::RSQ:
    if (value == 0.0)
      return {std::copysign(std::numeric_limits<double>::infinity(), value),
              ResultProvenance::VALUE};
    if (value < 0.0)
      return {std::numeric_limits<double>::quiet_NaN(), ResultProvenance::INVALID_DOMAIN};
    if (std::isinf(value))
      return {0.0, ResultProvenance::VALUE};
    return {1.0 / std::sqrt(value), ResultProvenance::VALUE};
  case Operation::SQRT:
    if (value < 0.0)
      return {std::numeric_limits<double>::quiet_NaN(), ResultProvenance::INVALID_DOMAIN};
    if (std::isinf(value))
      return {value, ResultProvenance::VALUE};
    return {std::sqrt(value), ResultProvenance::VALUE};
  }
  return {std::numeric_limits<double>::quiet_NaN(), ResultProvenance::VALUE};
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

/// @brief Apply F16 result modifiers and perform one direct F64-to-F16 rounding.
/// @details This is the supported policy surface for fused operations whose exact result is
/// representable in F64. It avoids exposing pseudo-scalar implementation details to other
/// execution helpers. CLAMP's NaN conversion is selected separately because older profiles
/// require MODE.DX10_CLAMP while GFX12 and gfx1250 always convert NaN to positive zero.
inline uint16_t round_f16_result(double value, uint32_t round_mode, uint32_t omod, bool clamp,
                                 bool fp16_ovfl, bool clamp_nan_to_zero) {
  const bool effective_clamp = clamp && (clamp_nan_to_zero || !std::isnan(value));
  const detail::EvaluationResult modified = detail::apply_output_modifiers(
      {value, detail::ResultProvenance::VALUE}, omod, effective_clamp);
  return detail::round_f64_to_f16(modified, round_mode, fp16_ovfl);
}

/// @brief Execute a pseudo-scalar F32 transcendental operation.
/// @details Source absolute value and negation are applied before input-denormal handling and
/// operation evaluation. OMOD is then applied before CLAMP, result rounding, and output-denormal
/// handling. Round modes are 0 for nearest-even, 1 for positive infinity, 2 for negative infinity,
/// and 3 for zero. Denormal mode bit 0 allows input denormals and bit 1 allows output denormals.
/// @param operation Transcendental operation to execute.
/// @param source Raw F32 source value.
/// @param absolute Whether to clear the source sign bit before evaluation.
/// @param negate Whether to toggle the source sign bit after applying absolute value.
/// @param round_mode Numeric MODE.FP_ROUND encoding for F32.
/// @param denorm_mode Numeric MODE.FP_DENORM encoding for F32.
/// @param omod Numeric VOP3 OMOD encoding: 0 unchanged, 1 multiply by 2, 2 multiply by 4, and 3
/// multiply by 0.5.
/// @param clamp Whether to clamp NaN and negative results to zero and results above one to one.
/// @returns Raw 32-bit F32 result encoding.
inline uint32_t execute_f32(Operation operation, float source, bool absolute, bool negate,
                            uint32_t round_mode, uint32_t denorm_mode, uint32_t omod, bool clamp) {
  source = detail::apply_source_modifiers(source, absolute, negate);
  source = detail::flush_input_f32(source, denorm_mode);
  source = detail::quiet_nan(source);
  const detail::EvaluationResult value = detail::apply_output_modifiers(
      detail::evaluate(operation, static_cast<double>(source)), omod, clamp);
  uint32_t result = detail::round_f64_to_f32(value, round_mode);
  if (((denorm_mode & 2u) == 0 || omod != 0) && (result & 0x7f800000u) == 0 &&
      (result & 0x007fffffu) != 0)
    result &= 0x80000000u;
  return result;
}

/// @brief Execute a pseudo-scalar F16 transcendental operation.
/// @details Source absolute value and negation are applied before input-denormal handling and
/// operation evaluation. OMOD is then applied before CLAMP, result rounding, and output-denormal
/// handling. Round modes are 0 for nearest-even, 1 for positive infinity, 2 for negative infinity,
/// and 3 for zero. Denormal mode bit 0 allows input denormals and bit 1 allows output denormals.
/// FP16_OVFL clamps finite overflow to signed maximum finite F16 regardless of round mode, but does
/// not clamp true infinity or divide-by-zero results.
/// @param operation Transcendental operation to execute.
/// @param source F16 source value represented exactly as an F32 value.
/// @param absolute Whether to clear the source sign bit before evaluation.
/// @param negate Whether to toggle the source sign bit after applying absolute value.
/// @param round_mode Numeric MODE.FP_ROUND encoding for F16.
/// @param denorm_mode Numeric MODE.FP_DENORM encoding for F16.
/// @param omod Numeric VOP3 OMOD encoding: 0 unchanged, 1 multiply by 2, 2 multiply by 4, and 3
/// multiply by 0.5.
/// @param clamp Whether to clamp NaN and negative results to zero and results above one to one.
/// @param fp16_ovfl Whether MODE.FP16_OVFL finite-overflow saturation is enabled.
/// @returns Raw F16 encoding in bits 15:0 with bits 31:16 cleared.
inline uint32_t execute_f16(Operation operation, float source, bool absolute, bool negate,
                            uint32_t round_mode, uint32_t denorm_mode, uint32_t omod, bool clamp,
                            bool fp16_ovfl) {
  source = detail::apply_source_modifiers(source, absolute, negate);
  source = detail::flush_input_f16(source, denorm_mode);
  source = detail::quiet_nan(source);
  const detail::EvaluationResult value = detail::apply_output_modifiers(
      detail::evaluate(operation, static_cast<double>(source)), omod, clamp);
  uint16_t result = detail::round_f64_to_f16(value, round_mode, fp16_ovfl);
  if (((denorm_mode & 2u) == 0 || omod != 0) && (result & 0x7c00u) == 0 && (result & 0x03ffu) != 0)
    result &= 0x8000u;
  return result;
}

} // namespace rocjitsu::amdgpu::pseudo_scalar
