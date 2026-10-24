// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Scalar/SIMD source modifier model inherited from rocjitsu.
// Works on raw encodings; guest FP controls never change the host FP environment.

#include "conversion/model/fp_format.h"

#include <bit>
#include <cstdint>
#include <type_traits>

namespace rocjitsu::arith::conversion_model::source_modifier {

/// @brief Apply ABS then NEG to one source.
template <typename Fmt, typename V> constexpr V apply(V bits, bool absolute, bool negate) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  if (absolute) {
    bits = bits & Fmt::kMagnitude;
  }
  if (negate) {
    bits = bits ^ Fmt::kSign;
  }
  return bits;
}

/// @brief Apply the modifier fields for source `index` of a VOP3 instruction.
/// @param abs VOP3 ABS field; bit i applies to source i.
/// @param neg VOP3 NEG field, with the same bit assignment.
template <typename Fmt, typename V>
constexpr V apply(V bits, unsigned index, uint32_t abs, uint32_t neg) {
  return apply<Fmt>(bits, ((abs >> index) & 1u) != 0, ((neg >> index) & 1u) != 0);
}

/// @brief Adapt a host float/double to the shared sign-bit operation.
/// @details Bit casts preserve the encoding, including a NaN's payload and quiet bit.
template <typename Float>
  requires(std::is_same_v<Float, float> || std::is_same_v<Float, double>)
constexpr Float apply_to_float(Float value, unsigned index, uint32_t abs, uint32_t neg) {
  using Fmt = std::conditional_t<std::is_same_v<Float, float>, fp_format::F32, fp_format::F64>;
  const auto bits = std::bit_cast<typename Fmt::Lane>(value);
  return std::bit_cast<Float>(apply<Fmt>(bits, index, abs, neg));
}

} // namespace rocjitsu::arith::conversion_model::source_modifier
