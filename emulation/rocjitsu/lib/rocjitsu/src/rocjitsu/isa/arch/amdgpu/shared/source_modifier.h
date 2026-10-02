// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file source_modifier.h
/// @brief ABS/NEG on scalar or SIMD floating-point encodings.
/// @details ABS clears the sign bit, then NEG flips it. The payload and
/// exponent are unchanged; input flushing belongs to the operation.

#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"

#include <cstdint>

namespace rocjitsu::amdgpu::source_modifier {

/// @brief Apply ABS then NEG to one source.
template <typename Fmt, typename V> constexpr V apply(V bits, bool absolute, bool negate) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  if (absolute)
    bits = bits & Fmt::kMagnitude;
  if (negate)
    bits = bits ^ Fmt::kSign;
  return bits;
}

/// @brief Apply the modifier fields for source `index` of a VOP3 instruction.
/// @param abs VOP3 ABS field; bit i applies to source i.
/// @param neg VOP3 NEG field, with the same bit assignment.
template <typename Fmt, typename V>
constexpr V apply(V bits, unsigned index, uint32_t abs, uint32_t neg) {
  return apply<Fmt>(bits, ((abs >> index) & 1u) != 0, ((neg >> index) & 1u) != 0);
}

} // namespace rocjitsu::amdgpu::source_modifier
