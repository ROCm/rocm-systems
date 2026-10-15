// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file input_denormal.h
/// @brief Input-denormal handling on scalar or SIMD floating-point encodings.
/// @details Callers select the source format and policy required by the instruction.
/// Apply this stage before widening: an F16 subnormal is normal when represented in F32.
/// Source modifiers and output-denormal handling are separate stages.

#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"

#include <cstdint>

namespace rocjitsu::amdgpu::input_denormal {

/// @brief Input-flush policy, fixed before any lane is evaluated.
struct Policy {
  bool flush_inputs = false;

  /// @brief Read the input control from the source format's MODE.FP_DENORM field.
  /// @details Bit 0 allows input denormals; the output control in bit 1 is ignored.
  static constexpr Policy make(uint32_t denorm_mode) { return {(denorm_mode & 1u) == 0}; }
};

namespace detail {

/// @brief All ones where the magnitude is below the smallest normal, zero elsewhere.
/// @details The subtraction borrows into the lane's top bit exactly when the
/// magnitude is smaller, which avoids a mask type.
template <typename Fmt, typename V> constexpr V below_normal(V bits) {
  using Lane = typename Fmt::Lane;
  const V magnitude = bits & Fmt::kMagnitude;
  return Lane{0} - ((magnitude - Fmt::kMinNormal) >> (8 * sizeof(Lane) - 1));
}

} // namespace detail

/// @brief Flush a subnormal source to a zero of the same sign when enabled.
/// @details NaN, infinity, zero and normal encodings pass through unchanged.
template <typename Fmt, typename V> constexpr V flush_input(V bits, const Policy &policy) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  if (!policy.flush_inputs)
    return bits;
  return bits & (~detail::below_normal<Fmt>(bits) | Fmt::kSign);
}

/// @brief Mask to the source format and apply input flushing.
template <typename Fmt, typename V> constexpr V prepare(V bits, const Policy &policy) {
  return flush_input<Fmt>(bits & Fmt::kBits, policy);
}

} // namespace rocjitsu::amdgpu::input_denormal
