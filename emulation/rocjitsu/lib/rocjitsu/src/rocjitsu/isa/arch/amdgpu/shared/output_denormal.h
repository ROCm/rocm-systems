// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file output_denormal.h
/// @brief Output-denormal handling on scalar or SIMD floating-point encodings.
/// @details Applies to a result already rounded to its destination format.
/// Operations that detect tininess while rounding call this at that point;
/// OMOD's own zero and subnormal rules belong to output_modifier.h.

#include "rocjitsu/isa/arch/amdgpu/shared/denormal.h"

#include <cstdint>

namespace rocjitsu::amdgpu::output_denormal {

/// @brief Output-flush policy, fixed before any lane is evaluated.
struct Policy {
  bool flush_outputs = false;

  /// @brief Read the output control from the result format's MODE.FP_DENORM field.
  /// @details Bit 1 allows output denormals; the input control in bit 0 is ignored.
  static constexpr Policy make(uint32_t denorm_mode) { return {(denorm_mode & 2u) == 0}; }
};

/// @brief Flush a subnormal result to a zero of the same sign when enabled.
template <typename Fmt, typename V> constexpr V flush_output(V bits, const Policy &policy) {
  return policy.flush_outputs ? denormal::flush<Fmt>(bits) : bits;
}

} // namespace rocjitsu::amdgpu::output_denormal
