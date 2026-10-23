// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Scalar/SIMD input denormal model inherited from rocjitsu.
// Works on raw encodings; guest FP controls never change the host FP environment.

#include "conversion/model/denormal.h"

#include <cstdint>

namespace rocjitsu::arith::conversion_model::input_denormal {

/// @brief Input-flush policy, fixed before any lane is evaluated.
struct Policy {
  bool flush_inputs = false;

  /// @brief Read the input control from the source format's MODE.FP_DENORM field.
  /// @details Bit 0 allows input denormals; the output control in bit 1 is ignored.
  static constexpr Policy make(uint32_t denorm_mode) { return {(denorm_mode & 1u) == 0}; }
};

/// @brief Flush a subnormal source to a zero of the same sign when enabled.
/// @details NaN, infinity, zero and normal encodings pass through unchanged.
template <typename Fmt, typename V> constexpr V flush_input(V bits, const Policy &policy) {
  return policy.flush_inputs ? denormal::flush<Fmt>(bits) : bits;
}

/// @brief Mask to the source format and apply input flushing.
template <typename Fmt, typename V> constexpr V prepare(V bits, const Policy &policy) {
  return flush_input<Fmt>(bits & Fmt::kBits, policy);
}

} // namespace rocjitsu::arith::conversion_model::input_denormal
