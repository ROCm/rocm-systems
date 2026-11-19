// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited scalar FP16 math references for decoded instruction tests. These are
// the previously validated legacy mappings; they assume default host FP controls.
#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/pseudo_scalar.h"
#include "util/amdgpu_rcp.h"
#include "util/amdgpu_rsq.h"
#include "util/amdgpu_sqrt.h"
#include "util/amdgpu_trig.h"
#include "util/data_types.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rocjitsu::amdgpu::half_reference {

namespace detail {
// The caller provides default host FP controls and flushes the source half
// under guest MODE before widening it.
// The returned F32 value represents the already rounded architectural half.
template <bool Logarithm>
inline float log_exp_f16_nearest(float x, uint32_t denorm_mode, bool fp16_ovfl, bool quiet_snan) {
  uint32_t bits = std::bit_cast<uint32_t>(x);
  uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) {
    return std::bit_cast<float>(bits | (quiet_snan ? 0x00400000u : 0u));
  }
  if (magnitude == 0x7f800000u) {
    if constexpr (Logarithm) {
      return bits & 0x80000000u ? std::bit_cast<float>(0xffc00000u) : x;
    }
    return bits & 0x80000000u ? 0.0f : x;
  }
  double value;
  if constexpr (Logarithm) {
    if (magnitude == 0) {
      return fp16_ovfl ? -65504.0f : -std::numeric_limits<float>::infinity();
    }
    if (bits & 0x80000000u) {
      return std::bit_cast<float>(0xffc00000u);
    }
    value = std::log2(static_cast<double>(x));
  } else {
    // Outside this interval every nearest half result is zero or overflows.
    // Keep the finite-input provenance instead of overflowing host libm.
    value = std::exp2(std::clamp(static_cast<double>(x), -32.0, 32.0));
  }
  // F32 evaluation can land on a half midpoint and round in the wrong direction.
  const uint16_t result = pseudo_scalar::round_f16_result(value, 0, 0, false, fp16_ovfl, false);
  return util::f16_to_f32(static_cast<uint16_t>(output_denormal::flush_output<fp_format::F16>(
      uint32_t{result}, output_denormal::Policy::make(denorm_mode))));
}
} // namespace detail

enum class HalfOperation { RCP, RSQ, SQRT, SIN, COS };

/// @brief Evaluate a prepared source, round to F16, and apply output-denormal policy.
/// @details The caller flushes the raw half before widening and owns OMOD/CLAMP.
/// Return an already-rounded half represented in F32. FP16_OVFL also saturates
/// infinity produced from zero or a flushed input.
template <HalfOperation Op>
inline float map_f16(float source, uint32_t denorm_mode, bool fp16_ovfl, bool quiet_snan) {
  float value;
  if constexpr (Op == HalfOperation::RCP) {
    value = util::amdgpu_rcp_f32(source, quiet_snan);
  } else if constexpr (Op == HalfOperation::RSQ) {
    value = util::amdgpu_rsq_f32(source, quiet_snan);
  } else if constexpr (Op == HalfOperation::SQRT) {
    value = util::amdgpu_sqrt_f32(source, quiet_snan);
  } else {
    value = util::amdgpu_trig_f32(source, Op == HalfOperation::COS, 3, quiet_snan);
  }
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  if (fp16_ovfl && (bits & 0x7fffffffu) == 0x7f800000u &&
      (std::bit_cast<uint32_t>(source) & 0x7fffffffu) < 0x7f800000u) {
    return std::bit_cast<float>((bits & 0x80000000u) | 0x477fe000u);
  }
  const uint16_t result = util::f32_to_f16_mode(value, fp16_ovfl);
  return util::f16_to_f32(static_cast<uint16_t>(output_denormal::flush_output<fp_format::F16>(
      uint32_t{result}, output_denormal::Policy::make(denorm_mode))));
}

} // namespace rocjitsu::amdgpu::half_reference
