// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/hwfloat/mul_f32.h"

namespace rocjitsu::amdgpu::hwfloat {

/// Explicit policy measured for DX9 MUL on concrete gfx1100 and gfx1201.
/// Other targets remain unqualified. Measured causes use the shared ALU trap
/// delivery machinery, independently of the numerical result.
struct Dx9MulF32Policy {
  uint32_t round = 0;
  uint32_t denorm = 0;
  bool rdna4 = false;
  bool ieee = false;
  bool dx10_clamp = false;
  uint32_t omod = 0;
  bool clamp = false;
};

struct Dx9MulF32Result {
  uint32_t bits = 0;
  uint32_t causes = 0;
};

/// Inputs already carry source ABS/NEG. All arithmetic and classification use
/// integer bits; no host floating-point state or exception flags are touched.
constexpr Dx9MulF32Result evaluate_dx9_mul_f32(uint32_t a, uint32_t b, Dx9MulF32Policy policy) {
  uint32_t omod = policy.omod;
  if (!policy.rdna4 && ((policy.denorm & 2u) || policy.ieee))
    omod = 0;
  const MulF32Policy base_policy{policy.round, omod ? (policy.denorm & 1u) : policy.denorm,
                                 policy.rdna4 || policy.ieee};
  auto result = multiply_f32(a, b, base_policy);
  const auto flushed_magnitude = [&](uint32_t bits) {
    const uint32_t magnitude = bits & 0x7fffffffu;
    return !(base_policy.denorm & 1u) && magnitude < 0x00800000u ? 0u : magnitude;
  };
  if (!flushed_magnitude(a) || !flushed_magnitude(b)) {
    result.bits = 0;
    // Zero annihilates infinity without INVALID, but does not hide a signaling
    // NaN. Keep the original input-denormal fact for preserved subnormals.
    const auto signaling_nan = [](uint32_t bits) {
      return (bits & 0x7fffffffu) > 0x7f800000u && !(bits & 0x00400000u);
    };
    if (!signaling_nan(a) && !signaling_nan(b))
      result.facts &= ~mul_f32_fact::kInvalid;
  }
  uint32_t bits = result.bits;
  uint32_t causes = plain_mul_f32_causes(result, base_policy);
  if (omod) {
    // Scaling takes +0 for either signed zero or a tiny intermediate result.
    if ((bits & 0x7fffffffu) < 0x00800000u)
      bits = 0;
    const uint32_t scale = omod == 1 ? 0x40000000u : omod == 2 ? 0x40800000u : 0x3f000000u;
    const auto scaled = multiply_f32(bits, scale, base_policy);
    bits = scaled.bits;
    causes |= plain_mul_f32_causes(scaled, base_policy);
    if (policy.rdna4)
      causes &= ~(mul_f32_cause::kUnderflow | mul_f32_cause::kInexact);
  }
  if (policy.clamp) {
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
      if (policy.rdna4 || policy.dx10_clamp)
        bits = 0;
    } else if ((bits >> 31) || !magnitude) {
      bits = 0;
    } else if (bits > 0x3f800000u) {
      bits = 0x3f800000u;
    }
  }
  // Encoded OMOD suppresses causes on gfx1100 even when MODE disables its
  // numeric effect. CLAMP suppresses causes on both qualified targets.
  if (policy.clamp || (!policy.rdna4 && (policy.omod || policy.dx10_clamp)))
    causes = 0;
  return {bits, causes};
}

} // namespace rocjitsu::amdgpu::hwfloat
