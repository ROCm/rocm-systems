// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_status.h"

namespace rocjitsu::rdna4 {

enum class BinaryFpOperation { None, Add, Sub, Subrev, Mul, Select };

// Inspect already acquired source bits. Only sums close to the FP32 overflow
// boundary need an additional arithmetic calculation in compatibility mode.
template <BinaryFpOperation Op>
uint32_t binary_causes(uint32_t a, uint32_t b, uint64_t modifiers, uint32_t round,
                       uint32_t denorm) {
  using F32 = amdgpu::fp_format::F32;
  uint32_t causes = amdgpu::fp_status::input_causes<F32>(a, denorm) |
                    amdgpu::fp_status::input_causes<F32>(b, denorm);
  if constexpr (Op == BinaryFpOperation::Select)
    return causes;

  const auto policy = amdgpu::input_denormal::Policy::make(denorm);
  a = amdgpu::input_denormal::prepare<F32>(a, policy);
  b = amdgpu::input_denormal::prepare<F32>(b, policy);
  const uint32_t ma = a & F32::kMagnitude, mb = b & F32::kMagnitude;
  if constexpr (Op == BinaryFpOperation::Mul) {
    if ((!ma && mb == F32::kInfinity) || (!mb && ma == F32::kInfinity))
      causes |= amdgpu::fp_status::kInvalid;
    return causes;
  }

  if (modifiers & RJ_ARITH_ALU_ABS_A)
    a = ma;
  if (modifiers & RJ_ARITH_ALU_ABS_B)
    b = mb;
  if (modifiers & RJ_ARITH_ALU_NEG_A)
    a ^= F32::kSign;
  if (modifiers & RJ_ARITH_ALU_NEG_B)
    b ^= F32::kSign;
  if constexpr (Op == BinaryFpOperation::Sub)
    b ^= F32::kSign;
  if constexpr (Op == BinaryFpOperation::Subrev)
    a ^= F32::kSign;
  if ((a ^ b) & F32::kSign) {
    if (ma == F32::kInfinity && mb == F32::kInfinity)
      causes |= amdgpu::fp_status::kInvalid;
    return causes;
  }
  if (ma >= F32::kInfinity || mb >= F32::kInfinity || std::max(ma, mb) < 0x7f000000u)
    return causes;
  amdgpu::fp_mode::ScopedEnvironment nearest(0);
  const double exact = amdgpu::fp_mode::detail::round_to_odd(amdgpu::fp_mode::detail::add_exact(
      amdgpu::fp_status::value<F32>(a), amdgpu::fp_status::value<F32>(b)));
  const uint32_t large_sum = amdgpu::fp_status::result_causes<F32>(exact, 0, round);
  if (large_sum & amdgpu::fp_status::kOverflow)
    causes |= amdgpu::fp_status::kOverflow |
              ((modifiers & RJ_ARITH_ALU_OMOD_MASK) ? 0 : amdgpu::fp_status::kInexact);
  return causes;
}

} // namespace rocjitsu::rdna4
