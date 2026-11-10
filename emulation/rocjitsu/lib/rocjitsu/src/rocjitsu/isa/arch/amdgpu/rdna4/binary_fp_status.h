// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_status.h"

namespace rocjitsu::amdgpu::rdna4_fp_status {

enum class Binary { None, Add, Sub, Subrev, Mul, Select };

// Inspect already acquired source bits. Only sums close to the FP32 overflow
// boundary need an additional arithmetic calculation in compatibility mode.
template <Binary Op>
uint32_t binary_causes(uint32_t a, uint32_t b, uint64_t modifiers, uint32_t round,
                       uint32_t denorm) {
  using F32 = fp_format::F32;
  uint32_t causes =
      fp_status::input_causes<F32>(a, denorm) | fp_status::input_causes<F32>(b, denorm);
  if constexpr (Op == Binary::Select)
    return causes;

  const auto policy = input_denormal::Policy::make(denorm);
  a = input_denormal::prepare<F32>(a, policy);
  b = input_denormal::prepare<F32>(b, policy);
  const uint32_t ma = a & F32::kMagnitude, mb = b & F32::kMagnitude;
  if constexpr (Op == Binary::Mul) {
    if ((!ma && mb == F32::kInfinity) || (!mb && ma == F32::kInfinity))
      causes |= fp_status::kInvalid;
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
  if constexpr (Op == Binary::Sub)
    b ^= F32::kSign;
  if constexpr (Op == Binary::Subrev)
    a ^= F32::kSign;
  if ((a ^ b) & F32::kSign) {
    if (ma == F32::kInfinity && mb == F32::kInfinity)
      causes |= fp_status::kInvalid;
    return causes;
  }
  if (ma >= F32::kInfinity || mb >= F32::kInfinity || std::max(ma, mb) < 0x7f000000u)
    return causes;
  fp_mode::ScopedEnvironment nearest(0);
  const double exact = fp_mode::detail::round_to_odd(
      fp_mode::detail::add_exact(fp_status::value<F32>(a), fp_status::value<F32>(b)));
  const uint32_t large_sum = fp_status::result_causes<F32>(exact, 0, round);
  if (large_sum & fp_status::kOverflow)
    causes |=
        fp_status::kOverflow | ((modifiers & RJ_ARITH_ALU_OMOD_MASK) ? 0 : fp_status::kInexact);
  return causes;
}

} // namespace rocjitsu::amdgpu::rdna4_fp_status
