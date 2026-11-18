// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_status.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

namespace rocjitsu::rdna4 {

// VOPD uses the same ALU causes as the corresponding single-issue operations.
// Normalize the supplied arithmetic result and inputs to a fused a*b+c.
inline uint32_t vopd_flags(unsigned op, uint32_t a, uint32_t b, uint32_t c, uint32_t result,
                           const amdgpu::Wavefront &wave) {
  using F32 = amdgpu::fp_format::F32;
  namespace status = amdgpu::fp_status;
  const auto semantics = wave.cu().arithmetic_flags() & RJ_ARITH_SEMANTICS_MASK;
  if (semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY &&
      semantics != RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL)
    return 0;
  auto denorm = wave.fp_denorm_mode_f32();
  constexpr uint32_t one = 0x3f800000u;
  switch (op) {
  case 0: // FMAC
  case 1: // FMAAK
    break;
  case 2: // FMAMK
    std::swap(b, c);
    break;
  case 3: // MUL
    c = (a ^ b) & F32::kSign;
    break;
  case 4: // ADD
  case 5: // SUB
  case 6: // SUBREV
    c = b ^ (op == 5 ? F32::kSign : 0);
    a ^= op == 6 ? F32::kSign : 0;
    b = one;
    break;
  case 7: // MUL_DX9_ZERO computes zero if either operand is zero.
    denorm = 0;
    if ((a & F32::kMagnitude) < F32::kMinNormal || (b & F32::kMagnitude) < F32::kMinNormal)
      return 0;
    c = (a ^ b) & F32::kSign;
    break;
  case 10: // MAX_NUM
  case 11: // MIN_NUM
    return status::input_causes<F32>(a, denorm) | status::input_causes<F32>(b, denorm);
  default:
    return 0;
  }
  // Compatibility handles exceptional operands and boundary results without
  // recomputing ordinary finite lanes. Strict mode also reports exactness.
  const uint32_t magnitude = result & F32::kMagnitude;
  if (semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY && magnitude < 0x7f7fffffu &&
      (a & F32::kMagnitude) < F32::kInfinity && (b & F32::kMagnitude) < F32::kInfinity &&
      (c & F32::kMagnitude) < F32::kInfinity)
    return status::input_causes<F32>(a, denorm) | status::input_causes<F32>(b, denorm) |
           status::input_causes<F32>(c, denorm);
  return status::fma<F32>(a, b, c, result, wave.fp_round_mode_f32(), denorm);
}

} // namespace rocjitsu::rdna4
