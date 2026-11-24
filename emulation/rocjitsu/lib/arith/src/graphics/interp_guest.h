// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Guest-controlled interpolation reuses the existing directed mixed-FMA arithmetic.
#pragma once

#include "rj_arith/detail/rj_arith_fp.h"
#include <stdint.h>

namespace rocjitsu::arith {

constexpr uint64_t interp_round_denorm_flags =
    RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
    RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK;
constexpr uint64_t interp_fp_flags =
    interp_round_denorm_flags | RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_DX10_CLAMP_OFF;

// Broadcast quad coefficients before invoking FMA; preserve aliases and packed destinations.
template <bool P2, bool Half, bool Rtz = false>
int interp_guest(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  uint32_t av[32], cv[32];
  for (int lane = 0; lane < 32; ++lane) {
    av[lane] = a[0][(lane & ~3) + (P2 ? 2 : 1)];
    cv[lane] = c[0][P2 ? lane : (lane & ~3)];
  }
  const uint32_t *ap = av, *cp = cv;
  flags = (flags & ~RJ_ARITH_SEMANTICS_MASK) | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
  if constexpr (Rtz) {
    flags = (flags & ~RJ_ARITH_FP_ROUND_MASK) | RJ_ARITH_FP_ROUND_ZERO;
  }
  if constexpr (Half) {
    mode |= RJ_ARITH_MIX_F16_A | (P2 ? 0 : RJ_ARITH_MIX_F16_C);
  }
  if constexpr (Half && P2) {
    const bool high = mode & RJ_ARITH_ALU_HIGH_D;
    mode &= ~RJ_ARITH_ALU_HIGH_D;
    auto call = high ? rj_arith_v_fma_mixhi_f16_w32 : rj_arith_v_fma_mixlo_f16_w32;
    return call(flags, exec_mask, mode, d, &ap, b, &cp, nullptr);
  }
  return rj_arith_v_fma_mix_f32_w32(flags, exec_mask, mode, d, &ap, b, &cp, nullptr);
}

} // namespace rocjitsu::arith
