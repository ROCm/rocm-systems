// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA FP16/BF16 vector dots: host FP32 pair sum followed by accumulation.
// Inherits rocjitsu's arithmetic order, not the distinct gfx11/gfx12 exact models.

#include "common/float_encoding.h"
#include "common/internal.h"
#include "common/simd.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_dot.h"

#include <stdint.h>

namespace {

// Compute selected lanes after validating the complete request. Guest FP modes and
// CLAMP do not affect this inherited model; unsupported exact requests leave D unchanged.
template <bool Bf16, bool Accumulate, typename Mask>
int dot(uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(flags, mode & ~(Accumulate ? 0ULL : 0x7ffULL), kMatrixIgnoredFpFlags)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) > RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int base = 0; base < int(sizeof(Mask) * 8); base += 32) {
    const uint32_t *av = a[0] + base, *bv = b[0] + base;
    uint32_t *dv = d[0] + base;
    const uint32_t *cv = Accumulate ? dv : c[0] + base;
    const uint32_t active = uint32_t(exec_mask >> base);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
    if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
      cdna_dot2_x86_64_v4(Bf16, active, uint32_t(mode), dv, av, bv, cv);
      continue;
    }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
      cdna_dot2_x86_64_v3(Bf16, active, uint32_t(mode), dv, av, bv, cv);
      continue;
    }
#endif
    const int a0 = mode & RJ_ARITH_DOT_LO_A_HIGH ? 16 : 0;
    const int b0 = mode & RJ_ARITH_DOT_LO_B_HIGH ? 16 : 0;
    const int a1 = mode & RJ_ARITH_DOT_HI_A_LOW ? 0 : 16;
    const int b1 = mode & RJ_ARITH_DOT_HI_B_LOW ? 0 : 16;
    auto factor = [](uint32_t bits, int shift, bool negate) {
      uint16_t half = uint16_t(bits >> shift) ^ (negate ? 0x8000U : 0U);
      return Bf16 ? bf16_to_float(half) : f16_to_float(half);
    };
    for (int lane = 0; lane < 32; ++lane) {
      if (!((active >> lane) & 1)) {
        continue;
      }
      const float low = factor(av[lane], a0, mode & RJ_ARITH_DOT_NEG_LO_A) *
                        factor(bv[lane], b0, mode & RJ_ARITH_DOT_NEG_LO_B);
      const float high = factor(av[lane], a1, mode & RJ_ARITH_DOT_NEG_HI_A) *
                         factor(bv[lane], b1, mode & RJ_ARITH_DOT_NEG_HI_B);
      const uint32_t acc = (cv[lane] & (mode & RJ_ARITH_DOT_ABS_C ? 0x7fffffffU : ~0U)) ^
                           (mode & RJ_ARITH_DOT_NEG_C ? 0x80000000U : 0U);
      dv[lane] = as_bits((low + high) + as_float(acc));
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2_f32_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c,
                                       [[maybe_unused]] uint32_t *trapsts) {
  return dot<false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2_f32_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c,
                                       [[maybe_unused]] uint32_t *trapsts) {
  return dot<false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2c_f32_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        [[maybe_unused]] uint32_t *trapsts) {
  return dot<false, true>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2c_f32_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        [[maybe_unused]] uint32_t *trapsts) {
  return dot<false, true>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2_f32_bf16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c,
                                        [[maybe_unused]] uint32_t *trapsts) {
  return dot<true, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2_f32_bf16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c,
                                        [[maybe_unused]] uint32_t *trapsts) {
  return dot<true, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2c_f32_bf16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         [[maybe_unused]] uint32_t *trapsts) {
  return dot<true, true>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2c_f32_bf16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         [[maybe_unused]] uint32_t *trapsts) {
  return dot<true, true>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}
