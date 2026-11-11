// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16/BF16 MFMA validation, baseline arithmetic and dispatch for
// gfx942/gfx950.
#include "common/float_formats.h"
#include "common/internal.h"
#include "mfma/backends.h"
#include "mfma/layout.h"
#include "rj_arith/rj_arith.h"

namespace {

// Evaluate a full MFMA wave after validating controls; errors do not access
// operands.
template <int M, int K, int Batch, bool Bf16>
int mfma(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
         const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(flags, instruction_flags & ~0x3ffULL,
                           kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    mfma_half_x86_64_v4<M, K, Batch, Bf16>(instruction_flags, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    mfma_half_x86_64_v3<M, K, Batch, Bf16>(instruction_flags, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  constexpr int Regs = M * M * Batch / 64;
  uint32_t result[Regs][64];
  // Convert each source once, even when multiple output rows reuse it.
  float left[M * K * Batch], right[M * K * Batch];
  for (int i = 0; i < M * K * Batch; ++i) {
    uint16_t x = uint16_t(a[i / 128][i / 2 % 64] >> (16 * (i % 2)));
    uint16_t y = uint16_t(b[i / 128][i / 2 % 64] >> (16 * (i % 2)));
    left[i] = Bf16 ? bf16_to_float(x) : f16_to_float(x);
    right[i] = Bf16 ? bf16_to_float(y) : f16_to_float(y);
  }
  for (int batch = 0; batch < Batch; ++batch) {
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < M; ++col) {
        auto out = mfma_output<M, Batch>(row, col, batch);
        float acc = as_float(c[out.reg][out.lane]);
        for (int k = 0; k < K; ++k) {
          int ai = mfma_permute_half(mfma_half_index<M, K, Batch>(row, k, batch), instruction_flags,
                                     true);
          int bi = mfma_permute_half(mfma_half_index<M, K, Batch>(col, k, batch), instruction_flags,
                                     false);
          // Preserve BF16 cancellation even when the separate product would
          // overflow.
          if constexpr (Bf16) {
            acc = __builtin_fmaf(left[ai], right[bi], acc);
          } else {
            acc += left[ai] * right[bi];
          }
        }
        result[out.reg][out.lane] = as_bits(acc);
      }
    }
  }
  for (int reg = 0; reg < Regs; ++reg) {
    for (int lane = 0; lane < 64; ++lane) {
      d[reg][lane] = result[reg][lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate 32x32x4 FP16 MFMA with 2 independent matrix block(s).
int rj_arith_v_mfma_f32_32x32x4_2b_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 4, 2, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 16x16x4 FP16 MFMA with 4 independent matrix block(s).
int rj_arith_v_mfma_f32_16x16x4_4b_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<16, 4, 4, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 4x4x4 FP16 MFMA with 16 independent matrix block(s).
int rj_arith_v_mfma_f32_4x4x4_16b_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<4, 4, 16, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 32x32x8 FP16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_32x32x8_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 8, 1, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 16x16x16 FP16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_16x16x16_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<16, 16, 1, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 16x16x32 FP16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_16x16x32_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<16, 32, 1, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 32x32x16 FP16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_32x32x16_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 16, 1, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 32x32x4 BF16 MFMA with 2 independent matrix block(s).
int rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                            uint32_t *const *d, const uint32_t *const *a,
                                            const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 4, 2, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 16x16x4 BF16 MFMA with 4 independent matrix block(s).
int rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                            uint32_t *const *d, const uint32_t *const *a,
                                            const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<16, 4, 4, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 4x4x4 BF16 MFMA with 16 independent matrix block(s).
int rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<4, 4, 16, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 32x32x8 BF16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_32x32x8_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 8, 1, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 16x16x16 BF16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_16x16x16_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<16, 16, 1, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 16x16x32 BF16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_16x16x32_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<16, 32, 1, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 32x32x16 BF16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_32x32x16_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 16, 1, true>(flags, instruction_flags, d, a, b, c);
}
