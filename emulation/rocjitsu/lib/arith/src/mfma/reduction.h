// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Baseline MFMA reduction over decoded operand elements, independent of their
// packing.
#pragma once
#include "common/internal.h"
#include "mfma/f32_model.h"
#include "mfma/layout.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Flush subnormal FP32 values to signed zero without changing the host environment.
inline float mfma_flush_f32(float value) {
  uint32_t bits = as_bits(value);
  return as_float((bits & 0x7f800000U) ? bits : bits & 0x80000000U);
}

// Reduce decoded inputs with the selected fusion policy and stage outputs
// before stores.
template <int M, int K, int Batch, int PerWord, bool Bf16 = false,
          MfmaF32Model Model = MfmaF32Model::Default>
void mfma_reduce_scalar(uint64_t instruction_flags, uint32_t *const *d, const float *left,
                        const float *right, const uint32_t *const *c, bool clamp_overflow = false,
                        uint64_t fp_flags = 0) {
  constexpr int ABlockLanes = Model == MfmaF32Model::Gfx90a ? M : 0;
  constexpr int Regs = M * M * Batch / 64;
  uint32_t result[Regs][64];
  for (int batch = 0; batch < Batch; ++batch) {
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < M; ++col) {
        auto out = mfma_output<M, Batch>(row, col, batch);
        float acc = as_float(c[out.reg][out.lane]);
        if constexpr (Model == MfmaF32Model::Gfx90a) {
          if (fp_flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
            acc = mfma_flush_f32(acc);
          }
        }
        for (int k = 0; k < K; ++k) {
          int ai = mfma_permute_input<PerWord, ABlockLanes>(
              mfma_input_index<M, K, Batch, PerWord>(row, k, batch), instruction_flags, true);
          int bi = mfma_permute_input<PerWord, ABlockLanes>(
              mfma_input_index<M, K, Batch, PerWord>(col, k, batch), instruction_flags, false);
          // Preserve fused BF16 and full-width FP32/XF32 cancellation when a
          // separate product would overflow, including on the older inherited models.
          if constexpr (Bf16 || PerWord == 1) {
            acc = __builtin_fmaf(left[ai], right[bi], acc);
          } else {
            acc += left[ai] * right[bi];
          }
        }
        if constexpr (Model == MfmaF32Model::Default) {
          if (clamp_overflow) {
            acc = mfma_clamp_overflow<M, K, Batch, PerWord>(acc, as_float(c[out.reg][out.lane]),
                                                            left, right, row, col, batch,
                                                            instruction_flags);
          }
        }
        if constexpr (Model == MfmaF32Model::Gfx90a) {
          if (fp_flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
            acc = mfma_flush_f32(acc);
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
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
