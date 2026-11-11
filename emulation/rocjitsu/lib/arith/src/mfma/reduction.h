// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Baseline MFMA reduction over decoded operand elements, independent of their
// packing.
#pragma once
#include "common/internal.h"
#include "mfma/layout.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Reduce decoded inputs with the selected fusion policy and stage outputs
// before stores.
template <int M, int K, int Batch, int PerWord, bool Bf16 = false>
void mfma_reduce_scalar(uint64_t instruction_flags, uint32_t *const *d, const float *left,
                        const float *right, const uint32_t *const *c) {
  constexpr int Regs = M * M * Batch / 64;
  uint32_t result[Regs][64];
  for (int batch = 0; batch < Batch; ++batch) {
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < M; ++col) {
        auto out = mfma_output<M, Batch>(row, col, batch);
        float acc = as_float(c[out.reg][out.lane]);
        for (int k = 0; k < K; ++k) {
          int ai = mfma_permute_input<PerWord>(
              mfma_input_index<M, K, Batch, PerWord>(row, k, batch), instruction_flags, true);
          int bi = mfma_permute_input<PerWord>(
              mfma_input_index<M, K, Batch, PerWord>(col, k, batch), instruction_flags, false);
          // Preserve fused BF16 and full-width FP32/XF32 cancellation when a
          // separate product would overflow.
          if constexpr (Bf16 || PerWord == 1) {
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
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
