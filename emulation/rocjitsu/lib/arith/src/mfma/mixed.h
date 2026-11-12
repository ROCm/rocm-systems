// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx950 mixed-format MFMA staging and reduction, isolated by CPU ISA.
#pragma once

#include "mfma/layout.h"
#include "mfma/mixed_formats.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#if defined(__AVX2__)
#include "mfma/x86_64/reduction.h"
#endif

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Decode an operand in the inherited byte-chunked or contiguous-subbyte lane layout.
template <int M, int K>
void mfma_decode_mixed(const uint32_t *const *input, int format, float output[M][K]) {
  constexpr int lanes_per_row = 64 / M, per_lane = K / lanes_per_row;
  for (int row = 0; row < M; ++row) {
    for (int k = 0; k < K; ++k) {
      int lane, element;
      if (matrix_format_bits(format) == 8 && per_lane > 16) {
        lane = (k / 16 % lanes_per_row) * M + row;
        element = (k / (16 * lanes_per_row)) * 16 + k % 16;
      } else {
        lane = (k / per_lane) * M + row;
        element = k % per_lane;
      }
      output[row][k] = matrix_read_packed(input, lane, element, format);
    }
  }
}

// Stage every output before stores; preserve inherited reductions and optional block scaling.
template <int M, int K, bool Scaled = false>
void mfma_mixed_kernel(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                       const uint32_t *const *b, const uint32_t *const *c,
                       const uint32_t *scale_a = nullptr, const uint32_t *scale_b = nullptr) {
  alignas(64) float left[M][K], right_rows[M][K], right[K][M], result[M][M];
  mfma_decode_mixed<M, K>(a, int(controls & 7), left);
  mfma_decode_mixed<M, K>(b, int(controls >> 3 & 7), right_rows);
  for (int k = 0; k < K; ++k) {
    for (int col = 0; col < M; ++col) {
      right[k][col] = right_rows[col][k];
    }
  }
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < M; ++col) {
      auto out = mfma_output<M, 1>(row, col, 0);
      uint32_t initial = c[out.reg][out.lane];
      if constexpr (Scaled) {
        if (controls & RJ_ARITH_MFMA_SCALE_ABS_C) {
          initial &= 0x7fffffffU;
        }
        if (controls & RJ_ARITH_MFMA_SCALE_NEG_C) {
          initial ^= 0x80000000U;
        }
      }
      result[row][col] = as_float(initial);
    }
    if constexpr (Scaled) {
      // Scale each fused 32-product partial once, then add it separately to C.
      for (int block = 0; block < K / 32; ++block) {
        alignas(64) float sums[M] = {};
#if defined(__AVX2__)
        for (int col = 0; col < M; col += kMfmaWidth) {
          auto sum = mfma_broadcast(0.0f);
          for (int k = 32 * block; k < 32 * (block + 1); ++k) {
            sum = mfma_multiply_add(mfma_broadcast(left[row][k]), mfma_load(right[k] + col), sum);
          }
          mfma_store(sums + col, sum);
        }
#else
        for (int col = 0; col < M; ++col) {
          for (int k = 32 * block; k < 32 * (block + 1); ++k) {
            sums[col] = __builtin_fmaf(left[row][k], right[k][col], sums[col]);
          }
        }
#endif
        int a_exp = int(scale_a[M * block + row] >> (8 * ((controls >> 6) & 3)) & 255);
        for (int col = 0; col < M; ++col) {
          int b_exp = int(scale_b[M * block + col] >> (8 * ((controls >> 8) & 3)) & 255);
          float scaled = a_exp == 255 || b_exp == 255
                             ? as_float(0x7fc00000U)
                             : __builtin_ldexpf(sums[col], a_exp + b_exp - 254);
          result[row][col] += scaled;
        }
      }
    } else {
#if defined(__AVX2__)
      for (int col = 0; col < M; col += kMfmaWidth) {
        auto acc = mfma_load(result[row] + col);
        for (int k = 0; k < K; ++k) {
          acc = mfma_multiply_add(mfma_broadcast(left[row][k]), mfma_load(right[k] + col), acc);
        }
        mfma_store(result[row] + col, acc);
      }
#else
      for (int col = 0; col < M; ++col) {
        for (int k = 0; k < K; ++k) {
          result[row][col] += left[row][k] * right[k][col];
        }
      }
#endif
    }
  }
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < M; ++col) {
      auto out = mfma_output<M, 1>(row, col, 0);
      d[out.reg][out.lane] = as_bits(result[row][col]);
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
