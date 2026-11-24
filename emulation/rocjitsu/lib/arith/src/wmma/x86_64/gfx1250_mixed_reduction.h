// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "mfma/x86_64/reduction.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include "wmma/x86_64/gfx1250_reduction.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// All subbyte encodings are finite. Byte operands occupy sixteen Wave32 VGPRs.
inline bool gfx1250_mixed_special_input(const uint32_t *const *input, int format) {
  if (format == RJ_ARITH_MATRIX_FORMAT_FP8) {
    return gfx1250_has_nonfinite<16, true, 0x7f, 0x7f>(input, input);
  }
  return format == RJ_ARITH_MATRIX_FORMAT_BF8 &&
         gfx1250_has_nonfinite<16, true, 0x7c, 0x7c>(input, input);
}

// A scale is NaN for byte 255 in E8M0 and unsigned E5M3 or a magnitude of 127 in E4M3.
template <int Block, int Lanes>
inline bool gfx1250_mixed_special_scale(const uint32_t *const *scale, int first, int format) {
  const auto mask = _mm256_set1_epi8(format == 2 ? 0x7f : -1);
  for (int reg = 0; reg < 32 / Block; ++reg) {
    for (int lane = first; lane < first + Lanes; lane += 8) {
      const auto bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(scale[reg] + lane));
      if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(bytes, mask), mask))) {
        return true;
      }
    }
  }
  return false;
}

template <int M, int Block> struct Gfx1250MixedScales {
  static constexpr int groups = Block ? 128 / Block : 1;
  alignas(64) float a[groups][M], b[groups][16];
};

// B vectors are shared by independent output rows. Keep one serial FMA chain per
// output and apply each block's two scales in the instruction-defined order.
template <int M, int Block, int ARows, int ACols>
[[gnu::noinline]] void
gfx1250_reduce_mixed_tile(const float (&left)[ARows][ACols], const float (&right)[128][16],
                          float (&result)[M][16], const Gfx1250MixedScales<M, Block> &scales) {
  constexpr int tile = Block == 0 ? 8 : kMfmaWidth / 2;
  constexpr int span = Block ? Block : 128;
  static_assert((ARows == M && ACols == 128) || (ARows == 128 && ACols == M));
  auto a_value = [&](int row, int k) {
    if constexpr (ARows == 128) {
      return left[k][row];
    } else {
      return left[row][k];
    }
  };
  for (int row = 0; row < M; row += tile) {
    for (int col = 0; col < 16; col += kMfmaWidth) {
      MfmaNative acc[tile];
      for (int r = 0; r < tile; ++r) {
        acc[r] = mfma_load(result[row + r] + col);
      }
      for (int block = 0; block < 128 / span; ++block) {
        MfmaNative sum[tile];
        for (int r = 0; r < tile; ++r) {
          sum[r] = Block ? mfma_broadcast(0) : acc[r];
        }
        for (int k = block * span; k < (block + 1) * span; ++k) {
          auto b = mfma_load(right[k] + col);
          RJ_ARITH_UNROLL(16)
          for (int r = 0; r < tile; ++r) {
            sum[r] = mfma_multiply_add(mfma_broadcast(a_value(row + r, k)), b, sum[r]);
          }
        }
        for (int r = 0; r < tile; ++r) {
          if constexpr (Block != 0) {
            acc[r] += (sum[r] * mfma_broadcast(scales.a[block][row + r])) *
                      mfma_load(scales.b[block] + col);
          } else {
            acc[r] = sum[r];
          }
        }
      }
      for (int r = 0; r < tile; ++r) {
        mfma_store(result[row + r] + col, acc[r]);
      }
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
