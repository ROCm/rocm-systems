// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 pair-aware mixed-format layouts and staged dense WMMA reductions.
#pragma once
#include "mfma/mixed_formats.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#if defined(__AVX2__)
#include "mfma/x86_64/mixed_decode.h"
#include "mfma/x86_64/reduction.h"
#include "wmma/x86_64/gfx1250_mixed_reduction.h"
#endif
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Read a logical element in the format-pair-dependent Wave32 layout.
inline float gfx1250_mixed_input(const uint32_t *const *input, int row, int k, int format,
                                 int other_format, bool scaled = false) {
  int bits = matrix_format_bits(format), lane, element;
  if (bits == 8) {
    lane = row % 16 + 16 * (k / 16 % 2);
    element = k / 32 * 16 + k % 16;
  } else if (scaled || other_format < 2) {
    lane = row % 16 + 16 * (k / 32 % 2);
    element = k / 64 * 32 + k % 32;
  } else {
    lane = row % 16 + 16 * (k / 4 % 2);
    element = k / 8 * 4 + k % 4;
  }
  // The 32-row FP4 A operand is two consecutive 16-row pieces.
  return matrix_read_packed(input + (row / 16) * 8, lane, element, format);
}
// Decode inherited E8M0, unsigned E5M3 or signed OCP E4M3 scale bytes.
inline float gfx1250_decode_scale(uint32_t raw, int format) {
  if (format == 2) {
    return fp8_to_float<false>(uint8_t(raw));
  }
  if (raw == 255) {
    return as_float(0x7fc00000U);
  }
  if (format == 0) {
    return as_float(raw ? raw << 23 : 0x00400000U);
  }
  int exponent = int(raw >> 3), fraction = int(raw & 7);
  return exponent ? as_float(uint32_t(exponent + 112) << 23 | uint32_t(fraction) << 20)
                  : float(fraction) * 0x1p-17f;
}
// Select one of the four/eight block scale bytes in one/two Wave32 VGPRs.
inline float gfx1250_scale(const uint32_t *const *input, int lane, int block, int format) {
  return gfx1250_decode_scale((input[block / 4][lane] >> (8 * (block % 4))) & 255, format);
}
#if defined(__AVX2__)
// Stage the scales before reducing. Special inputs use the row-wise kernel
// so competing NaNs retain the backend's original operand priority.
template <int M, int Block>
bool gfx1250_prepare_mixed_scales(uint64_t controls, const uint32_t *const *c,
                                  const uint32_t *const *scale_a, const uint32_t *const *scale_b,
                                  Gfx1250MixedScales<M, Block> &scales) {
  bool finite = true;
  if constexpr (Block != 0) {
    const int a_first = M == 32 ? 0 : 16 * int(controls >> 12 & 1);
    if (gfx1250_mixed_special_scale<Block, M>(scale_a, a_first, int(controls >> 8 & 3)) ||
        gfx1250_mixed_special_scale<Block, 16>(scale_b, 16 * int(controls >> 13 & 1),
                                               int(controls >> 10 & 3))) {
      return false;
    }
    for (int reg = 0; reg < M / 2; ++reg) {
      for (int lane = 0; lane < 32; ++lane) {
        finite &= (c[reg][lane] & 0x7f800000U) != 0x7f800000U;
      }
    }
    if (!finite) {
      return false;
    }
    for (int block = 0; block < scales.groups; ++block) {
      for (int row = 0; row < M; ++row) {
        int lane = M == 32 ? row : row + 16 * int(controls >> 12 & 1);
        scales.a[block][row] = gfx1250_scale(scale_a, lane, block, int(controls >> 8 & 3));
      }
      for (int col = 0; col < 16; ++col) {
        scales.b[block][col] = gfx1250_scale(scale_b, col + 16 * int(controls >> 13 & 1), block,
                                             int(controls >> 10 & 3));
      }
    }
  }
  return true;
}
#endif

// Preserve legacy scalar unfused/SIMD fused accumulation, consuming inputs before stores.
template <int M, int Block = 0, bool Interleaved = true>
#if defined(__AVX512F__)
// Keep the 16-row SCALE16 fallback throughput stable across linker layouts.
[[gnu::aligned(!Interleaved && M == 16 && Block == 16 ? 64 : 16)]]
#endif
    void
    gfx1250_mixed_kernel(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                         const uint32_t *const *b, const uint32_t *const *c,
                         const uint32_t *const *scale_a = nullptr,
                         const uint32_t *const *scale_b = nullptr) {
#if defined(__AVX2__)
  Gfx1250MixedScales<M, Block> scales;
  if constexpr (Interleaved) {
    if (!gfx1250_prepare_mixed_scales<M, Block>(controls, c, scale_a, scale_b, scales) ||
        (M == 16 && (gfx1250_mixed_special_input(a, int(controls & 7)) ||
                     gfx1250_mixed_special_input(b, int(controls >> 3 & 7))))) {
      return gfx1250_mixed_kernel<M, Block, false>(controls, d, a, b, c, scale_a, scale_b);
    }
  }
#endif
  int fa = M == 32 ? 4 : int(controls & 7), fb = M == 32 ? 4 : int(controls >> 3 & 7);
  alignas(64) float result[M][16];
#if defined(__AVX2__)
  alignas(64) float left[128][M], right[128][16];
  const bool interleave_four = Block == 0 && fa >= 2 && fb >= 2;
  mixed_x86::decode<2, 16>(a, fa, interleave_four, left);
  mixed_x86::decode<2, 16>(b, fb, interleave_four, right);
#else
  alignas(64) float left[M][128], right[128][16];
  for (int row = 0; row < M; ++row) {
    for (int k = 0; k < 128; ++k) {
      left[row][k] = gfx1250_mixed_input(a, row, k, fa, fb, Block != 0);
    }
  }
  for (int col = 0; col < 16; ++col) {
    for (int k = 0; k < 128; ++k) {
      right[k][col] = gfx1250_mixed_input(b, col, k, fb, fa, Block != 0);
    }
  }
#endif
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < 16; ++col) {
      uint32_t initial = c[8 * (row / 16) + row % 8][col + 16 * (row % 16 / 8)];
      if (controls & RJ_ARITH_WMMA_MIXED_ABS_C) {
        initial &= 0x7fffffffU;
      }
      if (controls & RJ_ARITH_WMMA_MIXED_NEG_C) {
        initial ^= 0x80000000U;
      }
      result[row][col] = as_float(initial);
    }
#if defined(__AVX2__)
    // The interleaved reduction consumes the staged accumulators for all rows.
    if constexpr (Interleaved) {
      continue;
    }
#endif
    if constexpr (Block != 0) {
      int a_lane = M == 32 ? row : row + 16 * int((controls >> 12) & 1);
      for (int block = 0; block < 128 / Block; ++block) {
        alignas(64) float sums[16] = {};
#if defined(__AVX2__)
        for (int col = 0; col < 16; col += kMfmaWidth) {
          auto sum = mfma_broadcast(0.0f);
          for (int k = block * Block; k < (block + 1) * Block; ++k) {
            sum = mfma_multiply_add(mfma_broadcast(left[k][row]), mfma_load(right[k] + col), sum);
          }
          mfma_store(sums + col, sum);
        }
#else
        for (int col = 0; col < 16; ++col) {
          for (int k = block * Block; k < (block + 1) * Block; ++k) {
            sums[col] = __builtin_fmaf(left[row][k], right[k][col], sums[col]);
          }
        }
#endif
        float as = gfx1250_scale(scale_a, a_lane, block, int(controls >> 8 & 3));
        for (int col = 0; col < 16; ++col) {
          float bs = gfx1250_scale(scale_b, col + 16 * int(controls >> 13 & 1), block,
                                   int(controls >> 10 & 3));
          // Preserve two rounded multiplies followed by a separate addition to C.
          float scaled = sums[col] * as;
          scaled *= bs;
          result[row][col] += scaled;
        }
      }
    } else {
#if defined(__AVX2__)
      for (int col = 0; col < 16; col += kMfmaWidth) {
        auto acc = mfma_load(result[row] + col);
        for (int k = 0; k < 128; ++k) {
          acc = mfma_multiply_add(mfma_broadcast(left[k][row]), mfma_load(right[k] + col), acc);
        }
        mfma_store(result[row] + col, acc);
      }
#else
      for (int col = 0; col < 16; ++col) {
        for (int k = 0; k < 128; ++k) {
          result[row][col] += left[row][k] * right[k][col];
        }
      }
#endif
    }
  }
#if defined(__AVX2__)
  if constexpr (Interleaved) {
    gfx1250_reduce_mixed_tile<M, Block>(left, right, result, scales);
  }
#endif
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < 16; ++col) {
      d[8 * (row / 16) + row % 8][col + 16 * (row % 16 / 8)] = as_bits(result[row][col]);
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
