// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Deinterleave packed MFMA and WMMA operands into K-major SIMD rows.
#pragma once

#include "common/compiler.h"
#include "common/x86_64/fp8.h"
#include "mfma/mixed_formats.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include <cstring>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::mixed_x86 {
#if defined(__AVX512F__)
using Native = __m512i;
constexpr int Width = 16;
#else
using Native = __m256i;
constexpr int Width = 8;
#endif
using Words [[gnu::vector_size(Width * 4)]] = uint32_t;
using Values [[gnu::vector_size(Width * 4)]] = float;

// Load adjacent wave lanes without assuming that the VGPR is SIMD-aligned.
inline Words load_words(const uint32_t *p) {
  Words v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

// The extracted encoding occupies the low bits. Each widening operation ignores the
// higher bits that belong to subsequent elements in the packed VGPR.
template <int Format> inline Values widen(Words code) {
  if constexpr (Format <= RJ_ARITH_MATRIX_FORMAT_BF8) {
    return widen_fp8<Format == RJ_ARITH_MATRIX_FORMAT_BF8>(bit_cast<Native>(code));
  } else if constexpr (Format == RJ_ARITH_MATRIX_FORMAT_FP4) {
#if defined(__AVX512F__)
    const auto values =
        _mm512_setr_ps(0, 0.5f, 1, 1.5, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5, -2, -3, -4, -6);
    return _mm512_permutexvar_ps(bit_cast<Native>(code), values);
#else
    const auto values = _mm256_setr_ps(0, 0.5f, 1, 1.5, 2, 3, 4, 6);
    auto mag = _mm256_permutevar8x32_ps(values, bit_cast<Native>(code));
    return bit_cast<Values>(bit_cast<Words>(mag) | ((code & 8U) << 28));
#endif
  } else {
#if defined(__AVX512F__)
    // Every FP6 magnitude is exactly representable in the upper half of FP32.
    // All 32 entries fit in one register, indexed by the low five bits of each code.
    struct Magnitudes {
      uint16_t bits[32]{};
      constexpr Magnitudes() {
        for (int i = 0; i < 32; ++i) {
          bits[i] = uint16_t(bit_cast<uint32_t>(matrix_decode_subbyte(i, Format)) >> 16);
        }
      }
    };
    alignas(64) static constexpr Magnitudes table;
    auto index = bit_cast<Native>(code);
    auto halves = _mm512_permutexvar_epi16(index, _mm512_load_si512(table.bits));
    return bit_cast<Values>((bit_cast<Words>(halves) << 16) | ((code & 32u) << 26));
#else
    constexpr int fraction = Format == RJ_ARITH_MATRIX_FORMAT_FP6 ? 3 : 2;
    constexpr int bias = Format == RJ_ARITH_MATRIX_FORMAT_FP6 ? 1 : 3;
    Words magnitude = code & 31u;
    Words normal = (magnitude << (23 - fraction)) + ((127u - bias) << 23);
    Values small = __builtin_convertvector(magnitude, Values) *
                   (Format == RJ_ARITH_MATRIX_FORMAT_FP6 ? 0x1p-3f : 0x1p-4f);
    Words value = magnitude < (1u << fraction) ? bit_cast<Words>(small) : normal;
    return bit_cast<Values>(value | ((code & 32u) << 26));
#endif
  }
}

// Consecutive lanes hold independent matrix rows. Decode directly into K-by-row
// storage. Only the six-bit codes straddling a VGPR need a second read.
template <int Format, int Block, int Groups, int RowsPerGroup, int Rows, int K>
void unpack(const uint32_t *const *input, float (&out)[K][Rows]) {
  constexpr int Bits = matrix_format_bits(Format);
  static_assert(RowsPerGroup % Width == 0 && Rows % RowsPerGroup == 0 && K % (Block * Groups) == 0);
  for (int k = 0; k < K; k += Block) {
    const int lane = (k / Block % Groups) * RowsPerGroup;
    const int first = (k / (Block * Groups)) * Block;
    for (int row = 0; row < Rows; row += Width) {
      // Wide gfx1250 FP4 uses a second bank of eight VGPRs for rows 16..31.
      const auto regs = input + (row / RowsPerGroup) * 8;
      const int first_lane = lane + row % RowsPerGroup;
      RJ_ARITH_UNROLL(32)
      for (int j = 0; j < Block; ++j) {
        int bit = (first + j) * Bits;
        auto word = load_words(regs[bit / 32] + first_lane);
        auto code = word >> (bit % 32);
        if constexpr (Bits == 6) {
          if (bit % 32 + Bits > 32) {
            code |= load_words(regs[bit / 32 + 1] + first_lane) << (32 - bit % 32);
          }
        }
        auto values = widen<Format>(code);
        std::memcpy(out[k + j] + row, &values, sizeof(values));
      }
    }
  }
}

// Each lane supplies 16 byte-format elements or 32 subbyte elements at a time.
// Unscaled gfx1250 subbyte pairs instead interleave groups of four.
template <int Groups, int RowsPerGroup, int Rows, int K>
void decode(const uint32_t *const *input, int format, bool interleave_four, float (&out)[K][Rows]) {
  auto decode_format = [&]<int Format>() {
    if constexpr (Format <= RJ_ARITH_MATRIX_FORMAT_BF8) {
      unpack<Format, 16, Groups, RowsPerGroup>(input, out);
    } else if (interleave_four) {
      unpack<Format, 4, Groups, RowsPerGroup>(input, out);
    } else {
      unpack<Format, 32, Groups, RowsPerGroup>(input, out);
    }
  };
  switch (format) {
  case RJ_ARITH_MATRIX_FORMAT_FP8:
    return decode_format.template operator()<RJ_ARITH_MATRIX_FORMAT_FP8>();
  case RJ_ARITH_MATRIX_FORMAT_BF8:
    return decode_format.template operator()<RJ_ARITH_MATRIX_FORMAT_BF8>();
  case RJ_ARITH_MATRIX_FORMAT_FP6:
    return decode_format.template operator()<RJ_ARITH_MATRIX_FORMAT_FP6>();
  case RJ_ARITH_MATRIX_FORMAT_BF6:
    return decode_format.template operator()<RJ_ARITH_MATRIX_FORMAT_BF6>();
  default:
    return decode_format.template operator()<RJ_ARITH_MATRIX_FORMAT_FP4>();
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::mixed_x86
