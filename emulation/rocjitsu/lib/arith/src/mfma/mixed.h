// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx950 mixed-format MFMA staging and reduction, isolated by CPU ISA.
//
// Reading order: mfma_mixed_kernel, at the bottom, decodes A and B and then dispatches to the
// plain reduction or to the block-scaled one. The vector reductions walk the output in tiles of
// four rows by one vector of columns, spelled in terms of the per-ISA primitives defined above
// them, so one tile body serves both vector widths.
#pragma once

#include "mfma/layout.h"
#include "mfma/mixed_formats.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include <cstring>
#if defined(__AVX2__)
#include "mfma/x86_64/mixed_decode.h"
#include "mfma/x86_64/reduction.h"
#endif

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// An E8M0 scale biases its exponent by 127 and reserves 255 for NaN, so one A scale and one B
// scale together carry twice that bias.
inline constexpr int kE8m0NanExponent = 255;
inline constexpr int kE8m0PairBias = 254;

// FP32 fields the scaled reductions test on the integer side, where a vector compare replaces a
// classification call.
inline constexpr uint32_t kFp32Exponent = 0x7f800000u;
inline constexpr uint32_t kFp32Magnitude = 0x7fffffffu;
inline constexpr uint32_t kFp32Sign = 0x80000000u;
inline constexpr uint32_t kFp32QuietNan = 0x7fc00000u;

// Rows per reduction tile. Four accumulators reuse each B load and keep four reduction chains in
// flight while the per-tile constants stay in registers. RJ_ARITH_UNROLL spells the same count
// where the pragma needs a literal.
inline constexpr int kMfmaRowTile = 4;

// Position of an operand's selected scale byte, from the two control bits that choose it.
inline int mfma_scale_byte_shift(uint64_t controls, uint64_t control_shift) {
  return 8 * int((controls >> control_shift) & 3);
}

// Extract the selected E8M0 byte of all 64 scale lanes once, before the reduction reads each of
// them for a whole row or column. Exponents stay signed so the reduction's index arithmetic and
// its vector loads share one type.
inline void mfma_scale_exponents(const uint32_t *scale, int byte_shift, int32_t exponents[64]) {
  for (int lane = 0; lane < 64; ++lane) {
    exponents[lane] = int32_t((scale[lane] >> byte_shift) & 0xffu);
  }
}

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

#if defined(__AVX2__)
// Primitives the scaled tile is written in terms of, beyond the ones shared with the other
// backends in mfma/x86_64/reduction.h. Each is one host instruction or a short fixed sequence,
// defined once per vector width.
#if defined(__AVX512F__)
using MfmaBits = __m512i;   // one output vector seen as raw FP32 bits
using MfmaMask = __mmask16; // one bit per output column

inline MfmaNative mfma_add(MfmaNative left, MfmaNative right) { return _mm512_add_ps(left, right); }
inline MfmaBits mfma_bits_broadcast(uint32_t value) { return _mm512_set1_epi32(int32_t(value)); }
inline MfmaBits mfma_bits_loadu(const uint32_t *source) { return _mm512_loadu_si512(source); }
inline MfmaBits mfma_exponents_load(const int32_t *source) { return _mm512_load_si512(source); }
inline MfmaNative mfma_bits_as_float(MfmaBits bits) { return _mm512_castsi512_ps(bits); }

// Apply the C-operand absolute-value and negate modifiers while C is still integer.
inline MfmaBits mfma_modify_c(MfmaBits bits, MfmaBits magnitude, MfmaBits sign_flip) {
  return _mm512_xor_si512(_mm512_and_si512(bits, magnitude), sign_flip);
}

// Select the columns holding an all-ones exponent, that is an infinity or a NaN.
inline MfmaMask mfma_nonfinite(MfmaBits bits) {
  const MfmaBits exponent = mfma_bits_broadcast(kFp32Exponent);
  return _mm512_cmpeq_epi32_mask(_mm512_and_si512(bits, exponent), exponent);
}
inline MfmaMask mfma_nonfinite(MfmaNative values) {
  return mfma_nonfinite(_mm512_castps_si512(values));
}
inline MfmaMask mfma_mask_or(MfmaMask left, MfmaMask right) { return left | right; }

// Scale a block dot by 2^(a_exp + b_exp - 254), or deliver a canonical quiet NaN where either
// scale is the E8M0 NaN. vscalefps applies the whole exponent with one rounding.
inline MfmaNative mfma_scale_e8m0(MfmaNative sum, int a_exp, MfmaBits b_exps) {
  const MfmaMask scalable =
      a_exp == kE8m0NanExponent
          ? MfmaMask{0}
          : _mm512_cmpneq_epi32_mask(b_exps, _mm512_set1_epi32(kE8m0NanExponent));
  const MfmaBits exponent = _mm512_add_epi32(_mm512_set1_epi32(a_exp - kE8m0PairBias), b_exps);
  return _mm512_mask_scalef_ps(mfma_bits_as_float(mfma_bits_broadcast(kFp32QuietNan)), scalable,
                               sum, _mm512_cvtepi32_ps(exponent));
}

// Saturate an overflow that finite inputs alone produced: step the infinity down to the largest
// finite FP32 of the same sign. Columns that saw a nonfinite input keep the infinity.
inline MfmaNative mfma_clamp_scaled_overflow(MfmaNative acc, MfmaMask nonfinite) {
  const MfmaBits bits = _mm512_castps_si512(acc);
  const MfmaMask at_infinity = _mm512_mask_cmpeq_epi32_mask(
      MfmaMask(~nonfinite), _mm512_and_si512(bits, mfma_bits_broadcast(kFp32Magnitude)),
      mfma_bits_broadcast(kFp32Exponent));
  return _mm512_castsi512_ps(_mm512_mask_sub_epi32(bits, at_infinity, bits, _mm512_set1_epi32(1)));
}
#else
using MfmaBits = __m256i; // one output vector seen as raw FP32 bits
using MfmaMask = __m256i; // all ones in each selected output column

inline MfmaNative mfma_add(MfmaNative left, MfmaNative right) { return _mm256_add_ps(left, right); }
inline MfmaBits mfma_bits_broadcast(uint32_t value) { return _mm256_set1_epi32(int32_t(value)); }
inline MfmaBits mfma_bits_loadu(const uint32_t *source) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(source));
}
inline MfmaBits mfma_exponents_load(const int32_t *source) {
  return _mm256_load_si256(reinterpret_cast<const __m256i *>(source));
}
inline MfmaNative mfma_bits_as_float(MfmaBits bits) { return _mm256_castsi256_ps(bits); }

// Apply the C-operand absolute-value and negate modifiers while C is still integer.
inline MfmaBits mfma_modify_c(MfmaBits bits, MfmaBits magnitude, MfmaBits sign_flip) {
  return _mm256_xor_si256(_mm256_and_si256(bits, magnitude), sign_flip);
}

// Select the columns holding an all-ones exponent, that is an infinity or a NaN.
inline MfmaMask mfma_nonfinite(MfmaBits bits) {
  const MfmaBits exponent = mfma_bits_broadcast(kFp32Exponent);
  return _mm256_cmpeq_epi32(_mm256_and_si256(bits, exponent), exponent);
}
inline MfmaMask mfma_nonfinite(MfmaNative values) {
  return mfma_nonfinite(_mm256_castps_si256(values));
}
inline MfmaMask mfma_mask_or(MfmaMask left, MfmaMask right) { return _mm256_or_si256(left, right); }

// Scale a block dot by 2^(a_exp + b_exp - 254), or deliver a canonical quiet NaN where either
// scale is the E8M0 NaN.
//
// Without vscalefps the exponent goes on as two multiplies, because one FP32 factor cannot span
// the 2^-254..2^254 the scale pair asks for. Splitting the pair exponent into halves that differ
// by at most one keeps both factors normal and keeps the intermediate product in range: every
// nonzero finite dot of 32 F8/F6/F4 products has magnitude inside 2^-32..2^36, so the intermediate
// can only leave FP32 range when the final result does as well, and every step in between is
// exact. The halves clamp at exponent 1 only for pair exponents below 2, where the final result
// underflows to zero for every dot in that range and the clamped form agrees anyway.
inline MfmaNative mfma_scale_e8m0(MfmaNative sum, int a_exp, MfmaBits b_exps) {
  const MfmaBits one = _mm256_set1_epi32(1);
  const MfmaBits a_exps = _mm256_set1_epi32(a_exp);
  const MfmaBits pair = _mm256_add_epi32(a_exps, b_exps);
  const MfmaBits low_half = _mm256_max_epi32(_mm256_srli_epi32(pair, 1), one);
  const MfmaBits high_half = _mm256_max_epi32(_mm256_sub_epi32(pair, low_half), one);
  // Each half is itself an E8M0 exponent, so placing it in the FP32 exponent field spells the
  // power of two it stands for.
  const MfmaNative first = mfma_bits_as_float(_mm256_slli_epi32(low_half, 23));
  const MfmaNative second = mfma_bits_as_float(_mm256_slli_epi32(high_half, 23));
  const MfmaNative scaled = _mm256_mul_ps(_mm256_mul_ps(sum, first), second);
  const MfmaMask nan_scale =
      _mm256_cmpeq_epi32(_mm256_max_epi32(a_exps, b_exps), _mm256_set1_epi32(kE8m0NanExponent));
  return _mm256_blendv_ps(scaled, mfma_bits_as_float(mfma_bits_broadcast(kFp32QuietNan)),
                          mfma_bits_as_float(nan_scale));
}

// Saturate an overflow that finite inputs alone produced: step the infinity down to the largest
// finite FP32 of the same sign. Columns that saw a nonfinite input keep the infinity.
inline MfmaNative mfma_clamp_scaled_overflow(MfmaNative acc, MfmaMask nonfinite) {
  const MfmaBits bits = _mm256_castps_si256(acc);
  const MfmaMask at_infinity =
      _mm256_cmpeq_epi32(_mm256_and_si256(bits, mfma_bits_broadcast(kFp32Magnitude)),
                         mfma_bits_broadcast(kFp32Exponent));
  // A selected column holds all ones, that is -1, so adding the selection is the single step from
  // the infinity down to the largest finite value.
  return mfma_bits_as_float(_mm256_add_epi32(bits, _mm256_andnot_si256(nonfinite, at_infinity)));
}
#endif

// Reduce one block-scaled MFMA with vectors: every tile of four rows by one vector of columns
// accumulates each block's scaled 32-product partial on top of C, then stores. ClampOverflow is a
// template parameter because the inherited saturation rule needs a mask that the plain form must
// not carry through its reduction.
template <int M, int K, bool ClampOverflow>
void mfma_mixed_scaled_simd(uint64_t controls, const float left[K][M], const float right[K][M],
                            const uint32_t *const *c, const int32_t a_exponents[64],
                            const int32_t b_exponents[64], float result[M][M]) {
  static_assert(M % kMfmaRowTile == 0 && M % kMfmaWidth == 0,
                "a scaled tile covers whole rows and whole vectors");
  const MfmaBits magnitude =
      mfma_bits_broadcast((controls & RJ_ARITH_MFMA_SCALE_ABS_C) ? kFp32Magnitude : ~uint32_t{0});
  const MfmaBits sign_flip =
      mfma_bits_broadcast((controls & RJ_ARITH_MFMA_SCALE_NEG_C) ? kFp32Sign : 0u);
  for (int col = 0; col < M; col += kMfmaWidth) {
    for (int row = 0; row < M; row += kMfmaRowTile) {
      MfmaNative acc[kMfmaRowTile];
      MfmaMask nonfinite[kMfmaRowTile];
      RJ_ARITH_UNROLL(4)
      for (int sub = 0; sub < kMfmaRowTile; ++sub) {
        const MfmaWord out = mfma_output<M, 1>(row + sub, col, 0);
        const MfmaBits initial =
            mfma_modify_c(mfma_bits_loadu(c[out.reg] + out.lane), magnitude, sign_flip);
        if constexpr (ClampOverflow) {
          nonfinite[sub] = mfma_nonfinite(initial);
        }
        acc[sub] = mfma_bits_as_float(initial);
      }
      // Scale each fused 32-product partial once, then add it separately to C.
      for (int block = 0; block < K / 32; ++block) {
        MfmaNative sum[kMfmaRowTile];
        RJ_ARITH_UNROLL(4)
        for (int sub = 0; sub < kMfmaRowTile; ++sub) {
          sum[sub] = mfma_broadcast(0.0f);
        }
        for (int k = 32 * block; k < 32 * (block + 1); ++k) {
          const MfmaNative rhs = mfma_load(right[k] + col);
          RJ_ARITH_UNROLL(4)
          for (int sub = 0; sub < kMfmaRowTile; ++sub) {
            sum[sub] = mfma_multiply_add(mfma_broadcast(left[k][row + sub]), rhs, sum[sub]);
          }
        }
        const MfmaBits b_exps = mfma_exponents_load(b_exponents + M * block + col);
        RJ_ARITH_UNROLL(4)
        for (int sub = 0; sub < kMfmaRowTile; ++sub) {
          if constexpr (ClampOverflow) {
            // A finite low-precision dot fits in FP32 before scaling, so a nonfinite block
            // identifies a nonfinite input to this output.
            nonfinite[sub] = mfma_mask_or(nonfinite[sub], mfma_nonfinite(sum[sub]));
          }
          acc[sub] = mfma_add(
              acc[sub], mfma_scale_e8m0(sum[sub], a_exponents[M * block + row + sub], b_exps));
        }
      }
      RJ_ARITH_UNROLL(4)
      for (int sub = 0; sub < kMfmaRowTile; ++sub) {
        if constexpr (ClampOverflow) {
          acc[sub] = mfma_clamp_scaled_overflow(acc[sub], nonfinite[sub]);
        }
        mfma_store(result[row + sub] + col, acc[sub]);
      }
    }
  }
}
#else
// Scale a block dot by 2^(a_exp + b_exp - 254), or deliver a canonical quiet NaN where either
// scale is the E8M0 NaN. The exponent goes on as two balanced power-of-two multiplies, for the
// reasons the AVX2 definition above records.
inline float mfma_scale_e8m0(float sum, int a_exp, int b_exp) {
  if (a_exp == kE8m0NanExponent || b_exp == kE8m0NanExponent) {
    return as_float(kFp32QuietNan);
  }
  const int pair = a_exp + b_exp;
  const int low_half = (pair >> 1) > 1 ? (pair >> 1) : 1;
  const int high_half = (pair - low_half) > 1 ? (pair - low_half) : 1;
  return (sum * as_float(uint32_t(low_half) << 23)) * as_float(uint32_t(high_half) << 23);
}
#endif

// Stage every output before stores; preserve inherited reductions and optional block scaling.
template <int M, int K, bool Scaled = false>
void mfma_mixed_kernel(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                       const uint32_t *const *b, const uint32_t *const *c,
                       const uint32_t *scale_a = nullptr, const uint32_t *scale_b = nullptr,
                       bool clamp_overflow = false) {
  alignas(64) float result[M][M];
#if defined(__AVX2__)
  alignas(64) float left[K][M], right[K][M];
  mixed_x86::decode<64 / M, M>(a, int(controls & 7), false, left);
  mixed_x86::decode<64 / M, M>(b, int(controls >> 3 & 7), false, right);
  if constexpr (Scaled) {
    alignas(64) int32_t a_exponents[64], b_exponents[64];
    mfma_scale_exponents(scale_a, mfma_scale_byte_shift(controls, RJ_ARITH_MFMA_SCALE_A_BYTE_SHIFT),
                         a_exponents);
    mfma_scale_exponents(scale_b, mfma_scale_byte_shift(controls, RJ_ARITH_MFMA_SCALE_B_BYTE_SHIFT),
                         b_exponents);
    if (clamp_overflow) {
      mfma_mixed_scaled_simd<M, K, true>(controls, left, right, c, a_exponents, b_exponents,
                                         result);
    } else {
      mfma_mixed_scaled_simd<M, K, false>(controls, left, right, c, a_exponents, b_exponents,
                                          result);
    }
  } else {
    // The unscaled reduction tiles the same way, so four rows share each B load.
    for (int col = 0; col < M; col += kMfmaWidth) {
      for (int row = 0; row < M; row += kMfmaRowTile) {
        MfmaNative acc[kMfmaRowTile];
        RJ_ARITH_UNROLL(4)
        for (int sub = 0; sub < kMfmaRowTile; ++sub) {
          const MfmaWord out = mfma_output<M, 1>(row + sub, col, 0);
          // C is not aligned to the vector width, and its words arrive as raw bits.
          acc[sub] = mfma_bits_as_float(mfma_bits_loadu(c[out.reg] + out.lane));
        }
        for (int k = 0; k < K; ++k) {
          const MfmaNative rhs = mfma_load(right[k] + col);
          RJ_ARITH_UNROLL(4)
          for (int sub = 0; sub < kMfmaRowTile; ++sub) {
            acc[sub] = mfma_multiply_add(mfma_broadcast(left[k][row + sub]), rhs, acc[sub]);
          }
        }
        RJ_ARITH_UNROLL(4)
        for (int sub = 0; sub < kMfmaRowTile; ++sub) {
          mfma_store(result[row + sub] + col, acc[sub]);
        }
      }
    }
  }
#else
  alignas(64) int32_t a_exponents[64], b_exponents[64];
  if constexpr (Scaled) {
    mfma_scale_exponents(scale_a, mfma_scale_byte_shift(controls, RJ_ARITH_MFMA_SCALE_A_BYTE_SHIFT),
                         a_exponents);
    mfma_scale_exponents(scale_b, mfma_scale_byte_shift(controls, RJ_ARITH_MFMA_SCALE_B_BYTE_SHIFT),
                         b_exponents);
  }
  alignas(64) float left[M][K], right_rows[M][K], right[K][M];
  mfma_decode_mixed<M, K>(a, int(controls & 7), left);
  mfma_decode_mixed<M, K>(b, int(controls >> 3 & 7), right_rows);
  for (int k = 0; k < K; ++k) {
    for (int col = 0; col < M; ++col) {
      right[k][col] = right_rows[col][k];
    }
  }
  for (int row = 0; row < M; ++row) {
    const MfmaWord out = mfma_output<M, 1>(row, 0, 0);
    bool clamp[M];
    for (int col = 0; col < M; ++col) {
      uint32_t initial = c[out.reg][out.lane + col];
      if constexpr (Scaled) {
        if (controls & RJ_ARITH_MFMA_SCALE_ABS_C) {
          initial &= kFp32Magnitude;
        }
        if (controls & RJ_ARITH_MFMA_SCALE_NEG_C) {
          initial ^= kFp32Sign;
        }
        clamp[col] = clamp_overflow && __builtin_isfinite(as_float(initial));
      }
      result[row][col] = as_float(initial);
    }
    if constexpr (Scaled) {
      // Scale each fused 32-product partial once, then add it separately to C.
      for (int block = 0; block < K / 32; ++block) {
        alignas(64) float sums[M] = {};
        for (int col = 0; col < M; ++col) {
          for (int k = 32 * block; k < 32 * (block + 1); ++k) {
            sums[col] = __builtin_fmaf(left[row][k], right[k][col], sums[col]);
          }
        }
        const int a_exp = a_exponents[M * block + row];
        for (int col = 0; col < M; ++col) {
          const float scaled = mfma_scale_e8m0(sums[col], a_exp, b_exponents[M * block + col]);
          // A finite low-precision dot fits in FP32 before scaling, so a nonfinite block
          // identifies a nonfinite input to this output.
          clamp[col] = clamp[col] && __builtin_isfinite(sums[col]);
          result[row][col] += scaled;
        }
      }
      for (int col = 0; col < M; ++col) {
        if (clamp[col] && __builtin_isinf(result[row][col])) {
          result[row][col] = as_float(as_bits(result[row][col]) - 1u);
        }
      }
    } else {
      for (int col = 0; col < M; ++col) {
        for (int k = 0; k < K; ++k) {
          result[row][col] += left[row][k] * right[k][col];
        }
      }
    }
  }
#endif
  // Each output row is a contiguous slice of one physical accumulator.
  for (int row = 0; row < M; ++row) {
    const MfmaWord out = mfma_output<M, 1>(row, 0, 0);
    std::memcpy(d[out.reg] + out.lane, result[row], M * sizeof(uint32_t));
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
