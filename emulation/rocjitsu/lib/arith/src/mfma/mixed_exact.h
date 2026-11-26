// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// MI350X-calibrated exact integer model for gfx950 f8f6f4 and scale_f8f6f4 MFMA.
#pragma once

#include "common/bits.h"
#include "mfma/mixed.h"
#include <cstdint>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
namespace mxfp_exact_detail {

constexpr int kMaxDim = 32;
constexpr int kMaxK = 128;
constexpr int kChunkProducts = 8;
constexpr int32_t kZeroProductExponent = -999;
constexpr uint32_t kInfNanExponentField = 0xffu;
constexpr uint32_t kPositiveInfinity = 0x7f800000u;
constexpr uint32_t kNegativeInfinity = 0xff800000u;
constexpr uint32_t kCanonicalNan = 0xffc00000u;

constexpr int32_t format_subnormal_exponent(int format) {
  return format == 0 ? -6 : format == 1 ? -14 : format == 2 ? 0 : format == 3 ? -2 : 0;
}

inline void decode_sources(const float *values, int count, int32_t subnormal_exp,
                           int32_t *mantissas, int32_t *exponents) {
  for (int i = 0; i < count; ++i) {
    const uint32_t bits = bit_cast<uint32_t>(values[i]);
    const uint32_t exp_field = (bits >> 23) & 0xffu;
    if (exp_field == kInfNanExponentField) {
      mantissas[i] = 0;
      exponents[i] = kZeroProductExponent;
      continue;
    }
    const int32_t exp = maximum(static_cast<int32_t>(exp_field), 1) - 127;
    const uint32_t f32_mant = (bits & 0x7fffffu) | (exp_field != 0 ? 0x800000u : 0u);
    const int32_t below = clamp(subnormal_exp - exp, 0, 31);
    const int32_t mag = static_cast<int32_t>((f32_mant >> below) >> 13);
    mantissas[i] = (bits >> 31) != 0 ? -mag : mag;
    exponents[i] = maximum(exp, subnormal_exp);
  }
}

// Align 8 products per column to the column maximum, truncating toward zero at 13 fraction bits,
// then scale back to 24 fraction bits.
inline void sum_chunk8(const int32_t *a_mant, const int32_t *a_exp, const int32_t *b_mant,
                       const int32_t *b_exp, int cols, int64_t *sums24, int32_t *top_exp) {
  int32_t prod_mant[kChunkProducts * kMaxDim];
  int32_t prod_exp[kChunkProducts * kMaxDim];
  for (int col = 0; col < cols; ++col) {
    sums24[col] = 0;
    top_exp[col] = kZeroProductExponent;
  }
  for (int i = 0; i < kChunkProducts; ++i) {
    const int32_t am = a_mant[i];
    const int32_t ae = a_exp[i];
    const int32_t *bm = &b_mant[i * cols];
    const int32_t *be = &b_exp[i * cols];
    for (int col = 0; col < cols; ++col) {
      const int32_t m = am * bm[col];
      const int32_t e = m != 0 ? ae + be[col] : kZeroProductExponent;
      prod_mant[i * cols + col] = m;
      prod_exp[i * cols + col] = e;
      top_exp[col] = maximum(top_exp[col], e);
    }
  }
  int64_t acc13[kMaxDim]{};
  for (int i = 0; i < kChunkProducts; ++i) {
    for (int col = 0; col < cols; ++col) {
      const int32_t m = prod_mant[i * cols + col];
      if (m == 0) {
        continue;
      }
      // 24 - 13 = 11 bits dropped below the 24-fraction-bit product scale.
      const int32_t rsh = minimum((top_exp[col] - prod_exp[i * cols + col]) + 11, 40);
      const int64_t scaled = static_cast<int64_t>(m) * 16;
      const int64_t toward_zero = scaled < 0 ? ((int64_t{1} << rsh) - 1) : 0;
      acc13[col] += (scaled + toward_zero) >> rsh;
    }
  }
  for (int col = 0; col < cols; ++col) {
    sums24[col] = acc13[col] << 11;
  }
}

inline void add_accumulators(const int64_t *dot24, const int32_t *dot_exp, const float *acc,
                             int cols, uint32_t *results) {
  for (int col = 0; col < cols; ++col) {
    const uint32_t cb = bit_cast<uint32_t>(acc[col]);
    const int64_t cef = (cb >> 23) & 0xff;
    const int64_t ce = maximum<int64_t>(cef, 1) - 127;
    const int64_t cmag = (cb & 0x7fffffu) | (cef != 0 ? 0x800000u : 0u);
    const int64_t cm = (cb >> 31) != 0 ? -cmag : cmag;

    const int64_t de = dot_exp[col];
    const int64_t te = maximum(de, ce);
    const int64_t dot_part = (dot24[col] * 256) >> minimum<int64_t>(te - de, 63);
    int64_t c_part = ((cm * 2) >> minimum<int64_t>(te - ce, 63)) * 256;
    c_part = ce < te - 25 ? 0 : c_part;
    const int64_t total = dot_part + c_part;

    const uint64_t total_mag = static_cast<uint64_t>(total < 0 ? -total : total);
    const int64_t total_top = 63 - countl_zero(total_mag | 1);
    const int64_t norm_exp = maximum<int64_t>(total_top + te - 32, -126);
    const int64_t right = norm_exp - te + 1;
    const int64_t lsh = clamp<int64_t>(-right, 0, 31);
    const int64_t rsh = clamp<int64_t>(right, 0, 63);
    const int64_t fixed = static_cast<int64_t>(static_cast<uint64_t>(total) << lsh) >> rsh;

    const uint64_t mag = static_cast<uint64_t>(fixed < 0 ? -fixed : fixed);
    const int64_t top = 63 - countl_zero(mag | 1);
    const int64_t exp = maximum<int64_t>(top + norm_exp - 31, -126);
    const int64_t dropped = clamp<int64_t>(exp - 23 - norm_exp + 31, 1, 63);
    const uint64_t kept = mag >> dropped;
    const uint64_t rem = mag & ((uint64_t{1} << dropped) - 1);
    const uint64_t half = uint64_t{1} << (dropped - 1);
    const uint64_t round_up = (rem > half || (rem == half && (kept & 1) != 0)) ? 1 : 0;
    const uint64_t bits = (static_cast<uint64_t>(exp + 126) << 23) + kept + round_up;
    const uint64_t bounded = minimum<uint64_t>(bits, uint64_t{kPositiveInfinity});
    const uint64_t signed_bits = bounded | (fixed < 0 ? uint64_t{0x80000000u} : uint64_t{0});
    results[col] = (mag != 0 && bounded != 0) ? static_cast<uint32_t>(signed_bits) : 0u;
  }
}

} // namespace mxfp_exact_detail

// Evaluate D = A*B + C (or block-scaled D) using the MI350X-characterized step pipeline:
//   - 8-product chunks aligned at 13 fraction bits (truncated toward zero), combined into
//     16-product passes whenever either input is 8-bit (FP8/BF8) and 32-product passes when
//     both inputs are subbyte (FP6/BF6/FP4).
//   - Block-scaled instructions shift any active pass exponent by (scale_a + scale_b - 254),
//     including passes where non-zero products cancel to zero, and produce NaN on scale 0xff.
//   - Accumulators more than 25 bits below the pass exponent are discarded; NaNs canonicalize
//     to 0xffc00000; RJ_ARITH_FP16_OVFL flushes NaN to +0.0 and clamps +-Inf to +-MAX_FLOAT
//     after each pass.
template <int M, int K, bool Scaled>
void mfma_mixed_exact(uint64_t controls, bool fp16_ovfl, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                      const uint32_t *scale_a = nullptr, const uint32_t *scale_b = nullptr) {
  using namespace mxfp_exact_detail;
  const int fmt_a = static_cast<int>(controls & 7);
  const int fmt_b = static_cast<int>((controls >> 3) & 7);
  float left[M][K], right_rows[M][K], right[K][M], acc[M][M];
  mfma_decode_mixed<M, K>(a, fmt_a, left);
  mfma_decode_mixed<M, K>(b, fmt_b, right_rows);
  for (int k = 0; k < K; ++k) {
    for (int col = 0; col < M; ++col) {
      right[k][col] = right_rows[col][k];
    }
  }
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < M; ++col) {
      const auto out = mfma_output<M, 1>(row, col, 0);
      acc[row][col] = as_float(c[out.reg][out.lane]);
    }
  }
  int sa_exp[64]{}, sb_exp[64]{};
  if constexpr (Scaled) {
    const int sa_shift = 8 * static_cast<int>((controls >> RJ_ARITH_MFMA_SCALE_A_BYTE_SHIFT) & 3);
    const int sb_shift = 8 * static_cast<int>((controls >> RJ_ARITH_MFMA_SCALE_B_BYTE_SHIFT) & 3);
    for (int lane = 0; lane < 64; ++lane) {
      sa_exp[lane] = static_cast<int>((scale_a[lane] >> sa_shift) & 0xffu);
      sb_exp[lane] = static_cast<int>((scale_b[lane] >> sb_shift) & 0xffu);
    }
  }

  int32_t a_mant[M * K], a_exp[M * K], b_mant[K * M], b_exp[K * M];
  decode_sources(&left[0][0], M * K, format_subnormal_exponent(fmt_a), a_mant, a_exp);
  decode_sources(&right[0][0], K * M, format_subnormal_exponent(fmt_b), b_mant, b_exp);

  const int products_per_step = (fmt_a < 2 || fmt_b < 2) ? 16 : 32;
  const int chunks_per_step = products_per_step / kChunkProducts;

  int64_t dot[kMaxDim], chunk_sum[kMaxDim];
  int32_t dot_exp[kMaxDim], chunk_exp[kMaxDim];
  uint32_t results[kMaxDim], prev_c[kMaxDim];

  for (int row = 0; row < M; ++row) {
    float *c_row = acc[row];
    for (int start = 0; start < K; start += products_per_step) {
      for (int col = 0; col < M; ++col) {
        dot[col] = 0;
        dot_exp[col] = kZeroProductExponent;
      }
      for (int g = 0; g < chunks_per_step; ++g) {
        const int offset = start + g * kChunkProducts;
        sum_chunk8(&a_mant[row * K + offset], &a_exp[row * K + offset], &b_mant[offset * M],
                   &b_exp[offset * M], M, chunk_sum, chunk_exp);
        for (int col = 0; col < M; ++col) {
          const int32_t exp = maximum(dot_exp[col], chunk_exp[col]);
          dot[col] = (dot[col] >> minimum(exp - dot_exp[col], 63)) +
                     (chunk_sum[col] >> minimum(exp - chunk_exp[col], 63));
          dot_exp[col] = exp;
        }
      }
      const int block = start / 32;
      const int ea = Scaled ? sa_exp[block * M + row] : 127;
      if constexpr (Scaled) {
        for (int col = 0; col < M; ++col) {
          const int eb = sb_exp[block * M + col];
          if (ea < 255 && eb < 255 && dot_exp[col] > kZeroProductExponent) {
            dot_exp[col] += ea + eb - 254;
          }
        }
      }
      for (int col = 0; col < M; ++col) {
        prev_c[col] = bit_cast<uint32_t>(c_row[col]);
      }
      add_accumulators(dot, dot_exp, c_row, M, results);

      for (int col = 0; col < M; ++col) {
        const int eb = Scaled ? sb_exp[block * M + col] : 127;
        bool nan = (ea == 255 || eb == 255);
        bool pos_inf = false;
        bool neg_inf = false;
        const auto add_inf = [&](bool sign) { (sign ? neg_inf : pos_inf) = true; };
        for (int i = 0; i < products_per_step; ++i) {
          const uint32_t ab = bit_cast<uint32_t>(left[row][start + i]);
          const uint32_t bb = bit_cast<uint32_t>(right[start + i][col]);
          const bool sign = ((ab ^ bb) >> 31) != 0;
          const bool a_spec = ((ab >> 23) & 0xffu) == kInfNanExponentField;
          const bool b_spec = ((bb >> 23) & 0xffu) == kInfNanExponentField;
          const bool a_nan = a_spec && (ab & 0x7fffffu) != 0;
          const bool b_nan = b_spec && (bb & 0x7fffffu) != 0;
          const bool a_zero = (ab & 0x7fffffffu) == 0;
          const bool b_zero = (bb & 0x7fffffffu) == 0;
          if (a_nan || b_nan || (a_spec && b_zero) || (b_spec && a_zero)) {
            nan = true;
          } else if (a_spec || b_spec) {
            add_inf(sign);
          }
        }
        const uint32_t cb = prev_c[col];
        if (((cb >> 23) & 0xffu) == kInfNanExponentField) {
          if ((cb & 0x7fffffu) != 0) {
            nan = true;
          } else {
            add_inf((cb >> 31) != 0);
          }
        }
        if (nan || (pos_inf && neg_inf)) {
          results[col] = kCanonicalNan;
        } else if (pos_inf || neg_inf) {
          results[col] = pos_inf ? kPositiveInfinity : kNegativeInfinity;
        }
        uint32_t bits = results[col];
        if (fp16_ovfl && (bits & 0x7f800000u) == 0x7f800000u) {
          bits = (bits & 0x007fffffu) != 0 ? 0u : bits - 1u;
        }
        c_row[col] = bit_cast<float>(bits);
      }
    }
  }

  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < M; ++col) {
      const auto out = mfma_output<M, 1>(row, col, 0);
      d[out.reg][out.lane] = as_bits(acc[row][col]);
    }
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
