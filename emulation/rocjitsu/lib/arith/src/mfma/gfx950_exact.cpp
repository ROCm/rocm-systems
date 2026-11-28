// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integer model of gfx950 (CDNA4 / MI350X) BF16 DOT2 and F16/BF16/OCP-FP8 dense & sparse MFMA,
// calibrated against MI350X hardware captures. Compiled for the baseline ISA only.
#include "mfma/gfx950_exact.h"
#include "common/float_formats.h"
#include "mfma/fp8_formats.h"
#include "mfma/layout.h"
#include "mfma/sparse_layout.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
namespace {

struct Gfx950MfmaArithmetic {
  int32_t subnormal_a;
  int32_t subnormal_b;
  int products_per_step;
  int chunk_products;
  int align_frac;
  bool drop_far_accumulator;
};

constexpr Gfx950MfmaArithmetic kGfx950Dot2Bf16{-126, -126, 2, 2, 24, false};
constexpr Gfx950MfmaArithmetic kGfx950MfmaF16{-14, -14, 8, 8, 24, false};
constexpr Gfx950MfmaArithmetic kGfx950MfmaBf16{-126, -126, 8, 8, 24, false};

constexpr Gfx950MfmaArithmetic gfx950_mfma_ocp_f8(bool a_is_fp8, bool b_is_fp8) {
  return {a_is_fp8 ? -6 : -14, b_is_fp8 ? -6 : -14, 16, 8, 13, true};
}

constexpr Gfx950MfmaArithmetic format_arithmetic(MfmaGfx950Format format) {
  switch (format) {
  case MfmaGfx950Format::F16:
    return kGfx950MfmaF16;
  case MfmaGfx950Format::Bf16:
    return kGfx950MfmaBf16;
  case MfmaGfx950Format::Fp8Fp8:
    return gfx950_mfma_ocp_f8(true, true);
  case MfmaGfx950Format::Fp8Bf8:
    return gfx950_mfma_ocp_f8(true, false);
  case MfmaGfx950Format::Bf8Fp8:
    return gfx950_mfma_ocp_f8(false, true);
  case MfmaGfx950Format::Bf8Bf8:
    return gfx950_mfma_ocp_f8(false, false);
  }
  return kGfx950MfmaF16;
}

namespace gfx950_mfma_detail {

constexpr int kMaxColumns = 32;
constexpr int kMaxSourceElements = 1024;
constexpr int kMaxChunkProducts = 8;
constexpr int32_t kZeroProductExponent = -999;
constexpr uint32_t kInfNanExponentField = 0xFFu;
constexpr uint32_t kPositiveInfinity = 0x7F800000u;
constexpr uint32_t kNegativeInfinity = 0xFF800000u;
constexpr uint32_t kNan = 0xFFC00000u;
constexpr int32_t kProductAtLeastTwoShift = 21;

inline bool decode_sources(const float *values, int count, int32_t subnormal_exponent,
                           int32_t *mantissas, int32_t *exponents) {
  uint32_t any_inf_nan = 0;
  for (int index = 0; index < count; ++index) {
    const uint32_t bits = bit_cast<uint32_t>(values[index]);
    const int32_t exponent_field = static_cast<int32_t>((bits >> 23) & 0xFFu);
    const int32_t exponent = maximum(exponent_field, 1) - 127;
    const uint32_t f32_mantissa = (bits & 0x7FFFFFu) | (exponent_field != 0 ? 0x800000u : 0u);
    const int32_t below_format = clamp(subnormal_exponent - exponent, 0, 31);
    const int32_t magnitude = static_cast<int32_t>((f32_mantissa >> below_format) >> 13);
    mantissas[index] = (bits >> 31) != 0 ? -magnitude : magnitude;
    exponents[index] = maximum(exponent, subnormal_exponent);
    any_inf_nan |= exponent_field == static_cast<int32_t>(kInfNanExponentField) ? 1u : 0u;
  }
  return any_inf_nan != 0;
}

// Aligns one contiguous chunk of products to the largest exponent in each column, truncates each
// product toward zero at align_frac fraction bits, and returns the sum scaled to 24 fraction bits.
inline int32_t sum_chunk(const int32_t *a_mantissas, const int32_t *a_exponents,
                         const int32_t *b_mantissas, const int32_t *b_exponents, int products,
                         int columns, int align_frac, int32_t *sums, int32_t *top_exponents) {
  alignas(64) int32_t product_mantissas[kMaxChunkProducts * kMaxColumns];
  alignas(64) int32_t product_exponents[kMaxChunkProducts * kMaxColumns];
  for (int col = 0; col < columns; ++col) {
    sums[col] = 0;
    top_exponents[col] = kZeroProductExponent;
  }
  int32_t may_overflow = 0;
  for (int index = 0; index < products; ++index) {
    const int32_t a_mantissa = a_mantissas[index];
    const int32_t a_exponent = a_exponents[index];
    const int32_t *b_mantissa = &b_mantissas[index * columns];
    const int32_t *b_exponent = &b_exponents[index * columns];
    int32_t *mantissa_row = &product_mantissas[index * columns];
    int32_t *exponent_row = &product_exponents[index * columns];
    for (int col = 0; col < columns; ++col) {
      const int32_t mantissa = a_mantissa * b_mantissa[col];
      const int32_t exponent = mantissa != 0 ? a_exponent + b_exponent[col] : kZeroProductExponent;
      mantissa_row[col] = mantissa;
      exponent_row[col] = exponent;
      top_exponents[col] = maximum(top_exponents[col], exponent);
      const int32_t magnitude = mantissa < 0 ? -mantissa : mantissa;
      may_overflow |= (exponent + (magnitude >> kProductAtLeastTwoShift)) >= 128 ? 1 : 0;
    }
  }
  int64_t aligned[kMaxColumns]{};
  for (int index = 0; index < products; ++index) {
    const int32_t *mantissa_row = &product_mantissas[index * columns];
    const int32_t *exponent_row = &product_exponents[index * columns];
    for (int col = 0; col < columns; ++col) {
      const int32_t mantissa = mantissa_row[col];
      if (mantissa == 0) {
        continue;
      }
      const int32_t rsh = (top_exponents[col] - exponent_row[col]) + (24 - align_frac);
      const int64_t scaled = static_cast<int64_t>(mantissa) * 16;
      if (rsh <= 0) {
        aligned[col] += scaled;
      } else if (rsh < 62) {
        const int64_t toward_zero = scaled < 0 ? ((int64_t{1} << rsh) - 1) : 0;
        aligned[col] += (scaled + toward_zero) >> rsh;
      }
    }
  }
  for (int col = 0; col < columns; ++col) {
    sums[col] = static_cast<int32_t>(static_cast<uint64_t>(aligned[col]) << (24 - align_frac));
  }
  return may_overflow;
}

inline bool add_accumulators(const int32_t *dot, const int32_t *dot_exponents,
                             const float *accumulators, bool drop_far_accumulator, int columns,
                             uint32_t *results) {
  uint32_t inf_nan = 0;
  for (int col = 0; col < columns; ++col) {
    const uint32_t c_bits = bit_cast<uint32_t>(accumulators[col]);
    const int64_t c_exponent_field = (c_bits >> 23) & 0xFFu;
    const int64_t c_exponent = maximum<int64_t>(c_exponent_field, 1) - 127;
    const int64_t c_magnitude = (c_bits & 0x7FFFFFu) | (c_exponent_field != 0 ? 0x800000u : 0u);
    const int64_t c_mantissa = (c_bits >> 31) != 0 ? -c_magnitude : c_magnitude;
    inf_nan |= c_exponent_field == kInfNanExponentField ? 1u : 0u;

    const int64_t dot_exponent = dot_exponents[col];
    const int64_t top_exponent = maximum(dot_exponent, c_exponent);
    const int64_t dot_part =
        (static_cast<int64_t>(dot[col]) * 256) >> minimum<int64_t>(top_exponent - dot_exponent, 63);
    int64_t c_part = ((c_mantissa * 2) >> minimum<int64_t>(top_exponent - c_exponent, 63)) * 256;
    if (drop_far_accumulator) {
      c_part = c_exponent < top_exponent - 25 ? 0 : c_part;
    }
    const int64_t total = dot_part + c_part;

    const uint64_t total_magnitude = static_cast<uint64_t>(total < 0 ? -total : total);
    const int64_t total_top_bit = 63 - countl_zero(total_magnitude | 1);
    const int64_t normal_exponent = maximum<int64_t>(total_top_bit + top_exponent - 32, -126);
    const int64_t right = normal_exponent - top_exponent + 1;
    const int64_t left_shift = clamp<int64_t>(-right, 0, 31);
    const int64_t right_shift = clamp<int64_t>(right, 0, 63);
    const int64_t fixed =
        static_cast<int64_t>(static_cast<uint64_t>(total) << left_shift) >> right_shift;

    const uint64_t magnitude = static_cast<uint64_t>(fixed < 0 ? -fixed : fixed);
    const int64_t top_bit = 63 - countl_zero(magnitude | 1);
    const int64_t exponent = maximum<int64_t>(top_bit + normal_exponent - 31, -126);
    const int64_t dropped = clamp<int64_t>(exponent - 23 - normal_exponent + 31, 1, 63);
    const uint64_t kept = magnitude >> dropped;
    const uint64_t remainder = magnitude & ((uint64_t{1} << dropped) - 1);
    const uint64_t half = uint64_t{1} << (dropped - 1);
    const uint64_t round_up = (remainder > half || (remainder == half && (kept & 1) != 0)) ? 1 : 0;
    const uint64_t bits = (static_cast<uint64_t>(exponent + 126) << 23) + kept + round_up;
    const uint64_t bounded = minimum<uint64_t>(bits, kPositiveInfinity);
    const uint64_t signed_bits = bounded | (fixed < 0 ? uint64_t{0x80000000u} : uint64_t{0});
    results[col] = (magnitude != 0 && bounded != 0) ? static_cast<uint32_t>(signed_bits) : 0u;
  }
  return inf_nan != 0;
}

inline void non_finite_result(const Gfx950MfmaArithmetic &arithmetic, const float *a,
                              const float *b, int columns, int products, float accumulator,
                              uint32_t &result) {
  bool nan = false, positive = false, negative = false;
  const auto add_infinity = [&](bool sign) { (sign ? negative : positive) = true; };
  for (int index = 0; index < products; ++index) {
    const uint32_t a_bits = bit_cast<uint32_t>(a[index]);
    const uint32_t b_bits = bit_cast<uint32_t>(b[index * columns]);
    const bool sign = ((a_bits ^ b_bits) >> 31) != 0;
    const bool a_special = ((a_bits >> 23) & 0xFFu) == kInfNanExponentField;
    const bool b_special = ((b_bits >> 23) & 0xFFu) == kInfNanExponentField;
    const bool a_nan = a_special && (a_bits & 0x7FFFFFu) != 0;
    const bool b_nan = b_special && (b_bits & 0x7FFFFFu) != 0;
    const bool a_zero = (a_bits & 0x7FFFFFFFu) == 0;
    const bool b_zero = (b_bits & 0x7FFFFFFFu) == 0;
    if (a_nan || b_nan || (a_special && b_zero) || (b_special && a_zero)) {
      nan = true;
    } else if (a_special || b_special) {
      add_infinity(sign);
    } else if (!a_zero && !b_zero) {
      int32_t a_mantissa = 0, a_exponent = 0, b_mantissa = 0, b_exponent = 0;
      decode_sources(&a[index], 1, arithmetic.subnormal_a, &a_mantissa, &a_exponent);
      decode_sources(&b[index * columns], 1, arithmetic.subnormal_b, &b_mantissa, &b_exponent);
      const int32_t mantissa = a_mantissa * b_mantissa;
      const int32_t magnitude = mantissa < 0 ? -mantissa : mantissa;
      if (magnitude != 0 &&
          a_exponent + b_exponent + (magnitude >> kProductAtLeastTwoShift) >= 128) {
        add_infinity(sign);
      }
    }
  }
  const uint32_t c_bits = bit_cast<uint32_t>(accumulator);
  if (((c_bits >> 23) & 0xFFu) == kInfNanExponentField) {
    if ((c_bits & 0x7FFFFFu) != 0) {
      nan = true;
    } else {
      add_infinity((c_bits >> 31) != 0);
    }
  }
  if (nan || (positive && negative)) {
    result = kNan;
  } else if (positive || negative) {
    result = positive ? kPositiveInfinity : kNegativeInfinity;
  }
}

} // namespace gfx950_mfma_detail

// D = A*B + C in place on C for one block of a CDNA4 MFMA/SMFMAC or BF16 DOT2, calibrated on
// MI350X:
//   - F16/BF16 MFMA & SMFMAC reduce 8 products per pass aligned to 24 fraction bits; BF16 DOT2
//     reduces 2 products in one pass with the same alignment and BF16 2^128 product overflow rule.
//   - OCP FP8/BF8 MFMA & SMFMAC reduce 16 products per pass split into two contiguous 8-product
//     chunks aligned at 13 fraction bits (truncated toward zero), floor-aligned to max(e0, e1),
//     and discard finite C more than 25 bits below the step exponent.
//   - Under RJ_ARITH_FP16_OVFL (MFMA/SMFMAC), each pass flushes NaN to +0.0 and clamps +-Inf
//     (including source or product infinities) to +-0x7f7fffff before the next pass.
inline void gfx950_mfma_matmul(const Gfx950MfmaArithmetic &arithmetic, int M, int N, int K,
                               bool fp16_ovfl, const float *A, const float *B, float *C) {
  using namespace gfx950_mfma_detail;
  alignas(64) int32_t a_mantissas[kMaxSourceElements], a_exponents[kMaxSourceElements];
  alignas(64) int32_t b_mantissas[kMaxSourceElements], b_exponents[kMaxSourceElements];
  const bool a_inf_nan = decode_sources(A, M * K, arithmetic.subnormal_a, a_mantissas, a_exponents);
  const bool b_inf_nan = decode_sources(B, K * N, arithmetic.subnormal_b, b_mantissas, b_exponents);

  alignas(64) int32_t dot[kMaxColumns], dot_exponents[kMaxColumns];
  alignas(64) int32_t chunk_sum[kMaxColumns], chunk_exponents[kMaxColumns];
  alignas(64) uint32_t results[kMaxColumns];
  for (int row = 0; row < M; ++row) {
    float *c_row = &C[row * N];
    for (int start = 0; start < K; start += arithmetic.products_per_step) {
      const int products = minimum(K - start, arithmetic.products_per_step);
      const int chunk = minimum(products, arithmetic.chunk_products);
      const int groups = products / chunk;
      int32_t may_overflow = 0;
      for (int col = 0; col < N; ++col) {
        dot[col] = 0;
        dot_exponents[col] = kZeroProductExponent;
      }
      for (int g = 0; g < groups; ++g) {
        const int offset = start + g * chunk;
        may_overflow |= sum_chunk(&a_mantissas[row * K + offset], &a_exponents[row * K + offset],
                                  &b_mantissas[offset * N], &b_exponents[offset * N], chunk, N,
                                  arithmetic.align_frac, chunk_sum, chunk_exponents);
        for (int col = 0; col < N; ++col) {
          const int32_t exponent = maximum(dot_exponents[col], chunk_exponents[col]);
          dot[col] = (dot[col] >> minimum(exponent - dot_exponents[col], 31)) +
                     (chunk_sum[col] >> minimum(exponent - chunk_exponents[col], 31));
          dot_exponents[col] = exponent;
        }
      }
      const bool c_inf_nan =
          add_accumulators(dot, dot_exponents, c_row, arithmetic.drop_far_accumulator, N, results);
      if (a_inf_nan || b_inf_nan || may_overflow != 0 || c_inf_nan) {
        for (int col = 0; col < N; ++col) {
          non_finite_result(arithmetic, &A[row * K + start], &B[start * N + col], N, products,
                            c_row[col], results[col]);
        }
      }
      for (int col = 0; col < N; ++col) {
        uint32_t bits = results[col];
        if (fp16_ovfl && (bits & 0x7F800000u) == 0x7F800000u) {
          bits = (bits & 0x007FFFFFu) != 0 ? 0u : bits - 1u;
        }
        c_row[col] = bit_cast<float>(bits);
      }
    }
  }
}

float source_value(MfmaGfx950Format format, bool operand_b, const uint32_t *const *source,
                   int index) {
  const bool half = format == MfmaGfx950Format::F16 || format == MfmaGfx950Format::Bf16;
  const int per_word = half ? 2 : 4;
  const uint32_t word = source[index / (64 * per_word)][index / per_word % 64];
  if (half) {
    const uint16_t bits = uint16_t(word >> (16 * (index % 2)));
    return format == MfmaGfx950Format::Bf16 ? bf16_to_float(bits) : f16_to_float(bits);
  }
  const uint8_t bits = uint8_t(word >> (8 * (index % 4)));
  const bool bf8 = format == MfmaGfx950Format::Bf8Bf8 ||
                   (format == MfmaGfx950Format::Fp8Bf8 && operand_b) ||
                   (format == MfmaGfx950Format::Bf8Fp8 && !operand_b);
  return bf8 ? mfma_decode_fp8<true, false>(bits) : mfma_decode_fp8<false, false>(bits);
}

template <int M, int K, bool Half>
void smfmac_gfx950_exact_shape(MfmaGfx950Format format, const Gfx950MfmaArithmetic &arithmetic,
                               uint64_t controls, bool fp16_ovfl, uint32_t *const *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               const uint32_t *const *index) {
  constexpr bool wide = (M * K * (Half ? 16 : 8)) == 16384;
  constexpr auto layout = Half ? (wide ? SmfmacLayout::Cdna4F16 : SmfmacLayout::Cdna3F16)
                               : (wide ? SmfmacLayout::Cdna4Fp8 : SmfmacLayout::Cdna3Fp8);
  constexpr int per_word = Half ? 2 : 4;
  constexpr int set_bits = M * K / 64;
  constexpr int output_regs = M * M / 64;
  constexpr int elems_per_lane = (K / 2) / (64 / M);
  const int shift = (controls & 3) == 0 ? set_bits * int((controls >> 3) & (32 / set_bits - 1)) : 0;
  uint32_t selected_index[64];
  for (int lane = 0; lane < 64; ++lane) {
    selected_index[lane] = index[0][lane] >> shift;
  }
  float left[M * K / 2], raw_right[M * K], right[K][M];
  for (int i = 0; i < M * K / 2; ++i) {
    left[i] = source_value(format, false, a, i);
  }
  for (int i = 0; i < M * K; ++i) {
    raw_right[i] = source_value(format, true, b, i);
  }
  for (int depth = 0; depth < K; ++depth) {
    for (int col = 0; col < M; ++col) {
      const auto loc = smfmac_dense_loc<layout, M>(depth, col);
      right[depth][col] =
          raw_right[(loc.element / per_word * 64 + loc.lane) * per_word + loc.element % per_word];
    }
  }
  uint32_t result[output_regs][64];
  for (int row = 0; row < M; ++row) {
    float row_a[K / 2], row_b[(K / 2) * M], row_c[M];
    // On gfx950, every SMFMAC shape (compact and wide, half and OCP FP8/BF8) feeds sparse
    // products in hardware (lane_group, register_element) order into 8-product chunks.
    for (int q = 0; q < K / 4; ++q) {
      for (int s = 0; s < 2; ++s) {
        const auto loc = smfmac_sparse_loc<layout, M, K>(row, q, s);
        const int t = (loc.lane / M) * elems_per_lane + loc.element;
        row_a[t] =
            left[(loc.element / per_word * 64 + loc.lane) * per_word + loc.element % per_word];
        const int sel = 4 * q + int((selected_index[loc.lane] >> (4 * loc.nibble + 2 * s)) & 3);
        for (int col = 0; col < M; ++col) {
          row_b[t * M + col] = right[sel][col];
        }
      }
    }
    for (int col = 0; col < M; ++col) {
      const auto out = mfma_output<M, 1>(row, col, 0);
      row_c[col] = as_float(d[out.reg][out.lane]);
    }
    gfx950_mfma_matmul(arithmetic, 1, M, K / 2, fp16_ovfl, row_a, row_b, row_c);
    for (int col = 0; col < M; ++col) {
      const auto out = mfma_output<M, 1>(row, col, 0);
      result[out.reg][out.lane] = as_bits(row_c[col]);
    }
  }
  for (int reg = 0; reg < output_regs; ++reg) {
    for (int lane = 0; lane < 64; ++lane) {
      d[reg][lane] = result[reg][lane];
    }
  }
}

} // namespace

void mfma_gfx950_exact(MfmaGfx950Format format, int m, int k, int batches, bool fp16_ovfl,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c) {
  const auto arithmetic = format_arithmetic(format);
  const int per_word = format == MfmaGfx950Format::F16 || format == MfmaGfx950Format::Bf16 ? 2 : 4;
  const int per_lane = k * m * batches / 64;
  const int output_regs = m * m * batches / 64;
  const int blocks_per_reg = (64 + m * m / 4 - 1) / (m * m / 4);
  uint32_t result[32][64];
  float left[512], right[512], accumulators[1024];
  for (int batch = 0; batch < batches; ++batch) {
    auto output = [&](int row, int col) {
      return MfmaWord{batch * (m * m / 64) + row / (4 * (64 / m)) * 4 + row % 4,
                      batch % blocks_per_reg * m + row / 4 % (64 / m) * blocks_per_reg * m + col};
    };
    for (int index = 0; index < m; ++index) {
      for (int depth = 0; depth < k; ++depth) {
        const int lane = batch * m + depth / per_lane * m * batches + index;
        const int local = depth % per_lane;
        const int source = (local / per_word * 64 + lane) * per_word + local % per_word;
        left[index * k + depth] = source_value(format, false, a, source);
        right[depth * m + index] = source_value(format, true, b, source);
      }
      for (int col = 0; col < m; ++col) {
        const auto location = output(index, col);
        accumulators[index * m + col] = as_float(c[location.reg][location.lane]);
      }
    }
    gfx950_mfma_matmul(arithmetic, m, m, k, fp16_ovfl, left, right, accumulators);
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < m; ++col) {
        const auto location = output(row, col);
        result[location.reg][location.lane] = as_bits(accumulators[row * m + col]);
      }
    }
  }
  for (int reg = 0; reg < output_regs; ++reg) {
    for (int lane = 0; lane < 64; ++lane) {
      d[reg][lane] = result[reg][lane];
    }
  }
}

void smfmac_gfx950_exact(MfmaGfx950Format format, int m, int k, uint64_t controls, bool fp16_ovfl,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *index) {
  const auto arithmetic = format_arithmetic(format);
  const bool half = format == MfmaGfx950Format::F16 || format == MfmaGfx950Format::Bf16;
  if (half) {
    if (m == 16 && k == 32) {
      smfmac_gfx950_exact_shape<16, 32, true>(format, arithmetic, controls, fp16_ovfl, d, a, b,
                                              index);
    } else if (m == 32 && k == 16) {
      smfmac_gfx950_exact_shape<32, 16, true>(format, arithmetic, controls, fp16_ovfl, d, a, b,
                                              index);
    } else if (m == 16 && k == 64) {
      smfmac_gfx950_exact_shape<16, 64, true>(format, arithmetic, controls, fp16_ovfl, d, a, b,
                                              index);
    } else {
      smfmac_gfx950_exact_shape<32, 32, true>(format, arithmetic, controls, fp16_ovfl, d, a, b,
                                              index);
    }
  } else {
    if (m == 16 && k == 64) {
      smfmac_gfx950_exact_shape<16, 64, false>(format, arithmetic, controls, fp16_ovfl, d, a, b,
                                               index);
    } else if (m == 32 && k == 32) {
      smfmac_gfx950_exact_shape<32, 32, false>(format, arithmetic, controls, fp16_ovfl, d, a, b,
                                               index);
    } else if (m == 16 && k == 128) {
      smfmac_gfx950_exact_shape<16, 128, false>(format, arithmetic, controls, fp16_ovfl, d, a, b,
                                                index);
    } else {
      smfmac_gfx950_exact_shape<32, 64, false>(format, arithmetic, controls, fp16_ovfl, d, a, b,
                                               index);
    }
  }
}

uint32_t dot2_bf16_gfx950_exact(uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1,
                                uint32_t c_bits) {
  const float a_vals[2] = {bf16_to_float(a0), bf16_to_float(a1)};
  const float b_vals[2] = {bf16_to_float(b0), bf16_to_float(b1)};
  float acc = as_float(c_bits);
  gfx950_mfma_matmul(kGfx950Dot2Bf16, 1, 1, 2, /*fp16_ovfl=*/false, a_vals, b_vals, &acc);
  return as_bits(acc);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
