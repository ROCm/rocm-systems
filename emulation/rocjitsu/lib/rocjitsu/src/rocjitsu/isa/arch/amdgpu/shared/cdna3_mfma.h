// SPDX-License-Identifier: MIT

#pragma once

/// @file cdna3_mfma.h
/// @brief CDNA3 MFMA arithmetic for FP16, BF16, XF32, FP8 and BF8 sources.
/// See tests/fixtures/float_dot/README.md for the hardware captures behind this model.

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief How one MFMA source reads an element once it has been staged as f32.
struct Cdna3MfmaSource {
  int32_t subnormal_exponent;
  bool ignores_low_13_bits;
};

inline constexpr Cdna3MfmaSource kCdna3MfmaSourceF16{-14, false};
inline constexpr Cdna3MfmaSource kCdna3MfmaSourceBf16{-126, false};
inline constexpr Cdna3MfmaSource kCdna3MfmaSourceXf32{-126, true};
inline constexpr Cdna3MfmaSource kCdna3MfmaSourceFp8Fnuz{-7, false};
inline constexpr Cdna3MfmaSource kCdna3MfmaSourceBf8Fnuz{-15, false};

/// @brief The arithmetic of one MFMA instruction.
struct Cdna3MfmaArithmetic {
  Cdna3MfmaSource source_a;
  Cdna3MfmaSource source_b;
  uint32_t products_per_step;
  bool even_odd_groups;
};

inline constexpr Cdna3MfmaArithmetic kCdna3MfmaF16{kCdna3MfmaSourceF16, kCdna3MfmaSourceF16, 8,
                                                   false};
inline constexpr Cdna3MfmaArithmetic kCdna3MfmaBf16{kCdna3MfmaSourceBf16, kCdna3MfmaSourceBf16, 8,
                                                    false};
inline constexpr Cdna3MfmaArithmetic kCdna3MfmaXf32{kCdna3MfmaSourceXf32, kCdna3MfmaSourceXf32, 4,
                                                    false};

/// @brief FP8 and BF8, as FNUZ, in any combination.
constexpr Cdna3MfmaArithmetic cdna3_mfma_f8(bool a_is_fp8, bool b_is_fp8) {
  return {a_is_fp8 ? kCdna3MfmaSourceFp8Fnuz : kCdna3MfmaSourceBf8Fnuz,
          b_is_fp8 ? kCdna3MfmaSourceFp8Fnuz : kCdna3MfmaSourceBf8Fnuz, 16, true};
}

namespace cdna3_mfma_detail {

inline constexpr uint32_t kMaxColumns = 32;
inline constexpr uint32_t kMaxSourceElements = 512;
inline constexpr uint32_t kMaxGroupProducts = 8;
inline constexpr int32_t kZeroProductExponent = -999;
inline constexpr uint32_t kInfNanExponentField = 0xFF;
inline constexpr uint32_t kPositiveInfinity = 0x7F800000u;
inline constexpr uint32_t kNegativeInfinity = 0xFF800000u;
inline constexpr uint32_t kNan = 0xFFC00000u;
// A product with 20 fraction bits is at least 2.0 when this shift leaves anything.
inline constexpr int32_t kProductAtLeastTwoShift = 21;

inline uint32_t source_bits(float value, Cdna3MfmaSource source) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return source.ignores_low_13_bits ? bits & 0xFFFFE000u : bits;
}

// Every source format has at most 10 fraction bits, so a product of mantissas fits in 32 bits.
// Returns whether any element is infinite or NaN. Their mantissas and exponents are meaningless,
// and any output that reads them is replaced by non_finite_result.
inline bool decode_sources(const float *values, uint32_t count, Cdna3MfmaSource source,
                           int32_t *mantissas, int32_t *exponents) {
  uint32_t any_inf_nan = 0;
  for (uint32_t index = 0; index < count; ++index) {
    const uint32_t bits = source_bits(values[index], source);
    const int32_t exponent_field = static_cast<int32_t>((bits >> 23) & 0xFF);
    const int32_t exponent = std::max(exponent_field, 1) - 127;
    const uint32_t f32_mantissa = (bits & 0x7FFFFFu) | (exponent_field != 0 ? 0x800000u : 0u);
    const int32_t below_format = std::clamp(source.subnormal_exponent - exponent, 0, 31);
    const int32_t magnitude = static_cast<int32_t>((f32_mantissa >> below_format) >> 13);
    mantissas[index] = (bits >> 31) != 0 ? -magnitude : magnitude;
    exponents[index] = std::max(exponent, source.subnormal_exponent);
    any_inf_nan |= exponent_field == static_cast<int32_t>(kInfNanExponentField) ? 1u : 0u;
  }
  return any_inf_nan != 0;
}

// Aligns one group of products to the largest in each column, truncates them toward zero to 24
// fraction bits and sums them. Returns nonzero if a product may have reached 2^128.
inline int32_t sum_products(const int32_t *a_mantissas, const int32_t *a_exponents,
                            const int32_t *b_mantissas, const int32_t *b_exponents,
                            uint32_t products, uint32_t step, uint32_t columns, int32_t *sums,
                            int32_t *top_exponents) {
  alignas(64) int32_t product_mantissas[kMaxGroupProducts * kMaxColumns];
  alignas(64) int32_t product_exponents[kMaxGroupProducts * kMaxColumns];
  for (uint32_t col = 0; col < columns; ++col) {
    sums[col] = 0;
    top_exponents[col] = kZeroProductExponent;
  }
  int32_t may_overflow = 0;
  for (uint32_t index = 0; index < products; ++index) {
    const int32_t a_mantissa = a_mantissas[index * step];
    const int32_t a_exponent = a_exponents[index * step];
    const int32_t *b_mantissa = &b_mantissas[index * step * columns];
    const int32_t *b_exponent = &b_exponents[index * step * columns];
    int32_t *mantissa_row = &product_mantissas[index * columns];
    int32_t *exponent_row = &product_exponents[index * columns];
    for (uint32_t col = 0; col < columns; ++col) {
      const int32_t mantissa = a_mantissa * b_mantissa[col];
      const int32_t exponent = mantissa != 0 ? a_exponent + b_exponent[col] : kZeroProductExponent;
      mantissa_row[col] = mantissa;
      exponent_row[col] = exponent;
      top_exponents[col] = std::max(top_exponents[col], exponent);
      const int32_t magnitude = mantissa < 0 ? -mantissa : mantissa;
      may_overflow |= (exponent + (magnitude >> kProductAtLeastTwoShift)) >= 128 ? 1 : 0;
    }
  }
  for (uint32_t index = 0; index < products; ++index) {
    const int32_t *mantissa_row = &product_mantissas[index * columns];
    const int32_t *exponent_row = &product_exponents[index * columns];
    for (uint32_t col = 0; col < columns; ++col) {
      // A shift of 30 clears any product: they are below 2^26 with 24 fraction bits.
      const int32_t shift = std::min(top_exponents[col] - exponent_row[col], 30);
      const int32_t scaled = mantissa_row[col] * 16;
      const int32_t toward_zero = (scaled >> 31) & ((1 << shift) - 1);
      sums[col] += (scaled + toward_zero) >> shift;
    }
  }
  return may_overflow;
}

// Adds each accumulator to its sum and rounds to f32. Returns whether any accumulator is
// infinite or NaN.
inline bool add_accumulators(const int32_t *dot, const int32_t *dot_exponents,
                             const float *accumulators, bool drop_far_accumulator, uint32_t columns,
                             uint32_t *results) {
  uint32_t inf_nan = 0;
  for (uint32_t col = 0; col < columns; ++col) {
    const uint32_t c_bits = std::bit_cast<uint32_t>(accumulators[col]);
    const int64_t c_exponent_field = (c_bits >> 23) & 0xFF;
    const int64_t c_exponent = std::max<int64_t>(c_exponent_field, 1) - 127;
    const int64_t c_magnitude = (c_bits & 0x7FFFFFu) | (c_exponent_field != 0 ? 0x800000u : 0u);
    const int64_t c_mantissa = (c_bits >> 31) != 0 ? -c_magnitude : c_magnitude;
    inf_nan |= c_exponent_field == kInfNanExponentField ? 1u : 0u;

    // In units of 2^-32 at the larger exponent. Right shifts of negative values round down.
    const int64_t dot_exponent = dot_exponents[col];
    const int64_t top_exponent = std::max(dot_exponent, c_exponent);
    const int64_t dot_part = (static_cast<int64_t>(dot[col]) * 256) >>
                             std::min<int64_t>(top_exponent - dot_exponent, 63);
    int64_t c_part = ((c_mantissa * 2) >> std::min<int64_t>(top_exponent - c_exponent, 63)) * 256;
    if (drop_far_accumulator)
      c_part = c_exponent < top_exponent - 25 ? 0 : c_part;
    const int64_t total = dot_part + c_part;

    // Normalize, rounding down to 31 fraction bits.
    const uint64_t total_magnitude = static_cast<uint64_t>(total < 0 ? -total : total);
    const int64_t total_top_bit = 63 - std::countl_zero(total_magnitude | 1);
    const int64_t normal_exponent = std::max<int64_t>(total_top_bit + top_exponent - 32, -126);
    const int64_t right = normal_exponent - top_exponent + 1;
    const int64_t left_shift = std::clamp<int64_t>(-right, 0, 31);
    const int64_t right_shift = std::clamp<int64_t>(right, 0, 63);
    const int64_t fixed =
        static_cast<int64_t>(static_cast<uint64_t>(total) << left_shift) >> right_shift;

    // Round to f32, nearest even, on the magnitude. Adding the mantissa with its implicit bit to
    // (biased exponent - 1) carries into the exponent where it should. Subnormals have that at 0.
    const uint64_t magnitude = static_cast<uint64_t>(fixed < 0 ? -fixed : fixed);
    const int64_t top_bit = 63 - std::countl_zero(magnitude | 1);
    const int64_t exponent = std::max<int64_t>(top_bit + normal_exponent - 31, -126);
    // Only a zero, whose result is discarded below, could ask for less than 8.
    const int64_t dropped = std::clamp<int64_t>(exponent - 23 - normal_exponent + 31, 1, 63);
    const uint64_t kept = magnitude >> dropped;
    const uint64_t remainder = magnitude & ((uint64_t{1} << dropped) - 1);
    const uint64_t half = uint64_t{1} << (dropped - 1);
    const uint64_t round_up = (remainder > half || (remainder == half && (kept & 1) != 0)) ? 1 : 0;
    const uint64_t bits = (static_cast<uint64_t>(exponent + 126) << 23) + kept + round_up;
    const uint64_t bounded = std::min<uint64_t>(bits, kPositiveInfinity);
    const uint64_t signed_bits = bounded | (fixed < 0 ? uint64_t{0x80000000u} : uint64_t{0});
    results[col] = magnitude != 0 ? static_cast<uint32_t>(signed_bits) : 0u;
  }
  return inf_nan != 0;
}

// With an infinite or NaN input, or a product reaching 2^128, an output is infinite or NaN
// whatever its other terms are. Returns whether this one is, setting @p result.
inline bool non_finite_result(const Cdna3MfmaArithmetic &arithmetic, const float *a, const float *b,
                              uint32_t columns, uint32_t products, float accumulator,
                              uint32_t &result) {
  bool nan = false, positive = false, negative = false;
  const auto add_infinity = [&](bool sign) { (sign ? negative : positive) = true; };
  for (uint32_t index = 0; index < products; ++index) {
    const uint32_t a_bits = source_bits(a[index], arithmetic.source_a);
    const uint32_t b_bits = source_bits(b[index * columns], arithmetic.source_b);
    const bool sign = ((a_bits ^ b_bits) >> 31) != 0;
    const bool a_special = ((a_bits >> 23) & 0xFF) == kInfNanExponentField;
    const bool b_special = ((b_bits >> 23) & 0xFF) == kInfNanExponentField;
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
      decode_sources(&a[index], 1, arithmetic.source_a, &a_mantissa, &a_exponent);
      decode_sources(&b[index * columns], 1, arithmetic.source_b, &b_mantissa, &b_exponent);
      const int32_t mantissa = a_mantissa * b_mantissa;
      const int32_t magnitude = mantissa < 0 ? -mantissa : mantissa;
      if (magnitude != 0 && a_exponent + b_exponent + (magnitude >> kProductAtLeastTwoShift) >= 128)
        add_infinity(sign);
    }
  }
  const uint32_t c_bits = std::bit_cast<uint32_t>(accumulator);
  if (((c_bits >> 23) & 0xFF) == kInfNanExponentField) {
    if ((c_bits & 0x7FFFFFu) != 0)
      nan = true;
    else
      add_infinity((c_bits >> 31) != 0);
  }
  if (nan || (positive && negative)) {
    result = kNan;
    return true;
  }
  if (positive || negative) {
    result = positive ? kPositiveInfinity : kNegativeInfinity;
    return true;
  }
  return false;
}

} // namespace cdna3_mfma_detail

/// @brief D = A*B + C for one block of a CDNA3 MFMA, in place on C, characterized on MI300X.
/// @details These instructions don't round after each multiply-accumulate. Each output takes its
/// K products in steps. In each step the hardware:
///   1. Aligns the products to the largest, truncates each toward zero to 24 fraction bits and
///      sums them exactly. FP8 and BF8 sum even and odd positions separately, each with its own
///      alignment, then align the two sums rounding down and add them.
///   2. Rounds that sum down to 32 fraction bits and C down to 24, both relative to the larger
///      exponent, and adds them. FP8 and BF8 discard a finite C more than 25 bits below it.
///   3. Normalizes, rounding down to 31 fraction bits, then rounds to f32 to nearest even.
/// XF32 sources lose their low 13 bits first. A product of at least 2^128 counts as infinite.
/// Every NaN result is written as 0xFFC00000.
///
/// This is the model of Xie et al., "Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate
/// Units" (MICRO 2026), implemented by MMA-Sim (github.com/microsoft/MMA-Sim, MIT). Where
/// MMA-Sim 2.1.1's FP8 path discards an infinite or NaN C next to a large sum, this keeps it, as
/// an MI300X does. Integer arithmetic makes results independent of the host FP state, and the
/// column loops are plain enough for the compiler to vectorize.
///
/// @param A A[row * K + k], staged as f32.
/// @param B B[k * N + col], staged as f32.
/// @param C C[row * N + col], read and written as f32 bits.
inline void cdna3_mfma_matmul(const Cdna3MfmaArithmetic &arithmetic, uint32_t M, uint32_t N,
                              uint32_t K, const float *A, const float *B, float *C) {
  using namespace cdna3_mfma_detail;
  assert(N <= kMaxColumns && M * K <= kMaxSourceElements && K * N <= kMaxSourceElements);
  alignas(64) int32_t a_mantissas[kMaxSourceElements], a_exponents[kMaxSourceElements];
  alignas(64) int32_t b_mantissas[kMaxSourceElements], b_exponents[kMaxSourceElements];
  const bool a_inf_nan = decode_sources(A, M * K, arithmetic.source_a, a_mantissas, a_exponents);
  const bool b_inf_nan = decode_sources(B, K * N, arithmetic.source_b, b_mantissas, b_exponents);

  alignas(64) int32_t dot[kMaxColumns], dot_exponents[kMaxColumns];
  alignas(64) int32_t odd[kMaxColumns], odd_exponents[kMaxColumns];
  alignas(64) uint32_t results[kMaxColumns];
  for (uint32_t row = 0; row < M; ++row) {
    float *c_row = &C[row * N];
    for (uint32_t start = 0; start < K; start += arithmetic.products_per_step) {
      const uint32_t products = std::min(K - start, arithmetic.products_per_step);
      const int32_t *step_a_mantissas = &a_mantissas[row * K + start];
      const int32_t *step_a_exponents = &a_exponents[row * K + start];
      const int32_t *step_b_mantissas = &b_mantissas[start * N];
      const int32_t *step_b_exponents = &b_exponents[start * N];
      int32_t may_overflow = 0;
      if (arithmetic.even_odd_groups) {
        may_overflow |=
            sum_products(step_a_mantissas, step_a_exponents, step_b_mantissas, step_b_exponents,
                         (products + 1) / 2, 2, N, dot, dot_exponents);
        may_overflow |=
            sum_products(step_a_mantissas + 1, step_a_exponents + 1, step_b_mantissas + N,
                         step_b_exponents + N, products / 2, 2, N, odd, odd_exponents);
        for (uint32_t col = 0; col < N; ++col) {
          const int32_t exponent = std::max(dot_exponents[col], odd_exponents[col]);
          dot[col] = (dot[col] >> std::min(exponent - dot_exponents[col], 31)) +
                     (odd[col] >> std::min(exponent - odd_exponents[col], 31));
          dot_exponents[col] = exponent;
        }
      } else {
        may_overflow |= sum_products(step_a_mantissas, step_a_exponents, step_b_mantissas,
                                     step_b_exponents, products, 1, N, dot, dot_exponents);
      }
      const bool c_inf_nan =
          add_accumulators(dot, dot_exponents, c_row, arithmetic.even_odd_groups, N, results);
      if (a_inf_nan || b_inf_nan || may_overflow != 0 || c_inf_nan) {
        for (uint32_t col = 0; col < N; ++col)
          non_finite_result(arithmetic, &A[row * K + start], &B[start * N + col], N, products,
                            c_row[col], results[col]);
      }
      for (uint32_t col = 0; col < N; ++col)
        c_row[col] = std::bit_cast<float>(results[col]);
    }
  }
}

} // namespace rocjitsu::amdgpu
