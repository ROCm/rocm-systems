// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/transcendental_simd.h"
#include "util/amdgpu_exp.h"
#include "util/amdgpu_log.h"
#include "util/amdgpu_rcp.h"
#include "util/amdgpu_rsq.h"
#include "util/amdgpu_sqrt.h"
#include "util/amdgpu_trig.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace rocjitsu::amdgpu::transcendental {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
namespace detail {
void evaluate_reciprocal_f32_simd(bool rsq, const uint32_t *input, uint32_t *output, size_t count,
                                  bool quiet_snan);
}
#endif
namespace {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
// Retain the scalar mappings' integer stages and rounding. Coefficient columns
// are derived from the same authoritative rows for register-table selection.
using V = __m512i;
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V
constant(uint64_t x) {
  return _mm512_set1_epi64(static_cast<long long>(x));
}
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V add(V a, V b) {
  return _mm512_add_epi64(a, b);
}
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V sub(V a, V b) {
  return _mm512_sub_epi64(a, b);
}
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V mul(V a, V b) {
  return _mm512_mullo_epi64(a, b);
}
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V band(V a,
                                                                                    uint64_t b) {
  return _mm512_and_si512(a, constant(b));
}
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V select(__mmask8 mask,
                                                                                      V yes, V no) {
  return _mm512_mask_blend_epi64(mask, no, yes);
}
template <unsigned Shift>
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V round_even(V value) {
  V whole = _mm512_srli_epi64(value, Shift);
  V rest = band(value, (uint64_t{1} << Shift) - 1);
  V half = constant(uint64_t{1} << (Shift - 1));
  __mmask8 increment =
      _mm512_cmpgt_epu64_mask(rest, half) |
      (_mm512_cmpeq_epi64_mask(rest, half) & _mm512_cmpneq_epi64_mask(band(whole, 1), constant(0)));
  return _mm512_mask_add_epi64(whole, increment, whole, constant(1));
}
template <typename Coefficient, size_t N> constexpr auto columns(const Coefficient (&rows)[N]) {
  std::array<std::array<uint32_t, N>, 4> result{};
  for (size_t i = 0; i < N; ++i) {
    result[0][i] = rows[i].constant;
    result[1][i] = rows[i].linear;
    result[2][i] = rows[i].quadratic;
    result[3][i] = rows[i].cubic;
  }
  return result;
}
inline constexpr auto exp_columns = columns(util::detail::exp::coefficients);
inline constexpr auto log_columns = columns(util::detail::log::coefficients);
inline constexpr auto sqrt_columns = columns(util::detail::sqrt::coefficients);
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V
table_coefficient(V index, const uint32_t *table) {
  auto indices = _mm512_zextsi256_si512(_mm512_cvtepi64_epi32(index));
  auto values =
      _mm512_permutex2var_epi32(_mm512_loadu_si512(table), indices, _mm512_loadu_si512(table + 16));
  return _mm512_cvtepu32_epi64(_mm512_castsi512_si256(values));
}
template <bool Logarithm, unsigned Field>
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V
coefficient(V index) {
  const uint32_t *table;
  if constexpr (Logarithm)
    table = log_columns[Field].data();
  else
    table = exp_columns[Field].data();
  V result = table_coefficient(index, table);
  if constexpr (Logarithm)
    result = select(_mm512_cmpeq_epi64_mask(index, constant(32)), constant(log_columns[Field][32]),
                    result);
  return result;
}
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline __mmask8
select_exp_range(V bits, V mag) {
  __mmask8 negative = _mm512_cmpneq_epi64_mask(band(bits, 0x80000000), constant(0));
  return (negative & _mm512_cmple_epu64_mask(mag, constant(0x42fc0000))) |
         (~negative & _mm512_cmplt_epu64_mask(mag, constant(0x43000000)));
}

[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V exp_normal(V bits) {
  V mag = band(bits, 0x7fffffff);
  V exponent = sub(_mm512_srli_epi64(mag, 23), constant(127));
  V mantissa = _mm512_or_si512(band(mag, 0x7fffff), constant(0x800000));
  V shift = add(exponent, constant(6));
  __mmask8 left = _mm512_cmpge_epi64_mask(shift, constant(0));
  V absolute = select(left, _mm512_sllv_epi64(mantissa, shift),
                      _mm512_srlv_epi64(mantissa, sub(constant(0), shift)));
  __mmask8 negative = _mm512_cmpneq_epi64_mask(band(bits, 0x80000000), constant(0));
  V phase = select(negative, _mm512_xor_si512(absolute, constant(~uint64_t{0})), absolute);
  V fraction = band(phase, 0xffffff);
  V index = band(_mm512_srli_epi64(phase, 24), 31);
  V c0 = coefficient<false, 0>(index);
  V c1 = coefficient<false, 1>(index);
  V c2 = coefficient<false, 2>(index);
  V c3 = coefficient<false, 3>(index);
  V lp = mul(c1, fraction);
  V linear = select(_mm512_cmpge_epu64_mask(lp, constant(uint64_t{1} << 47)), round_even<24>(lp),
                    _mm512_srli_epi64(round_even<23>(lp), 1));
  V coordinate = _mm512_srli_epi64(fraction, 6);
  V square = _mm512_srli_epi64(mul(coordinate, coordinate), 12);
  V inner = round_even<7>(add(_mm512_slli_epi64(c2, 7), mul(c3, coordinate)));
  V product = mul(inner, square);
  V quad = select(_mm512_cmpge_epu64_mask(product, constant(uint64_t{1} << 47)),
                  _mm512_slli_epi64(round_even<24>(product), 1), round_even<23>(product));
  V sum = add(_mm512_slli_epi64(add(c0, linear), 8), quad);
  return add(_mm512_slli_epi64(add(_mm512_srai_epi64(phase, 29), constant(126)), 23),
             round_even<13>(sum));
}

[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V
log_near_one_negative(V bits) {
  // Four-bit normalization and upward magnitude rounding below one.
  V k = sub(constant(0x3f800000), bits);
  __mmask8 upper = _mm512_cmpge_epu64_mask(k, constant(0x20000));
  V c1 = select(upper, constant(12102221), constant(12102203));
  V c2 = select(upper, constant(12097323), constant(12101979));
  V c3 = select(upper, constant(2088428), constant(2035248));
  V width = sub(constant(64), _mm512_lzcnt_epi64(k));
  V normalization = band(sub(constant(18), _mm512_min_epu64(width, constant(18))), ~uint64_t{3});
  V linear_shift = sub(constant(14), _mm512_min_epu64(normalization, constant(14)));
  V square_shift = sub(constant(12), _mm512_min_epu64(normalization, constant(12)));
  V linear = _mm512_sllv_epi64(_mm512_srlv_epi64(mul(_mm512_slli_epi64(c1, 1), k), linear_shift),
                               linear_shift);
  V square = _mm512_sllv_epi64(_mm512_srlv_epi64(mul(k, k), square_shift), square_shift);
  V inner = round_even<22>(add(_mm512_slli_epi64(c2, 22), mul(c3, sub(k, constant(1)))));
  V product_shift = sub(constant(38), normalization);
  V quadratic = _mm512_srlv_epi64(
      add(mul(inner, square), sub(_mm512_sllv_epi64(constant(1), product_shift), constant(1))),
      product_shift);
  V fixed = sub(add(_mm512_slli_epi64(linear, 2),
                    _mm512_sllv_epi64(quadratic, sub(constant(16), normalization))),
                select(upper, constant(uint64_t{100} << 16), constant(0)));
  V leading = sub(constant(63), _mm512_lzcnt_epi64(fixed));
  V shift = sub(leading, constant(23));
  V mantissa =
      _mm512_srlv_epi64(add(fixed, sub(_mm512_sllv_epi64(constant(1), shift), constant(1))), shift);
  __mmask8 carry = _mm512_cmpeq_epi64_mask(mantissa, constant(uint64_t{1} << 24));
  mantissa = select(carry, _mm512_srli_epi64(mantissa, 1), mantissa);
  leading = _mm512_mask_add_epi64(leading, carry, leading, constant(1));
  return _mm512_or_si512(
      constant(0x80000000),
      _mm512_or_si512(_mm512_slli_epi64(add(leading, constant(77)), 23), band(mantissa, 0x7fffff)));
}

[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V
log_near_one_positive(V bits) {
  // Above one, advance the truncated significand even on an exact grid hit.
  V i = sub(bits, constant(0x3f800000));
  __mmask8 middle = _mm512_cmpge_epu64_mask(i, constant(0x10000));
  __mmask8 upper = _mm512_cmpge_epu64_mask(i, constant(0x20000));
  V c0 = select(upper, constant(2856), select(middle, constant(192), constant(0)));
  V c1 = select(upper, constant(12102072), select(middle, constant(12102186), constant(12102203)));
  V c2 = select(upper, constant(12084310), select(middle, constant(12097614), constant(12102094)));
  V c3 = select(upper, constant(1883467), select(middle, constant(1948887), constant(1999052)));
  V width = sub(constant(64), _mm512_lzcnt_epi64(i));
  V normalization = select(middle, constant(0), band(sub(constant(18), width), ~uint64_t{3}));
  V normalized = _mm512_sllv_epi64(i, normalization);
  V linear = _mm512_slli_epi64(_mm512_srli_epi64(mul(c1, normalized), 13), 2);
  V square = _mm512_srli_epi64(mul(normalized, i), 12);
  V inner = round_even<22>(
      sub(_mm512_slli_epi64(c2, 22), mul(c3, add(_mm512_slli_epi64(i, 1), constant(1)))));
  V product = mul(inner, square);
  V quadratic = select(_mm512_cmpge_epu64_mask(product, constant(uint64_t{1} << 47)),
                       _mm512_slli_epi64(round_even<24>(product), 1), round_even<23>(product));
  V fixed = sub(add(c0, linear), quadratic);
  V leading = sub(constant(63), _mm512_lzcnt_epi64(fixed));
  V mantissa = add(_mm512_srlv_epi64(fixed, sub(leading, constant(23))), constant(1));
  __mmask8 carry = _mm512_cmpeq_epi64_mask(mantissa, constant(uint64_t{1} << 24));
  mantissa = select(carry, _mm512_srli_epi64(mantissa, 1), mantissa);
  leading = _mm512_mask_add_epi64(leading, carry, leading, constant(1));
  return _mm512_or_si512(_mm512_slli_epi64(sub(add(leading, constant(92)), normalization), 23),
                         band(mantissa, 0x7fffff));
}

[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V
log_ordinary(V bits) {
  V fraction = band(bits, 0x3ffff);
  V index = band(_mm512_srli_epi64(bits, 18), 31);
  index = select(_mm512_cmpeq_epi64_mask(index, constant(1)) &
                     _mm512_cmpge_epu64_mask(fraction, constant(0x20000)),
                 constant(32), index);
  V c0 = coefficient<true, 0>(index);
  V c1 = coefficient<true, 1>(index);
  V c2 = coefficient<true, 2>(index);
  V c3 = coefficient<true, 3>(index);
  V linear = _mm512_srli_epi64(mul(c1, fraction), 16);
  V inner = round_even<22>(
      sub(_mm512_slli_epi64(c2, 22), mul(c3, add(_mm512_slli_epi64(fraction, 1), constant(1)))));
  V square = _mm512_srli_epi64(mul(fraction, fraction), 12);
  V product = mul(inner, square);
  V quad = select(_mm512_cmpge_epu64_mask(product, constant(uint64_t{1} << 47)),
                  _mm512_slli_epi64(round_even<24>(product), 1), round_even<23>(product));
  V exponent = sub(_mm512_srli_epi64(bits, 23), constant(127));
  V fixed = add(_mm512_slli_epi64(exponent, 35), sub(_mm512_slli_epi64(add(c0, linear), 5), quad));
  __mmask8 negative = _mm512_cmplt_epi64_mask(fixed, constant(0));
  V magnitude = select(negative, sub(constant(0), fixed), fixed);
  V leading = sub(constant(63), _mm512_lzcnt_epi64(magnitude));
  V shift = sub(leading, constant(23));
  V bias = sub(_mm512_sllv_epi64(constant(1), shift), constant(1));
  V mantissa = _mm512_srlv_epi64(add(magnitude, select(negative, bias, constant(0))), shift);
  __mmask8 carry = _mm512_cmpeq_epi64_mask(mantissa, constant(uint64_t{1} << 24));
  mantissa = select(carry, _mm512_srli_epi64(mantissa, 1), mantissa);
  leading = _mm512_mask_add_epi64(leading, carry, leading, constant(1));
  return _mm512_or_si512(
      select(negative, constant(0x80000000), constant(0)),
      _mm512_or_si512(_mm512_slli_epi64(add(leading, constant(92)), 23), band(mantissa, 0x7fffff)));
}

template <bool Logarithm>
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::noinline]] void
batch(const uint32_t *input, uint32_t *output, bool quiet_snan) {
  __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input));
  V bits = _mm512_cvtepu32_epi64(packed);
  __mmask8 eligible;
  if constexpr (Logarithm) {
    eligible = _mm512_cmpge_epu64_mask(bits, constant(0x00800000)) &
               _mm512_cmplt_epu64_mask(bits, constant(0x7f800000)) &
               (_mm512_cmplt_epu64_mask(bits, constant(0x3f7c0000)) |
                _mm512_cmpge_epu64_mask(bits, constant(0x3f840000)));
  } else {
    V mag = band(bits, 0x7fffffff);
    eligible = _mm512_cmpge_epu64_mask(mag, constant(0x33800000)) & select_exp_range(bits, mag);
  }
  V magnitude = band(bits, 0x7fffffff);
  __mmask8 negative = _mm512_cmpneq_epi64_mask(band(bits, 0x80000000), constant(0));
  V result;
  if constexpr (Logarithm) {
    result = select(negative, constant(0xffc00000), constant(0x7f800000));
    result = select(_mm512_cmplt_epu64_mask(magnitude, constant(0x00800000)), constant(0xff800000),
                    result);
    result = select(_mm512_cmpeq_epi64_mask(bits, constant(0x3f800000)), constant(0), result);
  } else {
    result = constant(0x3f800000);
    result = select(negative & _mm512_cmpgt_epu64_mask(magnitude, constant(0x42fc0000)),
                    constant(0), result);
    result = select(~negative & _mm512_cmpge_epu64_mask(magnitude, constant(0x43000000)),
                    constant(0x7f800000), result);
  }
  result = select(_mm512_cmpgt_epu64_mask(magnitude, constant(0x7f800000)),
                  _mm512_or_si512(bits, constant(quiet_snan ? 0x00400000 : 0)), result);
  if (eligible) {
    V safe = select(eligible, bits, constant(Logarithm ? 0x40000000 : 0x3f000000));
    if constexpr (Logarithm)
      result = select(eligible, log_ordinary(safe), result);
    else
      result = select(eligible, exp_normal(safe), result);
  }
  if constexpr (Logarithm) {
    __mmask8 below = _mm512_cmpge_epu64_mask(bits, constant(0x3f7c0000)) &
                     _mm512_cmplt_epu64_mask(bits, constant(0x3f800000));
    __mmask8 above = _mm512_cmpgt_epu64_mask(bits, constant(0x3f800000)) &
                     _mm512_cmplt_epu64_mask(bits, constant(0x3f840000));
    if (below)
      result =
          select(below, log_near_one_negative(select(below, bits, constant(0x3f7fffff))), result);
    if (above)
      result =
          select(above, log_near_one_positive(select(above, bits, constant(0x3f800001))), result);
  }
  __m256i narrowed = _mm512_cvtepi64_epi32(result);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(output), narrowed);
}

inline constexpr auto trig_columns = [] {
  std::array<std::array<int32_t, 32>, 4> result{};
  for (unsigned i = 0; i < 32; ++i) {
    auto c = util::detail::trig::coefficients[i / 16][i % 16];
    result[0][i] = c.constant;
    result[1][i] = c.linear;
    result[2][i] = c.quadratic;
    result[3][i] = c.cubic;
  }
  return result;
}();
template <unsigned Field>
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V
signed_coefficient(V index) {
  auto indices = _mm512_zextsi256_si512(_mm512_cvtepi64_epi32(index));
  auto values = _mm512_permutex2var_epi32(_mm512_loadu_si512(trig_columns[Field].data()), indices,
                                          _mm512_loadu_si512(trig_columns[Field].data() + 16));
  return _mm512_cvtepi32_epi64(_mm512_castsi512_si256(values));
}
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V
round_variable(V value, V shift) {
  V whole = _mm512_srlv_epi64(value, shift);
  V rest = _mm512_and_si512(value, sub(_mm512_sllv_epi64(constant(1), shift), constant(1)));
  V half = _mm512_sllv_epi64(constant(1), sub(shift, constant(1)));
  __mmask8 increment =
      _mm512_cmpgt_epu64_mask(rest, half) |
      (_mm512_cmpeq_epi64_mask(rest, half) & _mm512_cmpneq_epi64_mask(band(whole, 1), constant(0)));
  return _mm512_mask_add_epi64(whole, increment, whole, constant(1));
}
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::always_inline]] inline V
ordinary_trig(V bits, bool cosine, __mmask8 &eligible) {
  // Reduction is in turns; normalized sine intervals retain extra product bits.
  V magnitude = band(bits, 0x7fffffff);
  eligible = _mm512_cmpge_epu64_mask(magnitude, constant(cosine ? 0x00800000 : 0x39c00000)) &
             _mm512_cmplt_epu64_mask(magnitude, constant(0x4b000000));
  bits = select(eligible, bits, constant(0x3dcccccd));
  magnitude = band(bits, 0x7fffffff);
  V exponent = sub(_mm512_srli_epi64(magnitude, 23), constant(127));
  V mantissa = _mm512_or_si512(band(magnitude, 0x7fffff), constant(0x800000));
  V shift = add(exponent, constant(8));
  V phase = select(_mm512_cmpge_epi64_mask(shift, constant(0)),
                   band(_mm512_sllv_epi64(mantissa, shift), 0x7fffffff),
                   _mm512_srlv_epi64(mantissa, sub(constant(0), shift)));
  V quadrant = _mm512_srli_epi64(phase, 29);
  V quarter_phase = band(phase, 0x1fffffff);
  __mmask8 reflected = _mm512_cmpge_epu64_mask(quarter_phase, constant(0x10000000));
  V reflection = select(reflected, constant(1), constant(0));
  V reduced = select(reflected, sub(constant(0x20000000), quarter_phase), quarter_phase);
  V adjusted = sub(reduced, reflection);
  V table = _mm512_xor_si512(_mm512_xor_si512(constant(cosine), band(quadrant, 1)), reflection);
  V index = _mm512_srli_epi64(adjusted, 24);
  V fraction = sub(reduced, _mm512_slli_epi64(index, 24));
  __mmask8 normalized =
      _mm512_cmpeq_epi64_mask(table, constant(0)) & _mm512_cmplt_epu64_mask(index, constant(2));
  V row = add(_mm512_slli_epi64(table, 4), index);
  V c0 = signed_coefficient<0>(row), c1 = signed_coefficient<1>(row);
  V c2 = signed_coefficient<2>(row), c3 = signed_coefficient<3>(row);
  V square_input = _mm512_srli_epi64(sub(fraction, reflection), 7);
  __mmask8 sine_coefficient = _mm512_cmple_epi64_mask(c3, constant(0));
  V inner_coordinate = select(sine_coefficient, sub(fraction, reflection),
                              sub(add(sub(constant(1u << 24), fraction), reflection), constant(1)));
  inner_coordinate = band(inner_coordinate, ~uint64_t{63});
  V inner = select(
      sine_coefficient, sub(_mm512_slli_epi64(sub(constant(0), c2), 24), mul(c3, inner_coordinate)),
      add(_mm512_slli_epi64(sub(constant(0), add(c2, c3)), 24), mul(c3, inner_coordinate)));
  V staged_inner = round_even<17>(inner);
  V value = sub(fraction, reflection);
  V rounded = _mm512_srli_epi64(add(value, constant(8)), 4);
  V limit = sub(_mm512_slli_epi64(add(_mm512_srli_epi64(value, 24), constant(1)), 20), constant(1));
  V linear_input = _mm512_slli_epi64(_mm512_min_epu64(rounded, limit), 4);
  V product = mul(c1, linear_input);
  __mmask8 negative_product = _mm512_cmplt_epi64_mask(product, constant(0));
  V absolute = select(negative_product, sub(constant(0), product), product);
  V linear = select(_mm512_cmpge_epu64_mask(absolute, constant(uint64_t{1} << 45)),
                    _mm512_slli_epi64(round_even<22>(absolute), 1), round_even<21>(absolute));
  linear = select(negative_product, sub(constant(0), linear), linear);
  V square = _mm512_srli_epi64(mul(square_input, square_input), 10);
  V quadratic_product = mul(staged_inner, square);
  V bias = select(_mm512_cmpge_epu64_mask(quadratic_product, constant(uint64_t{1} << 47)),
                  constant(uint64_t{1} << 23), constant(uint64_t{1} << 22));
  V quadratic = select(_mm512_cmple_epu64_mask(quadratic_product, bias), constant(0),
                       sub(constant(0), _mm512_srli_epi64(add(sub(quadratic_product, bias),
                                                              constant((uint64_t{1} << 29) - 1)),
                                                          29)));
  V sum = add(add(_mm512_slli_epi64(c0, 3), linear), _mm512_slli_epi64(quadratic, 1));
  V fractional = constant(29);
  if (normalized) {
    V width = sub(constant(64), _mm512_lzcnt_epi64(fraction));
    __mmask8 first = _mm512_cmpeq_epi64_mask(index, constant(0));
    V normalization =
        select(first, band(sub(constant(24), _mm512_min_epu64(width, constant(24))), ~uint64_t{3}),
               constant(0));
    V offset_shift =
        select(first, band(sub(constant(25), _mm512_min_epu64(width, constant(25))), ~uint64_t{3}),
               constant(0));
    V rounded2 = _mm512_srli_epi64(add(value, constant(2)), 2);
    V limit2 =
        sub(_mm512_slli_epi64(add(_mm512_srli_epi64(value, 22), constant(1)), 20), constant(1));
    V normalized_linear_input = select(first, _mm512_sllv_epi64(fraction, offset_shift),
                                       _mm512_slli_epi64(_mm512_min_epu64(rounded2, limit2), 2));
    __mmask8 saturated =
        first & reflected &
        _mm512_cmpeq_epi64_mask(band(_mm512_sllv_epi64(fraction, normalization), 0xfffff),
                                constant(0));
    normalized_linear_input = _mm512_mask_sub_epi64(normalized_linear_input, saturated,
                                                    normalized_linear_input, constant(1));
    V lp = select(first, add(constant(32), normalization), constant(30));
    V qp = add(constant(34), normalization);
    V normalized_linear = _mm512_srlv_epi64(mul(c1, normalized_linear_input),
                                            sub(add(constant(50), offset_shift), lp));
    V sp = _mm512_min_epu64(constant(34), add(constant(24), normalization));
    V staged_square = _mm512_srlv_epi64(mul(square_input, square_input), sub(constant(34), sp));
    V correction = round_variable(mul(staged_inner, staged_square), sub(add(constant(33), sp), qp));
    V normalized_sum = sub(add(_mm512_sllv_epi64(c0, sub(qp, constant(26))),
                               _mm512_sllv_epi64(normalized_linear, sub(qp, lp))),
                           correction);
    sum = select(normalized, normalized_sum, sum);
    fractional = select(normalized, qp, fractional);
  }
  V leading = sub(constant(63), _mm512_lzcnt_epi64(sum));
  V significand = select(_mm512_cmpge_epu64_mask(leading, constant(23)),
                         _mm512_srlv_epi64(sum, sub(leading, constant(23))),
                         _mm512_sllv_epi64(sum, sub(constant(23), leading)));
  V result = _mm512_or_si512(_mm512_slli_epi64(sub(add(leading, constant(127)), fractional), 23),
                             band(significand, 0x7fffff));
  result = select(_mm512_cmpeq_epi64_mask(sum, constant(0)), constant(0), result);
  result = select(_mm512_cmpneq_epi64_mask(table, constant(0)) &
                      _mm512_cmplt_epu64_mask(adjusted, constant(46592)),
                  constant(0x3f800000), result);
  result = select(_mm512_cmpeq_epi64_mask(table, constant(0)) &
                      _mm512_cmpeq_epi64_mask(reduced, constant(0)),
                  constant(0), result);
  V negative = _mm512_xor_si512(_mm512_srli_epi64(quadrant, 1),
                                cosine ? band(quadrant, 1) : _mm512_srli_epi64(bits, 31));
  negative = select(_mm512_cmpeq_epi64_mask(result, constant(0)), constant(0), negative);
  return _mm512_or_si512(result, _mm512_slli_epi64(negative, 31));
}
template <bool Cosine>
[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::noinline]] void
trig_batch(const uint32_t *input, uint32_t *output, unsigned denorm, bool quiet_snan) {
  __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input));
  V bits = _mm512_cvtepu32_epi64(packed), magnitude = band(bits, 0x7fffffff);
  V sign = band(bits, 0x80000000);
  V result = constant(Cosine ? 0x3f800000 : 0);
  if constexpr (!Cosine) {
    __mmask8 small = _mm512_cmplt_epu64_mask(magnitude, constant(0x39c00000)) &
                     _mm512_cmpneq_epi64_mask(magnitude, constant(0));
    if (!(denorm & 1))
      small &= _mm512_cmpge_epu64_mask(magnitude, constant(0x00800000));
    result = select(_mm512_cmpeq_epi64_mask(magnitude, constant(0)), sign, result);
    if (!(denorm & 1))
      result = select(_mm512_cmplt_epu64_mask(magnitude, constant(0x00800000)), sign, result);
    if (small) {
      V safe = select(small, magnitude, constant(0x00800000));
      V exponent_bits = _mm512_srli_epi64(safe, 23);
      __mmask8 normal = _mm512_cmpneq_epi64_mask(exponent_bits, constant(0));
      V mantissa =
          _mm512_or_si512(band(safe, 0x7fffff), select(normal, constant(0x800000), constant(0)));
      V exponent = select(normal, sub(exponent_bits, constant(127)), constant(uint64_t(-126)));
      V product = mul(mantissa, constant(0xc90fd5));
      V leading = sub(constant(63), _mm512_lzcnt_epi64(product));
      V result_exponent = add(sub(exponent, constant(44)), leading);
      __mmask8 subnormal = _mm512_cmplt_epi64_mask(result_exponent, constant(uint64_t(-126)));
      V rounded = round_variable(product, sub(leading, constant(23)));
      __mmask8 carry = _mm512_cmpeq_epi64_mask(rounded, constant(0x1000000));
      rounded = select(carry, _mm512_srli_epi64(rounded, 1), rounded);
      result_exponent = _mm512_mask_add_epi64(result_exponent, carry, result_exponent, constant(1));
      V packed_result = _mm512_or_si512(_mm512_slli_epi64(add(result_exponent, constant(127)), 23),
                                        band(rounded, 0x7fffff));
      V gradual = round_even<21>(product);
      if (!(denorm & 2))
        gradual =
            select(_mm512_cmplt_epu64_mask(gradual, constant(0x800000)), constant(0), gradual);
      result =
          select(small, _mm512_or_si512(select(subnormal, gradual, packed_result), sign), result);
    }
  }
  __mmask8 eligible =
      _mm512_cmpge_epu64_mask(magnitude, constant(Cosine ? 0x00800000 : 0x39c00000)) &
      _mm512_cmplt_epu64_mask(magnitude, constant(0x4b000000));
  if (eligible)
    result = select(eligible, ordinary_trig(bits, Cosine, eligible), result);
  result = select(_mm512_cmpeq_epi64_mask(magnitude, constant(0x7f800000)), constant(0xffc00000),
                  result);
  result = select(_mm512_cmpgt_epu64_mask(magnitude, constant(0x7f800000)),
                  _mm512_or_si512(bits, constant(quiet_snan ? 0x00400000 : 0)), result);
  __m256i narrowed = _mm512_cvtepi64_epi32(result);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(output), narrowed);
}

[[gnu::target("avx512f,avx512dq,avx512cd,avx2"), gnu::noinline]] void
sqrt_batch(const uint32_t *input, uint32_t *output, bool quiet_snan) {
  V bits = _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(input)));
  V magnitude = band(bits, 0x7fffffff), sign = band(bits, 0x80000000);
  V exponent = _mm512_srli_epi64(magnitude, 23);
  V parity = band(_mm512_xor_si512(exponent, constant(1)), 1);
  V index = _mm512_or_si512(_mm512_slli_epi64(parity, 23), band(bits, 0x7fffff));
  V segment = _mm512_srli_epi64(index, 19), fraction = band(index, 0x7ffff);
  V product = mul(table_coefficient(segment, sqrt_columns[1].data()), fraction);
  V bias = select(_mm512_cmpge_epu64_mask(product, constant(uint64_t{1} << 40)), constant(1u << 16),
                  constant(1u << 15));
  V linear = _mm512_srli_epi64(add(product, bias), 17);
  V square_input = band(fraction, ~uint64_t{3});
  V square = _mm512_srli_epi64(mul(square_input, square_input), 14);
  V inner =
      add(_mm512_slli_epi64(table_coefficient(segment, sqrt_columns[2].data()), 8),
          _mm512_srai_epi64(
              sub(constant(256), mul(table_coefficient(segment, sqrt_columns[3].data()), fraction)),
              10));
  V quadratic = mul(inner, square);
  V negated =
      select(_mm512_cmpge_epu64_mask(quadratic, constant(uint64_t{1} << 47)),
             _mm512_slli_epi64(_mm512_srai_epi64(sub(constant(1u << 23), quadratic), 24), 1),
             _mm512_srai_epi64(sub(constant(1u << 22), quadratic), 23));
  V sum = add(_mm512_slli_epi64(table_coefficient(segment, sqrt_columns[0].data()), 22),
              add(_mm512_slli_epi64(linear, 19), _mm512_slli_epi64(negated, 12)));
  V result = add(constant(0x3f800000), round_even<24>(sum));
  V adjustment = _mm512_srai_epi64(sub(sub(exponent, constant(127)), parity), 1);
  result = add(result, _mm512_slli_epi64(adjustment, 23));
  result = select(_mm512_cmpneq_epi64_mask(sign, constant(0)), constant(0xffc00000), result);
  result = select(_mm512_cmplt_epu64_mask(magnitude, constant(0x00800000)), sign, result);
  result = select(_mm512_cmpeq_epi64_mask(bits, constant(0x7f800000)), bits, result);
  result = select(_mm512_cmpgt_epu64_mask(magnitude, constant(0x7f800000)),
                  _mm512_or_si512(bits, constant(quiet_snan ? 0x00400000 : 0)), result);

  V remainder = band(sum, (1u << 24) - 1);
  unsigned corrections = _mm512_cmpge_epu64_mask(remainder, constant((1u << 23) - 8192)) &
                         _mm512_cmple_epu64_mask(remainder, constant((1u << 23) + 8192)) &
                         _mm512_cmpge_epu64_mask(bits, constant(0x00800000)) &
                         _mm512_cmplt_epu64_mask(bits, constant(0x7f800000));
  // Capture indices before storing: input and output can alias.
  uint32_t indices[8];
  if (corrections)
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(indices), _mm512_cvtepi64_epi32(index));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(output), _mm512_cvtepi64_epi32(result));
  while (corrections) {
    const unsigned lane = std::countr_zero(corrections);
    corrections &= corrections - 1;
    output[lane] = util::detail::amdgpu_sqrt_correct(indices[lane], output[lane]);
  }
}

#endif
uint32_t scalar(F32Operation operation, uint32_t bits, unsigned denorm, bool quiet) {
  switch (operation) {
  case F32Operation::Log:
    return util::detail::log::evaluate(bits, quiet);
  case F32Operation::Exp:
    return util::detail::exp::evaluate(bits, quiet);
  case F32Operation::Sin:
    return util::detail::trig::evaluate(bits, false, denorm, quiet);
  case F32Operation::Cos:
    return util::detail::trig::evaluate(bits, true, denorm, quiet);
  case F32Operation::Rcp:
    return std::bit_cast<uint32_t>(util::amdgpu_rcp_f32(std::bit_cast<float>(bits), quiet));
  case F32Operation::Rsq:
    return std::bit_cast<uint32_t>(util::amdgpu_rsq_f32(std::bit_cast<float>(bits), quiet));
  case F32Operation::Sqrt:
    return util::detail::amdgpu_sqrt_bits(bits, quiet);
  }
  return 0;
}
} // namespace

bool supports_f32_simd() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
         __builtin_cpu_supports("avx512cd") && __builtin_cpu_supports("avx2");
#else
  return false;
#endif
}

void evaluate_f32_simd(F32Operation operation, const uint32_t *input, uint32_t *output,
                       std::size_t count, unsigned denorm, bool quiet_snan) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if (supports_f32_simd() && count % 8 == 0) {
    switch (operation) {
    case F32Operation::Log:
      for (std::size_t i = 0; i < count; i += 8)
        batch<true>(input + i, output + i, quiet_snan);
      break;
    case F32Operation::Exp:
      for (std::size_t i = 0; i < count; i += 8)
        batch<false>(input + i, output + i, quiet_snan);
      break;
    case F32Operation::Sin:
      for (std::size_t i = 0; i < count; i += 8)
        trig_batch<false>(input + i, output + i, denorm, quiet_snan);
      break;
    case F32Operation::Cos:
      for (std::size_t i = 0; i < count; i += 8)
        trig_batch<true>(input + i, output + i, denorm, quiet_snan);
      break;
    case F32Operation::Sqrt:
      for (std::size_t i = 0; i < count; i += 8)
        sqrt_batch(input + i, output + i, quiet_snan);
      break;
    case F32Operation::Rcp:
    case F32Operation::Rsq:
      detail::evaluate_reciprocal_f32_simd(operation == F32Operation::Rsq, input, output, count,
                                           quiet_snan);
      break;
    }
    return;
  }
#endif
  for (std::size_t i = 0; i < count; ++i)
    output[i] = scalar(operation, input[i], denorm, quiet_snan);
}
} // namespace rocjitsu::amdgpu::transcendental
