// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/amdgpu_exp.h"
#include "util/amdgpu_log.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rocjitsu::amdgpu::fp_math::detail {

template <typename Coefficient, std::size_t N>
constexpr auto coefficient_columns(const Coefficient (&rows)[N]) {
  std::array<std::array<std::uint32_t, N>, 4> result{};
  for (std::size_t i = 0; i < N; ++i) {
    result[0][i] = rows[i].constant;
    result[1][i] = rows[i].linear;
    result[2][i] = rows[i].quadratic;
    result[3][i] = rows[i].cubic;
  }
  return result;
}

inline constexpr auto exp_columns = coefficient_columns(util::detail::exp::coefficients);
inline constexpr auto log_columns = coefficient_columns(util::detail::log::coefficients);

// Ops supplies packed 64-bit integer arithmetic. No host floating-point
// operation or per-lane libm call participates in either mapping.
template <typename Ops> struct ExpLog {
  using V = typename Ops::V;
  using M = typename Ops::M;

  static V c(std::uint64_t value) { return Ops::set(value); }

  static V round_even(V value, unsigned shift) {
    const V whole = Ops::shr(value, shift);
    const V rest = Ops::band(value, Ops::sub(Ops::shl(c(1), shift), c(1)));
    const V half = Ops::shl(c(1), shift - 1);
    const M increment =
        Ops::mask_or(Ops::gt_u(rest, half),
                     Ops::mask_and(Ops::eq(rest, half), Ops::ne(Ops::band(whole, c(1)), c(0))));
    return Ops::add(whole, Ops::select(increment, c(1), c(0)));
  }

  static V exp_normal(V bits) {
    const V mag = Ops::band(bits, c(0x7fffffff));
    const V exponent = Ops::sub(Ops::shr(mag, 23), c(127));
    const V mantissa = Ops::bit_or(Ops::band(mag, c(0x7fffff)), c(0x800000));
    const V shift = Ops::add(exponent, c(6));
    const M left = Ops::ge_s(shift, c(0));
    const V absolute =
        Ops::select(left, Ops::shlv(mantissa, shift), Ops::shrv(mantissa, Ops::sub(c(0), shift)));
    const M negative = Ops::ne(Ops::band(bits, c(0x80000000)), c(0));
    const V phase = Ops::select(negative, Ops::bit_xor(absolute, c(~std::uint64_t{0})), absolute);
    const V fraction = Ops::band(phase, c(0xffffff));
    const V index = Ops::band(Ops::shr(phase, 24), c(31));
    const V c0 = Ops::template coefficient<false, 0>(index);
    const V c1 = Ops::template coefficient<false, 1>(index);
    const V c2 = Ops::template coefficient<false, 2>(index);
    const V c3 = Ops::template coefficient<false, 3>(index);
    const V lp = Ops::mul(c1, fraction);
    const V linear = Ops::select(Ops::ge_u(lp, c(std::uint64_t{1} << 47)), round_even(lp, 24),
                                 Ops::shr(round_even(lp, 23), 1));
    const V coordinate = Ops::shr(fraction, 6);
    const V square = Ops::shr(Ops::mul(coordinate, coordinate), 12);
    const V inner = round_even(Ops::add(Ops::shl(c2, 7), Ops::mul(c3, coordinate)), 7);
    const V product = Ops::mul(inner, square);
    const V quad = Ops::select(Ops::ge_u(product, c(std::uint64_t{1} << 47)),
                               Ops::shl(round_even(product, 24), 1), round_even(product, 23));
    const V sum = Ops::add(Ops::shl(Ops::add(c0, linear), 8), quad);
    return Ops::add(Ops::shl(Ops::add(Ops::sar(phase, 29), c(126)), 23), round_even(sum, 13));
  }

  static V log_near_one_negative(V bits) {
    const V k = Ops::sub(c(0x3f800000), bits);
    const M upper = Ops::ge_u(k, c(0x20000));
    const V c1 = Ops::select(upper, c(12102221), c(12102203));
    const V c2 = Ops::select(upper, c(12097323), c(12101979));
    const V c3 = Ops::select(upper, c(2088428), c(2035248));
    const V width = Ops::sub(c(64), Ops::lzcnt(k));
    const V normalization =
        Ops::band(Ops::sub(c(18), Ops::min_u(width, c(18))), c(~std::uint64_t{3}));
    const V linear_shift = Ops::sub(c(14), Ops::min_u(normalization, c(14)));
    const V square_shift = Ops::sub(c(12), Ops::min_u(normalization, c(12)));
    const V linear_product = Ops::mul(Ops::shl(c1, 1), k);
    const V linear = Ops::shlv(Ops::shrv(linear_product, linear_shift), linear_shift);
    const V square_product = Ops::mul(k, k);
    const V square = Ops::shlv(Ops::shrv(square_product, square_shift), square_shift);
    const V inner = round_even(Ops::add(Ops::shl(c2, 22), Ops::mul(c3, Ops::sub(k, c(1)))), 22);
    const V product_shift = Ops::sub(c(38), normalization);
    const V quadratic =
        Ops::shrv(Ops::add(Ops::mul(inner, square), Ops::sub(Ops::shlv(c(1), product_shift), c(1))),
                  product_shift);
    const V fixed = Ops::sub(
        Ops::add(Ops::shl(linear, 2), Ops::shlv(quadratic, Ops::sub(c(16), normalization))),
        Ops::select(upper, c(std::uint64_t{100} << 16), c(0)));
    V leading = Ops::sub(c(63), Ops::lzcnt(fixed));
    const V shift = Ops::sub(leading, c(23));
    V mantissa = Ops::shrv(Ops::add(fixed, Ops::sub(Ops::shlv(c(1), shift), c(1))), shift);
    const M carry = Ops::eq(mantissa, c(std::uint64_t{1} << 24));
    mantissa = Ops::select(carry, Ops::shr(mantissa, 1), mantissa);
    leading = Ops::add(leading, Ops::select(carry, c(1), c(0)));
    return Ops::bit_or(c(0x80000000), Ops::bit_or(Ops::shl(Ops::add(leading, c(77)), 23),
                                                  Ops::band(mantissa, c(0x7fffff))));
  }

  static V log_near_one_positive(V bits) {
    const V i = Ops::sub(bits, c(0x3f800000));
    const M middle = Ops::ge_u(i, c(0x10000));
    const M upper = Ops::ge_u(i, c(0x20000));
    const V c0 = Ops::select(upper, c(2856), Ops::select(middle, c(192), c(0)));
    const V c1 = Ops::select(upper, c(12102072), Ops::select(middle, c(12102186), c(12102203)));
    const V c2 = Ops::select(upper, c(12084310), Ops::select(middle, c(12097614), c(12102094)));
    const V c3 = Ops::select(upper, c(1883467), Ops::select(middle, c(1948887), c(1999052)));
    const V width = Ops::sub(c(64), Ops::lzcnt(i));
    const V normalization =
        Ops::select(middle, c(0), Ops::band(Ops::sub(c(18), width), c(~std::uint64_t{3})));
    const V normalized = Ops::shlv(i, normalization);
    const V linear = Ops::shl(Ops::shr(Ops::mul(c1, normalized), 13), 2);
    const V square = Ops::shr(Ops::mul(normalized, i), 12);
    const V inner =
        round_even(Ops::sub(Ops::shl(c2, 22), Ops::mul(c3, Ops::add(Ops::shl(i, 1), c(1)))), 22);
    const V product = Ops::mul(inner, square);
    const V quadratic = Ops::select(Ops::ge_u(product, c(std::uint64_t{1} << 47)),
                                    Ops::shl(round_even(product, 24), 1), round_even(product, 23));
    const V fixed = Ops::sub(Ops::add(c0, linear), quadratic);
    V leading = Ops::sub(c(63), Ops::lzcnt(fixed));
    V mantissa = Ops::add(Ops::shrv(fixed, Ops::sub(leading, c(23))), c(1));
    const M carry = Ops::eq(mantissa, c(std::uint64_t{1} << 24));
    mantissa = Ops::select(carry, Ops::shr(mantissa, 1), mantissa);
    leading = Ops::add(leading, Ops::select(carry, c(1), c(0)));
    return Ops::bit_or(Ops::shl(Ops::sub(Ops::add(leading, c(92)), normalization), 23),
                       Ops::band(mantissa, c(0x7fffff)));
  }

  static V log_ordinary(V bits) {
    const V fraction = Ops::band(bits, c(0x3ffff));
    V index = Ops::band(Ops::shr(bits, 18), c(31));
    index = Ops::select(Ops::mask_and(Ops::eq(index, c(1)), Ops::ge_u(fraction, c(0x20000))), c(32),
                        index);
    const V c0 = Ops::template coefficient<true, 0>(index);
    const V c1 = Ops::template coefficient<true, 1>(index);
    const V c2 = Ops::template coefficient<true, 2>(index);
    const V c3 = Ops::template coefficient<true, 3>(index);
    const V linear = Ops::shr(Ops::mul(c1, fraction), 16);
    const V inner = round_even(
        Ops::sub(Ops::shl(c2, 22), Ops::mul(c3, Ops::add(Ops::shl(fraction, 1), c(1)))), 22);
    const V square = Ops::shr(Ops::mul(fraction, fraction), 12);
    const V product = Ops::mul(inner, square);
    const V quad = Ops::select(Ops::ge_u(product, c(std::uint64_t{1} << 47)),
                               Ops::shl(round_even(product, 24), 1), round_even(product, 23));
    const V exponent = Ops::sub(Ops::shr(bits, 23), c(127));
    const V fixed =
        Ops::add(Ops::shl(exponent, 35), Ops::sub(Ops::shl(Ops::add(c0, linear), 5), quad));
    const M negative = Ops::lt_s(fixed, c(0));
    const V magnitude = Ops::select(negative, Ops::sub(c(0), fixed), fixed);
    V leading = Ops::sub(c(63), Ops::lzcnt(magnitude));
    const V shift = Ops::sub(leading, c(23));
    const V bias = Ops::sub(Ops::shlv(c(1), shift), c(1));
    V mantissa = Ops::shrv(Ops::add(magnitude, Ops::select(negative, bias, c(0))), shift);
    const M carry = Ops::eq(mantissa, c(std::uint64_t{1} << 24));
    mantissa = Ops::select(carry, Ops::shr(mantissa, 1), mantissa);
    leading = Ops::add(leading, Ops::select(carry, c(1), c(0)));
    return Ops::bit_or(
        Ops::select(negative, c(0x80000000), c(0)),
        Ops::bit_or(Ops::shl(Ops::add(leading, c(92)), 23), Ops::band(mantissa, c(0x7fffff))));
  }

  template <bool Logarithm> static V batch(V bits, bool quiet_snan) {
    const V magnitude = Ops::band(bits, c(0x7fffffff));
    const M negative = Ops::ne(Ops::band(bits, c(0x80000000)), c(0));
    M eligible;
    V result;
    if constexpr (Logarithm) {
      eligible = Ops::mask_and(
          Ops::mask_and(Ops::ge_u(bits, c(0x00800000)), Ops::lt_u(bits, c(0x7f800000))),
          Ops::mask_or(Ops::lt_u(bits, c(0x3f7c0000)), Ops::ge_u(bits, c(0x3f840000))));
      result = Ops::select(negative, c(0xffc00000), c(0x7f800000));
      result = Ops::select(Ops::lt_u(magnitude, c(0x00800000)), c(0xff800000), result);
      result = Ops::select(Ops::eq(bits, c(0x3f800000)), c(0), result);
    } else {
      eligible =
          Ops::mask_and(Ops::ge_u(magnitude, c(0x33800000)),
                        Ops::mask_or(Ops::mask_and(negative, Ops::le_u(magnitude, c(0x42fc0000))),
                                     Ops::mask_and(Ops::mask_not(negative),
                                                   Ops::lt_u(magnitude, c(0x43000000)))));
      result = c(0x3f800000);
      result =
          Ops::select(Ops::mask_and(negative, Ops::gt_u(magnitude, c(0x42fc0000))), c(0), result);
      result =
          Ops::select(Ops::mask_and(Ops::mask_not(negative), Ops::ge_u(magnitude, c(0x43000000))),
                      c(0x7f800000), result);
    }
    result = Ops::select(Ops::gt_u(magnitude, c(0x7f800000)),
                         Ops::bit_or(bits, c(quiet_snan ? 0x00400000 : 0)), result);
    if (Ops::mask_any(eligible)) {
      const V safe = Ops::select(eligible, bits, c(Logarithm ? 0x40000000 : 0x3f000000));
      result = Ops::select(eligible, Logarithm ? log_ordinary(safe) : exp_normal(safe), result);
    }
    if constexpr (Logarithm) {
      const M below = Ops::mask_and(Ops::ge_u(bits, c(0x3f7c0000)), Ops::lt_u(bits, c(0x3f800000)));
      const M above = Ops::mask_and(Ops::gt_u(bits, c(0x3f800000)), Ops::lt_u(bits, c(0x3f840000)));
      if (Ops::mask_any(below))
        result = Ops::select(below, log_near_one_negative(Ops::select(below, bits, c(0x3f7fffff))),
                             result);
      if (Ops::mask_any(above))
        result = Ops::select(above, log_near_one_positive(Ops::select(above, bits, c(0x3f800001))),
                             result);
    }
    return result;
  }
};

} // namespace rocjitsu::amdgpu::fp_math::detail
