// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_dot.h"
#include "util/simd.h"

// Vectorize independent F32 outputs, preserving each architecture's reduction boundaries.
// No floating-point operations, host-specific intrinsics, or scalar lane fallback.
#if __has_include(<experimental/simd>)
namespace rocjitsu::amdgpu::rdna_dot_simd {
namespace sx = util::stdx;
using U = util::native<uint32_t>;
using I = util::native<int32_t>;
using M = U::mask_type;
inline constexpr size_t width = U::size();

inline U u(I v) { return sx::static_simd_cast<U>(v); }
inline I i(U v) { return sx::static_simd_cast<I>(v); }
inline M m(I::mask_type v) {
  I bits(0);
  sx::where(v, bits) = I(-1);
  return u(bits) != U(0);
}
inline U choose(M mask, U yes, U no) {
  sx::where(mask, no) = yes;
  return no;
}
inline I choose(M mask, I yes, I no) { return i(choose(mask, u(yes), u(no))); }

// Defined for zero (returns zero); callers handle their own zero sentinel.
inline I top_bit(U x) {
  U count(0);
  for (unsigned shift : {16u, 8u, 4u, 2u, 1u}) {
    const M high = x >= U(uint32_t{1} << shift);
    x = choose(high, x >> shift, x);
    count += choose(high, U(shift), U(0));
  }
  return i(count);
}

// Every shift operand is clamped even in lanes later discarded by a mask.
template <bool Left> inline U shift_magnitude(U x, I shift) {
  const U count = u(sx::min(sx::max(shift, I(0)), I(31)));
  U result = x;
  if constexpr (width >= 8) {
    if constexpr (Left)
      result = x << count;
    else
      result = x >> count;
  } else {
    // Some four-lane stdx implementations emulate variable integer shifts with
    // floating-point conversions. Use fixed shifts to preserve FP state and
    // avoid an out-of-range float-to-int conversion for a shift of 31.
    for (unsigned bit : {1u, 2u, 4u, 8u, 16u}) {
      const U shifted = Left ? result << bit : result >> bit;
      result = choose((count & U(bit)) != U(0), shifted, result);
    }
  }
  return choose(m(shift >= I(32)), U(0), result);
}
inline U left(U x, I shift) { return shift_magnitude<true>(x, shift); }
inline U right(U x, I shift) { return shift_magnitude<false>(x, shift); }
inline U round_even(U x, I shift) {
  const U truncated = right(x, shift);
  // x < 2^29, so any right shift >= 32 rounds to zero.
  const U mask = left(U(1), shift) - U(1);
  const U tail = x & mask;
  const U half = left(U(1), shift - I(1));
  const M up = (tail > half) || ((tail == half) && ((truncated & U(1)) != U(0)));
  const U rounded = truncated + choose(up, U(1), U(0));
  return choose(m(shift <= I(0)), left(x, -shift), choose(m(shift >= I(32)), U(0), rounded));
}

struct Term {
  U significand;
  I exponent;
  M negative;
};

inline I align(Term term, I grid, bool accumulator = false) {
  const I shift = term.exponent - grid;
  const U magnitude =
      choose(m(shift < I(0)), right(term.significand, -shift), left(term.significand, shift));
  const U tail_mask = left(U(1), -shift) - U(1);
  const M tail = (term.significand != U(0)) && m(shift < I(0)) &&
                 (m(shift <= I(-32)) || ((term.significand & tail_mask) != U(0)));
  const M floor = tail && (M(!accumulator) || (magnitude != U(0)));
  return choose(term.negative, -i(magnitude) - choose(floor, I(1), I(0)), i(magnitude));
}

inline U pack(I units, I grid) {
  const M negative = m(units < I(0));
  const U sign = choose(negative, U(0x80000000u), U(0));
  const U magnitude = u(choose(negative, -units, units));
  const I top = top_bit(magnitude);
  I exponent = top + grid;
  U significand = round_even(magnitude, top - I(23));
  const M carry = significand == U(0x1000000);
  significand = choose(carry, significand >> 1, significand);
  exponent += choose(carry, I(1), I(0));
  U result = sign | (u(exponent + I(127)) << 23) | (significand & U(0x7fffff));
  result = choose(m(exponent > I(127)), sign | U(0x7f800000), result);
  const U subnormal = round_even(magnitude, I(-149) - grid);
  result =
      choose(m(top + grid < I(-126)), choose(subnormal != U(0), sign | subnormal, U(0)), result);
  return choose(magnitude == U(0), U(0), result);
}

template <bool Bf16> inline U dot4(const std::array<U, 4> &a, const std::array<U, 4> &b, U acc) {
  constexpr int fraction_bits = Bf16 ? 7 : 10;
  constexpr int bias = Bf16 ? 127 : 15;
  constexpr uint32_t infinity = Bf16 ? 0x7f80 : 0x7c00;
  constexpr uint32_t fraction_mask = (1u << fraction_bits) - 1;
  std::array<Term, 4> products;
  std::array<I, 4> grids;
  I product_grid(-1024);
  M factor_nan(false), invalid(false), positive_inf(false), negative_inf(false);
  for (size_t j = 0; j != 4; ++j) {
    const U aa = a[j] & U(0x7fff), bb = b[j] & U(0x7fff);
    factor_nan |= (aa > U(infinity)) || (bb > U(infinity));
    const M ai = aa == U(infinity), bi = bb == U(infinity);
    const U ae = aa >> fraction_bits, be = bb >> fraction_bits;
    const U am = (aa & U(fraction_mask)) | choose(ae != U(0), U(1u << fraction_bits), U(0));
    const U bm = (bb & U(fraction_mask)) | choose(be != U(0), U(1u << fraction_bits), U(0));
    const M negative = ((a[j] ^ b[j]) & U(0x8000)) != U(0);
    invalid |= (ai && (bm == U(0))) || (bi && (am == U(0)));
    negative_inf |= (ai || bi) && negative;
    positive_inf |= (ai || bi) && !negative;
    const I e = i(sx::max(ae, U(1)) + sx::max(be, U(1))) - I(2 * bias);
    products[j] = {am * bm, e - I(2 * fraction_bits), negative};
    grids[j] = choose(products[j].significand != U(0), e - I(24), I(-1024));
    product_grid = sx::max(product_grid, grids[j]);
  }
  invalid |= positive_inf && negative_inf;
  const U acc_exp = (acc >> 23) & U(255);
  const Term c{(acc & U(0x7fffff)) | choose(acc_exp != U(0), U(0x800000), U(0)),
               i(sx::max(acc_exp, U(1))) - I(150), (acc >> 31) != U(0)};
  const I leading = choose(c.significand != U(0), c.exponent + top_bit(c.significand), I(-1024));
  const I grid = sx::max(product_grid, sx::max(I(-126), leading) - I(26));
  I total = align(c, grid, true);
  for (size_t j = 0; j != 4; j += 2) {
    const I pair_grid = sx::max(grids[j], grids[j + 1]);
    const I pair = align(products[j], pair_grid) + align(products[j + 1], pair_grid);
    const M negative = m(pair < I(0));
    total += align(Term{u(choose(negative, -pair, pair)), pair_grid, negative}, grid);
  }
  U result = pack(total, grid);
  positive_inf |= acc == U(0x7f800000);
  negative_inf |= acc == U(0xff800000);
  result = choose(positive_inf || negative_inf,
                  choose(positive_inf && negative_inf, U(0xffc00000),
                         choose(negative_inf, U(0xff800000), U(0x7f800000))),
                  result);
  result = choose((acc & U(0x7fffffff)) > U(0x7f800000), acc | U(0x400000), result);
  result = choose(invalid, U(0xffc00000), result);
  return choose(factor_nan, U(0xffc00a3d), result);
}

// GFX11 uses C's sign frame and complements opposing magnitudes after
// truncation, including signed zeros. BF16 inputs and F32 C flush subnormals.
template <bool Bf16> inline U dot2(const std::array<U, 2> &a, const std::array<U, 2> &b, U acc) {
  constexpr int bits = Bf16 ? 7 : 10, bias = Bf16 ? 127 : 15;
  constexpr uint32_t infinity = Bf16 ? 0x7f80 : 0x7c00;
  std::array<Term, 2> products;
  I product_grid(-1048), leading(-1024);
  M factor_nan(false), invalid(false), positive_inf(false), negative_inf(false);
  for (size_t j = 0; j != 2; ++j) {
    const U aa = a[j] & U(0x7fff), bb = b[j] & U(0x7fff);
    factor_nan |= (aa > U(infinity)) || (bb > U(infinity));
    const U ae = aa >> bits, be = bb >> bits;
    U am = (aa & U((1u << bits) - 1)) | choose(ae != U(0), U(1u << bits), U(0));
    U bm = (bb & U((1u << bits) - 1)) | choose(be != U(0), U(1u << bits), U(0));
    if constexpr (Bf16) {
      am = choose(ae == U(0), U(0), am);
      bm = choose(be == U(0), U(0), bm);
    }
    const M ai = aa == U(infinity), bi = bb == U(infinity);
    const M negative = ((a[j] ^ b[j]) & U(0x8000)) != U(0);
    invalid |= (ai && (bm == U(0))) || (bi && (am == U(0)));
    negative_inf |= (ai || bi) && negative;
    positive_inf |= (ai || bi) && !negative;
    const I e = i(sx::max(ae, U(1)) + sx::max(be, U(1))) - I(2 * bias);
    products[j] = {am * bm, e - I(2 * bits), negative};
    const auto &p = products[j];
    product_grid = sx::max(product_grid, choose(p.significand != U(0), e - I(24), I(-1048)));
    leading = sx::max(leading,
                      choose(p.significand != U(0), p.exponent + top_bit(p.significand), I(-1024)));
  }
  invalid |= positive_inf && negative_inf;
  const U ce = (acc >> 23) & U(255);
  const Term c{choose(ce != U(0), (acc & U(0x7fffff)) | U(0x800000), U(0)), i(ce) - I(150),
               (acc >> 31) != U(0)};
  leading = sx::max(leading,
                    choose(c.significand != U(0), c.exponent + top_bit(c.significand), I(-1024)));
  const I grid = sx::max(product_grid, leading - I(26));
  I total(0);
  for (const Term &term : {c, products[0], products[1]}) {
    const I shift = term.exponent - grid;
    const U mag =
        choose(m(shift < I(0)), right(term.significand, -shift), left(term.significand, shift));
    total += choose(term.negative != c.negative, -i(mag) - I(1), i(mag));
  }
  // All-zero terms bypass the complement contributions in the scalar model.
  total = choose((c.significand | products[0].significand | products[1].significand) == U(0), I(0),
                 total);
  const M negative = m(total < I(0));
  const U mag = u(choose(negative, -total, total));
  const U sign = choose(negative != c.negative, U(0x80000000), U(0));
  const I top = top_bit(mag);
  I exponent = top + grid;
  U significand = round_even(mag, top - I(23));
  const M carry = significand == U(0x1000000);
  significand = choose(carry, significand >> 1, significand);
  exponent += choose(carry, I(1), I(0));
  U result = sign | (u(exponent + I(127)) << 23) | (significand & U(0x7fffff));
  result = choose(m(exponent > I(127)), sign | U(0x7f800000), result);
  result = choose(m(exponent < I(-126)) || (mag == U(0)), U(0), result);
  positive_inf |= acc == U(0x7f800000);
  negative_inf |= acc == U(0xff800000);
  result = choose(positive_inf || negative_inf,
                  choose(positive_inf && negative_inf, U(0xffc00000),
                         choose(negative_inf, U(0xff800000), U(0x7f800000))),
                  result);
  result = choose((acc & U(0x7fffffff)) > U(0x7f800000), acc | U(0x400000), result);
  result = choose(invalid, U(0xffc00000), result);
  return choose(factor_nan, U(0xffc00a3d), result);
}

// One row of independent outputs. A is K-major, B is [K][column].
template <bool Gfx12, bool Bf16>
inline void row(const uint16_t *a, const uint16_t b[16][16], uint32_t *acc) {
  constexpr size_t group = Gfx12 ? 4 : 2;
  for (size_t col = 0; col < 16; col += width) {
    U c;
    c.copy_from(acc + col, sx::element_aligned);
    for (size_t k = 0; k < 16; k += group) {
      std::array<U, group> av, bv;
      for (size_t j = 0; j != group; ++j) {
        av[j] = U(a[k + j]);
        bv[j] = U([&](auto lane) { return b[k + j][col + lane]; });
      }
      if constexpr (Gfx12)
        c = dot4<Bf16>(av, bv, c);
      else
        c = dot2<Bf16>(av, bv, c);
    }
    c.copy_to(acc + col, sx::element_aligned);
  }
}
} // namespace rocjitsu::amdgpu::rdna_dot_simd
#endif
