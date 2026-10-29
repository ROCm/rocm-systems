// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86 SIMD vector implementation of empirical DOT alignment, accumulation, and result packing.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "common/bits.h"
#include "dot/scalar_repair.h"
#include "dot/x86_64/exact_dot_simd.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::exact_dot {

// Independent SIMD lanes of the inherited integer DOT model. Products have
// at most 22 significant bits, aligned terms at most 28; sums fit signed i32.
// Exponents remain signed i32. This is vectorization, not a new FP model.
template <typename V> struct Term {
  V magnitude, exponent, negative;
};

// Return a per-lane mask indicating nonzero integer values.
template <typename V> inline V nonzero(V value) {
  return bit_xor(equal(value, constant<V>(0)), constant<V>(-1));
}

// Only called on factor products and C significands, each <= 24 bits. Their
// integer-to-float conversion is exact, including at powers of two.
template <typename V> inline V leading(Term<V> value) {
  auto top = sub(float_exponent(value.magnitude), constant<V>(127));
  return choose(nonzero(value.magnitude), add(value.exponent, top), constant<V>(-1024));
}

// Align a product magnitude to the supplied accumulation grid; sign remains separate.
template <typename V> inline V shifted(Term<V> value, V grid) {
  auto shift = sub(value.exponent, grid);
  return bit_or(shift_left(value.magnitude, shift),
                shift_right(value.magnitude, sub(constant<V>(0), shift)));
}

// GFX12 floors negative terms, except an accumulator whose entire magnitude
// was discarded. Variable shifts >=32 yield zero; magnitudes here fit in 28 bits.
template <typename V> inline V align(Term<V> value, V grid, V accumulator = constant<V>(0)) {
  auto shift = sub(grid, value.exponent);
  auto magnitude = shifted(value, grid);
  auto restored = shift_left(magnitude, shift);
  auto tail = bit_and(greater(shift, constant<V>(0)), nonzero(bit_xor(value.magnitude, restored)));
  tail = andnot(andnot(nonzero(magnitude), accumulator), tail);
  auto negative = sub(sub(constant<V>(0), magnitude), bit_and(tail, constant<V>(1)));
  return choose(value.negative, negative, magnitude);
}

// Form vector product terms, update the accumulation grid, and flag exceptional inputs in bad.
template <bool Bf16, bool Rdna3, typename V> inline Term<V> product(V a, V b, V &grid, V &bad) {
  constexpr int fraction = Bf16 ? 7 : 10, bias = Bf16 ? 127 : 15;
  constexpr int infinity = Bf16 ? 0x7f80 : 0x7c00;
  auto am = bit_and(a, constant<V>(0x7fff));
  auto bm = bit_and(b, constant<V>(0x7fff));
  bad = bit_or(
      bad, bit_or(greater(am, constant<V>(infinity - 1)), greater(bm, constant<V>(infinity - 1))));
  auto ae = shift_right(am, fraction), be = shift_right(bm, fraction);
  // Restore the implicit bit for normals and apply the format's subnormal policy.
  auto significand = [](V m, V e) {
    auto fraction_bits = bit_and(m, constant<V>((1 << fraction) - 1));
    auto normal = bit_or(fraction_bits, constant<V>(1 << fraction));
    return choose(nonzero(e), normal, Bf16 && Rdna3 ? constant<V>(0) : fraction_bits);
  };
  auto magnitude = mul(significand(am, ae), significand(bm, be));
  auto exponent =
      sub(add(maximum(ae, constant<V>(1)), maximum(be, constant<V>(1))), constant<V>(2 * bias));
  grid = choose(nonzero(magnitude), sub(exponent, constant<V>(24)), constant<V>(-1024));
  auto negative = sub(constant<V>(0), shift_right(bit_xor(a, b), 15));
  return {magnitude, sub(exponent, constant<V>(2 * fraction)), negative};
}

// Round the inherited signed-integer sum onto the selected output grid.
// Mirrors the scalar packers: gfx11 rounds before flushing; gfx12 rounds
// directly on the subnormal grid.
template <bool Bf16, bool Rdna3, typename V> inline V pack(V total, V grid, V frame_negative) {
  constexpr int Fraction = 23, Bias = 127;
  auto magnitude = absolute(total);
  auto top = highest_bit(magnitude);
  auto exponent = add(top, grid);
  auto tiny = greater(constant<V>(1 - Bias), exponent);
  auto shift = sub(top, constant<V>(Fraction));
  if constexpr (!Rdna3) {
    shift = choose(tiny, sub(constant<V>(1 - Bias - Fraction), grid), shift);
  }
  auto rounded =
      bit_or(shift_right(magnitude, shift), shift_left(magnitude, sub(constant<V>(0), shift)));
  auto tail = bit_and(magnitude, sub(shift_left(constant<V>(1), shift), constant<V>(1)));
  auto halfway = shift_left(constant<V>(1), sub(shift, constant<V>(1)));
  auto increment = bit_or(greater(tail, halfway),
                          bit_and(equal(tail, halfway), nonzero(bit_and(rounded, constant<V>(1)))));
  increment = bit_and(increment, greater(shift, constant<V>(0)));
  rounded = add(rounded, bit_and(increment, constant<V>(1)));

  // Sums fit in 29 magnitude bits, so shifts >=32 always round to zero.
  rounded = andnot(greater(shift, constant<V>(31)), rounded);
  auto carry = equal(rounded, constant<V>(1 << (Fraction + 1)));
  auto significand = choose(carry, shift_right(rounded, 1), rounded);
  exponent = add(exponent, bit_and(carry, constant<V>(1)));
  auto bits = bit_or(shift_left(add(exponent, constant<V>(Bias)), Fraction),
                     bit_and(significand, constant<V>((1 << Fraction) - 1)));
  auto overflow = constant<V>(0x7f800000);
  bits = choose(greater(exponent, constant<V>(Bias)), overflow, bits);
  if constexpr (!Rdna3) {
    bits = choose(tiny, rounded, bits);
  } else {
    bits = andnot(greater(constant<V>(-126), exponent), bits);
  }
  auto sign = signed_shift_right(total, 31);
  if constexpr (Rdna3) {
    sign = bit_xor(sign, frame_negative);
  }
  bits = bit_or(bits, bit_and(sign, constant<V>(int(0x80000000U))));

  // These models return positive zero on exact zero and on flushed underflow.
  bits = bit_and(bits, bit_and(nonzero(total), nonzero(bit_and(bits, constant<V>(0x7fffffff)))));

  return bits;
}

// Return the empirical vector DOT result for the selected architecture with FP32 output.
template <bool Bf16, bool Rdna3, typename V, int N>
inline V dot(const V (&a)[N], const V (&b)[N], V acc) {
  static_assert(Rdna3 ? N == 2 : (N == 2 || N == 4));
  constexpr int Columns = sizeof(V) / sizeof(int32_t);
  Term<V> products[N];
  V grids[N], product_grid = constant<V>(-1024);
  auto bad = greater(bit_and(acc, constant<V>(0x7fffffff)), constant<V>(0x7f7fffff));
  for (int i = 0; i < N; ++i) {
    products[i] = product<Bf16, Rdna3>(a[i], b[i], grids[i], bad);
    product_grid = maximum(product_grid, grids[i]);
  }
  auto ce = bit_and(shift_right(acc, 23), constant<V>(255));
  auto cf = bit_and(acc, constant<V>(0x7fffff));
  Term<V> c{choose(nonzero(ce), bit_or(cf, constant<V>(0x800000)), Rdna3 ? constant<V>(0) : cf),
            sub(maximum(ce, constant<V>(Rdna3 ? 0 : 1)), constant<V>(150)),
            signed_shift_right(acc, 31)};
  auto top = leading(c);
  if constexpr (Rdna3) {

    for (auto term : products) {
      top = maximum(top, leading(term));
    }
  } else {
    top = maximum(top, constant<V>(-126));
  }
  auto grid = maximum(product_grid, sub(top, constant<V>(26)));
  V total;
  if constexpr (Rdna3) {
    total = shifted(c, grid);
    auto any = c.magnitude;
    for (auto term : products) {
      auto magnitude = shifted(term, grid);
      auto opposite = bit_xor(term.negative, c.negative);
      total = add(total, bit_xor(magnitude, opposite));
      any = bit_or(any, term.magnitude);
    }
    total = bit_and(total, nonzero(any));
  } else {
    auto accumulator = greater(grid, constant<V>(-126));
    total = align(c, grid, accumulator);
    for (int i = 0; i < N; i += 2) {
      auto group_grid = maximum(grids[i], grids[i + 1]);
      auto group = add(align(products[i], group_grid), align(products[i + 1], group_grid));
      Term<V> sum{absolute(group), group_grid, signed_shift_right(group, 31)};
      total = add(total, align(sum, grid));
    }
  }
  auto result = pack<Bf16, Rdna3>(total, grid, c.negative);

  // Only exceptional columns use the scalar model; a NaN in one column must
  // not send the remaining columns (or the entire matrix) off SIMD.
  int exceptional = lane_mask(bad);
  if (exceptional) {
    alignas(64) uint32_t left[N][Columns], right[N][Columns], accumulators[Columns],
        values[Columns];
    store(values, result);
    store(accumulators, acc);
    for (int i = 0; i < N; ++i) {
      store(left[i], a[i]);
      store(right[i], b[i]);
    }
    for (int lane = 0; lane < Columns; ++lane) {
      if (!(exceptional & (1 << lane))) {
        continue;
      }
      uint16_t av[N], bv[N];
      for (int i = 0; i < N; ++i) {
        av[i] = uint16_t(left[i][lane]);
        bv[i] = uint16_t(right[i][lane]);
      }
      values[lane] = repair_wmma_dot(av, bv, accumulators[lane], Bf16, Rdna3);
    }
    result = load<V>(values);
  }
  return result;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::exact_dot
