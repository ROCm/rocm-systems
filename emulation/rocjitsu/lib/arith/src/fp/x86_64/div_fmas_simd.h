// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86 SIMD multiword vector arithmetic for exact division FMA and conditional scaling.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

// Integer SIMD form of the rocjitsu-derived fused division model.

#pragma once

#include "common/bits.h"
#include "fp/float_output.h"
#include "rj_arith/rj_arith.h"

#include <initializer_list>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

template <class O> struct DivisionVector128 {
  using V = typename O::V;
  V lo, hi;
};

// Shift each two-word unsigned 128-bit value left by its per-lane count.
template <class O>
inline DivisionVector128<O> division_vector_left(DivisionVector128<O> a, typename O::V shift) {
  auto cross = O::shr(a.lo, O::sub(O::set(64), shift));
  auto hi = O::bor(O::shl(a.hi, shift), cross);
  hi = O::bor(hi, O::shl(a.lo, O::sub(shift, O::set(64))));
  return {O::shl(a.lo, shift), hi};
}

// Shift each two-word unsigned 128-bit value right by its per-lane count.
template <class O>
inline DivisionVector128<O> division_vector_right(DivisionVector128<O> a, typename O::V shift) {
  auto cross = O::shl(a.hi, O::sub(O::set(64), shift));
  auto lo = O::bor(O::shr(a.lo, shift), cross);
  lo = O::bor(lo, O::shr(a.hi, O::sub(shift, O::set(64))));
  return {lo, O::shr(a.hi, shift)};
}

// Shift each 128-bit value right, OR-ing any discarded nonzero bits into bit zero.
template <class O>
inline DivisionVector128<O> division_vector_jam(DivisionVector128<O> a, typename O::V shift) {
  auto r = division_vector_right<O>(a, shift);
  auto restored = division_vector_left<O>(r, shift);
  auto lost = O::either(O::inverse(O::eq(a.lo, restored.lo)), O::inverse(O::eq(a.hi, restored.hi)));
  r.lo = O::bor(r.lo, O::select(lost, O::set(1), O::set(0)));
  return r;
}

// Return each 128-bit value's highest set-bit index; zero maps to zero.
template <class O> inline typename O::V division_vector_top(DivisionVector128<O> a) {
  auto high = O::inverse(O::eq(a.hi, O::set(0)));
  auto word = O::select(high, a.hi, a.lo), top = O::select(high, O::set(64), O::set(0));
  constexpr unsigned shifts[] = {32u, 16u, 8u, 4u, 2u, 1u};
  for (unsigned shift : shifts) {
    auto upper = O::shr(word, O::set(shift));
    auto present = O::inverse(O::eq(upper, O::set(0)));
    word = O::select(present, upper, word);
    top = O::add(top, O::select(present, O::set(shift), O::set(0)));
  }
  return top;
}

// Subtract two-word values, propagating the low-word borrow to the high word.
template <class O>
inline DivisionVector128<O> division_vector_sub(DivisionVector128<O> a, DivisionVector128<O> b) {
  return {O::sub(a.lo, b.lo),
          O::sub(O::sub(a.hi, b.hi), O::select(O::ugt(b.lo, a.lo), O::set(1), O::set(0)))};
}

// Return exact DIV_FMAS bits with conditional post-scaling and output modifiers. If causes is
// non-null, accumulate FP causes from active lanes only.
template <int Width, class O>
inline typename O::V division_vector_fmas(typename O::V a, typename O::V b, typename O::V c,
                                          typename O::Mask post, uint32_t mode, bool flush_output,
                                          typename O::Mask active, uint32_t *causes) {
  using F = IeeeFormat<Width>;
  using V = typename O::V;
  using U = DivisionVector128<O>;
  auto zero = O::set(0), one = O::set(1), signbit = O::set(F::sign), inf = O::set(F::infinity),
       hidden = O::set(uint64_t(1) << F::fraction), fracmask = O::sub(hidden, one);
  auto am = O::band(a, O::set(F::sign - 1)), bm = O::band(b, O::set(F::sign - 1)),
       cm = O::band(c, O::set(F::sign - 1));
  auto ae = O::shr(am, O::set(F::fraction)), be = O::shr(bm, O::set(F::fraction)),
       ce = O::shr(cm, O::set(F::fraction));
  auto as = O::bor(O::band(a, fracmask), O::select(O::eq(ae, zero), zero, hidden));
  auto bs = O::bor(O::band(b, fracmask), O::select(O::eq(be, zero), zero, hidden));
  auto cs = O::bor(O::band(c, fracmask), O::select(O::eq(ce, zero), zero, hidden));
  U product = {O::mul32(as, bs), zero}, addend = {cs, zero};
  if constexpr (Width == 64) {
    auto mask32 = O::set(UINT32_MAX), a1 = O::shr(as, O::set(32)), b1 = O::shr(bs, O::set(32));
    auto cross = O::add(O::mul32(a1, bs), O::shr(product.lo, O::set(32)));
    auto middle = O::add(O::mul32(as, b1), O::band(cross, mask32));
    product.hi =
        O::add(O::mul32(a1, b1), O::add(O::shr(cross, O::set(32)), O::shr(middle, O::set(32))));
    product.lo = O::bor(O::shl(middle, O::set(32)), O::band(product.lo, mask32));
  }
  auto pe = O::sub(O::add(O::select(O::eq(ae, zero), one, ae), O::select(O::eq(be, zero), one, be)),
                   O::set(2 * (F::bias + F::fraction)));
  auto se = O::sub(O::select(O::eq(ce, zero), one, ce), O::set(F::bias + F::fraction));

  // Align both terms to a common 128-bit grid before the signed addition; jam discarded bits for
  // final rounding.
  auto pt = division_vector_top<O>(product), ct = division_vector_top<O>(addend);
  auto pzero = O::eq(O::bor(product.lo, product.hi), zero), czero = O::eq(cs, zero);
  auto pexp = O::select(pzero, O::set(uint64_t(-8192)), O::add(pe, pt));
  auto cexp = O::select(czero, O::set(uint64_t(-8192)), O::add(se, ct));
  auto exponent = O::select(O::gt(pexp, cexp), pexp, cexp);
  product = division_vector_left<O>(product, O::sub(O::set(126), pt));
  addend = division_vector_left<O>(addend, O::sub(O::set(126), ct));

  // Zero terms can have an exponent above the chosen nonzero term; clamp their
  // unused shift counts so all variable shifts retain unsigned-count semantics.
  auto pd = O::sub(exponent, O::add(pe, pt)), cd = O::sub(exponent, O::add(se, ct));
  pd = O::select(O::gt(zero, pd), zero, pd);
  cd = O::select(O::gt(zero, cd), zero, cd);
  product = division_vector_jam<O>(product, pd);
  addend = division_vector_jam<O>(addend, cd);
  auto psign = O::band(O::bxor(a, b), signbit), csign = O::band(c, signbit);
  auto same = O::eq(psign, csign);
  auto larger = O::either(O::ugt(product.hi, addend.hi),
                          O::both(O::eq(product.hi, addend.hi), O::ugt(product.lo, addend.lo)));
  auto equal = O::both(O::eq(product.hi, addend.hi), O::eq(product.lo, addend.lo));
  auto sign = O::select(same, psign, O::select(equal, zero, O::select(larger, psign, csign)));
  U sum = {O::add(product.lo, addend.lo), O::add(product.hi, addend.hi)};
  sum.hi = O::add(sum.hi, O::select(O::ugt(product.lo, sum.lo), one, zero));
  auto pminus = division_vector_sub<O>(product, addend),
       cminus = division_vector_sub<O>(addend, product);
  sum.lo = O::select(same, sum.lo, O::select(larger, pminus.lo, cminus.lo));
  sum.hi = O::select(same, sum.hi, O::select(larger, pminus.hi, cminus.hi));
  auto adjustment = O::select(post,
                              O::select(O::gt(ce, O::set(F::bias)), O::set(Width == 32 ? 64 : 128),
                                        O::set(uint64_t(Width == 32 ? -64 : -128))),
                              zero);
  exponent = O::add(O::sub(exponent, O::set(126)), adjustment);
  auto highest = division_vector_top<O>(sum);
  auto shift = O::sub(highest, O::set(F::fraction));
  bool omod = (mode & RJ_ARITH_ALU_OMOD_MASK) != 0;
  if (!omod) {
    auto tiny_shift = O::sub(O::set(uint64_t(1 - F::bias - F::fraction)), exponent);
    shift = O::select(O::gt(tiny_shift, shift), tiny_shift, shift);
  }
  // Keep guard and sticky information separate so the single final rounding handles both normal
  // and subnormal results.
  auto positive = O::gt(shift, zero);
  auto right = division_vector_right<O>(sum, shift),
       left = division_vector_left<O>(sum, O::sub(zero, shift));
  V kept = O::select(positive, right.lo, left.lo);
  auto guard_shift = O::sub(shift, one);
  auto guard_value = division_vector_right<O>(sum, guard_shift);
  auto restored = division_vector_left<O>(guard_value, guard_shift);
  auto sticky =
      O::either(O::inverse(O::eq(sum.lo, restored.lo)), O::inverse(O::eq(sum.hi, restored.hi)));
  auto guard = O::inverse(O::eq(O::band(guard_value.lo, one), zero));
  auto odd = O::inverse(O::eq(O::band(kept, one), zero));
  kept =
      O::add(kept, O::select(O::both(positive, O::both(guard, O::either(sticky, odd))), one, zero));
  auto result_exponent = O::add(O::add(exponent, shift), O::set(F::fraction));
  auto carry = O::ugt(kept, O::sub(O::shl(hidden, one), one));
  kept = O::select(carry, O::shr(kept, one), kept);
  result_exponent = O::add(result_exponent, O::select(carry, one, zero));
  auto result =
      O::bor(sign, O::bor(O::shl(O::add(result_exponent, O::set(F::bias)), O::set(F::fraction)),
                          O::band(kept, fracmask)));
  result = O::select(O::ugt(hidden, kept), O::bor(sign, kept), result);
  result = O::select(O::gt(result_exponent, O::set(F::bias)), O::bor(sign, inf), result);
  if (omod) {
    result = O::select(O::gt(O::set(uint64_t(1 - F::bias)), result_exponent), zero, result);
  }
  result = O::select(O::eq(O::bor(sum.lo, sum.hi), zero), sign, result);

  // Reuse the scalar division_round classifier's discarded bits and tininess
  // threshold. Reporting is optional; numeric arithmetic is shared unchanged.
  if (causes && !(mode & RJ_ARITH_ALU_CLAMP)) {
    auto finite = O::both(O::ugt(inf, am), O::both(O::ugt(inf, bm), O::ugt(inf, cm)));
    auto participating =
        O::both(active, O::both(finite, O::inverse(O::eq(O::bor(sum.lo, sum.hi), zero))));
    auto overflow = O::either(O::gt(O::add(highest, exponent), O::set(F::bias)),
                              O::gt(result_exponent, O::set(F::bias)));
    if (O::any(O::both(participating, overflow))) {
      *causes |= RJ_ARITH_EXCEPTION_OVERFLOW | (omod ? 0 : RJ_ARITH_EXCEPTION_INEXACT);
    }
    if (!omod) {
      auto inexact = O::both(positive, O::either(guard, sticky));
      if (O::any(O::both(participating, inexact))) {
        *causes |= RJ_ARITH_EXCEPTION_INEXACT;
      }
      auto normal_shift = O::sub(highest, O::set(F::fraction));
      auto midpoint =
          division_vector_sub<O>(division_vector_left<O>({one, zero}, O::add(highest, one)),
                                 division_vector_left<O>({one, zero}, O::sub(normal_shift, one)));
      auto at_midpoint =
          O::either(O::ugt(sum.hi, midpoint.hi),
                    O::both(O::eq(sum.hi, midpoint.hi), O::inverse(O::ugt(midpoint.lo, sum.lo))));
      auto edge = O::both(O::eq(O::add(highest, exponent), O::set(uint64_t(-F::bias))),
                          O::both(O::gt(normal_shift, zero), at_midpoint));
      auto tiny = O::both(O::gt(O::set(uint64_t(1 - F::bias)), O::add(highest, exponent)),
                          O::inverse(edge));
      if (O::any(O::both(participating, O::both(inexact, tiny)))) {
        *causes |= RJ_ARITH_EXCEPTION_UNDERFLOW;
      }
    } else if (((mode >> 6) & 3) != 3) {
      auto magnitude = O::band(result, O::set(F::sign - 1));
      auto scaled_overflow =
          O::both(O::ugt(inf, magnitude),
                  O::gt(O::add(O::shr(magnitude, O::set(F::fraction)), O::set((mode >> 6) & 3)),
                        O::set(2 * F::bias)));
      if (O::any(O::both(participating, scaled_overflow))) {
        *causes |= RJ_ARITH_EXCEPTION_OVERFLOW;
      }
    }
  }
  // Restore exceptional operands in ISA priority order, before output modifiers.
  auto pinf = O::either(O::eq(am, inf), O::eq(bm, inf));
  result = O::select(O::eq(cm, inf), c, result);
  result = O::select(pinf, O::bor(psign, inf), result);
  result = O::select(O::ugt(cm, inf), O::bor(c, O::set(F::quiet)), result);
  auto invalid =
      O::either(O::both(O::eq(am, inf), O::eq(bm, zero)), O::both(O::eq(bm, inf), O::eq(am, zero)));
  invalid = O::either(invalid, O::both(O::both(pinf, O::eq(cm, inf)), O::inverse(same)));
  result = O::select(invalid, O::set(F::sign | F::infinity | F::quiet), result);
  result = O::select(O::ugt(bm, inf), O::bor(b, O::set(F::quiet)), result);
  result = O::select(O::ugt(am, inf), O::bor(a, O::set(F::quiet)), result);
  if (flush_output) {
    result = O::select(O::eq(O::band(result, inf), zero), O::band(result, signbit), result);
  }
  unsigned output_scale = (mode >> 6) & 3;
  if (output_scale) {
    auto e = O::band(result, inf), outsign = O::band(result, signbit);
    V scaled;
    if (output_scale == 3) {
      scaled = O::select(O::eq(e, hidden), outsign, O::sub(result, hidden));
    } else {
      auto increment = O::set(uint64_t(output_scale) << F::fraction);
      scaled = O::select(O::inverse(O::ugt(O::sub(inf, increment), e)), O::bor(outsign, inf),
                         O::add(result, increment));
    }
    result = O::select(O::eq(e, inf), result, scaled);
    result = O::select(O::eq(e, zero), zero, result);
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    auto negative = O::inverse(O::eq(O::band(result, signbit), zero));
    result = O::select(O::either(negative, O::ugt(result, inf)), zero, result);
    auto one_value = O::set(uint64_t(F::bias) << F::fraction);
    result = O::select(O::ugt(result, one_value), one_value, result);
  }
  return result;
}

// Evaluate all lanes, then commit masked D0/D1 stores after all source reads.
template <int Width, class O>
inline void division_vector_run(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c, uint32_t condition, bool flush_output,
                                uint32_t *state) {
  using F = IeeeFormat<Width>;
  uint32_t staged[2][32], causes = 0;
  for (int lane = 0; lane < 32; lane += O::lanes) {
    typename O::V inputs[3];
    const uint32_t *const *sources[] = {a, b, c};
    for (int i = 0; i < 3; ++i) {
      inputs[i] = O::load_words(sources[i][0] + lane);
      if constexpr (Width == 64) {
        inputs[i] = O::bor(inputs[i], O::shl(O::load_words(sources[i][1] + lane), O::set(32)));
      }
      if (mode & (RJ_ARITH_ALU_ABS_A << i)) {
        inputs[i] = O::band(inputs[i], O::set(~uint64_t(F::sign)));
      }
      if (mode & (RJ_ARITH_ALU_NEG_A << i)) {
        inputs[i] = O::bxor(inputs[i], O::set(F::sign));
      }
    }
    auto result = division_vector_fmas<Width, O>(
        inputs[0], inputs[1], inputs[2], O::lane_mask(condition >> lane), mode, flush_output,
        O::lane_mask(exec_mask >> lane), state ? &causes : nullptr);
    O::store_words(staged[0] + lane, result);
    if constexpr (Width == 64) {
      O::store_words(staged[1] + lane, O::shr(result, O::set(32)));
    }
  }
  for (int reg = 0; reg < (Width == 64 ? 2 : 1); ++reg) {
    for (int lane = 0; lane < 32; lane += O::lanes) {
      O::masked_words(d[reg] + lane, staged[reg] + lane, exec_mask >> lane);
    }
  }
  if (state) {
    *state |= causes;
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
