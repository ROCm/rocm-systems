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

template <typename Ops> struct DivisionVector128 {
  using V = typename Ops::V;
  V lo, hi;
};

// Shift each two-word unsigned 128-bit value left by its per-lane count.
template <typename Ops>
inline DivisionVector128<Ops> division_vector_left(DivisionVector128<Ops> a,
                                                   typename Ops::V shift) {
  auto cross = Ops::shr(a.lo, Ops::sub(Ops::set(64), shift));
  auto hi = Ops::bor(Ops::shl(a.hi, shift), cross);
  hi = Ops::bor(hi, Ops::shl(a.lo, Ops::sub(shift, Ops::set(64))));
  return {Ops::shl(a.lo, shift), hi};
}

// Shift each two-word unsigned 128-bit value right by its per-lane count.
template <typename Ops>
inline DivisionVector128<Ops> division_vector_right(DivisionVector128<Ops> a,
                                                    typename Ops::V shift) {
  auto cross = Ops::shl(a.hi, Ops::sub(Ops::set(64), shift));
  auto lo = Ops::bor(Ops::shr(a.lo, shift), cross);
  lo = Ops::bor(lo, Ops::shr(a.hi, Ops::sub(shift, Ops::set(64))));
  return {lo, Ops::shr(a.hi, shift)};
}

// Shift each 128-bit value right, OR-ing any discarded nonzero bits into bit zero.
template <typename Ops>
inline DivisionVector128<Ops> division_vector_jam(DivisionVector128<Ops> a, typename Ops::V shift) {
  auto r = division_vector_right<Ops>(a, shift);
  auto restored = division_vector_left<Ops>(r, shift);
  auto lost = Ops::either(Ops::inverse(Ops::eq(a.lo, restored.lo)),
                          Ops::inverse(Ops::eq(a.hi, restored.hi)));
  r.lo = Ops::bor(r.lo, Ops::select(lost, Ops::set(1), Ops::set(0)));
  return r;
}

// Return each 128-bit value's highest set-bit index; zero maps to zero.
template <typename Ops> inline typename Ops::V division_vector_top(DivisionVector128<Ops> a) {
  auto high = Ops::inverse(Ops::eq(a.hi, Ops::set(0)));
  auto word = Ops::select(high, a.hi, a.lo), top = Ops::select(high, Ops::set(64), Ops::set(0));
  constexpr unsigned shifts[] = {32u, 16u, 8u, 4u, 2u, 1u};
  for (unsigned shift : shifts) {
    auto upper = Ops::shr(word, Ops::set(shift));
    auto present = Ops::inverse(Ops::eq(upper, Ops::set(0)));
    word = Ops::select(present, upper, word);
    top = Ops::add(top, Ops::select(present, Ops::set(shift), Ops::set(0)));
  }
  return top;
}

// Subtract two-word values, propagating the low-word borrow to the high word.
template <typename Ops>
inline DivisionVector128<Ops> division_vector_sub(DivisionVector128<Ops> a,
                                                  DivisionVector128<Ops> b) {
  return {
      Ops::sub(a.lo, b.lo),
      Ops::sub(Ops::sub(a.hi, b.hi), Ops::select(Ops::ugt(b.lo, a.lo), Ops::set(1), Ops::set(0)))};
}

// Return exact DIV_FMAS bits with conditional post-scaling and output modifiers. If causes is
// non-null, accumulate FP causes from active lanes only.
template <int Width, typename Ops>
inline typename Ops::V division_vector_fmas(typename Ops::V a, typename Ops::V b, typename Ops::V c,
                                            typename Ops::Mask post, uint32_t mode, uint64_t flags,
                                            typename Ops::Mask active, uint32_t *causes) {
  using F = IeeeFormat<Width>;
  using V = typename Ops::V;
  using U = DivisionVector128<Ops>;
  auto zero = Ops::set(0), one = Ops::set(1), signbit = Ops::set(F::sign),
       inf = Ops::set(F::infinity), hidden = Ops::set(uint64_t(1) << F::fraction),
       fracmask = Ops::sub(hidden, one);
  auto am = Ops::band(a, Ops::set(F::sign - 1)), bm = Ops::band(b, Ops::set(F::sign - 1)),
       cm = Ops::band(c, Ops::set(F::sign - 1));
  auto ae = Ops::shr(am, Ops::set(F::fraction)), be = Ops::shr(bm, Ops::set(F::fraction)),
       ce = Ops::shr(cm, Ops::set(F::fraction));
  auto as = Ops::bor(Ops::band(a, fracmask), Ops::select(Ops::eq(ae, zero), zero, hidden));
  auto bs = Ops::bor(Ops::band(b, fracmask), Ops::select(Ops::eq(be, zero), zero, hidden));
  auto cs = Ops::bor(Ops::band(c, fracmask), Ops::select(Ops::eq(ce, zero), zero, hidden));
  U product = {Ops::mul32(as, bs), zero}, addend = {cs, zero};
  if constexpr (Width == 64) {
    auto mask32 = Ops::set(UINT32_MAX), a1 = Ops::shr(as, Ops::set(32)),
         b1 = Ops::shr(bs, Ops::set(32));
    auto cross = Ops::add(Ops::mul32(a1, bs), Ops::shr(product.lo, Ops::set(32)));
    auto middle = Ops::add(Ops::mul32(as, b1), Ops::band(cross, mask32));
    product.hi = Ops::add(Ops::mul32(a1, b1),
                          Ops::add(Ops::shr(cross, Ops::set(32)), Ops::shr(middle, Ops::set(32))));
    product.lo = Ops::bor(Ops::shl(middle, Ops::set(32)), Ops::band(product.lo, mask32));
  }
  auto pe = Ops::sub(
      Ops::add(Ops::select(Ops::eq(ae, zero), one, ae), Ops::select(Ops::eq(be, zero), one, be)),
      Ops::set(2 * (F::bias + F::fraction)));
  auto se = Ops::sub(Ops::select(Ops::eq(ce, zero), one, ce), Ops::set(F::bias + F::fraction));

  // Align both terms to a common 128-bit grid before the signed addition; jam discarded bits for
  // final rounding.
  auto pt = division_vector_top<Ops>(product), ct = division_vector_top<Ops>(addend);
  auto pzero = Ops::eq(Ops::bor(product.lo, product.hi), zero), czero = Ops::eq(cs, zero);
  auto pexp = Ops::select(pzero, Ops::set(uint64_t(-8192)), Ops::add(pe, pt));
  auto cexp = Ops::select(czero, Ops::set(uint64_t(-8192)), Ops::add(se, ct));
  auto exponent = Ops::select(Ops::gt(pexp, cexp), pexp, cexp);
  product = division_vector_left<Ops>(product, Ops::sub(Ops::set(126), pt));
  addend = division_vector_left<Ops>(addend, Ops::sub(Ops::set(126), ct));

  // Zero terms can have an exponent above the chosen nonzero term; clamp their
  // unused shift counts so all variable shifts retain unsigned-count semantics.
  auto pd = Ops::sub(exponent, Ops::add(pe, pt)), cd = Ops::sub(exponent, Ops::add(se, ct));
  pd = Ops::select(Ops::gt(zero, pd), zero, pd);
  cd = Ops::select(Ops::gt(zero, cd), zero, cd);
  product = division_vector_jam<Ops>(product, pd);
  addend = division_vector_jam<Ops>(addend, cd);
  auto psign = Ops::band(Ops::bxor(a, b), signbit), csign = Ops::band(c, signbit);
  auto same = Ops::eq(psign, csign);
  auto larger =
      Ops::either(Ops::ugt(product.hi, addend.hi),
                  Ops::both(Ops::eq(product.hi, addend.hi), Ops::ugt(product.lo, addend.lo)));
  auto equal = Ops::both(Ops::eq(product.hi, addend.hi), Ops::eq(product.lo, addend.lo));
  auto sign = Ops::select(same, psign, Ops::select(equal, zero, Ops::select(larger, psign, csign)));
  U sum = {Ops::add(product.lo, addend.lo), Ops::add(product.hi, addend.hi)};
  sum.hi = Ops::add(sum.hi, Ops::select(Ops::ugt(product.lo, sum.lo), one, zero));
  auto pminus = division_vector_sub<Ops>(product, addend),
       cminus = division_vector_sub<Ops>(addend, product);
  sum.lo = Ops::select(same, sum.lo, Ops::select(larger, pminus.lo, cminus.lo));
  sum.hi = Ops::select(same, sum.hi, Ops::select(larger, pminus.hi, cminus.hi));
  auto adjustment =
      Ops::select(post,
                  Ops::select(Ops::gt(ce, Ops::set(F::bias)), Ops::set(Width == 32 ? 64 : 128),
                              Ops::set(uint64_t(Width == 32 ? -64 : -128))),
                  zero);
  exponent = Ops::add(Ops::sub(exponent, Ops::set(126)), adjustment);
  auto highest = division_vector_top<Ops>(sum);
  auto shift = Ops::sub(highest, Ops::set(F::fraction));
  bool omod = (mode & RJ_ARITH_ALU_OMOD_MASK) != 0;
  if (!omod) {
    auto tiny_shift = Ops::sub(Ops::set(uint64_t(1 - F::bias - F::fraction)), exponent);
    shift = Ops::select(Ops::gt(tiny_shift, shift), tiny_shift, shift);
  }
  // Keep guard and sticky information separate so the single final rounding handles both normal
  // and subnormal results.
  auto positive = Ops::gt(shift, zero);
  auto right = division_vector_right<Ops>(sum, shift),
       left = division_vector_left<Ops>(sum, Ops::sub(zero, shift));
  V kept = Ops::select(positive, right.lo, left.lo);
  auto guard_shift = Ops::sub(shift, one);
  auto guard_value = division_vector_right<Ops>(sum, guard_shift);
  auto restored = division_vector_left<Ops>(guard_value, guard_shift);
  auto sticky = Ops::either(Ops::inverse(Ops::eq(sum.lo, restored.lo)),
                            Ops::inverse(Ops::eq(sum.hi, restored.hi)));
  auto guard = Ops::inverse(Ops::eq(Ops::band(guard_value.lo, one), zero));
  auto odd = Ops::inverse(Ops::eq(Ops::band(kept, one), zero));
  kept = Ops::add(kept, Ops::select(Ops::both(positive, Ops::both(guard, Ops::either(sticky, odd))),
                                    one, zero));
  auto result_exponent = Ops::add(Ops::add(exponent, shift), Ops::set(F::fraction));
  auto carry = Ops::ugt(kept, Ops::sub(Ops::shl(hidden, one), one));
  kept = Ops::select(carry, Ops::shr(kept, one), kept);
  result_exponent = Ops::add(result_exponent, Ops::select(carry, one, zero));
  auto result = Ops::bor(
      sign, Ops::bor(Ops::shl(Ops::add(result_exponent, Ops::set(F::bias)), Ops::set(F::fraction)),
                     Ops::band(kept, fracmask)));
  result = Ops::select(Ops::ugt(hidden, kept), Ops::bor(sign, kept), result);
  result = Ops::select(Ops::gt(result_exponent, Ops::set(F::bias)), Ops::bor(sign, inf), result);
  if (omod) {
    result = Ops::select(Ops::gt(Ops::set(uint64_t(1 - F::bias)), result_exponent), zero, result);
  }
  result = Ops::select(Ops::eq(Ops::bor(sum.lo, sum.hi), zero), sign, result);

  // Reuse the scalar division_round classifier's discarded bits and tininess
  // threshold. Reporting is optional; numeric arithmetic is shared unchanged.
  if (causes && !(mode & RJ_ARITH_ALU_CLAMP)) {
    auto finite = Ops::both(Ops::ugt(inf, am), Ops::both(Ops::ugt(inf, bm), Ops::ugt(inf, cm)));
    auto participating =
        Ops::both(active, Ops::both(finite, Ops::inverse(Ops::eq(Ops::bor(sum.lo, sum.hi), zero))));
    auto overflow = Ops::either(Ops::gt(Ops::add(highest, exponent), Ops::set(F::bias)),
                                Ops::gt(result_exponent, Ops::set(F::bias)));
    if (Ops::any(Ops::both(participating, overflow))) {
      *causes |= RJ_ARITH_EXCEPTION_OVERFLOW | (omod ? 0 : RJ_ARITH_EXCEPTION_INEXACT);
    }
    if (!omod) {
      auto inexact = Ops::both(positive, Ops::either(guard, sticky));
      if (Ops::any(Ops::both(participating, inexact))) {
        *causes |= RJ_ARITH_EXCEPTION_INEXACT;
      }
      auto normal_shift = Ops::sub(highest, Ops::set(F::fraction));
      auto midpoint = division_vector_sub<Ops>(
          division_vector_left<Ops>({one, zero}, Ops::add(highest, one)),
          division_vector_left<Ops>({one, zero}, Ops::sub(normal_shift, one)));
      auto at_midpoint = Ops::either(
          Ops::ugt(sum.hi, midpoint.hi),
          Ops::both(Ops::eq(sum.hi, midpoint.hi), Ops::inverse(Ops::ugt(midpoint.lo, sum.lo))));
      auto edge = Ops::both(Ops::eq(Ops::add(highest, exponent), Ops::set(uint64_t(-F::bias))),
                            Ops::both(Ops::gt(normal_shift, zero), at_midpoint));
      auto tiny = Ops::both(Ops::gt(Ops::set(uint64_t(1 - F::bias)), Ops::add(highest, exponent)),
                            Ops::inverse(edge));
      if (Ops::any(Ops::both(participating, Ops::both(inexact, tiny)))) {
        *causes |= RJ_ARITH_EXCEPTION_UNDERFLOW;
      }
    } else if (((mode >> 6) & 3) != 3) {
      auto magnitude = Ops::band(result, Ops::set(F::sign - 1));
      auto scaled_overflow = Ops::both(
          Ops::ugt(inf, magnitude),
          Ops::gt(Ops::add(Ops::shr(magnitude, Ops::set(F::fraction)), Ops::set((mode >> 6) & 3)),
                  Ops::set(2 * F::bias)));
      if (Ops::any(Ops::both(participating, scaled_overflow))) {
        *causes |= RJ_ARITH_EXCEPTION_OVERFLOW;
      }
    }
  }
  // Restore exceptional operands in ISA priority order, before output modifiers.
  auto pinf = Ops::either(Ops::eq(am, inf), Ops::eq(bm, inf));
  result = Ops::select(Ops::eq(cm, inf), c, result);
  result = Ops::select(pinf, Ops::bor(psign, inf), result);
  result = Ops::select(Ops::ugt(cm, inf), Ops::bor(c, Ops::set(F::quiet)), result);
  auto invalid = Ops::either(Ops::both(Ops::eq(am, inf), Ops::eq(bm, zero)),
                             Ops::both(Ops::eq(bm, inf), Ops::eq(am, zero)));
  invalid = Ops::either(invalid, Ops::both(Ops::both(pinf, Ops::eq(cm, inf)), Ops::inverse(same)));
  result = Ops::select(invalid, Ops::set(F::sign | F::infinity | F::quiet), result);
  result = Ops::select(Ops::ugt(bm, inf), Ops::bor(b, Ops::set(F::quiet)), result);
  result = Ops::select(Ops::ugt(am, inf), Ops::bor(a, Ops::set(F::quiet)), result);
  if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
    result = Ops::select(Ops::eq(Ops::band(result, inf), zero), Ops::band(result, signbit), result);
  }
  unsigned output_scale = (mode >> 6) & 3;
  if (output_scale) {
    auto e = Ops::band(result, inf), outsign = Ops::band(result, signbit);
    V scaled;
    if (output_scale == 3) {
      scaled = Ops::select(Ops::eq(e, hidden), outsign, Ops::sub(result, hidden));
    } else {
      auto increment = Ops::set(uint64_t(output_scale) << F::fraction);
      scaled = Ops::select(Ops::inverse(Ops::ugt(Ops::sub(inf, increment), e)),
                           Ops::bor(outsign, inf), Ops::add(result, increment));
    }
    result = Ops::select(Ops::eq(e, inf), result, scaled);
    result = Ops::select(Ops::eq(e, zero), zero, result);
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    const auto unclamped = result;
    auto negative = Ops::inverse(Ops::eq(Ops::band(result, signbit), zero));
    result = Ops::select(Ops::either(negative, Ops::ugt(result, inf)), zero, result);
    auto one_value = Ops::set(uint64_t(F::bias) << F::fraction);
    result = Ops::select(Ops::ugt(result, one_value), one_value, result);
    if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
      result = Ops::select(Ops::ugt(Ops::band(unclamped, Ops::set(F::sign - 1)), inf), unclamped,
                           result);
    }
  }
  return result;
}

// Evaluate all lanes, then commit masked D0/D1 stores after all source reads.
template <int Width, typename Ops>
inline void division_vector_run(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c, uint32_t condition, uint64_t flags,
                                uint32_t *state) {
  using F = IeeeFormat<Width>;
  uint32_t staged[2][32], causes = 0;
  for (int lane = 0; lane < 32; lane += Ops::lanes) {
    typename Ops::V inputs[3];
    const uint32_t *const *sources[] = {a, b, c};
    for (int i = 0; i < 3; ++i) {
      inputs[i] = Ops::load_words(sources[i][0] + lane);
      if constexpr (Width == 64) {
        inputs[i] =
            Ops::bor(inputs[i], Ops::shl(Ops::load_words(sources[i][1] + lane), Ops::set(32)));
      }
      if (mode & (RJ_ARITH_ALU_ABS_A << i)) {
        inputs[i] = Ops::band(inputs[i], Ops::set(~uint64_t(F::sign)));
      }
      if (mode & (RJ_ARITH_ALU_NEG_A << i)) {
        inputs[i] = Ops::bxor(inputs[i], Ops::set(F::sign));
      }
    }
    auto result = division_vector_fmas<Width, Ops>(
        inputs[0], inputs[1], inputs[2], Ops::lane_mask(condition >> lane), mode, flags,
        Ops::lane_mask(exec_mask >> lane), state ? &causes : nullptr);
    Ops::store_words(staged[0] + lane, result);
    if constexpr (Width == 64) {
      Ops::store_words(staged[1] + lane, Ops::shr(result, Ops::set(32)));
    }
  }
  for (int reg = 0; reg < (Width == 64 ? 2 : 1); ++reg) {
    for (int lane = 0; lane < 32; lane += Ops::lanes) {
      Ops::masked_words(d[reg] + lane, staged[reg] + lane, exec_mask >> lane);
    }
  }
  if (state) {
    *state |= causes;
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
