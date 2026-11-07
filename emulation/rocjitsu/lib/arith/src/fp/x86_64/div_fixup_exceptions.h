// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86 SIMD shared SIMD classification of division-fixup exception causes.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "common/bits.h"
#include "fp/float_output.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Vector classification with explicit legacy versus captured-RDNA precedence.
// Legacy retains denormal causes beside NaNs; both prioritize underflow recovery
// over provisional-quotient overflow.
// Reuses the numeric FIXUP backend's integer lane operations at either width.
template <int Width, typename Ops>
uint32_t fixup_exception_lanes(typename Ops::V p, typename Ops::V b, typename Ops::V c,
                               uint32_t mode, uint32_t exec_mask, bool legacy = false) {
  using F = IeeeFormat<Width>;
  auto zero = Ops::set(0), inf = Ops::set(F::infinity), magnitude = Ops::set(F::sign - 1);
  p = Ops::band(p, magnitude);
  b = Ops::band(b, magnitude);
  c = Ops::band(c, magnitude);
  auto bn = Ops::gt(b, inf), cn = Ops::gt(c, inf);
  auto nan = Ops::either(bn, cn);
  auto snan = Ops::either(Ops::both(bn, Ops::eq(Ops::band(b, Ops::set(F::quiet)), zero)),
                          Ops::both(cn, Ops::eq(Ops::band(c, Ops::set(F::quiet)), zero)));
  auto bz = Ops::eq(b, zero), cz = Ops::eq(c, zero);
  auto bi = Ops::eq(b, inf), ci = Ops::eq(c, inf);
  auto invalid = Ops::either(
      snan, Ops::both(Ops::inverse(nan), Ops::either(Ops::both(bz, cz), Ops::both(bi, ci))));
  auto div0 = Ops::both(bz, Ops::both(Ops::gt(c, zero), Ops::gt(inf, c)));
  auto remaining = Ops::inverse(Ops::either(nan, Ops::either(invalid, div0)));
  auto tiny = Ops::set(uint64_t(1) << F::fraction);
  auto denorm = Ops::either(Ops::both(Ops::gt(b, zero), Ops::gt(tiny, b)),
                            Ops::both(Ops::gt(c, zero), Ops::gt(tiny, c)));
  // Legacy CDNA retains input-denormal causes alongside exceptional operands.
  if (!legacy) {
    denorm = Ops::both(remaining, denorm);
  }
  auto finite = Ops::both(remaining, Ops::both(Ops::both(Ops::gt(b, zero), Ops::gt(inf, b)),
                                               Ops::both(Ops::gt(c, zero), Ops::gt(inf, c))));
  auto underflow = Ops::eq(zero, Ops::set(1));
  if constexpr (Width != 16) {
    auto delta = Ops::sub(Ops::shr(c, F::fraction), Ops::shr(b, F::fraction));
    underflow =
        Ops::both(finite, Ops::gt(Ops::set(uint64_t(-int64_t(F::bias + F::fraction))), delta));
  }
  finite = Ops::both(finite, Ops::inverse(underflow));
  auto overflow = Ops::both(finite, Ops::inverse(Ops::gt(inf, p)));
  unsigned omod = (mode >> 6) & 3;
  auto inexact = Ops::either(underflow, overflow);
  if (omod == 1 || omod == 2) {
    overflow = Ops::either(
        overflow, Ops::both(finite, Ops::gt(Ops::add(Ops::shr(p, F::fraction), Ops::set(omod)),
                                            Ops::set(2 * F::bias))));
  }
  uint32_t causes = 0;
  // Accumulate cause only when at least one participating lane satisfies mask.
  auto accumulate = [&](typename Ops::Mask mask, uint32_t cause) {
    if (Ops::bits(mask) & exec_mask) {
      causes |= cause;
    }
  };
  accumulate(invalid, RJ_ARITH_EXCEPTION_INVALID);
  accumulate(div0, RJ_ARITH_EXCEPTION_FLOAT_DIV0);
  accumulate(denorm, RJ_ARITH_EXCEPTION_INPUT_DENORM);
  accumulate(overflow, RJ_ARITH_EXCEPTION_OVERFLOW);
  if (!omod) {
    accumulate(underflow, RJ_ARITH_EXCEPTION_UNDERFLOW);
    accumulate(inexact, RJ_ARITH_EXCEPTION_INEXACT);
  }
  return causes;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
