// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86 SIMD shared SIMD classification of division-fixup exception causes.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "fp/float_output.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Vector form of fixup_exceptions: preserve its precedence, especially NaN
// suppression of denormal causes and underflow precedence over quotient overflow.
// Reuses the numeric FIXUP backend's integer lane operations at either width.
template <int Width, class O>
uint32_t fixup_exception_lanes(typename O::V p, typename O::V b, typename O::V c, uint32_t mode,
                               uint32_t exec_mask) {
  using F = IeeeFormat<Width>;
  auto zero = O::set(0), inf = O::set(F::infinity), magnitude = O::set(F::sign - 1);
  p = O::band(p, magnitude);
  b = O::band(b, magnitude);
  c = O::band(c, magnitude);
  auto bn = O::gt(b, inf), cn = O::gt(c, inf);
  auto nan = O::either(bn, cn);
  auto snan = O::either(O::both(bn, O::eq(O::band(b, O::set(F::quiet)), zero)),
                        O::both(cn, O::eq(O::band(c, O::set(F::quiet)), zero)));
  auto bz = O::eq(b, zero), cz = O::eq(c, zero);
  auto bi = O::eq(b, inf), ci = O::eq(c, inf);
  auto invalid =
      O::either(snan, O::both(O::inverse(nan), O::either(O::both(bz, cz), O::both(bi, ci))));
  auto div0 = O::both(bz, O::both(O::gt(c, zero), O::gt(inf, c)));
  auto remaining = O::inverse(O::either(nan, O::either(invalid, div0)));
  auto tiny = O::set(uint64_t(1) << F::fraction);
  auto denorm = O::both(remaining, O::either(O::both(O::gt(b, zero), O::gt(tiny, b)),
                                             O::both(O::gt(c, zero), O::gt(tiny, c))));
  auto finite = O::both(remaining, O::both(O::both(O::gt(b, zero), O::gt(inf, b)),
                                           O::both(O::gt(c, zero), O::gt(inf, c))));
  auto underflow = O::eq(zero, O::set(1));
  if constexpr (Width != 16) {
    auto delta = O::sub(O::shr(c, F::fraction), O::shr(b, F::fraction));
    underflow = O::both(finite, O::gt(O::set(uint64_t(-int64_t(F::bias + F::fraction))), delta));
  }
  finite = O::both(finite, O::inverse(underflow));
  auto overflow = O::both(finite, O::inverse(O::gt(inf, p)));
  unsigned omod = (mode >> 6) & 3;
  auto inexact = O::either(underflow, overflow);
  if (omod == 1 || omod == 2) {
    overflow = O::either(
        overflow,
        O::both(finite, O::gt(O::add(O::shr(p, F::fraction), O::set(omod)), O::set(2 * F::bias))));
  }
  uint32_t causes = 0;
  // Accumulate cause only when at least one participating lane satisfies mask.
  auto accumulate = [&](typename O::Mask mask, uint32_t cause) {
    if (O::bits(mask) & exec_mask) {
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

} // namespace rocjitsu::arith
