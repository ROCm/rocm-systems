// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86 SIMD shared SIMD division-fixup rules parameterized by format and vector operations.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// RDNA3's promoted-half OMOD followed by nearest-even half narrowing and
// finalize_omod_f16, adapted from rocjitsu's generated VOP3 handler. Each
// vector element contains one zero-extended half result, with NaNs quieted.
template <class O>
inline typename O::V fixup_half_omod(typename O::V result, uint32_t omod, bool saturate) {
  auto magnitude = O::band(result, O::set(0x7fff));
  auto exponent = O::band(result, O::set(0x7c00));
  typename O::V scaled;
  if (omod == 3) {
    auto significand = O::bor(O::band(magnitude, O::set(0x3ff)),
                              O::select(O::eq(exponent, O::set(0)), O::set(0), O::set(0x400)));

    // Halving at the subnormal boundary: odd halfway values round to even.
    auto tiny =
        O::add(O::shr(significand, 1),
               O::select(O::eq(O::band(significand, O::set(3)), O::set(3)), O::set(1), O::set(0)));
    scaled = O::select(O::gt(O::set(0x800), magnitude), tiny, O::sub(magnitude, O::set(0x400)));
  } else {
    auto tiny = omod == 1 ? O::shl(magnitude, 1) : O::shl(magnitude, 2);

    // Multiplying a subnormal by four may cross a second exponent boundary.
    if (omod == 2) {
      tiny = O::select(O::gt(tiny, O::set(0x7ff)), O::add(O::shr(tiny, 1), O::set(0x400)), tiny);
    }
    scaled = O::select(O::eq(exponent, O::set(0)), tiny, O::add(magnitude, O::set(omod << 10)));
    scaled = O::select(O::gt(scaled, O::set(0x7bff)), O::set(saturate ? 0x7bff : 0x7c00), scaled);
  }
  scaled = O::bor(O::band(result, O::set(0x8000)), scaled);
  result = O::select(O::gt(O::set(0x7c00), magnitude), scaled, result);
  return O::select(O::eq(O::band(result, O::set(0x7c00)), O::set(0)), O::set(0), result);
}

} // namespace rocjitsu::arith
