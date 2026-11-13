// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// SIMD output scaling for CDNA's promoted-half division-fixup model.
// Shared by the AVX2 and AVX-512 kernels in their ISA-local namespaces.

#pragma once

#include "common/isa.h"
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Scale a zero-extended half, round to nearest-even, then flush tiny results
// to positive zero. Saturation applies only to finite overflow during scaling.
template <typename Ops>
inline typename Ops::V fixup_cdna_half_omod(typename Ops::V result, uint32_t omod, bool saturate) {
  auto magnitude = Ops::band(result, Ops::set(0x7fff));
  auto exponent = Ops::band(result, Ops::set(0x7c00));
  typename Ops::V scaled;
  if (omod == 3) {
    auto significand =
        Ops::bor(Ops::band(magnitude, Ops::set(0x3ff)),
                 Ops::select(Ops::eq(exponent, Ops::set(0)), Ops::set(0), Ops::set(0x400)));

    // Halving at the subnormal boundary: odd halfway values round to even.
    auto tiny = Ops::add(Ops::shr(significand, 1),
                         Ops::select(Ops::eq(Ops::band(significand, Ops::set(3)), Ops::set(3)),
                                     Ops::set(1), Ops::set(0)));
    scaled = Ops::select(Ops::gt(Ops::set(0x800), magnitude), tiny,
                         Ops::sub(magnitude, Ops::set(0x400)));
  } else {
    auto tiny = omod == 1 ? Ops::shl(magnitude, 1) : Ops::shl(magnitude, 2);

    // Multiplying a subnormal by four may cross a second exponent boundary.
    if (omod == 2) {
      tiny = Ops::select(Ops::gt(tiny, Ops::set(0x7ff)),
                         Ops::add(Ops::shr(tiny, 1), Ops::set(0x400)), tiny);
    }
    scaled = Ops::select(Ops::eq(exponent, Ops::set(0)), tiny,
                         Ops::add(magnitude, Ops::set(omod << 10)));
    scaled = Ops::select(Ops::gt(scaled, Ops::set(0x7bff)), Ops::set(saturate ? 0x7bff : 0x7c00),
                         scaled);
  }
  scaled = Ops::bor(Ops::band(result, Ops::set(0x8000)), scaled);
  result = Ops::select(Ops::gt(Ops::set(0x7c00), magnitude), scaled, result);
  return Ops::select(Ops::eq(Ops::band(result, Ops::set(0x7c00)), Ops::set(0)), Ops::set(0),
                     result);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
