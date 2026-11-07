// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited format rounding shared by short and wide scaled conversions.
#pragma once

#include "conversion/finite_narrow.h"
#include "conversion/fp8_narrow_impl.h"
#include "conversion/scaled_narrow.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Apply the selected format's inherited rounding policy to one scaled value.
template <ScaledNarrowDestination Dst, bool Stochastic>
inline uint32_t scaled_narrow_encode(uint32_t raw, uint32_t seed, bool saturate) {
  if constexpr (Dst == ScaledNarrowDestination::Fp4) {
    return narrow_finite<2, 1, Stochastic>(raw, seed);
  } else if constexpr (Dst == ScaledNarrowDestination::Fp6 || Dst == ScaledNarrowDestination::Bf6) {
    constexpr bool bf6 = Dst == ScaledNarrowDestination::Bf6;
    return narrow_finite<(bf6 ? 3 : 2), (bf6 ? 2 : 3), Stochastic>(raw, seed);
  } else {
    return narrow_fp8<Dst == ScaledNarrowDestination::Bf8, Stochastic>(raw, seed, saturate,
                                                                       Fp8NarrowPolicy::Ocp);
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
