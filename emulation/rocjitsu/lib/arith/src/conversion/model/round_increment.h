// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Scalar/SIMD round increment model inherited from rocjitsu.
// Works on raw encodings; guest FP controls never change the host FP environment.

#include "common/bits.h"
#include <cstdint>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::conversion_model::round_increment {

/// @brief Whether discarding `rest` from `kept` rounds the magnitude up.
/// @param round_mode MODE.FP_ROUND: 0 nearest even, 1 toward +inf, 2 toward
/// -inf, 3 toward zero. Directed modes round up only a value of that sign.
/// @param half The weight of the most significant discarded bit.
template <typename Fmt, typename V, typename Mask>
constexpr Mask rounds_up(uint32_t round_mode, const Mask &negative, V kept, V rest, V half) {
  using Lane = typename Fmt::Lane;
  const V zero(Lane{0});
  switch (round_mode & 3u) {
  case 0:
    return rest > half || (rest == half && (kept & Lane{1}) != zero);
  case 1:
    return rest != zero && !negative;
  case 2:
    return rest != zero && negative;
  default:
    // Toward zero never rounds up; the self-comparison builds a false of the
    // caller's scalar or SIMD mask type.
    return rest != rest;
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::conversion_model::round_increment
