// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file round_increment.h
/// @brief Whether discarding low bits rounds a magnitude up under MODE.FP_ROUND.
/// @details Shared by the raw-bit FRACT and LDEXP stages, for scalar lanes or
/// SIMD vectors. Callers keep their own conversion of the rounding mode.

#include <cstdint>

namespace rocjitsu::amdgpu::round_increment {

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
    return rest != rest;
  }
}

} // namespace rocjitsu::amdgpu::round_increment
