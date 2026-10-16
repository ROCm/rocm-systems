// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Scalar/SIMD denormal model inherited from rocjitsu.
// Works on raw encodings; guest FP controls never change the host FP environment.

#include "conversion/model/fp_format.h"

namespace rocjitsu::arith::conversion_model::denormal {

namespace detail {

/// @brief All ones where the magnitude is below the smallest normal, zero elsewhere.
/// @details The subtraction borrows into the lane's top bit exactly when the
/// magnitude is smaller, which avoids a mask type.
template <typename Fmt, typename V> constexpr V below_normal(V bits) {
  using Lane = typename Fmt::Lane;
  const V magnitude = bits & Fmt::kMagnitude;
  return Lane{0} - ((magnitude - Fmt::kMinNormal) >> (8 * sizeof(Lane) - 1));
}

} // namespace detail

/// @brief Replace a subnormal with a zero of the same sign.
/// @details NaN, infinity, zero and normal encodings pass through unchanged.
template <typename Fmt, typename V> constexpr V flush(V bits) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  return bits & (~detail::below_normal<Fmt>(bits) | Fmt::kSign);
}

} // namespace rocjitsu::arith::conversion_model::denormal
