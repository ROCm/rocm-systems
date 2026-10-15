// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file denormal.h
/// @brief Flush a subnormal encoding to a zero of the same sign.
/// @details This is the policy-free primitive. input_denormal.h and
/// output_denormal.h decide from MODE.FP_DENORM whether a source or a result
/// is flushed; instructions that always flush call this directly.
/// denormal_value.h adapts it to host floating values.
/// Works on raw encodings in unsigned scalar or SIMD lanes, so host DAZ/FTZ
/// cannot change the result. F16 must be flushed before it is widened: an F16
/// subnormal is normal when represented in F32.

#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"

namespace rocjitsu::amdgpu::denormal {

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

} // namespace rocjitsu::amdgpu::denormal
