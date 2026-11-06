// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file denormal_value.h
/// @brief Adapt denormal::flush to host floating values, scalar or SIMD.

#include "rocjitsu/isa/arch/amdgpu/shared/denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "util/simd.h"

#include <bit>
#include <type_traits>

namespace rocjitsu::amdgpu::denormal {

namespace detail {

template <typename V> struct Element {
  using type = V;
};
template <typename V>
  requires requires { typename V::value_type; }
struct Element<V> {
  using type = typename V::value_type;
};

} // namespace detail

/// @brief Flush a host float or double, or a SIMD vector of them, on its encoding.
/// @details For callers that hold floating values. F16 widened to F32 must not
/// use this; flush its half encoding instead.
template <typename V> inline V flush_value(V value) {
  using Element = typename detail::Element<V>::type;
  static_assert(std::is_same_v<Element, float> || std::is_same_v<Element, double>);
  using Fmt = std::conditional_t<std::is_same_v<Element, float>, fp_format::F32, fp_format::F64>;
  using Bits = std::conditional_t<std::is_arithmetic_v<V>, typename Fmt::Lane,
                                  util::native<typename Fmt::Lane>>;
  return std::bit_cast<V>(flush<Fmt>(std::bit_cast<Bits>(value)));
}

} // namespace rocjitsu::amdgpu::denormal
