// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file denormal_value.h
/// @brief Adapt denormal::flush to scalar host floating values.

#include "rocjitsu/isa/arch/amdgpu/shared/denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"

#include <bit>
#include <type_traits>

namespace rocjitsu::amdgpu::denormal {

// Flush a scalar float or double on its encoding; F16 must be flushed before widening.
template <typename V> inline V flush_value(V value) {
  static_assert(std::is_same_v<V, float> || std::is_same_v<V, double>);
  using Fmt = std::conditional_t<std::is_same_v<V, float>, fp_format::F32, fp_format::F64>;
  using Bits = typename Fmt::Lane;
  return std::bit_cast<V>(flush<Fmt>(std::bit_cast<Bits>(value)));
}

} // namespace rocjitsu::amdgpu::denormal
