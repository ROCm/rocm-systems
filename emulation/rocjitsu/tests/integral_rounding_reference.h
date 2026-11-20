// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Independent FP32/FP64 integral-rounding references under default host FP controls.
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rocjitsu::amdgpu::integral_reference {

// Round to an integral value, preserving signed zero and quieting the input NaN payload.
// Operations are TRUNC, CEIL, nearest-even and FLOOR, in that order.
template <typename Float> Float evaluate(int operation, Float value) {
  static_assert(std::is_same_v<Float, float> || std::is_same_v<Float, double>);
  using UInt = std::conditional_t<sizeof(Float) == 4, uint32_t, uint64_t>;
  constexpr UInt quiet = UInt(1) << (std::numeric_limits<Float>::digits - 2);
  if (std::isnan(value)) {
    return std::bit_cast<Float>(std::bit_cast<UInt>(value) | quiet);
  }
  switch (operation) {
  case 0:
    return std::trunc(value);
  case 1:
    return std::ceil(value);
  case 2:
    return std::nearbyint(value);
  default:
    return std::floor(value);
  }
}

} // namespace rocjitsu::amdgpu::integral_reference
