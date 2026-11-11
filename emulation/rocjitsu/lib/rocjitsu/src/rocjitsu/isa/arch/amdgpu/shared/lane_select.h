// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file lane_select.h
/// @brief Per-lane selection shared by the scalar and SIMD floating-point stages.

#include <type_traits>

namespace rocjitsu::amdgpu::lane {

/// @brief `when_true` where `take` holds, else `when_false`, for a scalar or SIMD lane.
/// @details Both candidates are evaluated; a SIMD mask selects independently per lane.
template <typename V, typename Mask>
constexpr V choose(const Mask &take, V when_true, V when_false) {
  if constexpr (std::is_same_v<Mask, bool>) {
    return take ? when_true : when_false;
  } else {
    where(take, when_false) = when_true;
    return when_false;
  }
}

} // namespace rocjitsu::amdgpu::lane
