// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Scalar/SIMD lane select model inherited from rocjitsu.
// Works on raw encodings; guest FP controls never change the host FP environment.

#include "common/bits.h"
#include <type_traits>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::conversion_model::lane {

/// @brief `when_true` where `take` holds, else `when_false`, for a scalar or SIMD lane.
/// @details Both candidates are evaluated; a SIMD mask selects independently per lane.
template <typename V, typename Mask>
constexpr V choose(const Mask &take, V when_true, V when_false) {
  if constexpr (std::is_same_v<Mask, bool>) {
    return take ? when_true : when_false;
  } else {
    return select(take, when_true, when_false);
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::conversion_model::lane
