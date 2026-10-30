// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local scalar helpers for conversion32.
#pragma once

#include "common/isa.h"
#include "conversion/conversion32.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Truncate and saturate to 32-bit integer bits without out-of-range casts.
// Signed NaNs yield nan_result; unsigned NaNs and negative values yield zero.
template <bool Signed> uint32_t truncate_integer(float x, uint32_t nan_result = 0) {
  // Check NaN and range before any C++ float-to-integer cast, whose out-of-range behavior is
  // undefined.
  if constexpr (Signed) {
    return __builtin_isnan(x)    ? nan_result
           : x >= 2147483648.0f  ? uint32_t(INT32_MAX)
           : x <= -2147483648.0f ? uint32_t(INT32_MIN)
                                 : uint32_t(int32_t(x));
  } else {
    return !(x > 0) ? 0 : x >= 4294967296.0f ? UINT32_MAX : uint32_t(x);
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
