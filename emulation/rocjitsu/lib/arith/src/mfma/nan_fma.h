// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Ordered matrix FMA with the inherited Rocjitsu NaN priority.
#pragma once

#include "common/internal.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Preserve the legacy matrix NaN priority: A, then B, then accumulator.
inline float matrix_nan_fma(float a, float b, float c) {
  float result = __builtin_fmaf(a, b, c);
  if ((as_bits(result) & 0x7fffffffU) <= 0x7f800000U) {
    return result;
  }
  if ((as_bits(a) & 0x7fffffffU) > 0x7f800000U) {
    return as_float(as_bits(a) | 0x00400000U);
  }
  if ((as_bits(b) & 0x7fffffffU) > 0x7f800000U) {
    return as_float(as_bits(b) | 0x00400000U);
  }
  if ((as_bits(c) & 0x7fffffffU) > 0x7f800000U) {
    return as_float(as_bits(c) | 0x00400000U);
  }
  return as_float(0xffc00000U);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
