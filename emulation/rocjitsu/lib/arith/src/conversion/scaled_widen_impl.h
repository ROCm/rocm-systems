// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared output rounding for scaled widening conversions.
#pragma once

#include "common/float_encoding.h"
#include "common/isa.h"
#include "conversion/scaled_widen.h"
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Round the scaled FP32 value into the requested destination format. Saturate
// finite FP32 overflow only: an infinity produced by scaling remains infinity.
template <ScaledDestination Format> uint32_t encode_scaled_widen(float value, bool saturate) {
  if constexpr (Format == ScaledDestination::F32) {
    return as_bits(value);
  } else if constexpr (Format == ScaledDestination::F16) {
    return float_to_f16(value, saturate);
  } else {
    uint16_t result = float_to_bf16(value);
    if (saturate && (as_bits(value) & 0x7fffffffU) < 0x7f800000U && (result & 0x7fff) == 0x7f80) {
      --result;
    }
    return result;
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
