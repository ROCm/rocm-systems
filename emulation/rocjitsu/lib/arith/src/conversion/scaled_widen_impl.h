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
template <ScaledDestination Format>
uint32_t encode_scaled_widen(float value, bool saturate, bool flush_output = false) {
  if constexpr (Format == ScaledDestination::F32) {
    uint32_t bits = as_bits(value);
    return flush_output && (bits & 0x7f800000U) == 0 ? (bits & 0x80000000U) : bits;
  } else if constexpr (Format == ScaledDestination::F16) {
    uint16_t result = float_to_f16(value, saturate);
    return flush_output && (result & 0x7c00U) == 0 ? uint16_t(result & 0x8000U) : result;
  } else {
    uint16_t result = float_to_bf16(value);
    if (saturate && (as_bits(value) & 0x7fffffffU) < 0x7f800000U && (result & 0x7fff) == 0x7f80) {
      --result;
    }
    return flush_output && (result & 0x7f80U) == 0 ? uint16_t(result & 0x8000U) : result;
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
