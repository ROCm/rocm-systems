// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16/BF16 narrowing models inherited from rocjitsu util/data_types.h.
#pragma once

#include "common/float_encoding.h"
#include "common/isa.h"
#include "conversion/float16_conversion.h"
#include "conversion/model/denormal.h"
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Narrow FP32 with the inherited nearest-even or seeded stochastic rule.
// Finite overflow may saturate; infinity and NaN representations follow that rule.
inline uint32_t narrow_bf16(uint32_t raw, uint32_t seed, bool stochastic, bool saturate) {
  uint32_t magnitude = raw & 0x7fffffff;
  uint32_t result;
  if (stochastic) {
    if (magnitude > 0x7f800000) {
      return (raw >> 16) | 0x7fff;
    }
    if (!magnitude || magnitude == 0x7f800000) {
      return raw >> 16;
    }
    // BF16 retains the FP32 exponent, including subnormal inputs.
    result = (raw + (seed >> 16)) >> 16;
  } else if (magnitude >= 0x7f800000) {
    result = (raw >> 16) | ((raw & 0xffff) != 0);
  } else {
    result = (raw + 0x7fff + ((raw >> 16) & 1)) >> 16;
  }
  if (saturate && magnitude < 0x7f800000 && (result & 0x7fff) == 0x7f80) {
    result = (result & 0x8000) | 0x7f7f;
  }
  return result;
}

// Seeded FP16 narrowing inherited from util::f32_to_binary_float_sr. The random
// increment is added after aligning subnormal mantissas, as in that model.
inline uint32_t narrow_f16_sr(uint32_t raw, uint32_t seed, bool saturate) {
  uint32_t magnitude = raw & 0x7fffffff, sign = (raw >> 16) & 0x8000;
  if (magnitude >= 0x7f800000) {
    return sign | (magnitude > 0x7f800000 ? 0x7fff : 0x7c00);
  }
  if (!magnitude) {
    return sign;
  }
  int exponent = int(magnitude >> 23) - 127;
  uint32_t mantissa = raw & 0x7fffff;
  if (exponent > 15) {
    return sign | (saturate ? 0x7bff : 0x7c00);
  }
  bool subnormal = exponent < -14;
  if (subnormal) {
    int shift = -14 - exponent;
    mantissa = shift >= 32 ? 0 : (mantissa | 0x800000) >> shift;
    exponent = 0;
  }
  mantissa += seed >> 19;
  exponent += mantissa >= 0x800000;
  if (exponent > 15) {
    return sign | (saturate ? 0x7bff : 0x7c00);
  }
  return sign | (uint32_t(subnormal ? exponent : exponent + 15) << 10) | ((mantissa >> 13) & 0x3ff);
}

// Select the inherited format's narrowing rule without duplicating packing or dispatch.
template <Float16Format Format>
inline uint32_t narrow_float16(uint32_t raw, uint32_t seed, bool sr, bool saturate) {
  if constexpr (Format == Float16Format::Bf16) {
    return narrow_bf16(raw, seed, sr, saturate);
  } else {
    return sr ? narrow_f16_sr(raw, seed, saturate) : float_to_f16(as_float(raw), saturate);
  }
}

// gfx1250 flushes an exact 2^-20 packed result. Use the rounded FP16
// encoding to model the unmeasured boundary between normal and subnormal.
template <typename V> inline V pack_rounded_halves(V low, V high, bool flush_f16) {
  if (flush_f16) {
    using F16 = conversion_model::fp_format::F16;
    low = conversion_model::denormal::flush<F16>(low);
    high = conversion_model::denormal::flush<F16>(high);
  }
  return low | (high << 16);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
