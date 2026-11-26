// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16/BF16 narrowing models inherited from rocjitsu util/data_types.h.
#pragma once

#include "common/alu.h"
#include "common/float_encoding.h"
#include "common/isa.h"
#include "conversion/float16_conversion.h"
#include "conversion/model/conversion.h"
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

// MI350X (gfx950) BF16-to-FP32 widening: honor source/output modifiers, sNaN quieting, and guest
// FP32 denormal/clamp/rounding controls.
template <typename V> inline V widen_bf16_gfx950(V raw, uint64_t flags, uint32_t mode) {
  using F32 = conversion_model::fp_format::F32;
  using conversion_model::lane::choose;
  const uint32_t effective = uint32_t(rdna3_output_modifiers(flags, mode));
  V x = (raw >> V(uint32_t((effective & RJ_ARITH_ALU_HIGH_A) ? 16 : 0))) << V(uint32_t{16});
  if (effective & RJ_ARITH_ALU_ABS_A) {
    x = x & V(0x7fffffffU);
  }
  if (effective & RJ_ARITH_ALU_NEG_A) {
    x = x ^ V(0x80000000U);
  }
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    x = conversion_model::denormal::flush<F32>(x);
  }
  x = choose((x & V(0x7fffffffU)) > V(0x7f800000U), x | V(0x00400000U), x);
  const conversion_model::output_modifier::Policy out{
      (effective >> 6) & 3U, bool(effective & RJ_ARITH_ALU_CLAMP),
      !(flags & RJ_ARITH_FP_DX10_CLAMP_OFF), uint32_t((flags >> 24) & 3U), false};
  x = conversion_model::output_modifier::apply<F32>(x, out);
  if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
    x = conversion_model::denormal::flush<F32>(x);
  }
  return x;
}

// MI350X (gfx950) FP32-to-BF16 nearest-even and stochastic narrowing: quiet NaNs while preserving
// sign and upper payload bits; saturate +-Inf on SR when FP16_OVFL is set.
template <typename V>
inline V narrow_bf16_gfx950(V raw, V seed, bool sr, bool saturate, bool flush_in) {
  using F32 = conversion_model::fp_format::F32;
  using conversion_model::lane::choose;
  if (flush_in) {
    raw = conversion_model::denormal::flush<F32>(raw);
  }
  const V sign = (raw >> V(uint32_t{16})) & V(0x8000U);
  const V mag = raw & V(0x7fffffffU);
  V res;
  if (sr) {
    res = (mag + (seed >> V(uint32_t{16}))) >> V(uint32_t{16});
    res = choose(mag == V(uint32_t{0}), V(uint32_t{0}), res);
  } else {
    const V odd = (mag >> V(uint32_t{16})) & V(uint32_t{1});
    res = (mag + V(0x7fffU) + odd) >> V(uint32_t{16});
  }
  if (saturate) {
    res = choose(res >= V(0x7f80U), V(0x7f7fU), res);
  }
  res = sign | res;
  const V inf_res = sign | V(uint32_t(sr && saturate ? 0x7f7fU : 0x7f80U));
  res = choose(mag == V(0x7f800000U), inf_res, res);
  const V nan_res = (raw >> V(uint32_t{16})) | V(0x0040U);
  return choose(mag > V(0x7f800000U), nan_res, res);
}

// MI350X (gfx950) packed FP32-to-FP16 narrowing: honor guest FP16 rounding, output-denormal
// flushing, FP16_OVFL finite saturation, and quieted NaN payload preservation.
template <typename V> inline V narrow_f16_pk_gfx950(V raw, uint64_t flags) {
  using F32 = conversion_model::fp_format::F32;
  using F16 = conversion_model::fp_format::F16;
  conversion_model::conversion::FloatRounding policy;
  policy.direction = conversion_model::conversion::mode_rounding(uint32_t(flags >> 24));
  policy.flush_tiny = (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) != 0;
  policy.saturate = (flags & RJ_ARITH_FP16_OVFL) != 0;
  return conversion_model::conversion::narrow<F32, F16>(raw, policy);
}

// MI350X (gfx950) stochastic FP32-to-FP16 narrowing: quiet NaNs while preserving upper payload
// bits, and saturate +-Inf as well as finite overflow when FP16_OVFL is set.
template <typename V> inline V narrow_f16_sr_gfx950(V raw, V seed, bool saturate) {
  using conversion_model::lane::choose;
  const V sign = (raw >> V(uint32_t{16})) & V(0x8000U);
  const V mag = raw & V(0x7fffffffU);
  const V field = mag >> V(uint32_t{23});
  const V sig =
      (mag & V(0x7fffffU)) | choose(field == V(uint32_t{0}), V(uint32_t{0}), V(0x800000U));
  const auto sub = field < V(uint32_t{113});
  V sh = choose(sub, V(uint32_t{113}) - field, V(uint32_t{0}));
  sh = choose(sh > V(uint32_t{25}), V(uint32_t{25}), sh);
  const V aligned = sig >> sh;
  const V exp_bits = choose(sub, V(uint32_t{0}), (field - V(uint32_t{113})) << V(uint32_t{10}));
  V res = exp_bits + ((aligned + (seed >> V(uint32_t{19}))) >> V(uint32_t{13}));
  res = choose(field >= V(uint32_t{143}), V(0x7c00U), res);
  const V limit(uint32_t(saturate ? 0x7bffU : 0x7c00U));
  res = choose(res > limit, limit, res);
  res = choose(mag == V(uint32_t{0}), V(uint32_t{0}), res);
  res = sign | res;
  res = choose(mag == V(0x7f800000U), sign | limit, res);
  const V nan_res = sign | V(0x7e00U) | ((mag >> V(uint32_t{13})) & V(0x01ffU));
  return choose(mag > V(0x7f800000U), nan_res, res);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
