// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TRANSCENDENTAL_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TRANSCENDENTAL_H_

/// @file Shared transcendental function implementations for AMDGPU ISAs.
///
/// These reference implementations produce results within the ULP accuracy
/// specified by the ISA manuals (typically 1 ULP for f32, 2 ULP for f64).
/// They are used by the simulator's execute() bodies for V_RCP_F32,
/// V_RSQ_F32, V_RSQ_F16, V_SQRT_F32, V_LOG_F32, V_EXP_F32, V_SIN_F32, V_COS_F32,
/// V_RCP_F64, V_RSQ_F64, V_SQRT_F64.
/// F32 reciprocal and F32/F16 reciprocal square root match the captured RDNA3/4 mappings.
/// F16 RSQ applies the half input-denormal policy after promotion to F32.
/// F32 LOG/EXP and SIN/COS use staged integer arithmetic modeled from RDNA3/4 captures,
/// including coordinate truncation and intermediate product rounding.
///
/// All functions handle special cases (NaN, Inf, denormals, ±0) per the
/// AMD ISA specification.

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "util/amdgpu_exp.h"
#include "util/amdgpu_log.h"
#include "util/amdgpu_rcp.h"
#include "util/amdgpu_rsq.h"
#include "util/amdgpu_trig.h"
#include "util/simd.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rocjitsu {
namespace amdgpu {
namespace transcendental {

/// @brief Flush f32 denormals to sign-preserving zero.
///
/// @details SQRT flushes input denormals regardless of the shader's
/// denorm mode. LOG and EXP handle denormals inside their shared integer
/// mappings and do not use this helper.
inline float flush_denorm_f32(float x) {
  uint32_t bits = std::bit_cast<uint32_t>(x);
  if ((bits & 0x7F800000u) == 0 && (bits & 0x007FFFFFu) != 0)
    return std::copysign(0.0f, x);
  return x;
}

/// @brief AMD single-precision reciprocal matching physical RDNA3/4 (within 1 ULP).
inline float rcp_f32(float x) { return util::amdgpu_rcp_f32(x); }

/// @brief AMD single-precision reciprocal square root matching physical RDNA3/4 (within 1 ULP).
inline float rsq_f32(float x) { return util::amdgpu_rsq_f32(x); }

/// @brief F16 reciprocal square root in the promoted F32 domain, with F16 input policy.
inline float rsq_f16(float x, uint32_t denorm_mode) { return util::amdgpu_rsq_f16(x, denorm_mode); }

/// @brief sqrt(x) (single-precision square root, correctly-rounded).
inline float sqrt_f32(float x) {
  x = flush_denorm_f32(x);
  if (std::isnan(x))
    return std::bit_cast<float>(std::bit_cast<uint32_t>(x) | 0x00400000u);
  if (x < 0.0f)
    return std::numeric_limits<float>::quiet_NaN();
  return std::sqrt(x);
}

/// @brief log2(x) using the captured RDNA3/4 reduction and staged approximation.
inline float log_f32(float x, bool quiet_snan = true) {
  return util::amdgpu_log_f32(x, quiet_snan);
}

/// @brief 2^x using the captured RDNA3/4 reduction and staged approximation.
inline float exp_f32(float x, bool quiet_snan = true) {
  return util::amdgpu_exp_f32(x, quiet_snan);
}

namespace detail {
// The caller establishes nearest rounding and preserves the host environment.
// The returned F32 value represents the already rounded architectural half.
template <bool Logarithm>
inline float log_exp_f16_nearest(float x, uint32_t denorm_mode, bool fp16_ovfl, bool quiet_snan) {
  uint32_t bits = std::bit_cast<uint32_t>(x);
  uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u)
    return std::bit_cast<float>(bits | (quiet_snan ? 0x00400000u : 0u));
  if (magnitude == 0x7f800000u) {
    if constexpr (Logarithm)
      return bits & 0x80000000u ? std::bit_cast<float>(0xffc00000u) : x;
    return bits & 0x80000000u ? 0.0f : x;
  }
  if (!(denorm_mode & 1u) && magnitude < 0x38800000u) {
    bits &= 0x80000000u;
    magnitude = 0;
    x = std::bit_cast<float>(bits);
  }
  double value;
  if constexpr (Logarithm) {
    if (magnitude == 0)
      return fp16_ovfl ? -65504.0f : -std::numeric_limits<float>::infinity();
    if (bits & 0x80000000u)
      return std::bit_cast<float>(0xffc00000u);
    value = std::log2(static_cast<double>(x));
  } else {
    // Outside this interval every nearest half result is zero or overflows.
    // Keep the finite-input provenance instead of overflowing host libm.
    value = std::exp2(std::clamp(static_cast<double>(x), -32.0, 32.0));
  }
  // F32 evaluation can land on a half midpoint and round in the wrong direction.
  uint16_t result = pseudo_scalar::round_f16_result(value, 0, 0, false, fp16_ovfl, false);
  if (!(denorm_mode & 2u) && (result & 0x7c00u) == 0)
    result &= 0x8000u;
  return util::f16_to_f32(result);
}
} // namespace detail

/// @brief Evaluate LOG/EXP with half input, output and finite-overflow policies.
/// @details Round once to F16 before any output scaling. Guest rounding is ignored.
template <bool Logarithm>
inline float log_exp_f16(float x, uint32_t denorm_mode, bool fp16_ovfl, bool quiet_snan) {
  fp_mode::ScopedEnvironment environment(0);
  return detail::log_exp_f16_nearest<Logarithm>(x, denorm_mode, fp16_ovfl, quiet_snan);
}

/// @brief Evaluate a native batch with one saved host floating-point environment.
template <bool Logarithm>
inline util::native<float> log_exp_f16_simd(util::native<float> x, uint32_t denorm_mode,
                                            bool fp16_ovfl, bool quiet_snan) {
  fp_mode::ScopedEnvironment environment(0);
  return util::map_native_convert_scalar<float, float>(x, [=](float lane) {
    return detail::log_exp_f16_nearest<Logarithm>(lane, denorm_mode, fp16_ovfl, quiet_snan);
  });
}

/// @brief sin(2*pi*x) using full-range reduction and a captured RDNA3/4 approximation.
///
/// @details The AMD ISA computes sin(2*pi*x), NOT sin(x). Input is in
/// units of 2*pi radians. Output range is [-1.0, 1.0].
inline float sin_f32(float x, uint32_t denorm_mode = 3, bool quiet_snan = true) {
  return util::amdgpu_trig_f32(x, false, denorm_mode, quiet_snan);
}

/// @brief cos(2*pi*x) using full-range reduction and a captured RDNA3/4 approximation.
///
/// @details The AMD ISA computes cos(2*pi*x), NOT cos(x). Input is in
/// units of 2*pi radians. Output range is [-1.0, 1.0].
inline float cos_f32(float x, uint32_t denorm_mode = 3, bool quiet_snan = true) {
  return util::amdgpu_trig_f32(x, true, denorm_mode, quiet_snan);
}

/// @brief Hyperbolic tangent (single-precision, correctly-rounded libm reference).
inline float tanh_f32(float x) {
  if (std::isnan(x))
    return std::bit_cast<float>(std::bit_cast<uint32_t>(x) | 0x00400000u);
  return std::tanh(x);
}

/// @brief 1.0 / x (double-precision reciprocal, ~1 ULP).
inline double rcp_f64(double x) {
  if (std::isnan(x))
    return x;
  if (x == 0.0)
    return std::copysign(std::numeric_limits<double>::infinity(), x);
  if (std::isinf(x))
    return std::copysign(0.0, x);
  return 1.0 / x;
}

/// @brief 1.0 / sqrt(x) (double-precision reciprocal square root, ~2 ULP).
inline double rsq_f64(double x) {
  if (std::isnan(x))
    return x;
  if (x == 0.0)
    return std::copysign(std::numeric_limits<double>::infinity(), x);
  if (x < 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  if (std::isinf(x))
    return 0.0;
  return 1.0 / std::sqrt(x);
}

/// @brief sqrt(x) (double-precision square root, correctly-rounded).
inline double sqrt_f64(double x) {
  if (std::isnan(x))
    return x;
  if (x < 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  return std::sqrt(x);
}

} // namespace transcendental
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TRANSCENDENTAL_H_
