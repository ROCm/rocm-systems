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
///
/// All functions handle special cases (NaN, Inf, denormals, ±0) per the
/// AMD ISA specification.

#include "util/amdgpu_rcp.h"
#include "util/amdgpu_rsq.h"
#include "util/amdgpu_trig.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rocjitsu {
namespace amdgpu {
namespace transcendental {

/// @brief Flush f32 denormals to sign-preserving zero.
///
/// @details AMD transcendental micro-ops always operate in FTZ mode
/// regardless of the shader's denorm mode.  This helper reproduces
/// that behaviour for sqrt, exp, and log.
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

/// @brief log2(x) (single-precision base-2 logarithm, ~1 ULP).
inline float log_f32(float x) {
  x = flush_denorm_f32(x);
  if (std::isnan(x))
    return std::bit_cast<float>(std::bit_cast<uint32_t>(x) | 0x00400000u);
  if (x == 0.0f)
    return -std::numeric_limits<float>::infinity();
  if (x < 0.0f)
    return std::numeric_limits<float>::quiet_NaN();
  if (std::isinf(x))
    return std::numeric_limits<float>::infinity();
  return std::log2(x);
}

/// @brief 2^x (single-precision base-2 exponential, ~1 ULP).
inline float exp_f32(float x) {
  x = flush_denorm_f32(x);
  if (std::isnan(x))
    return std::bit_cast<float>(std::bit_cast<uint32_t>(x) | 0x00400000u);
  if (x == -std::numeric_limits<float>::infinity())
    return 0.0f;
  if (x == std::numeric_limits<float>::infinity())
    return std::numeric_limits<float>::infinity();
  return flush_denorm_f32(std::exp2(x));
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
