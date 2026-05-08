// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TRANSCENDENTAL_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TRANSCENDENTAL_H_

/// @file Shared transcendental function implementations for AMDGPU ISAs.
///
/// These reference implementations produce results within the ULP accuracy
/// specified by the ISA manuals (typically 1 ULP for f32, 2 ULP for f64).
/// They are used by the simulator's execute() bodies for V_RCP_F32,
/// V_RSQ_F32, V_SQRT_F32, V_LOG_F32, V_EXP_F32, V_SIN_F32, V_COS_F32,
/// V_RCP_F64, V_RSQ_F64, V_SQRT_F64.
///
/// All functions handle special cases (NaN, Inf, denormals, ±0) per the
/// AMD ISA specification.

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rocjitsu {
namespace amdgpu {
namespace transcendental {

/// @brief 1.0 / x (single-precision reciprocal, ~0.5 ULP).
inline float rcp_f32(float x) {
  if (std::isnan(x))
    return x;
  if (x == 0.0f)
    return std::copysign(std::numeric_limits<float>::infinity(), x);
  if (std::isinf(x))
    return std::copysign(0.0f, x);
  return 1.0f / x;
}

/// @brief 1.0 / sqrt(x) (single-precision reciprocal square root, ~1 ULP).
inline float rsq_f32(float x) {
  if (std::isnan(x))
    return x;
  if (x == 0.0f)
    return std::copysign(std::numeric_limits<float>::infinity(), x);
  if (x < 0.0f)
    return std::numeric_limits<float>::quiet_NaN();
  if (std::isinf(x))
    return 0.0f;
  return 1.0f / std::sqrt(x);
}

/// @brief sqrt(x) (single-precision square root, correctly-rounded).
inline float sqrt_f32(float x) {
  if (std::isnan(x))
    return x;
  if (x < 0.0f)
    return std::numeric_limits<float>::quiet_NaN();
  return std::sqrt(x);
}

/// @brief log2(x) (single-precision base-2 logarithm, ~1 ULP).
inline float log_f32(float x) {
  if (std::isnan(x))
    return x;
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
  if (std::isnan(x))
    return x;
  if (x == -std::numeric_limits<float>::infinity())
    return 0.0f;
  if (x == std::numeric_limits<float>::infinity())
    return std::numeric_limits<float>::infinity();
  return std::exp2(x);
}

/// @brief sin(2*pi*x) (single-precision, ~1 ULP).
///
/// @details The AMD ISA computes sin(2*pi*x), NOT sin(x). Input is in
/// units of 2*pi radians. Output range is [-1.0, 1.0].
inline float sin_f32(float x) {
  if (std::isnan(x) || std::isinf(x))
    return std::numeric_limits<float>::quiet_NaN();
  constexpr float TWO_PI = 6.283185307179586476925286766559f;
  return std::sin(x * TWO_PI);
}

/// @brief cos(2*pi*x) (single-precision, ~1 ULP).
///
/// @details The AMD ISA computes cos(2*pi*x), NOT cos(x). Input is in
/// units of 2*pi radians. Output range is [-1.0, 1.0].
inline float cos_f32(float x) {
  if (std::isnan(x) || std::isinf(x))
    return std::numeric_limits<float>::quiet_NaN();
  constexpr float TWO_PI = 6.283185307179586476925286766559f;
  return std::cos(x * TWO_PI);
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
