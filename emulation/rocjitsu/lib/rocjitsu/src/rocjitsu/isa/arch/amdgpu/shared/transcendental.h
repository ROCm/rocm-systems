// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TRANSCENDENTAL_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TRANSCENDENTAL_H_

/// @file transcendental.h
/// @brief Shared transcendental evaluation for AMDGPU execute bodies.
/// @details F32 reciprocal/root/log/exp/trigonometric mappings use RDNA3/4
/// capture-based implementations; F64 and TANH use host arithmetic. These have
/// separate accuracy and hardware-coverage contracts.
/// F16 callers flush the raw source half under MODE before widening, including
/// for RSQ/SQRT. The half helpers round and flush the result; callers apply
/// OMOD/CLAMP afterward. See output_modifier.h for migration scope and evidence.

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_modifier.h"
#include "util/amdgpu_exp.h"
#include "util/amdgpu_log.h"
#include "util/amdgpu_rcp.h"
#include "util/amdgpu_rsq.h"
#include "util/amdgpu_sqrt.h"
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

/// @brief AMD single-precision reciprocal matching physical RDNA3/4 (within 1 ULP).
inline float rcp_f32(float x, bool quiet_snan = true) {
  return util::amdgpu_rcp_f32(x, quiet_snan);
}

/// @brief AMD single-precision reciprocal square root matching physical RDNA3/4 (within 1 ULP).
inline float rsq_f32(float x, bool quiet_snan = true) {
  return util::amdgpu_rsq_f32(x, quiet_snan);
}

/// @brief Single-precision square root matching physical RDNA3/4 (within 1 ULP).
inline float sqrt_f32(float x, bool quiet_snan = true) {
  return util::amdgpu_sqrt_f32(x, quiet_snan);
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
// The caller establishes nearest rounding and preserves the host environment,
// and flushes the source half under MODE before widening it (input_denormal.h).
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
  const uint16_t result = pseudo_scalar::round_f16_result(value, 0, 0, false, fp16_ovfl, false);
  return util::f16_to_f32(static_cast<uint16_t>(output_denormal::flush_output<fp_format::F16>(
      uint32_t{result}, output_denormal::Policy::make(denorm_mode))));
}
} // namespace detail

/// @brief Evaluate LOG/EXP, round to F16, and apply output-denormal policy.
/// @details The caller flushes the raw source half before widening. Return an
/// already-rounded half represented in F32; the caller owns OMOD/CLAMP.
/// Guest rounding is ignored; FP16_OVFL controls finite overflow.
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

/// @brief Half transcendental mappings evaluated through their F32 counterparts.
enum class HalfOperation { RCP, RSQ, SQRT, SIN, COS };

/// @brief Evaluate a prepared source, round to F16, and apply output-denormal policy.
/// @details The caller flushes the raw half before widening and owns OMOD/CLAMP.
/// Return an already-rounded half represented in F32. FP16_OVFL also saturates
/// infinity produced from zero or a flushed input.
template <HalfOperation Op>
inline float map_f16(float source, uint32_t denorm_mode, bool fp16_ovfl, bool quiet_snan) {
  float value;
  if constexpr (Op == HalfOperation::RCP)
    value = util::amdgpu_rcp_f32(source, quiet_snan);
  else if constexpr (Op == HalfOperation::RSQ)
    value = util::amdgpu_rsq_f32(source, quiet_snan);
  else if constexpr (Op == HalfOperation::SQRT)
    value = util::amdgpu_sqrt_f32(source, quiet_snan);
  else
    value = util::amdgpu_trig_f32(source, Op == HalfOperation::COS, 3, quiet_snan);
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  if (fp16_ovfl && (bits & 0x7fffffffu) == 0x7f800000u &&
      (std::bit_cast<uint32_t>(source) & 0x7fffffffu) < 0x7f800000u)
    return std::bit_cast<float>((bits & 0x80000000u) | 0x477fe000u);
  const uint16_t result = util::f32_to_f16_mode(value, fp16_ovfl);
  return util::f16_to_f32(static_cast<uint16_t>(output_denormal::flush_output<fp_format::F16>(
      uint32_t{result}, output_denormal::Policy::make(denorm_mode))));
}

/// @brief Map a native batch lane by lane with the same F16 result policy.
template <HalfOperation Op>
inline util::native<float> map_f16_simd(util::native<float> source, uint32_t denorm_mode,
                                        bool fp16_ovfl, bool quiet_snan) {
  return util::map_native_convert_scalar<float, float>(
      source, [=](float lane) { return map_f16<Op>(lane, denorm_mode, fp16_ovfl, quiet_snan); });
}

/// @brief Apply the vector half mapping and result stages to a pseudo-scalar op.
/// @details Physical gfx1201 ignores guest rounding for every half transcendental.
/// Half rounding and output-denormal handling precede OMOD, just as for vectors.
/// The caller flushes the source half under MODE before widening it.
inline uint32_t execute_pseudo_f16(pseudo_scalar::Operation operation, float source, bool absolute,
                                   bool negate, [[maybe_unused]] uint32_t round_mode,
                                   uint32_t denorm_mode, uint32_t omod, bool clamp,
                                   bool fp16_ovfl) {
  source = pseudo_scalar::detail::apply_source_modifiers(source, absolute, negate);
  float value = 0.0f;
  switch (operation) {
  case pseudo_scalar::Operation::LOG2:
    value = log_exp_f16<true>(source, denorm_mode, fp16_ovfl, true);
    break;
  case pseudo_scalar::Operation::EXP2:
    value = log_exp_f16<false>(source, denorm_mode, fp16_ovfl, true);
    break;
  case pseudo_scalar::Operation::RCP:
    value = map_f16<HalfOperation::RCP>(source, denorm_mode, fp16_ovfl, true);
    break;
  case pseudo_scalar::Operation::RSQ:
    value = map_f16<HalfOperation::RSQ>(source, denorm_mode, fp16_ovfl, true);
    break;
  case pseudo_scalar::Operation::SQRT:
    value = map_f16<HalfOperation::SQRT>(source, denorm_mode, fp16_ovfl, true);
    break;
  }
  // The mapped value is already a half. TRANS OMOD overflow rounds to nearest,
  // and CLAMP turns a NaN into +0 on every target with these instructions.
  const output_modifier::Policy policy{
      .omod = omod, .clamp = clamp, .clamp_nan_to_zero = true, .fp16_ovfl = fp16_ovfl};
  return output_modifier::apply<fp_format::F16>(uint32_t{util::f32_to_f16(value)}, policy);
}

/// @brief Evaluate single-precision TANH using host std::tanh.
inline float tanh_f32(float x) {
  if (std::isnan(x))
    return std::bit_cast<float>(std::bit_cast<uint32_t>(x) | 0x00400000u);
  return std::tanh(x);
}

/// @brief Evaluate reciprocal using host double division.
inline double rcp_f64(double x) {
  if (std::isnan(x))
    return x;
  if (x == 0.0)
    return std::copysign(std::numeric_limits<double>::infinity(), x);
  if (std::isinf(x))
    return std::copysign(0.0, x);
  return 1.0 / x;
}

/// @brief Evaluate reciprocal square root using host double sqrt/division.
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

/// @brief Evaluate double-precision square root using host std::sqrt.
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
