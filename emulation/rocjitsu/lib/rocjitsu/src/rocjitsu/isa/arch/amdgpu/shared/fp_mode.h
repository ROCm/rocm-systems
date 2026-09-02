// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file fp_mode.h
/// @brief Shared MODE-aware floating-point execution helpers that run the
/// arithmetic under the guest's MODE.FP_ROUND.
///
/// @details Everything here delegates rounding to the host FPU: an operation is
/// wrapped in a detail::ScopedFenv, which calls fesetround() and restores the
/// previous environment on the way out. That is only correct if the compiler
/// has been told the rounding mode is dynamic; otherwise it is entitled to
/// assume round-to-nearest and move the arithmetic across the fesetround call,
/// which silently returns the round-to-nearest answer under every directed
/// mode. See RJ_STRICT_FP_ROUNDING_OPTIONS in emulation/rocjitsu/CMakeLists.txt.
///
/// Helpers that do not touch the host FP environment live in fp_mode_no_fenv.h,
/// which this header pulls in, and carry no such requirement.

// The definition travels with the flag in RJ_STRICT_FP_ROUNDING_OPTIONS, so a
// target that has not opted in fails here. Because these helpers are `inline`,
// the alternative to failing is worse than an unflagged call site: the linker
// picks one of the emitted copies, so a single unflagged translation unit can
// supply the definition the whole image uses and reintroduce the miscompile
// everywhere. Include fp_mode_no_fenv.h instead if the fenv-free helpers are
// all that is needed.
#ifndef ROCJITSU_STRICT_FP_ROUNDING
#error                                                                                             \
    "fp_mode.h requires the strict rounding-mode build options (RJ_STRICT_FP_ROUNDING_OPTIONS); include fp_mode_no_fenv.h for the helpers that do not use the host FP environment"
#endif

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode_no_fenv.h"
#include "rocjitsu/isa/arch/amdgpu/shared/pseudo_scalar.h"
#include "util/data_types.h"

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>

namespace rocjitsu::amdgpu::fp_mode {

namespace detail {

inline int host_round_mode(uint32_t round_mode) {
  switch (round_mode & 3u) {
  case 1:
    return FE_UPWARD;
  case 2:
    return FE_DOWNWARD;
  case 3:
    return FE_TOWARDZERO;
  default:
    return FE_TONEAREST;
  }
}

class ScopedFenv {
public:
  explicit ScopedFenv(uint32_t round_mode) : saved_(std::feholdexcept(&environment_) == 0) {
    if (saved_)
      std::fesetround(host_round_mode(round_mode));
  }

  ScopedFenv(const ScopedFenv &) = delete;
  ScopedFenv &operator=(const ScopedFenv &) = delete;

  ~ScopedFenv() {
    if (saved_)
      std::fesetenv(&environment_);
  }

private:
  std::fenv_t environment_{};
  bool saved_;
};

} // namespace detail

/// @brief Execute an F16 fused multiply-add and return its raw F16 encoding.
/// @details The intermediate is computed in double and rounded to F16 in
/// software by pseudo_scalar, but the double fma is still subject to whatever
/// rounding mode is in effect when this runs, so it belongs on this side of the
/// split.
inline uint16_t fma_f16(uint16_t src0, uint16_t src1, uint16_t src2, bool abs0, bool abs1,
                        bool abs2, bool neg0, bool neg1, bool neg2, uint32_t round_mode,
                        uint32_t denorm_mode, uint32_t omod, bool clamp, bool fp16_ovfl,
                        bool clamp_nan_to_zero) {
  src0 = detail::flush_input_f16(detail::modify_f16(src0, abs0, neg0), denorm_mode);
  src1 = detail::flush_input_f16(detail::modify_f16(src1, abs1, neg1), denorm_mode);
  src2 = detail::flush_input_f16(detail::modify_f16(src2, abs2, neg2), denorm_mode);

  const double multiplicand = static_cast<double>(util::f16_to_f32(src0));
  const double multiplier = static_cast<double>(util::f16_to_f32(src1));
  const double addend = static_cast<double>(util::f16_to_f32(src2));
  uint16_t result =
      pseudo_scalar::round_f16_result(std::fma(multiplicand, multiplier, addend), round_mode, omod,
                                      clamp, fp16_ovfl, clamp_nan_to_zero);
  if ((denorm_mode & 2u) == 0 && (result & 0x7c00u) == 0 && (result & 0x03ffu) != 0)
    result &= 0x8000u;
  return finalize_omod_f16(result, omod);
}

/// @brief Execute an F64 fused multiply-add under MODE.FP_ROUND and MODE.FP_DENORM.
inline uint64_t fma_f64(uint64_t src0, uint64_t src1, uint64_t src2, uint32_t round_mode,
                        uint32_t denorm_mode) {
  if ((denorm_mode & 1u) == 0) {
    src0 = detail::flush_f64(src0);
    src1 = detail::flush_f64(src1);
    src2 = detail::flush_f64(src2);
  }

  uint64_t result;
  {
    detail::ScopedFenv environment(round_mode);
    const double value = std::fma(std::bit_cast<double>(src0), std::bit_cast<double>(src1),
                                  std::bit_cast<double>(src2));
    result = std::bit_cast<uint64_t>(value);
  }
  if ((denorm_mode & 2u) == 0)
    result = detail::flush_f64(result);
  return result;
}

/// @brief Apply F64 OMOD/CLAMP under the architectural rounding mode.
/// @details A nonzero OMOD flushes a denormal result and converts either signed zero to +0.
inline uint64_t finish_f64(uint64_t value, uint32_t round_mode, uint32_t omod, bool clamp,
                           bool clamp_nan_to_zero) {
  double result;
  {
    detail::ScopedFenv environment(round_mode);
    result = std::bit_cast<double>(value);
    if (omod == 1)
      result *= 2.0;
    else if (omod == 2)
      result *= 4.0;
    else if (omod == 3)
      result *= 0.5;
    if (clamp) {
      if ((clamp_nan_to_zero && std::isnan(result)) || result <= 0.0)
        result = 0.0;
      else if (result > 1.0)
        result = 1.0;
    }
  }
  return std::bit_cast<uint64_t>(finalize_omod_f64(result, omod));
}

/// @brief Scale an exact unsigned 53-bit significand using round-toward-zero.
/// @details The host floating-point environment is restored before returning.
inline uint64_t scale_u53_f64_rtz(uint64_t significand, int exponent) {
  detail::ScopedFenv environment(3);
  return std::bit_cast<uint64_t>(std::ldexp(static_cast<double>(significand), exponent));
}

} // namespace rocjitsu::amdgpu::fp_mode
