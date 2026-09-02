// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file fp_mode.h
/// @brief Shared MODE-aware floating-point execution helpers.

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/pseudo_scalar.h"
#include "util/data_types.h"

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rocjitsu::amdgpu::fp_mode {

namespace detail {

inline uint16_t modify_f16(uint16_t value, bool absolute, bool negate) {
  if (absolute)
    value &= 0x7fffu;
  if (negate)
    value ^= 0x8000u;
  return value;
}

inline uint16_t flush_input_f16(uint16_t value, uint32_t denorm_mode) {
  if ((denorm_mode & 1u) == 0 && (value & 0x7c00u) == 0 && (value & 0x03ffu) != 0)
    return value & 0x8000u;
  return value;
}

inline uint64_t flush_f64(uint64_t value) {
  if ((value & 0x7ff0000000000000ULL) == 0 && (value & 0x000fffffffffffffULL) != 0)
    return value & 0x8000000000000000ULL;
  return value;
}

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

/// @brief Return the OMOD value supported by an ordinary floating-point result.
inline uint32_t effective_omod(rj_code_arch_t arch, uint32_t denorm_mode, bool ieee_mode,
                               uint32_t omod) {
  if (omod == 0)
    return 0;
  if (arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5)
    return omod;
  return (denorm_mode & 2u) == 0 && !ieee_mode ? omod : 0;
}

/// @brief Return the OMOD value that applies to an F16 result on the selected ISA.
/// @details GFX11+ packed-F16 results explicitly ignore OMOD. Older profiles expose
/// OMOD on the promoted VOP3 form of V_PK_FMAC_F16, subject to their ordinary
/// output-denormal and MODE.IEEE restrictions. GFX12 and gfx1250 allow OMOD on
/// non-packed F16 results regardless of output-denormal mode.
inline uint32_t effective_f16_omod(rj_code_arch_t arch, uint32_t denorm_mode, bool ieee_mode,
                                   bool packed_result, uint32_t omod) {
  if (omod == 0)
    return 0;
  if (packed_result && (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
                        arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5))
    return 0;
  return effective_omod(arch, denorm_mode, ieee_mode, omod);
}

/// @brief Apply the result-format rules required by an active OMOD.
/// @details OMOD always flushes an output subnormal and maps either signed zero
/// to positive zero. These helpers operate after the result has been rounded to
/// its architectural destination format.
inline uint16_t finalize_omod_f16(uint16_t value, uint32_t omod) {
  if (omod == 0)
    return value;
  if ((value & 0x7c00u) == 0 && (value & 0x03ffu) != 0)
    value &= 0x8000u;
  return (value & 0x7fffu) == 0 ? 0 : value;
}

inline uint16_t finalize_omod_bf16(uint16_t value, uint32_t omod) {
  if (omod == 0)
    return value;
  if ((value & 0x7f80u) == 0 && (value & 0x007fu) != 0)
    value &= 0x8000u;
  return (value & 0x7fffu) == 0 ? 0 : value;
}

inline float finalize_omod_f32(float value, uint32_t omod) {
  if (omod == 0)
    return value;
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0)
    bits &= 0x80000000u;
  if ((bits & 0x7fffffffu) == 0)
    bits = 0;
  return std::bit_cast<float>(bits);
}

inline double finalize_omod_f64(double value, uint32_t omod) {
  if (omod == 0)
    return value;
  uint64_t bits = detail::flush_f64(std::bit_cast<uint64_t>(value));
  if ((bits & 0x7fffffffffffffffULL) == 0)
    bits = 0;
  return std::bit_cast<double>(bits);
}

/// @brief Execute an F16 fused multiply-add and return its raw F16 encoding.
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

/// @brief Binary F64 operations implemented by the shared MODE-aware helper.
enum class BinaryF64Op { Add, Multiply, MaximumNumber, MinimumNumber };

/// @brief Execute a binary F64 operation under MODE.FP_ROUND and MODE.FP_DENORM.
/// @details MaximumNumber and MinimumNumber return the numeric operand for a
/// one-NaN input and explicitly select +0/-0 for signed-zero ties respectively.
inline uint64_t binary_f64(uint64_t src0, uint64_t src1, BinaryF64Op operation, uint32_t round_mode,
                           uint32_t denorm_mode) {
  if ((denorm_mode & 1u) == 0) {
    src0 = detail::flush_f64(src0);
    src1 = detail::flush_f64(src1);
  }

  uint64_t result;
  {
    detail::ScopedFenv environment(round_mode);
    const double lhs = std::bit_cast<double>(src0);
    const double rhs = std::bit_cast<double>(src1);
    const double value = [&] {
      switch (operation) {
      case BinaryF64Op::Add:
        return lhs + rhs;
      case BinaryF64Op::Multiply:
        return lhs * rhs;
      case BinaryF64Op::MaximumNumber:
        if (std::isnan(lhs))
          return std::isnan(rhs) ? std::numeric_limits<double>::quiet_NaN() : rhs;
        if (std::isnan(rhs))
          return lhs;
        if ((src0 & 0x7fffffffffffffffULL) == 0 && (src1 & 0x7fffffffffffffffULL) == 0)
          return std::bit_cast<double>((src0 & src1) & 0x8000000000000000ULL);
        return std::fmax(lhs, rhs);
      case BinaryF64Op::MinimumNumber:
        if (std::isnan(lhs))
          return std::isnan(rhs) ? std::numeric_limits<double>::quiet_NaN() : rhs;
        if (std::isnan(rhs))
          return lhs;
        if ((src0 & 0x7fffffffffffffffULL) == 0 && (src1 & 0x7fffffffffffffffULL) == 0)
          return std::bit_cast<double>((src0 | src1) & 0x8000000000000000ULL);
        return std::fmin(lhs, rhs);
      }
      return std::numeric_limits<double>::quiet_NaN();
    }();
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
