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

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <xmmintrin.h>
#endif

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
  explicit ScopedFenv(uint32_t round_mode) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    saved_mxcsr_ = _mm_getcsr();
#endif
    saved_ = std::feholdexcept(&environment_) == 0;
    if (saved_)
      std::fesetround(host_round_mode(round_mode));
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    constexpr uint32_t kDazMask = 1u << 6;
    constexpr uint32_t kFtzMask = 1u << 15;
    _mm_setcsr(_mm_getcsr() & ~(kDazMask | kFtzMask));
#endif
  }

  ScopedFenv(const ScopedFenv &) = delete;
  ScopedFenv &operator=(const ScopedFenv &) = delete;

  ~ScopedFenv() {
    if (saved_)
      std::fesetenv(&environment_);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    _mm_setcsr(saved_mxcsr_);
#endif
  }

private:
  std::fenv_t environment_{};
  bool saved_ = false;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  uint32_t saved_mxcsr_ = 0;
#endif
};

struct ExactF64Sum {
  double value;
  double error;
};

/// @brief Add two finite F64 values while retaining the exact rounding residual.
inline ExactF64Sum add_exact(double lhs, double rhs) {
  const double value = lhs + rhs;
  const double rhs_virtual = value - lhs;
  const double error = (lhs - (value - rhs_virtual)) + (rhs - rhs_virtual);
  return {value, error};
}

/// @brief Compare an exact two-component sum with an exactly representable value.
inline int compare_exact(ExactF64Sum sum, double value) {
  if (sum.value < value)
    return -1;
  if (sum.value > value)
    return 1;
  return sum.error < 0.0 ? -1 : sum.error > 0.0 ? 1 : 0;
}

inline uint16_t next_up_bf16(uint16_t value) {
  if ((value & 0x7fffu) > 0x7f80u || value == 0x7f80u)
    return value;
  if (value == 0xff80u)
    return 0xff7fu;
  if ((value & 0x7fffu) == 0)
    return 0x0001u;
  return static_cast<uint16_t>((value & 0x8000u) != 0 ? value - 1u : value + 1u);
}

inline uint16_t next_down_bf16(uint16_t value) {
  if ((value & 0x7fffu) > 0x7f80u || value == 0xff80u)
    return value;
  if (value == 0x7f80u)
    return 0x7f7fu;
  if ((value & 0x7fffu) == 0)
    return 0x8001u;
  return static_cast<uint16_t>((value & 0x8000u) != 0 ? value + 1u : value - 1u);
}

inline uint16_t round_exact_to_bf16(ExactF64Sum sum, uint32_t round_mode) {
  constexpr double kMaxBf16 = 0x1.fep127;
  constexpr double kRneOverflowThreshold = 0x1.ffp127;
  const int zero_cmp = compare_exact(sum, 0.0);

  if (compare_exact(sum, kMaxBf16) > 0) {
    if ((round_mode & 3u) == 1u ||
        ((round_mode & 3u) == 0u && compare_exact(sum, kRneOverflowThreshold) >= 0))
      return 0x7f80u;
    return 0x7f7fu;
  }
  if (compare_exact(sum, -kMaxBf16) < 0) {
    if ((round_mode & 3u) == 2u ||
        ((round_mode & 3u) == 0u && compare_exact(sum, -kRneOverflowThreshold) <= 0))
      return 0xff80u;
    return 0xff7fu;
  }

  const float approximation = static_cast<float>(sum.value);
  const uint16_t candidate = util::f32_to_bf16_rne(approximation);
  const double candidate_value = util::bf16_to_f32(candidate);
  const int candidate_cmp = compare_exact(sum, candidate_value);
  if (candidate_cmp == 0)
    return candidate;

  const uint16_t lower = candidate_cmp > 0 ? candidate : next_down_bf16(candidate);
  const uint16_t upper = candidate_cmp < 0 ? candidate : next_up_bf16(candidate);
  switch (round_mode & 3u) {
  case 0: {
    const double midpoint =
        (static_cast<double>(util::bf16_to_f32(lower)) + util::bf16_to_f32(upper)) * 0.5;
    const int midpoint_cmp = compare_exact(sum, midpoint);
    if (midpoint_cmp < 0)
      return lower;
    if (midpoint_cmp > 0)
      return upper;
    return (lower & 1u) == 0 ? lower : upper;
  }
  case 1:
    return upper;
  case 2:
    return lower;
  case 3:
    return zero_cmp < 0 ? upper : lower;
  default:
    return candidate;
  }
}

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

namespace detail {

/// @brief Execute F32-source fused multiply-add in an established clean nearest environment.
inline uint16_t fma_f32_to_bf16_nearest_environment(float multiplicand, float multiplier,
                                                    float addend, uint32_t round_mode, bool clamp,
                                                    bool clamp_nan_to_zero) {
  if (!std::isfinite(multiplicand) || !std::isfinite(multiplier) || !std::isfinite(addend)) {
    float result = std::fma(multiplicand, multiplier, addend);
    if (clamp) {
      if ((clamp_nan_to_zero && std::isnan(result)) || result <= 0.0f)
        result = 0.0f;
      else if (result > 1.0f)
        result = 1.0f;
    }
    return util::f32_to_bf16_round(result, round_mode);
  }

  const double product = static_cast<double>(multiplicand) * static_cast<double>(multiplier);
  const detail::ExactF64Sum exact = detail::add_exact(product, static_cast<double>(addend));
  if (exact.value == 0.0 && exact.error == 0.0) {
    if (clamp)
      return 0;
    detail::ScopedFenv result_environment(round_mode);
    return util::f32_to_bf16_round(std::fma(multiplicand, multiplier, addend), round_mode);
  }
  if (clamp) {
    if (detail::compare_exact(exact, 0.0) <= 0)
      return 0;
    if (detail::compare_exact(exact, 1.0) > 0)
      return 0x3f80u;
  }
  return detail::round_exact_to_bf16(exact, round_mode);
}

} // namespace detail

/// @brief Execute an F32-source fused multiply-add and round once to BF16.
/// @details An F32 product is exact in F64. The error-free sum retains the exact addend residual,
/// which is needed when the rounded F64 value lands on a BF16 boundary or midpoint.
inline uint16_t fma_f32_to_bf16(float multiplicand, float multiplier, float addend,
                                uint32_t round_mode, bool clamp, bool clamp_nan_to_zero) {
  detail::ScopedFenv nearest_environment(0);
  return detail::fma_f32_to_bf16_nearest_environment(multiplicand, multiplier, addend, round_mode,
                                                     clamp, clamp_nan_to_zero);
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
