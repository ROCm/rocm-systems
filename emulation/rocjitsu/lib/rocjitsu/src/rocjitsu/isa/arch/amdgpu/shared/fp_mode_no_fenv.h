// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file fp_mode_no_fenv.h
/// @brief The MODE-aware helpers that do not touch the host FP environment.
///
/// @details These are split out of fp_mode.h so that a translation unit which
/// only needs, say, effective_omod does not have to opt in to the strict
/// rounding-mode contract fp_mode.h requires. Everything here is integer or
/// bit-pattern work: it neither reads nor writes the host rounding mode, so it
/// is correct under any compiler floating-point settings. Anything that reaches
/// for detail::ScopedFenv belongs in fp_mode.h instead.

#include "rocjitsu/code/rj_code.h"

#include <bit>
#include <cstdint>

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

} // namespace rocjitsu::amdgpu::fp_mode
