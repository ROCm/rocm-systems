// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP8/BF8 widening to FP32 or FP16, including packed pairs.
// Owns public entry points and their validation/dispatch policy.

#include "conversion/fp8_conversion.h"
#include "common/float_encoding.h"
#include "common/validation.h"
#include "register/dpp.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <stdint.h>

namespace {

using rocjitsu::arith::Fp8WidenPolicy;

// Widen FP8/BF8 factors to FP32/FP16 for the selected lanes. Reject invalid controls before
// changing outputs.
template <bool Bf8, bool Packed, bool Half = false>
int convert(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, Fp8WidenPolicy policy = Fp8WidenPolicy::Gfx12) {
  if (mode >> 32) {
    if constexpr (Packed) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    } else {
      return rocjitsu::arith::execute_dpp(
          flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
            return convert<Bf8, Packed, Half>(flags, exec_mask, uint32_t(mode), d, source, policy);
          });
    }
  }
  const uint32_t known = (Packed ? RJ_ARITH_ALU_HIGH_A : RJ_ARITH_CVT_BYTE_3) |
                         (policy == Fp8WidenPolicy::Gfx1250 ? RJ_ARITH_ALU_CLAMP : 0) |
                         (Half ? RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A |
                                     RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP
                               : 0);
  if (int error = rocjitsu::arith::validate(flags, mode & ~known)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  if (policy == Fp8WidenPolicy::Gfx1250) {
    policy = mode & RJ_ARITH_ALU_CLAMP ? Fp8WidenPolicy::E5M3 : Fp8WidenPolicy::Gfx950;
    mode &= ~uint64_t(RJ_ARITH_ALU_CLAMP);
  }
  unsigned shift = Packed ? (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0) : 8 * ((mode >> 16) & 3);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::fp8_conversion_x86_64_v4<Bf8, Packed, Half>(exec_mask, shift, d, a[0], policy,
                                                                 flags, uint32_t(mode));
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::fp8_conversion_x86_64_v3<Bf8, Packed, Half>(exec_mask, shift, d, a[0], policy,
                                                                 flags, uint32_t(mode));
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[Packed && !Half ? 2 : 1][32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t raw = a[0][lane] >> shift;

    // RX 9070 widening canonicalizes either NaN sign to the negative quiet NaN.
    auto widen = [policy](uint8_t code) {
      if (policy == Fp8WidenPolicy::E5M3) {
        if (code == 255) {
          return 0x7fc00000U;
        }
        if (code < 8) {
          return rocjitsu::arith::as_bits(float(code) * 0x1p-17f);
        }
        return (uint32_t(code) << 20) + (112U << 23);
      }
      if (policy == Fp8WidenPolicy::Gfx942) {
        return rocjitsu::arith::as_bits(rocjitsu::arith::fp8_to_float<Bf8, true>(code));
      }
      unsigned magnitude = code & 127;
      if (policy == Fp8WidenPolicy::Gfx12 && (Bf8 ? magnitude > 124 : magnitude == 127)) {
        return 0xffc00000U;
      }
      return rocjitsu::arith::as_bits(rocjitsu::arith::fp8_to_float<Bf8>(code));
    };
    uint32_t low = widen(uint8_t(raw));
    if constexpr (Half) {
      uint32_t value =
          rocjitsu::arith::float_to_f16(rocjitsu::arith::as_float(low), flags & RJ_ARITH_FP16_OVFL);
      if constexpr (Packed) {
        uint32_t high = widen(uint8_t(raw >> 8));
        value |= uint32_t(rocjitsu::arith::float_to_f16(rocjitsu::arith::as_float(high),
                                                        flags & RJ_ARITH_FP16_OVFL))
                 << 16;
      } else {
        int destination_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
        value = (d[0][lane] & ~(0xffffU << destination_shift)) | (value << destination_shift);
      }
      result[0][lane] = value;
    } else {
      result[0][lane] = low;
      if constexpr (Packed) {
        result[1][lane] = widen(uint8_t(raw >> 8));
      }
    }
  }
  for (int reg = 0; reg < (Packed && !Half ? 2 : 1); ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[reg][lane] = result[reg][lane];
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Widen FP8/BF8 factors to FP32/FP16 for Wave64. Reject invalid controls before changing outputs.
template <bool Bf8, bool Packed, bool Half = false>
int convert_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, Fp8WidenPolicy policy = Fp8WidenPolicy::Gfx12) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = convert<Bf8, Packed, Half>(flags, 0, mode, nullptr, nullptr, policy)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags = rocjitsu::arith::dpp_half_wave_instruction_flags(mode, half);
    uint32_t *dest[Packed && !Half ? 2 : 1];
    for (int reg = 0; reg < (Packed && !Half ? 2 : 1); ++reg) {
      dest[reg] = d[reg] + half * 32;
    }
    const uint32_t *source = a[0] + half * 32;
    if (int error = convert<Bf8, Packed, Half>(flags, uint32_t(exec_mask >> (half * 32)),
                                               half_instruction_flags, dest, &source, policy)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Convert FP8 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_fp8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<false, false>(flags, exec_mask, instruction_flags, d, a);
}

// Convert BF8 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_bf8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<true, false>(flags, exec_mask, instruction_flags, d, a);
}

// Convert FP8 to F32 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_pk_f32_fp8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<false, true>(flags, exec_mask, instruction_flags, d, a);
}

// Convert BF8 to F32 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_pk_f32_bf8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<true, true>(flags, exec_mask, instruction_flags, d, a);
}

// Convert FP8 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_fp8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<false, false>(flags, exec_mask, instruction_flags, d, a);
}

// Convert BF8 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_bf8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<true, false>(flags, exec_mask, instruction_flags, d, a);
}

// Convert FP8 to F32 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_pk_f32_fp8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<false, true>(flags, exec_mask, instruction_flags, d, a);
}

// Convert BF8 to F32 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_pk_f32_bf8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<true, true>(flags, exec_mask, instruction_flags, d, a);
}

// Widen with the inherited gfx942 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_f32_fp8_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<false, false>(flags, exec_mask, instruction_flags, d, a, Fp8WidenPolicy::Gfx942);
}

// Widen with the inherited gfx942 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_f32_fp8_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<false, false>(flags, exec_mask, instruction_flags, d, a,
                                      Fp8WidenPolicy::Gfx942);
}

// Widen with the inherited gfx942 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_f32_bf8_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<true, false>(flags, exec_mask, instruction_flags, d, a, Fp8WidenPolicy::Gfx942);
}

// Widen with the inherited gfx942 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_f32_bf8_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<true, false>(flags, exec_mask, instruction_flags, d, a,
                                     Fp8WidenPolicy::Gfx942);
}

// Widen with the inherited gfx942 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_pk_f32_fp8_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<false, true>(flags, exec_mask, instruction_flags, d, a, Fp8WidenPolicy::Gfx942);
}

// Widen with the inherited gfx942 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_pk_f32_fp8_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<false, true>(flags, exec_mask, instruction_flags, d, a,
                                     Fp8WidenPolicy::Gfx942);
}

// Widen with the inherited gfx942 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_pk_f32_bf8_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<true, true>(flags, exec_mask, instruction_flags, d, a, Fp8WidenPolicy::Gfx942);
}

// Widen with the inherited gfx942 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_pk_f32_bf8_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<true, true>(flags, exec_mask, instruction_flags, d, a,
                                    Fp8WidenPolicy::Gfx942);
}

// Widen with the inherited gfx950 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_f32_fp8_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<false, false>(flags, exec_mask, instruction_flags, d, a, Fp8WidenPolicy::Gfx950);
}

// Widen with the inherited gfx950 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_f32_fp8_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<false, false>(flags, exec_mask, instruction_flags, d, a,
                                      Fp8WidenPolicy::Gfx950);
}

// Widen with the inherited gfx950 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_f32_bf8_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<true, false>(flags, exec_mask, instruction_flags, d, a, Fp8WidenPolicy::Gfx950);
}

// Widen with the inherited gfx950 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_f32_bf8_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<true, false>(flags, exec_mask, instruction_flags, d, a,
                                     Fp8WidenPolicy::Gfx950);
}

// Widen with the inherited gfx950 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_pk_f32_fp8_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<false, true>(flags, exec_mask, instruction_flags, d, a, Fp8WidenPolicy::Gfx950);
}

// Widen with the inherited gfx950 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_pk_f32_fp8_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<false, true>(flags, exec_mask, instruction_flags, d, a,
                                     Fp8WidenPolicy::Gfx950);
}

// Widen with the inherited gfx950 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_pk_f32_bf8_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<true, true>(flags, exec_mask, instruction_flags, d, a, Fp8WidenPolicy::Gfx950);
}

// Widen with the inherited gfx950 format/NaN policy, retaining masked destination lanes.
int rj_arith_v_cvt_pk_f32_bf8_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<true, true>(flags, exec_mask, instruction_flags, d, a,
                                    Fp8WidenPolicy::Gfx950);
}

// Widen OCP E4M3FN, or unsigned E5M3 when the format-select CLAMP bit is set.
int rj_arith_v_cvt_f32_fp8_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<false, false>(flags, exec_mask, instruction_flags, d, a, Fp8WidenPolicy::Gfx1250);
}

// Widen OCP E4M3FN, or unsigned E5M3 when the format-select CLAMP bit is set.
int rj_arith_v_cvt_f32_fp8_w64_gfx1250(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<false, false>(flags, exec_mask, instruction_flags, d, a,
                                      Fp8WidenPolicy::Gfx1250);
}

// Decode one selected byte to FP16; preserve inactive lanes; preserve the other destination half.
int rj_arith_v_cvt_f16_fp8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<false, false, true>(flags, exec_mask, instruction_flags, d, a,
                                     Fp8WidenPolicy::Gfx1250);
}

// Decode one selected byte to FP16; preserve inactive lanes; preserve the other destination half.
int rj_arith_v_cvt_f16_fp8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<false, false, true>(flags, exec_mask, instruction_flags, d, a,
                                            Fp8WidenPolicy::Gfx1250);
}

// Decode two selected bytes to FP16; preserve inactive lanes.
int rj_arith_v_cvt_pk_f16_fp8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<false, true, true>(flags, exec_mask, instruction_flags, d, a,
                                    Fp8WidenPolicy::Gfx950);
}

// Decode two selected bytes to FP16; preserve inactive lanes.
int rj_arith_v_cvt_pk_f16_fp8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<false, true, true>(flags, exec_mask, instruction_flags, d, a,
                                           Fp8WidenPolicy::Gfx950);
}

// Decode one selected byte to FP16; preserve inactive lanes; preserve the other destination half.
int rj_arith_v_cvt_f16_bf8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<true, false, true>(flags, exec_mask, instruction_flags, d, a,
                                    Fp8WidenPolicy::Gfx950);
}

// Decode one selected byte to FP16; preserve inactive lanes; preserve the other destination half.
int rj_arith_v_cvt_f16_bf8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<true, false, true>(flags, exec_mask, instruction_flags, d, a,
                                           Fp8WidenPolicy::Gfx950);
}

// Decode two selected bytes to FP16; preserve inactive lanes.
int rj_arith_v_cvt_pk_f16_bf8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<true, true, true>(flags, exec_mask, instruction_flags, d, a,
                                   Fp8WidenPolicy::Gfx950);
}

// Decode two selected bytes to FP16; preserve inactive lanes.
int rj_arith_v_cvt_pk_f16_bf8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<true, true, true>(flags, exec_mask, instruction_flags, d, a,
                                          Fp8WidenPolicy::Gfx950);
}
