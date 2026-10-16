// SPDX-License-Identifier: MIT

// FP8/BF8 widening to FP32, including packed pairs.
// Owns public entry points and their validation/dispatch policy.

#include "conversion/fp8_conversion.h"
#include "common/float_formats.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Widen FP8/BF8 factors to FP32 for the selected lanes. Reject invalid controls before
// changing outputs.
template <bool Bf8, bool Packed>
int convert(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a) {
  if (mode >> 32) {
    if constexpr (Packed) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    } else {
      return rocjitsu::arith::execute_dpp(
          flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
            return convert<Bf8, Packed>(flags, exec_mask, uint32_t(mode), d, source);
          });
    }
  }
  const uint32_t known = Packed ? RJ_ARITH_ALU_HIGH_A : RJ_ARITH_CVT_BYTE_3;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known, true)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  unsigned shift = Packed ? (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0) : 8 * (mode >> 16);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::fp8_conversion_x86_64_v4<Bf8, Packed>(exec_mask, shift, d, a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::fp8_conversion_x86_64_v3<Bf8, Packed>(exec_mask, shift, d, a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[Packed ? 2 : 1][32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t raw = a[0][lane] >> shift;

    // RX 9070 widening canonicalizes either NaN sign to the negative quiet NaN.
    auto widen = [](uint8_t code) {
      unsigned magnitude = code & 127;
      if (Bf8 ? magnitude > 124 : magnitude == 127) {
        return 0xffc00000U;
      }
      return rocjitsu::arith::as_bits(rocjitsu::arith::fp8_to_float<Bf8>(code));
    };
    result[0][lane] = widen(uint8_t(raw));
    if constexpr (Packed) {
      result[1][lane] = widen(uint8_t(raw >> 8));
    }
  }
  for (int reg = 0; reg < (Packed ? 2 : 1); ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[reg][lane] = result[reg][lane];
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Widen FP8/BF8 factors to FP32 for Wave64. Reject invalid controls before changing outputs.
template <bool Bf8, bool Packed>
int convert_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = convert<Bf8, Packed>(flags, 0, mode, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_mode =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest[Packed ? 2 : 1];
    for (int reg = 0; reg < (Packed ? 2 : 1); ++reg) {
      dest[reg] = d[reg] + half * 32;
    }
    const uint32_t *source = a[0] + half * 32;
    if (int error = convert<Bf8, Packed>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, dest,
                                         &source)) {
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
