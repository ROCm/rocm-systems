// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16/FP32 conversion to signed/unsigned normalized 16-bit encodings.
// Owns public entry points and their validation/dispatch policy.

#include "conversion/normalized.h"
#include "common/float_formats.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace {

// Convert one selected FP16/FP32 value to normalized signed/unsigned 16-bit bits, honoring
// input flushing.
template <bool Unsigned, bool Half>
uint16_t normalized(uint32_t raw, uint32_t mode, bool flush_input) {
  if constexpr (Half) {
    uint16_t bits = uint16_t(raw >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
    if (flush_input && (bits & 0x7fff) < 0x400) {
      bits &= 0x8000;
    }
    raw = rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(bits));
  }
  if (mode & RJ_ARITH_ALU_ABS_A) {
    raw &= 0x7fffffff;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    raw ^= 0x80000000;
  }
  double value = rocjitsu::arith::as_float(raw);
  if (std::isnan(value)) {
    return 0;
  }
  constexpr double scale = Unsigned ? 65535 : 32767;
  value = value < (Unsigned ? 0 : -1) ? (Unsigned ? 0 : -1) : value > 1 ? 1 : value;

  // rocjitsu's normalized model uses an exact double product before rounding
  // to nearest-even, avoiding FP32 products that round onto a false tie.
  double product = value * scale;
  double lower = std::floor(product), fraction = product - lower;
  int32_t result = int32_t(lower);
  result += fraction > 0.5 || (fraction == 0.5 && (uint32_t(result) & 1));
  return uint16_t(result);
}

// Convert to normalized 16-bit values for the selected lanes. Reject invalid controls before
// changing outputs.
template <bool Unsigned, rocjitsu::arith::NormalizedForm Form>
int convert(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return convert<Unsigned, Form>(flags, exec_mask, uint32_t(mode), d, source, b);
        });
  }
  constexpr bool unary = Form == rocjitsu::arith::NormalizedForm::Half;
  constexpr bool half = Form != rocjitsu::arith::NormalizedForm::PackedFloat;
  const uint32_t known =
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_CLAMP |
      (half ? RJ_ARITH_ALU_HIGH_A : 0) |
      (unary ? RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_OMOD_HALF
             : RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_B | (half ? RJ_ARITH_ALU_HIGH_B : 0));
  if (int error =
          rocjitsu::arith::validate(flags, mode & ~known, false,
                                    RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                        RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }

  // Flush half inputs before widening; widened subnormals would be normal FP32 values.
  const bool flush_input = half && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::normalized_x86_64_v4<Unsigned, Form>(exec_mask, mode, d[0], a[0],
                                                          unary ? nullptr : b[0], flush_input);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::normalized_x86_64_v3<Unsigned, Form>(exec_mask, mode, d[0], a[0],
                                                          unary ? nullptr : b[0], flush_input);
    return RJ_ARITH_SUCCESS;
  }
#endif
  // Stage results before writing D; unary forms also preserve its unselected halfword.
  uint32_t result[32];
  unsigned shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t low = normalized<Unsigned, half>(a[0][lane], mode, flush_input);
    if constexpr (unary) {
      result[lane] = (d[0][lane] & ~(65535u << shift)) | (low << shift);
    } else {
      result[lane] =
          low | (uint32_t(normalized<Unsigned, half>(b[0][lane], mode >> 1, flush_input)) << 16);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Convert to normalized 16-bit values for Wave64. Reject invalid controls before changing
// outputs.
template <bool Unsigned, rocjitsu::arith::NormalizedForm Form>
int convert_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = convert<Unsigned, Form>(flags, 0, mode, nullptr, nullptr, nullptr)) {
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
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source_a = a[0] + half * 32, *source_b = nullptr;
    if constexpr (Form != rocjitsu::arith::NormalizedForm::Half) {
      source_b = b[0] + half * 32;
    }
    if (int error = convert<Unsigned, Form>(flags, uint32_t(exec_mask >> (half * 32)), half_mode,
                                            &dest, &source_a, &source_b)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Convert F32 to I16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_pk_norm_i16_f32_w32(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<false, rocjitsu::arith::NormalizedForm::PackedFloat>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Convert F32 to U16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_pk_norm_u16_f32_w32(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<true, rocjitsu::arith::NormalizedForm::PackedFloat>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Convert F16 to I16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_norm_i16_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                             uint64_t instruction_flags, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert<false, rocjitsu::arith::NormalizedForm::PackedHalf>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Convert F16 to U16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_norm_u16_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                             uint64_t instruction_flags, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert<true, rocjitsu::arith::NormalizedForm::PackedHalf>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Convert F16 to I16 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_norm_i16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<false, rocjitsu::arith::NormalizedForm::Half>(flags, exec_mask, instruction_flags,
                                                               d, a, nullptr);
}

// Convert F16 to U16 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_norm_u16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<true, rocjitsu::arith::NormalizedForm::Half>(flags, exec_mask, instruction_flags,
                                                              d, a, nullptr);
}

// Convert F32 to I16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_pk_norm_i16_f32_w64(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<false, rocjitsu::arith::NormalizedForm::PackedFloat>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Convert F32 to U16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_pk_norm_u16_f32_w64(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<true, rocjitsu::arith::NormalizedForm::PackedFloat>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Convert F16 to I16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_norm_i16_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                             uint64_t instruction_flags, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert_wave64<false, rocjitsu::arith::NormalizedForm::PackedHalf>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Convert F16 to U16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_norm_u16_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                             uint64_t instruction_flags, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert_wave64<true, rocjitsu::arith::NormalizedForm::PackedHalf>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Convert F16 to I16 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_norm_i16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<false, rocjitsu::arith::NormalizedForm::Half>(
      flags, exec_mask, instruction_flags, d, a, nullptr);
}

// Convert F16 to U16 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_norm_u16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<true, rocjitsu::arith::NormalizedForm::Half>(
      flags, exec_mask, instruction_flags, d, a, nullptr);
}

// Convert F16 to I16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_norm_i16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b,
                                       [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_pk_norm_i16_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Convert F16 to I16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_norm_i16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b,
                                       [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_pk_norm_i16_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Convert F16 to U16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_norm_u16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b,
                                       [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_pk_norm_u16_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Convert F16 to U16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_norm_u16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b,
                                       [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_pk_norm_u16_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}
