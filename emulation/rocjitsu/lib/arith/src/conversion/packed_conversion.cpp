// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Pairwise FP32 packing into FP16/BF16 with format-specific rounding.
// Owns public entry points and their validation/dispatch policy.

#include "conversion/packed_conversion.h"
#include "common/half_conversion.h"
#include "common/internal.h"
#include "conversion/float_conversion.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace {

// Apply source modifiers and convert FP32 to the selected packed result format.
template <rocjitsu::arith::PackedConversion Op>
uint16_t narrow(uint32_t raw, uint32_t mode, uint64_t flags) {
  if (mode & RJ_ARITH_ALU_ABS_A) {
    raw &= 0x7fffffff;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    raw ^= 0x80000000;
  }
  if constexpr (Op == rocjitsu::arith::PackedConversion::HalfRtz) {
    return rocjitsu::arith::conversion_detail::float_policy<
        rocjitsu::arith::conversion_detail::fmt::F32, rocjitsu::arith::conversion_detail::fmt::F16>(
        (flags & ~RJ_ARITH_FP_ROUND_MASK) | RJ_ARITH_FP_ROUND_ZERO, 0)(raw);
  } else {
    float value = rocjitsu::arith::as_float(raw);
    if constexpr (Op == rocjitsu::arith::PackedConversion::Unsigned) {
      return !(value > 0) ? 0 : value >= 65535 ? 65535 : uint16_t(value);
    } else {
      return std::isnan(value) ? 0
             : value >= 32767  ? 32767
             : value <= -32768 ? 32768
                               : uint16_t(int16_t(value));
    }
  }
}

// Pack two FP32 conversions into one word for the selected lanes. Reject invalid controls
// before changing outputs.
template <rocjitsu::arith::PackedConversion Op>
int convert(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return convert<Op>(flags, exec_mask, uint32_t(mode), d, source, b);
        });
  }
  const uint32_t known =
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B |
      RJ_ARITH_ALU_CLAMP |
      (Op == rocjitsu::arith::PackedConversion::HalfRtz ? RJ_ARITH_ALU_OMOD_HALF : 0);
  if (int error =
          rocjitsu::arith::validate(flags, mode & ~known, false,
                                    RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                        RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::packed_conversion_x86_64_v4<Op>(flags, exec_mask, mode, d[0], a[0], b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::packed_conversion_x86_64_v3<Op>(flags, exec_mask, mode, d[0], a[0], b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = narrow<Op>(a[0][lane], mode, flags) |
                   (uint32_t(narrow<Op>(b[0][lane], mode >> 1, flags)) << 16);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Pack two FP32 conversions into one word for Wave64. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::PackedConversion Op>
int convert_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = convert<Op>(flags, 0, mode, nullptr, nullptr, nullptr)) {
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
    const uint32_t *source_a = a[0] + half * 32, *source_b = b[0] + half * 32;
    if (int error = convert<Op>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                &source_a, &source_b)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Convert F32 to F16 and pack results with packed elements, round-to-zero conversion for
// selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_rtz_f16_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
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
  return convert<rocjitsu::arith::PackedConversion::HalfRtz>(flags, exec_mask, instruction_flags, d,
                                                             a, b);
}

// Convert F32 to I16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_i16_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
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
  return convert<rocjitsu::arith::PackedConversion::Signed>(flags, exec_mask, instruction_flags, d,
                                                            a, b);
}

// Convert F32 to U16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_u16_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
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
  return convert<rocjitsu::arith::PackedConversion::Unsigned>(flags, exec_mask, instruction_flags,
                                                              d, a, b);
}

// Convert F32 to F16 and pack results with packed elements, round-to-zero conversion for
// selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_rtz_f16_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
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
  return convert_wave64<rocjitsu::arith::PackedConversion::HalfRtz>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Convert F32 to I16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_i16_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
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
  return convert_wave64<rocjitsu::arith::PackedConversion::Signed>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Convert F32 to U16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_u16_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
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
  return convert_wave64<rocjitsu::arith::PackedConversion::Unsigned>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Convert F32 to F16 and pack results with packed elements, round-to-zero conversion for
// selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_rtz_f16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
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
  return rj_arith_v_cvt_pk_rtz_f16_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Convert F32 to F16 and pack results with packed elements, round-to-zero conversion for
// selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_rtz_f16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
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
  return rj_arith_v_cvt_pk_rtz_f16_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Convert F32 to I16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_i16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_pk_i16_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Convert F32 to I16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_i16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_pk_i16_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Convert F32 to U16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_u16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_pk_u16_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Convert F32 to U16 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_u16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_pk_u16_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}
