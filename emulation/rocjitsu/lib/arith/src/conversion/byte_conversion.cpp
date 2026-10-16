// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Byte-to-FP32 conversion, signed nibble offsets, and FP32-to-byte packing.
// Owns public entry points and their validation/dispatch policy.

#include "conversion/byte_conversion.h"
#include "common/alu.h"
#include "common/internal.h"
#include "conversion/float_conversion.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace {

// Convert selected bytes to FP32 for the selected lanes. Reject invalid controls before
// changing outputs.
template <int Byte>
int convert(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return convert<Byte>(flags, exec_mask, uint32_t(mode), d, source);
        });
  }
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~(RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP), true,
          RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
              RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::byte_conversion_x86_64_v4<Byte>(exec_mask, mode, d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::byte_conversion_x86_64_v3<Byte>(exec_mask, mode, d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  // rocjitsu's execute_v_cvt_f32_ubyte*_vop3 extracts an unsigned byte,
  // converts it exactly, then applies OMOD and CLAMP.
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float value = float((a[0][lane] >> (8 * Byte)) & 255);
    result[lane] = rocjitsu::arith::as_bits(rocjitsu::arith::alu_output(value, mode));
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Convert UBYTE0 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte0_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<0>(flags, exec_mask, instruction_flags, d, a);
}

// Convert UBYTE1 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte1_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<1>(flags, exec_mask, instruction_flags, d, a);
}

// Convert UBYTE2 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte2_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<2>(flags, exec_mask, instruction_flags, d, a);
}

// Convert UBYTE3 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte3_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<3>(flags, exec_mask, instruction_flags, d, a);
}

// Convert I4 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_off_f32_i4_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                        uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return rj_arith_v_cvt_off_f32_i4_w32_gfx12(flags, exec_mask, uint32_t(mode), d, source);
        });
  }
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~(RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP), true,
          RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
              RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::nibble_offset_x86_64_v4(exec_mask, mode, d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::nibble_offset_x86_64_v3(exec_mask, mode, d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    int nibble = int(a[0][lane] & 15);
    if (nibble & 8) {
      nibble -= 16;
    }
    result[lane] =
        rocjitsu::arith::as_bits(rocjitsu::arith::alu_output(float(nibble) * 0.0625f, mode));
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Convert the selected input format and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_u8_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c,
                                       [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return rj_arith_v_cvt_pk_u8_f32_w32_gfx12(flags, exec_mask, uint32_t(mode), d, source, b,
                                                    c, nullptr);
        });
  }
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~(RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_CLAMP), false,
          RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
              RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::byte_pack_x86_64_v4(flags, exec_mask, mode, d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::byte_pack_x86_64_v3(flags, exec_mask, mode, d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = rocjitsu::arith::conversion_detail::byte_pack_policy(flags, mode)(
        a[0][lane], b[0][lane], c[0][lane]);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

namespace {

// Byte 4 denotes the signed-nibble offset conversion.
template <int Byte>
int convert_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a) {
  const auto instruction = Byte == 4 ? rj_arith_v_cvt_off_f32_i4_w32_gfx12 : convert<Byte % 4>;

  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = instruction(flags, 0, mode, nullptr, nullptr)) {
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
    const uint32_t *source = a[0] + half * 32;
    if (int error =
            instruction(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest, &source)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Convert UBYTE0 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte0_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                        uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<0>(flags, exec_mask, mode, d, a);
}

// Convert UBYTE1 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte1_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                        uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<1>(flags, exec_mask, mode, d, a);
}

// Convert UBYTE2 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte2_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                        uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<2>(flags, exec_mask, mode, d, a);
}

// Convert UBYTE3 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte3_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                        uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<3>(flags, exec_mask, mode, d, a);
}

// Convert I4 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_off_f32_i4_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                        uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<4>(flags, exec_mask, mode, d, a);
}

// Convert the selected input format and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_u8_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c,
                                       uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (int error = rj_arith_v_cvt_pk_u8_f32_w32_gfx12(flags, 0, mode, nullptr, nullptr, nullptr,
                                                     nullptr, excp_flag_user)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & RJ_ARITH_DPP16) {
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_mode =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source_a = a[0] + half * 32, *source_b = b[0] + half * 32,
                   *source_c = c[0] + half * 32;
    if (int error = rj_arith_v_cvt_pk_u8_f32_w32_gfx12(flags, uint32_t(exec_mask >> (half * 32)),
                                                       half_mode, &dest, &source_a, &source_b,
                                                       &source_c, excp_flag_user)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Convert UBYTE0 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte0_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return rj_arith_v_cvt_f32_ubyte0_w32_gfx12(flags, exec_mask, mode, d, a);
}

// Convert UBYTE0 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte0_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return rj_arith_v_cvt_f32_ubyte0_w64_gfx12(flags, exec_mask, mode, d, a);
}

// Convert UBYTE1 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte1_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return rj_arith_v_cvt_f32_ubyte1_w32_gfx12(flags, exec_mask, mode, d, a);
}

// Convert UBYTE1 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte1_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return rj_arith_v_cvt_f32_ubyte1_w64_gfx12(flags, exec_mask, mode, d, a);
}

// Convert UBYTE2 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte2_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return rj_arith_v_cvt_f32_ubyte2_w32_gfx12(flags, exec_mask, mode, d, a);
}

// Convert UBYTE2 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte2_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return rj_arith_v_cvt_f32_ubyte2_w64_gfx12(flags, exec_mask, mode, d, a);
}

// Convert UBYTE3 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte3_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return rj_arith_v_cvt_f32_ubyte3_w32_gfx12(flags, exec_mask, mode, d, a);
}

// Convert UBYTE3 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_ubyte3_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return rj_arith_v_cvt_f32_ubyte3_w64_gfx12(flags, exec_mask, mode, d, a);
}

// Convert I4 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_off_f32_i4_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return rj_arith_v_cvt_off_f32_i4_w32_gfx12(flags, exec_mask, mode, d, a);
}

// Convert I4 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_off_f32_i4_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return rj_arith_v_cvt_off_f32_i4_w64_gfx12(flags, exec_mask, mode, d, a);
}

// Convert the selected input format and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_u8_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_pk_u8_f32_w32_gfx12(flags, exec_mask, instruction_flags, d, a, b, c,
                                            nullptr);
}

// Convert the selected input format and pack results with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_pk_u8_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_pk_u8_f32_w64_gfx12(flags, exec_mask, instruction_flags, d, a, b, c,
                                            nullptr);
}
