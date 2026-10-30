// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar and packed sums of absolute differences with integer accumulators.
// Owns public entry points and their validation/dispatch policy.
// SAD and saturation rules adapted from rocjitsu shared/simd_glue.h and
// lib/python/amdisa/codegen/execute/vector_special.py.

#include "common/internal.h"
#include "integer/sad_impl.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Sum unsigned absolute differences of corresponding packed fields; Masked skips zero A fields.
template <int Bits, bool Masked> uint32_t difference(uint32_t a, uint32_t b) {
  const uint32_t field_mask = UINT32_MAX >> (32 - Bits);
  uint32_t sum = 0;
  for (int field = 0; field < 32 / Bits; ++field) {
    uint32_t x = (a >> (field * Bits)) & field_mask;
    uint32_t y = (b >> (field * Bits)) & field_mask;
    if (!Masked || y) {
      sum += x > y ? x - y : y - x;
    }
  }
  return sum;
}

// Accumulate absolute differences for the selected lanes. Reject invalid controls before
// changing outputs.
template <rocjitsu::arith::Sad Op>
int sad(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    if constexpr (rocjitsu::arith::sad_quad(Op)) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    } else {
      return rocjitsu::arith::execute_dpp(
          flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
            return sad<Op>(flags, exec_mask, uint32_t(mode), d, source, b, c);
          });
    }
  }
  if (int error = rocjitsu::arith::validate(flags, mode & ~RJ_ARITH_ALU_CLAMP)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  const bool clamp = mode & RJ_ARITH_ALU_CLAMP;
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::sad_x86_64_v3<Op>(exec_mask, clamp, d, a, b[0], c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  constexpr int outputs = rocjitsu::arith::sad_outputs(Op);
  constexpr int bits = Op == rocjitsu::arith::Sad::U32   ? 32
                       : Op == rocjitsu::arith::Sad::U16 ? 16
                                                         : 8;
  constexpr int windows = rocjitsu::arith::sad_quad(Op) ? 4 : 1;
  uint32_t result[outputs][32] = {};
  for (int lane = 0; lane < 32; ++lane) {
    uint64_t source = a[0][lane];
    if constexpr (rocjitsu::arith::sad_quad(Op)) {
      source |= uint64_t(a[1][lane]) << 32;
    }

    // Quad forms slide a four-byte window across two source VGPRs and pack adjacent accumulators
    // into halfwords.
    for (int window = 0; window < windows; ++window) {
      uint32_t sum = difference<bits, rocjitsu::arith::sad_masked(Op)>(
          uint32_t(source >> (8 * window)), b[0][lane]);
      if constexpr (Op == rocjitsu::arith::Sad::HighU8) {
        sum <<= 16;
      }
      if constexpr (rocjitsu::arith::sad_packed(Op)) {
        const unsigned shift = 16 * (window % 2);
        uint32_t value = sum + ((c[window / 2][lane] >> shift) & 0xffff);
        value = clamp ? rocjitsu::arith::minimum(value, 0xffffU) : value & 0xffff;
        result[window / 2][lane] |= value << shift;
      } else {
        uint64_t value = uint64_t(sum) + c[window][lane];
        result[window][lane] =
            uint32_t(clamp ? rocjitsu::arith::minimum(value, uint64_t(UINT32_MAX)) : value);
      }
    }
  }
  for (int reg = 0; reg < outputs; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[reg][lane] = result[reg][lane];
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Accumulate absolute differences for Wave64. Reject invalid controls before changing outputs.
template <rocjitsu::arith::Sad Op>
int sad_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = sad<Op>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  constexpr int inputs = rocjitsu::arith::sad_quad(Op) ? 2 : 1;
  constexpr int outputs = rocjitsu::arith::sad_outputs(Op);
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_instruction_flags =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    const uint32_t *source_a[inputs], *source_b = b[0] + half * 32, *source_c[outputs];
    uint32_t *dest[outputs];
    for (int reg = 0; reg < inputs; ++reg) {
      source_a[reg] = a[reg] + half * 32;
    }
    for (int reg = 0; reg < outputs; ++reg) {
      source_c[reg] = c[reg] + half * 32;
      dest[reg] = d[reg] + half * 32;
    }
    if (int error = sad<Op>(flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags, dest,
                            source_a, &source_b, source_c)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Accumulate absolute differences for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sad_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad<rocjitsu::arith::Sad::U8>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate absolute differences for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sad_hi_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad<rocjitsu::arith::Sad::HighU8>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate absolute differences for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sad_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad<rocjitsu::arith::Sad::U16>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate absolute differences for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sad_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad<rocjitsu::arith::Sad::U32>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate masked byte absolute differences for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_msad_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad<rocjitsu::arith::Sad::MaskedU8>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate quad absolute differences with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_qsad_pk_u16_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad<rocjitsu::arith::Sad::QuadU16>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate masked quad absolute differences with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mqsad_pk_u16_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad<rocjitsu::arith::Sad::MaskedQuadU16>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate masked quad absolute differences for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mqsad_u32_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad<rocjitsu::arith::Sad::MaskedQuadU32>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate absolute differences for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sad_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad_wave64<rocjitsu::arith::Sad::U8>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate absolute differences for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sad_hi_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad_wave64<rocjitsu::arith::Sad::HighU8>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate absolute differences for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sad_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad_wave64<rocjitsu::arith::Sad::U16>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate absolute differences for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sad_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad_wave64<rocjitsu::arith::Sad::U32>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate masked byte absolute differences for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_msad_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad_wave64<rocjitsu::arith::Sad::MaskedU8>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate quad absolute differences with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_qsad_pk_u16_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad_wave64<rocjitsu::arith::Sad::QuadU16>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate masked quad absolute differences with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mqsad_pk_u16_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad_wave64<rocjitsu::arith::Sad::MaskedQuadU16>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate masked quad absolute differences for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mqsad_u32_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return sad_wave64<rocjitsu::arith::Sad::MaskedQuadU32>(flags, exec_mask, mode, d, a, b, c);
}
