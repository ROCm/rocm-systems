// SPDX-License-Identifier: MIT

// 16-bit multiply-add and three-input integer selections.
// Owns public entry points and their validation/dispatch policy.

#include "integer/integer16_ternary.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <stdint.h>

namespace {

// Extract the selected halfword and widen it with the requested signedness.
template <bool Signed> int64_t input(uint32_t word, int half) {
  int64_t value = uint16_t(word >> (16 * half));
  if constexpr (Signed) {
    value -= (value & 0x8000) ? 65536 : 0;
  }
  return value;
}

// Evaluate three-input 16-bit integer arithmetic for the selected lanes. Reject invalid
// controls before changing outputs.
template <bool Signed,
          rocjitsu::arith::Integer16Ternary Op = rocjitsu::arith::Integer16Ternary::Mad,
          bool Packed = true>
int ternary(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    if constexpr (!Packed) {
      return rocjitsu::arith::execute_dpp(
          flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
            return ternary<Signed, Op, Packed>(flags, exec_mask, uint32_t(mode), d, source, b, c);
          });
    } else {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }

  const uint32_t known = Packed
                             ? RJ_ARITH_PK_LO_A_HIGH | RJ_ARITH_PK_LO_B_HIGH |
                                   RJ_ARITH_PK_LO_C_HIGH | RJ_ARITH_PK_HI_A_LOW |
                                   RJ_ARITH_PK_HI_B_LOW | RJ_ARITH_PK_HI_C_LOW | RJ_ARITH_PK_CLAMP
                             : RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_C |
                                   RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_CLAMP;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known, true)) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::integer16_ternary_x86_64_v3<Signed, Op, Packed>(exec_mask, mode, d[0], a[0],
                                                                     b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  // Translate scalar halfword selectors into the packed selector layout so both forms share the
  // arithmetic loop.
  uint32_t arithmetic_mode = mode;
  if constexpr (!Packed) {
    arithmetic_mode = (mode & RJ_ARITH_ALU_HIGH_A ? RJ_ARITH_PK_LO_A_HIGH : 0) |
                      (mode & RJ_ARITH_ALU_HIGH_B ? RJ_ARITH_PK_LO_B_HIGH : 0) |
                      (mode & RJ_ARITH_ALU_HIGH_C ? RJ_ARITH_PK_LO_C_HIGH : 0) |
                      (mode & RJ_ARITH_ALU_CLAMP ? RJ_ARITH_PK_CLAMP : 0);
  }
  uint32_t result[32] = {};
  for (int lane = 0; lane < 32; ++lane) {
    for (int half = 0; half < (Packed ? 2 : 1); ++half) {
      int sa = half ^ bool(arithmetic_mode & (half ? RJ_ARITH_PK_HI_A_LOW : RJ_ARITH_PK_LO_A_HIGH));
      int sb = half ^ bool(arithmetic_mode & (half ? RJ_ARITH_PK_HI_B_LOW : RJ_ARITH_PK_LO_B_HIGH));
      int sc = half ^ bool(arithmetic_mode & (half ? RJ_ARITH_PK_HI_C_LOW : RJ_ARITH_PK_LO_C_HIGH));
      int64_t x = input<Signed>(a[0][lane], sa), y = input<Signed>(b[0][lane], sb),
              z = input<Signed>(c[0][lane], sc), value;
      if constexpr (Op == rocjitsu::arith::Integer16Ternary::Mad) {
        value = x * y + z;
        if (arithmetic_mode & RJ_ARITH_PK_CLAMP) {
          value = std::clamp(value, Signed ? INT64_C(-32768) : INT64_C(0),
                             Signed ? INT64_C(32767) : INT64_C(65535));
        }
      } else if constexpr (Op == rocjitsu::arith::Integer16Ternary::Min) {
        value = std::min(std::min(x, y), z);
      } else if constexpr (Op == rocjitsu::arith::Integer16Ternary::Max) {
        value = std::max(std::max(x, y), z);
      } else {
        value = std::max(std::min(x, y), std::min(std::max(x, y), z));
      }
      result[lane] |= uint32_t(uint16_t(value)) << (16 * half);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if (exec_mask >> lane & 1) {
      if constexpr (Packed) {
        d[0][lane] = result[lane];
      } else {
        int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
        d[0][lane] = (d[0][lane] & ~(65535U << shift)) | (result[lane] << shift);
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate three-input 16-bit integer arithmetic for Wave64. Reject invalid controls before
// changing outputs.
template <bool Signed,
          rocjitsu::arith::Integer16Ternary Op = rocjitsu::arith::Integer16Ternary::Mad,
          bool Packed = true>
int ternary_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = ternary<Signed, Op, Packed>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
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
    const uint32_t *source_c = c[0] + half * 32;
    if (int error =
            ternary<Signed, Op, Packed>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                        &source_a, &source_b, &source_c)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Multiply-add integer operands with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_mad_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return ternary<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Multiply-add integer operands with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_mad_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return ternary<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<false, rocjitsu::arith::Integer16Ternary::Mad, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<true, rocjitsu::arith::Integer16Ternary::Mad, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the minimum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min3_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<false, rocjitsu::arith::Integer16Ternary::Min, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the minimum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min3_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<true, rocjitsu::arith::Integer16Ternary::Min, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the maximum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max3_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<false, rocjitsu::arith::Integer16Ternary::Max, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the maximum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max3_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<true, rocjitsu::arith::Integer16Ternary::Max, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the median of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_med3_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<false, rocjitsu::arith::Integer16Ternary::Median, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the median of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_med3_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<true, rocjitsu::arith::Integer16Ternary::Median, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Multiply-add integer operands with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_mad_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return ternary_wave64<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Multiply-add integer operands with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_mad_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return ternary_wave64<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<false, rocjitsu::arith::Integer16Ternary::Mad, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<true, rocjitsu::arith::Integer16Ternary::Mad, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the minimum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min3_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<false, rocjitsu::arith::Integer16Ternary::Min, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the minimum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min3_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<true, rocjitsu::arith::Integer16Ternary::Min, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the maximum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max3_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<false, rocjitsu::arith::Integer16Ternary::Max, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the maximum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max3_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<true, rocjitsu::arith::Integer16Ternary::Max, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the median of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_med3_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<false, rocjitsu::arith::Integer16Ternary::Median, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the median of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_med3_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<true, rocjitsu::arith::Integer16Ternary::Median, false>(
      flags, exec_mask, instruction_flags, d, a, b, c);
}
