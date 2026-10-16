// SPDX-License-Identifier: MIT

// 16-bit integer add/subtract, multiply, shifts, and min/max.
// Owns public entry points and their validation/dispatch policy.

#include "integer/integer16.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <stdint.h>

namespace {

// Evaluate 16-bit integer arithmetic for the selected lanes. Reject invalid controls before
// changing outputs.
template <rocjitsu::arith::Integer16 Op, bool Signed, bool Packed = true>
int arithmetic(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    if constexpr (!Packed) {
      return rocjitsu::arith::execute_dpp(
          flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
            return arithmetic<Op, Signed, Packed>(flags, exec_mask, uint32_t(mode), d, source, b);
          });
    } else {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }

  const uint32_t known =
      Packed ? RJ_ARITH_PK_LO_A_HIGH | RJ_ARITH_PK_LO_B_HIGH | RJ_ARITH_PK_HI_A_LOW |
                   RJ_ARITH_PK_HI_B_LOW | RJ_ARITH_PK_CLAMP
             : RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_CLAMP;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known, true)) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::integer16_x86_64_v3<Op, Signed, Packed>(exec_mask, mode, d[0], a[0], b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t arithmetic_mode = mode;
  if constexpr (!Packed) {
    arithmetic_mode = (mode & RJ_ARITH_ALU_HIGH_A ? RJ_ARITH_PK_LO_A_HIGH : 0) |
                      (mode & RJ_ARITH_ALU_HIGH_B ? RJ_ARITH_PK_LO_B_HIGH : 0) |
                      (mode & RJ_ARITH_ALU_CLAMP ? RJ_ARITH_PK_CLAMP : 0);
  }
  uint32_t result[32] = {};
  for (int lane = 0; lane < 32; ++lane) {
    for (int half = 0; half < (Packed ? 2 : 1); ++half) {
      int sa = half ^ bool(arithmetic_mode & (half ? RJ_ARITH_PK_HI_A_LOW : RJ_ARITH_PK_LO_A_HIGH));
      int sb = half ^ bool(arithmetic_mode & (half ? RJ_ARITH_PK_HI_B_LOW : RJ_ARITH_PK_LO_B_HIGH));
      int64_t x = uint16_t(a[0][lane] >> (16 * sa));
      int64_t y = uint16_t(b[0][lane] >> (16 * sb));
      if constexpr (Signed) {
        x -= (x & 0x8000) ? 65536 : 0;
        y -= (y & 0x8000) ? 65536 : 0;
      }
      int64_t value;
      if constexpr (Op == rocjitsu::arith::Integer16::Add) {
        value = x + y;
      }
      if constexpr (Op == rocjitsu::arith::Integer16::Sub) {
        value = x - y;
      }
      if constexpr (Op == rocjitsu::arith::Integer16::Min) {
        value = std::min(x, y);
      }
      if constexpr (Op == rocjitsu::arith::Integer16::Max) {
        value = std::max(x, y);
      }
      if constexpr (Op == rocjitsu::arith::Integer16::Mul) {
        value = x * y;
      }
      if constexpr (Op == rocjitsu::arith::Integer16::ShiftLeft) {
        value = uint32_t(y) << (uint32_t(x) & 15);
      }

      // Signed Y is already extended to 32 bits. At most 15 right shifts
      // cannot move a zero from above bit 31 into the retained low 16 bits.
      if constexpr (Op == rocjitsu::arith::Integer16::ShiftRight) {
        value = uint32_t(y) >> (uint32_t(x) & 15);
      }

      if constexpr (Op == rocjitsu::arith::Integer16::Add ||
                    Op == rocjitsu::arith::Integer16::Sub) {
        if (arithmetic_mode & RJ_ARITH_PK_CLAMP) {
          value = std::clamp(value, Signed ? INT64_C(-32768) : INT64_C(0),
                             Signed ? INT64_C(32767) : INT64_C(65535));
        }
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

// Evaluate 16-bit integer arithmetic for Wave64. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::Integer16 Op, bool Signed, bool Packed = true>
int arithmetic_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = arithmetic<Op, Signed, Packed>(flags, 0, mode, nullptr, nullptr, nullptr)) {
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
    if (int error = arithmetic<Op, Signed, Packed>(flags, uint32_t(exec_mask >> (half * 32)),
                                                   half_mode, &dest, &source_a, &source_b)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Add the operands with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_add_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Integer16::Add, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Subtract B from A with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_sub_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Integer16::Sub, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Add the operands with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_add_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Integer16::Add, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b);
}

// Subtract B from A with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_sub_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Integer16::Sub, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b);
}

// Select the minimum with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_min_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Integer16::Min, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Select the maximum with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_max_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Integer16::Max, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Select the minimum with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_min_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Integer16::Min, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b);
}

// Select the maximum with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_max_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Integer16::Max, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b);
}

// Multiply the operands with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_mul_lo_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Integer16::Mul, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b);
}

// Shift B left by A with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_lshlrev_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Integer16::ShiftLeft, false>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Shift B right logically by A with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_lshrrev_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Integer16::ShiftRight, false>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Shift B right arithmetically by A with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_ashrrev_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Integer16::ShiftRight, true>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add_nc_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::Integer16::Add, true, false>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sub_nc_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::Integer16::Sub, true, false>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add_nc_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::Integer16::Add, false, false>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sub_nc_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::Integer16::Sub, false, false>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Select the minimum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::Integer16::Min, true, false>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Select the maximum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::Integer16::Max, true, false>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Select the minimum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::Integer16::Min, false, false>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Select the maximum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::Integer16::Max, false, false>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_lo_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::Integer16::Mul, false, false>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Shift B left by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshlrev_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::Integer16::ShiftLeft, false, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Shift B right logically by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshrrev_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::Integer16::ShiftRight, false, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Shift B right arithmetically by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_ashrrev_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::Integer16::ShiftRight, true, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Add the operands with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_add_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Add, true>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Subtract B from A with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_sub_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Sub, true>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Add the operands with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_add_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Add, false>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Subtract B from A with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_sub_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Sub, false>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Select the minimum with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_min_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Min, true>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Select the maximum with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_max_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Max, true>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Select the minimum with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_min_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Min, false>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Select the maximum with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_max_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Max, false>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Multiply the operands with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_mul_lo_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Mul, false>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Shift B left by A with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_lshlrev_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::ShiftLeft, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Shift B right logically by A with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_lshrrev_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::ShiftRight, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Shift B right arithmetically by A with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_pk_ashrrev_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::ShiftRight, true>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add_nc_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Add, true, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sub_nc_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Sub, true, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add_nc_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Add, false, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sub_nc_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Sub, false, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Select the minimum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Min, true, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Select the maximum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Max, true, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Select the minimum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Min, false, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Select the maximum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Max, false, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_lo_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::Mul, false, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Shift B left by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshlrev_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::ShiftLeft, false, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Shift B right logically by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshrrev_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::ShiftRight, false, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Shift B right arithmetically by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_ashrrev_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::Integer16::ShiftRight, true, false>(
      flags, exec_mask, instruction_flags, d, a, b);
}
