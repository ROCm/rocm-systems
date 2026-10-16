// SPDX-License-Identifier: MIT

// Three-input integer arithmetic, selection, and shift/boolean combinations.
// Owns public entry points and their validation/dispatch policy.

#include "integer/integer_ternary.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Evaluate three-input integer operations.
template <rocjitsu::arith::IntegerTernary Op>
uint32_t evaluate(uint32_t a, uint32_t b, uint32_t c) {
  if constexpr (Op == rocjitsu::arith::IntegerTernary::ShiftAdd) {
    return (a << (b & 31)) + c;
  }
  if constexpr (Op == rocjitsu::arith::IntegerTernary::AddShift) {
    return (a + b) << (c & 31);
  }
  if constexpr (Op == rocjitsu::arith::IntegerTernary::ShiftOr) {
    return (a << (b & 31)) | c;
  }
  if constexpr (Op == rocjitsu::arith::IntegerTernary::AndOr) {
    return (a & b) | c;
  }
  if constexpr (Op == rocjitsu::arith::IntegerTernary::Or3) {
    return a | b | c;
  }
  if constexpr (Op == rocjitsu::arith::IntegerTernary::Xor3) {
    return a ^ b ^ c;
  }
  if constexpr (Op == rocjitsu::arith::IntegerTernary::XorAdd) {
    return (a ^ b) + c;
  }
  if constexpr (Op == rocjitsu::arith::IntegerTernary::Lerp) {
    // Per-byte rounding follows rocjitsu's vector_alu.py LERP implementation.
    uint32_t result = 0;
    for (int byte = 0; byte < 4; ++byte) {
      unsigned shift = unsigned(8 * byte);
      unsigned sum = ((a >> shift) & 255) + ((b >> shift) & 255) + ((c >> shift) & 1);
      result |= (sum >> 1) << shift;
    }
    return result;
  }
}

// Evaluate three-input integer operations for the selected lanes. Reject invalid controls
// before changing outputs.
template <rocjitsu::arith::IntegerTernary Op>
int ternary(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return ternary<Op>(flags, exec_mask, uint32_t(mode), d, source, b, c);
        });
  }

  if (int error = rocjitsu::arith::validate(flags, mode & ~uint64_t(RJ_ARITH_ALU_CLAMP), true)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::integer_ternary_x86_64_v4<Op>(exec_mask, d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::integer_ternary_x86_64_v3<Op>(exec_mask, d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  if constexpr (Op == rocjitsu::arith::IntegerTernary::ShiftAdd ||
                Op == rocjitsu::arith::IntegerTernary::AddShift ||
                Op == rocjitsu::arith::IntegerTernary::ShiftOr) {
    // Clang's baseline vector shifts use FP conversions for powers of two;
    // converting 2^31 raises FE_INVALID. Explicit SIMD paths use integer shifts.
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
    for (int lane = 0; lane < 32; ++lane) {
      result[lane] = evaluate<Op>(a[0][lane], b[0][lane], c[0][lane]);
    }
  } else {
    for (int lane = 0; lane < 32; ++lane) {
      result[lane] = evaluate<Op>(a[0][lane], b[0][lane], c[0][lane]);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate three-input integer operations for Wave64. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::IntegerTernary Op>
int ternary_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = ternary<Op>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
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
    if (int error = ternary<Op>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                &source_a, &source_b, &source_c)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Shift left for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshl_add_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<rocjitsu::arith::IntegerTernary::ShiftAdd>(flags, exec_mask, mode, d, a, b, c);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add_lshl_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<rocjitsu::arith::IntegerTernary::AddShift>(flags, exec_mask, mode, d, a, b, c);
}

// Shift left for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshl_or_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<rocjitsu::arith::IntegerTernary::ShiftOr>(flags, exec_mask, mode, d, a, b, c);
}

// Compute bitwise AND for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_and_or_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<rocjitsu::arith::IntegerTernary::AndOr>(flags, exec_mask, mode, d, a, b, c);
}

// Execute OR3_B32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_or3_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<rocjitsu::arith::IntegerTernary::Or3>(flags, exec_mask, mode, d, a, b, c);
}

// Execute XOR3_B32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_xor3_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<rocjitsu::arith::IntegerTernary::Xor3>(flags, exec_mask, mode, d, a, b, c);
}

// Execute XAD_U32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_xad_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<rocjitsu::arith::IntegerTernary::XorAdd>(flags, exec_mask, mode, d, a, b, c);
}

// Execute LERP_U8 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lerp_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary<rocjitsu::arith::IntegerTernary::Lerp>(flags, exec_mask, mode, d, a, b, c);
}

// Shift left for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshl_add_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<rocjitsu::arith::IntegerTernary::ShiftAdd>(flags, exec_mask, mode, d, a, b,
                                                                   c);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add_lshl_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<rocjitsu::arith::IntegerTernary::AddShift>(flags, exec_mask, mode, d, a, b,
                                                                   c);
}

// Shift left for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshl_or_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<rocjitsu::arith::IntegerTernary::ShiftOr>(flags, exec_mask, mode, d, a, b,
                                                                  c);
}

// Compute bitwise AND for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_and_or_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<rocjitsu::arith::IntegerTernary::AndOr>(flags, exec_mask, mode, d, a, b, c);
}

// Execute OR3_B32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_or3_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<rocjitsu::arith::IntegerTernary::Or3>(flags, exec_mask, mode, d, a, b, c);
}

// Execute XOR3_B32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_xor3_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<rocjitsu::arith::IntegerTernary::Xor3>(flags, exec_mask, mode, d, a, b, c);
}

// Execute XAD_U32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_xad_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<rocjitsu::arith::IntegerTernary::XorAdd>(flags, exec_mask, mode, d, a, b,
                                                                 c);
}

// Execute LERP_U8 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lerp_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ternary_wave64<rocjitsu::arith::IntegerTernary::Lerp>(flags, exec_mask, mode, d, a, b, c);
}
