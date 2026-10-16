// SPDX-License-Identifier: MIT

// Wordwise boolean operations, including three-input truth tables.
// Owns public entry points and their validation/dispatch policy.

#include "integer/boolean.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Evaluate wordwise boolean operations for the selected lanes. Reject invalid controls before
// changing outputs.
template <rocjitsu::arith::Boolean Op, bool Half>
int boolean(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return boolean<Op, Half>(flags, exec_mask, uint32_t(mode), d, source, b);
        });
  }

  const uint32_t known = RJ_ARITH_ALU_CLAMP |
                         (Half ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D |
                                     (Op == rocjitsu::arith::Boolean::Not ? 0 : RJ_ARITH_ALU_HIGH_B)
                               : 0);
  if (int error = rocjitsu::arith::validate(flags, mode & ~known, true)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  const uint32_t *bp = nullptr;
  if constexpr (Op != rocjitsu::arith::Boolean::Not) {
    bp = b[0];
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::boolean_x86_64_v4<Op, Half>(exec_mask, mode, d[0], a[0], bp);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  // Half OR/XOR/NOT are faster as portable half-word operations. Keep the
  // decision opcode-based: half selectors and EXEC do not alter dispatch.
  if constexpr (!Half || Op == rocjitsu::arith::Boolean::And) {
    if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
      rocjitsu::arith::boolean_x86_64_v3<Op, Half>(exec_mask, mode, d[0], a[0], bp);
      return RJ_ARITH_SUCCESS;
    }
  }
#endif

  unsigned sa = Half && (mode & RJ_ARITH_ALU_HIGH_A) ? 16 : 0;
  unsigned sb = Half && (mode & RJ_ARITH_ALU_HIGH_B) ? 16 : 0;
  unsigned sd = Half && (mode & RJ_ARITH_ALU_HIGH_D) ? 16 : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t x = a[0][lane] >> sa, y = bp ? bp[lane] >> sb : 0;
    if constexpr (Op == rocjitsu::arith::Boolean::And) {
      result[lane] = x & y;
    }
    if constexpr (Op == rocjitsu::arith::Boolean::Or) {
      result[lane] = x | y;
    }
    if constexpr (Op == rocjitsu::arith::Boolean::Xor) {
      result[lane] = x ^ y;
    }
    if constexpr (Op == rocjitsu::arith::Boolean::Xnor) {
      result[lane] = ~(x ^ y);
    }
    if constexpr (Op == rocjitsu::arith::Boolean::Not) {
      result[lane] = ~x;
    }
    if constexpr (Half) {
      result[lane] =
          (d[0][lane] & ~(uint32_t(65535) << sd)) | (uint32_t(uint16_t(result[lane])) << sd);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate wordwise boolean operations for Wave64. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::Boolean Op, bool Half>
int boolean_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = boolean<Op, Half>(flags, 0, mode, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }

  // DPP stays within 16-lane rows, so whole-VGPR aliases are safe across halves.
  for (int half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_mode =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source_a = a[0] + half * 32;
    const uint32_t *source_b = nullptr;
    if constexpr (Op != rocjitsu::arith::Boolean::Not) {
      source_b = b[0] + half * 32;
    }
    if (int error = boolean<Op, Half>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                      &source_a, &source_b)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Compute bitwise AND for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_and_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean<rocjitsu::arith::Boolean::And, false>(flags, exec_mask, instruction_flags, d, a,
                                                       b);
}

// Compute bitwise OR for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_or_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean<rocjitsu::arith::Boolean::Or, false>(flags, exec_mask, instruction_flags, d, a, b);
}

// Compute bitwise XOR for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_xor_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean<rocjitsu::arith::Boolean::Xor, false>(flags, exec_mask, instruction_flags, d, a,
                                                       b);
}

// Complement the source bits for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_not_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean<rocjitsu::arith::Boolean::Not, false>(flags, exec_mask, instruction_flags, d, a,
                                                       nullptr);
}

// Compute bitwise AND for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_and_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean<rocjitsu::arith::Boolean::And, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Compute bitwise OR for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_or_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean<rocjitsu::arith::Boolean::Or, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Compute bitwise XOR for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_xor_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean<rocjitsu::arith::Boolean::Xor, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Complement the source bits for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_not_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean<rocjitsu::arith::Boolean::Not, true>(flags, exec_mask, instruction_flags, d, a,
                                                      nullptr);
}

// Compute bitwise equivalence for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_xnor_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean<rocjitsu::arith::Boolean::Xnor, false>(flags, exec_mask, instruction_flags, d, a,
                                                        b);
}

// Compute bitwise AND for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_and_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean_wave64<rocjitsu::arith::Boolean::And, false>(flags, exec_mask, instruction_flags,
                                                              d, a, b);
}

// Compute bitwise OR for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_or_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean_wave64<rocjitsu::arith::Boolean::Or, false>(flags, exec_mask, instruction_flags, d,
                                                             a, b);
}

// Compute bitwise XOR for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_xor_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean_wave64<rocjitsu::arith::Boolean::Xor, false>(flags, exec_mask, instruction_flags,
                                                              d, a, b);
}

// Complement the source bits for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_not_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean_wave64<rocjitsu::arith::Boolean::Not, false>(flags, exec_mask, instruction_flags,
                                                              d, a, nullptr);
}

// Compute bitwise AND for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_and_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean_wave64<rocjitsu::arith::Boolean::And, true>(flags, exec_mask, instruction_flags, d,
                                                             a, b);
}

// Compute bitwise OR for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_or_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean_wave64<rocjitsu::arith::Boolean::Or, true>(flags, exec_mask, instruction_flags, d,
                                                            a, b);
}

// Compute bitwise XOR for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_xor_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean_wave64<rocjitsu::arith::Boolean::Xor, true>(flags, exec_mask, instruction_flags, d,
                                                             a, b);
}

// Complement the source bits for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_not_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean_wave64<rocjitsu::arith::Boolean::Not, true>(flags, exec_mask, instruction_flags, d,
                                                             a, nullptr);
}

// Compute bitwise equivalence for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_xnor_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return boolean_wave64<rocjitsu::arith::Boolean::Xnor, false>(flags, exec_mask, instruction_flags,
                                                               d, a, b);
}
