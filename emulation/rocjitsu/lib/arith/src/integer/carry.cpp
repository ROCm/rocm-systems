// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integer add/subtract with per-lane carry/borrow input and output masks.
// Owns public entry points and their validation/dispatch policy.

#include "integer/carry.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <cstring>
#include <stdint.h>

namespace {

// Add/subtract with carry or borrow masks for the selected lanes. Reject invalid controls
// before changing outputs.
template <rocjitsu::arith::CarryOp Op, bool WithCarry>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d, uint32_t *carry,
        const uint32_t *const *a, const uint32_t *const *b, uint32_t input_carry) {
  if ((mode >> 32) &&
      (!(mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16)) || !rocjitsu::arith::valid_dpp(mode))) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, uint32_t(mode) & ~RJ_ARITH_ALU_CLAMP, true)) {
    return error;
  }
  if (!exec_mask) {
    *carry = 0;
    return RJ_ARITH_SUCCESS;
  }
  if (mode >> 32) {
    uint32_t permuted[32];
    const uint32_t *source = permuted;

    // SUBREV reverses arithmetic operands, so its routed source is B rather than A.
    const uint32_t *input = Op == rocjitsu::arith::CarryOp::Subrev ? b[0] : a[0];
    if (mode & RJ_ARITH_DPP8) {
      rocjitsu::arith::dpp8_source(flags, exec_mask, mode, permuted, input);
    } else {
      exec_mask = rocjitsu::arith::dpp16_source(flags, exec_mask, mode, permuted, input);
    }
    return run<Op, WithCarry>(flags, exec_mask, uint32_t(mode), d, carry,
                              Op == rocjitsu::arith::CarryOp::Subrev ? a : &source,
                              Op == rocjitsu::arith::CarryOp::Subrev ? &source : b, input_carry);
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    *carry = rocjitsu::arith::carry_x86_64_v4<Op, WithCarry>(exec_mask, mode, d, a, b, input_carry);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    *carry = rocjitsu::arith::carry_x86_64_v3<Op, WithCarry>(exec_mask, mode, d, a, b, input_carry);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32], output_carry = 0;
  for (int lane = 0; lane < 32; ++lane) {
    int64_t av = a[0][lane], bv = b[0][lane];
    uint32_t ci = WithCarry ? ((input_carry >> lane) & 1) : 0;
    int64_t wide;
    if constexpr (Op == rocjitsu::arith::CarryOp::Add) {
      wide = av + bv + ci;
    } else if constexpr (Op == rocjitsu::arith::CarryOp::Sub) {
      wide = av - bv - ci;
    } else {
      wide = bv - av - ci;
    }
    bool co = wide < 0 || wide > UINT32_MAX;
    result[lane] = (mode & RJ_ARITH_ALU_CLAMP) && co
                       ? (Op == rocjitsu::arith::CarryOp::Add ? UINT32_MAX : 0)
                       : uint32_t(wide);
    output_carry |= uint32_t(co) << lane;
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }

  // Inactive destination words are preserved, but their carry-result bits are cleared.
  *carry = output_carry & exec_mask;
  return RJ_ARITH_SUCCESS;
}

// Add/subtract with carry or borrow masks for Wave64. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::CarryOp Op, bool WithCarry>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               uint64_t *carry, const uint32_t *const *a, const uint32_t *const *b,
               uint64_t input_carry) {
  uint32_t output[2] = {};

  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = run<Op, WithCarry>(flags, 0, mode, nullptr, output, nullptr, nullptr, 0)) {
    return error;
  }
  if (exec_mask) {
    for (int half = 0; half < 2; ++half) {
      uint64_t half_mode = mode;
      if (mode & RJ_ARITH_DPP16) {
        uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
        half_mode = (mode & ~RJ_ARITH_DPP_ROW_MASK) |
                    (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
      }
      uint32_t *dest = d[0] + half * 32;
      const uint32_t *source_a = a[0] + half * 32, *source_b = b[0] + half * 32;
      if (int error = run<Op, WithCarry>(flags, uint32_t(exec_mask >> (half * 32)), half_mode,
                                         &dest, output + half, &source_a, &source_b,
                                         uint32_t(input_carry >> (half * 32)))) {
        return error;
      }
    }
  }
  // Delay the scalar output until both halves have consumed all their sources.
  uint64_t result = output[0] | (uint64_t(output[1]) << 32);

  // The mask may start at an unaligned byte within aliased VGPR storage.
  // Erase the pointer alignment before the byte copy (also for UBSan).
  std::memcpy(static_cast<void *>(carry), &result, sizeof result);
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add_co_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::CarryOp::Add, false>(flags, exec_mask, instruction_flags, d, carry, a,
                                                   b, 0);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sub_co_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::CarryOp::Sub, false>(flags, exec_mask, instruction_flags, d, carry, a,
                                                   b, 0);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_subrev_co_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                                 const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::CarryOp::Subrev, false>(flags, exec_mask, instruction_flags, d, carry,
                                                      a, b, 0);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add_co_ci_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t input_carry) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::CarryOp::Add, true>(flags, exec_mask, instruction_flags, d, carry, a,
                                                  b, input_carry);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sub_co_ci_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t input_carry) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::CarryOp::Sub, true>(flags, exec_mask, instruction_flags, d, carry, a,
                                                  b, input_carry);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_subrev_co_ci_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t input_carry) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::CarryOp::Subrev, true>(flags, exec_mask, instruction_flags, d, carry,
                                                     a, b, input_carry);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add_co_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<rocjitsu::arith::CarryOp::Add, false>(flags, exec_mask, instruction_flags, d,
                                                          carry, a, b, 0);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sub_co_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<rocjitsu::arith::CarryOp::Sub, false>(flags, exec_mask, instruction_flags, d,
                                                          carry, a, b, 0);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_subrev_co_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                                 const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<rocjitsu::arith::CarryOp::Subrev, false>(flags, exec_mask, instruction_flags, d,
                                                             carry, a, b, 0);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add_co_ci_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                                 const uint32_t *const *b, uint64_t input_carry) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<rocjitsu::arith::CarryOp::Add, true>(flags, exec_mask, instruction_flags, d,
                                                         carry, a, b, input_carry);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sub_co_ci_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                                 const uint32_t *const *b, uint64_t input_carry) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<rocjitsu::arith::CarryOp::Sub, true>(flags, exec_mask, instruction_flags, d,
                                                         carry, a, b, input_carry);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_subrev_co_ci_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                                    const uint32_t *const *b, uint64_t input_carry) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<rocjitsu::arith::CarryOp::Subrev, true>(flags, exec_mask, instruction_flags, d,
                                                            carry, a, b, input_carry);
}
