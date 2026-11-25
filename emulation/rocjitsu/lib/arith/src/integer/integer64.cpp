// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// 64-bit integer arithmetic over pairs of VGPRs, including narrow-count shift-add.
// Public dispatch preserves the inherited saturation and wrapping behavior.

#include "integer/integer64.h"
#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_integer.h"

#include <stdint.h>

namespace {
using rocjitsu::arith::Integer64;

// Compute one wide result using wrapping arithmetic; only add/subtract honor CLAMP.
template <Integer64 Op> uint64_t evaluate(uint64_t a, uint64_t b, uint64_t c, bool clamp) {
  if constexpr (Op == Integer64::Add) {
    uint64_t result = a + b;
    return clamp && result < a ? UINT64_MAX : result;
  } else if constexpr (Op == Integer64::Sub) {
    return clamp && a < b ? 0 : a - b;
  } else if constexpr (Op == Integer64::Mul) {
    return a * b;
  } else if constexpr (Op == Integer64::ShiftAdd) {
    // Only S1[2:0] is decoded; encodings 5..7 select a shift of zero.
    uint64_t shift = b & 7;
    return (a << (shift <= 4 ? shift : 0)) + c;
  } else {
    constexpr bool signed_order = Op == Integer64::MinI || Op == Integer64::MaxI;
    constexpr bool maximum = Op == Integer64::MaxU || Op == Integer64::MaxI;
    uint64_t bias = signed_order ? 1ULL << 63 : 0;
    bool less = (a ^ bias) < (b ^ bias);
    return less != maximum ? a : b;
  }
}

// Validate before reading operands, and stage every input before writing either destination word.
template <Integer64 Op>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = rocjitsu::arith::validate(flags, mode & ~uint64_t(RJ_ARITH_ALU_CLAMP))) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::integer64_x86_64_v4<Op>(exec_mask, mode, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::integer64_x86_64_v3<Op>(exec_mask, mode, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint64_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint64_t x = a[0][lane] | (uint64_t(a[1][lane]) << 32);
    uint64_t y = b[0][lane], z = 0;
    if constexpr (Op == Integer64::ShiftAdd) {
      z = c[0][lane] | (uint64_t(c[1][lane]) << 32);
    } else {
      y |= uint64_t(b[1][lane]) << 32;
    }
    result[lane] = evaluate<Op>(x, y, z, mode & RJ_ARITH_ALU_CLAMP);
  }
  for (int reg = 0; reg < 2; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      if (exec_mask >> lane & 1) {
        d[reg][lane] = uint32_t(result[lane] >> (32 * reg));
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Split Wave64 into two validated halves, preserving the same register-alias contract.
template <Integer64 Op>
int run64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = run<Op>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    int offset = half * 32;
    uint32_t *dest[] = {d[0] + offset, d[1] + offset};
    const uint32_t *x[] = {a[0] + offset, a[1] + offset};
    const uint32_t *y[] = {b[0] + offset, nullptr}, *z[] = {nullptr, nullptr};
    if constexpr (Op == Integer64::ShiftAdd) {
      z[0] = c[0] + offset;
      z[1] = c[1] + offset;
    } else {
      y[1] = b[1] + offset;
    }
    if (int error = run<Op>(flags, uint32_t(exec_mask >> offset), mode, dest, x, y, z)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_add_nc_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<Integer64::Add>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_add_nc_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run64<Integer64::Add>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_sub_nc_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<Integer64::Sub>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_sub_nc_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run64<Integer64::Sub>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_mul_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<Integer64::Mul>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_mul_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run64<Integer64::Mul>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_min_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<Integer64::MinU>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_min_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run64<Integer64::MinU>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_max_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<Integer64::MaxU>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_max_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run64<Integer64::MaxU>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_min_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<Integer64::MinI>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_min_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run64<Integer64::MinI>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_max_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<Integer64::MaxI>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_max_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run64<Integer64::MaxI>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_lshl_add_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<Integer64::ShiftAdd>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate wide integer arithmetic for active lanes; invalid controls leave outputs unchanged.
int rj_arith_v_lshl_add_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run64<Integer64::ShiftAdd>(flags, exec_mask, instruction_flags, d, a, b, c);
}
