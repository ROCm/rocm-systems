// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// 32-bit add/subtract and three-input addition, with optional saturation.
// Owns public entry points and their validation/dispatch policy.

#include "integer/integer_add.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <stdint.h>

namespace {

// Interpret a 32-bit word as signed or unsigned and widen it to int64_t.
template <bool Signed> int64_t value(uint32_t bits) {
  return int64_t(bits) - (Signed && (bits & 0x80000000) ? INT64_C(4294967296) : 0);
}

// Add or subtract integers for the selected lanes. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::IntegerAdd Op, bool Signed>
int arithmetic(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    // GFX1201 applies SUBREV's DPP to B before subtracting lane-local A.
    constexpr bool reverse = Op == rocjitsu::arith::IntegerAdd::Subrev;
    return rocjitsu::arith::execute_dpp(flags, exec_mask, mode, reverse ? b : a,
                                        [&](uint32_t exec_mask, const uint32_t *const *source) {
                                          return arithmetic<Op, Signed>(
                                              flags, exec_mask, uint32_t(mode), d,
                                              reverse ? a : source, reverse ? source : b, c);
                                        });
  }

  constexpr bool three = Op == rocjitsu::arith::IntegerAdd::Add3;
  const uint32_t known = RJ_ARITH_ALU_CLAMP;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known, true)) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::integer_add_x86_64_v4<Op, Signed>(exec_mask, mode, d[0], a[0], b[0],
                                                       three ? c[0] : nullptr);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    if constexpr (!three) {
      if (mode & RJ_ARITH_ALU_CLAMP) {
        rocjitsu::arith::integer_add_sat_x86_64_v3<Op, Signed>(exec_mask, d[0], a[0], b[0]);
        return RJ_ARITH_SUCCESS;
      }
    }
    rocjitsu::arith::integer_add_wrap_x86_64_v3<Op>(exec_mask, d[0], a[0], b[0],
                                                    three ? c[0] : nullptr);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    int64_t x = value<Signed>(a[0][lane]), y = value<Signed>(b[0][lane]), sum;
    if constexpr (Op == rocjitsu::arith::IntegerAdd::Add) {
      sum = x + y;
    }
    if constexpr (Op == rocjitsu::arith::IntegerAdd::Sub) {
      sum = x - y;
    }
    if constexpr (Op == rocjitsu::arith::IntegerAdd::Subrev) {
      sum = y - x;
    }
    if constexpr (three) {
      sum = x + y + c[0][lane];
    }
    if (!three && (mode & RJ_ARITH_ALU_CLAMP)) {
      sum = std::clamp(sum, Signed ? int64_t(INT32_MIN) : INT64_C(0),
                       Signed ? int64_t(INT32_MAX) : int64_t(UINT32_MAX));
    }
    result[lane] = uint32_t(sum);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if (exec_mask >> lane & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Add or subtract integers for Wave64. Reject invalid controls before changing outputs.
template <rocjitsu::arith::IntegerAdd Op, bool Signed>
int arithmetic_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = arithmetic<Op, Signed>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
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
    const uint32_t *source_c = nullptr;
    if constexpr (Op == rocjitsu::arith::IntegerAdd::Add3) {
      source_c = c[0] + half * 32;
    }
    if (int error = arithmetic<Op, Signed>(flags, uint32_t(exec_mask >> (half * 32)), half_mode,
                                           &dest, &source_a, &source_b, &source_c)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add_nc_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::IntegerAdd::Add, false>(flags, exec_mask, instruction_flags, d,
                                                             a, b, nullptr);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sub_nc_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::IntegerAdd::Sub, false>(flags, exec_mask, instruction_flags, d,
                                                             a, b, nullptr);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_subrev_nc_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::IntegerAdd::Subrev, false>(flags, exec_mask, instruction_flags,
                                                                d, a, b, nullptr);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add_nc_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::IntegerAdd::Add, true>(flags, exec_mask, instruction_flags, d,
                                                            a, b, nullptr);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sub_nc_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::IntegerAdd::Sub, true>(flags, exec_mask, instruction_flags, d,
                                                            a, b, nullptr);
}

// Sum three operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add3_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic<rocjitsu::arith::IntegerAdd::Add3, false>(flags, exec_mask, instruction_flags,
                                                              d, a, b, c);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add_nc_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::IntegerAdd::Add, false>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sub_nc_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::IntegerAdd::Sub, false>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_subrev_nc_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::IntegerAdd::Subrev, false>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add_nc_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::IntegerAdd::Add, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sub_nc_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::IntegerAdd::Sub, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Sum three operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_add3_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return arithmetic_wave64<rocjitsu::arith::IntegerAdd::Add3, false>(flags, exec_mask,
                                                                     instruction_flags, d, a, b, c);
}
