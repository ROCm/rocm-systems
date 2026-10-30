// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Low/high-half integer multiplication with signed or unsigned operands.
// Owns public entry points and their validation/dispatch policy.
// 24-bit truncation/sign extension follow rocjitsu shared/simd_glue.h.

#include "integer/integer_mul.h"
#include "common/bits.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Multiply integers and select result halves for the selected lanes. Reject invalid controls
// before changing outputs.
template <int Bits, bool Signed, bool High>
int multiply(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
             const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    if constexpr (Bits == 24) {
      return rocjitsu::arith::execute_dpp(
          flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
            return multiply<Bits, Signed, High>(flags, exec_mask, uint32_t(mode), d, source, b);
          });
    } else {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }

  const uint32_t known = Bits == 24 && !High ? RJ_ARITH_ALU_CLAMP : 0;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known)) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::integer_mul_x86_64_v4<Bits, Signed, High>(exec_mask, mode, d[0], a[0], b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::integer_mul_x86_64_v3<Bits, Signed, High>(exec_mask, mode, d[0], a[0], b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint64_t product;
    if constexpr (Signed) {
      int64_t value = rocjitsu::arith::extend_integer<Bits, true>(a[0][lane]) *
                      rocjitsu::arith::extend_integer<Bits, true>(b[0][lane]);
      if (mode & RJ_ARITH_ALU_CLAMP) {
        value = rocjitsu::arith::clamp(value, int64_t(INT32_MIN), int64_t(INT32_MAX));
      }
      product = uint64_t(value);
    } else {
      const uint32_t input_mask = Bits == 24 ? 0x00ffffffU : UINT32_MAX;
      product = uint64_t(a[0][lane] & input_mask) * (b[0][lane] & input_mask);
      if (mode & RJ_ARITH_ALU_CLAMP) {
        product = rocjitsu::arith::minimum(product, uint64_t(UINT32_MAX));
      }
    }
    // Select the result half after widening and optional saturation; low products otherwise
    // wrap.
    result[lane] = uint32_t(High ? product >> 32 : product);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if (exec_mask >> lane & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Multiply integers and select result halves for Wave64. Reject invalid controls before
// changing outputs.
template <int Bits, bool Signed, bool High>
int multiply_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                    const uint32_t *const *a, const uint32_t *const *b) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = multiply<Bits, Signed, High>(flags, 0, mode, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_instruction_flags =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source_a = a[0] + half * 32, *source_b = b[0] + half * 32;
    if (int error =
            multiply<Bits, Signed, High>(flags, uint32_t(exec_mask >> (half * 32)),
                                         half_instruction_flags, &dest, &source_a, &source_b)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_lo_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
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
  return multiply<32, false, false>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_hi_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
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
  return multiply<32, false, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_hi_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
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
  return multiply<32, true, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_i32_i24_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return multiply<24, true, false>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_hi_i32_i24_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return multiply<24, true, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_u32_u24_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return multiply<24, false, false>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_hi_u32_u24_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return multiply<24, false, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_lo_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
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
  return multiply_wave64<32, false, false>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_hi_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
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
  return multiply_wave64<32, false, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_hi_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
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
  return multiply_wave64<32, true, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_i32_i24_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return multiply_wave64<24, true, false>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_hi_i32_i24_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return multiply_wave64<24, true, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_u32_u24_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return multiply_wave64<24, false, false>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mul_hi_u32_u24_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return multiply_wave64<24, false, true>(flags, exec_mask, instruction_flags, d, a, b);
}
