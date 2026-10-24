// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// 24-bit and packed 16-bit integer multiply-add.
// Owns public entry points and their validation/dispatch policy.
// Wide intermediates and integer saturation follow rocjitsu shared/simd_glue.h.

#include "integer/integer_mad.h"
#include "common/bits.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <stdint.h>

namespace {

// Multiply-add narrow integer factors for the selected lanes. Reject invalid controls before
// changing outputs.
template <int Bits, bool Signed>
int mad(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return mad<Bits, Signed>(flags, exec_mask, uint32_t(mode), d, source, b, c);
        });
  }

  const uint32_t known =
      RJ_ARITH_ALU_CLAMP | (Bits == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0);
  if (int error = rocjitsu::arith::validate(flags, mode & ~known, true)) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::integer_mad_x86_64_v3<Bits, Signed>(exec_mask, mode, d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  const int sa = Bits == 16 && (mode & RJ_ARITH_ALU_HIGH_A) ? 16 : 0;
  const int sb = Bits == 16 && (mode & RJ_ARITH_ALU_HIGH_B) ? 16 : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    int64_t value = rocjitsu::arith::extend_integer<Bits, Signed>(a[0][lane] >> sa) *
                        rocjitsu::arith::extend_integer<Bits, Signed>(b[0][lane] >> sb) +
                    rocjitsu::arith::extend_integer<32, Signed>(c[0][lane]);
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = std::clamp(value, Signed ? int64_t(INT32_MIN) : INT64_C(0),
                         Signed ? int64_t(INT32_MAX) : int64_t(UINT32_MAX));
    }
    result[lane] = uint32_t(value);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if (exec_mask >> lane & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Multiply-add narrow integer factors for Wave64. Reject invalid controls before changing
// outputs.
template <int Bits, bool Signed>
int mad_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = mad<Bits, Signed>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
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
    const uint32_t *source_a = a[0] + half * 32, *source_b = b[0] + half * 32,
                   *source_c = c[0] + half * 32;
    if (int error = mad<Bits, Signed>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                      &source_a, &source_b, &source_c)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_u32_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mad<16, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_i32_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mad<16, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_u32_u24_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mad<24, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_i32_i24_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mad<24, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_u32_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mad_wave64<16, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_i32_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mad_wave64<16, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_u32_u24_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mad_wave64<24, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_i32_i24_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mad_wave64<24, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}
