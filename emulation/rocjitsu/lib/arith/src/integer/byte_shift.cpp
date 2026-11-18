// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Saturating byte packing after arithmetic right shifts.
// Owns public entry points and their validation/dispatch policy.

#include "integer/byte_shift.h"
#include "common/bits.h"
#include "common/validation.h"
#include "register/dpp.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_integer.h"

#include <stdint.h>

namespace {

// Shift signed inputs and pack saturated bytes for the selected lanes. Reject invalid controls
// before changing outputs.
template <bool Signed>
int pack(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return pack<Signed>(flags, exec_mask, uint32_t(mode), d, source, b, c);
        });
  }

  const uint32_t known = RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_CLAMP;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known)) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::byte_shift_x86_64_v4<Signed>(exec_mask, mode, d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::byte_shift_x86_64_v3<Signed>(exec_mask, mode, d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    int count = c[0][lane] & 31;
    int32_t x = int32_t(a[0][lane]) >> count, y = int32_t(b[0][lane]) >> count;
    x = rocjitsu::arith::clamp(x, Signed ? -128 : 0, Signed ? 127 : 255);
    y = rocjitsu::arith::clamp(y, Signed ? -128 : 0, Signed ? 127 : 255);
    uint32_t packed = uint8_t(x) | (uint32_t(uint8_t(y)) << 8);
    result[lane] = (d[0][lane] & ~(0xffffU << shift)) | (packed << shift);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if (exec_mask >> lane & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Shift signed inputs and pack saturated bytes for Wave64. Reject invalid controls before changing
// outputs.
template <bool Signed>
int pack_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = pack<Signed>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
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
    const uint32_t *source_a = a[0] + half * 32, *source_b = b[0] + half * 32,
                   *source_c = c[0] + half * 32;
    if (int error = pack<Signed>(flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags,
                                 &dest, &source_a, &source_b, &source_c)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Shift both signed inputs, saturate to bytes, and replace the selected destination half.
int rj_arith_v_ashr_pk_u8_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return pack<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Shift both signed inputs, saturate to bytes, and replace the selected destination half.
int rj_arith_v_ashr_pk_i8_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return pack<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Shift both signed inputs, saturate to bytes, and replace the selected destination half.
int rj_arith_v_ashr_pk_u8_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return pack_wave64<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Shift both signed inputs, saturate to bytes, and replace the selected destination half.
int rj_arith_v_ashr_pk_i8_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return pack_wave64<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}
