// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed INT8/INT4 dot products with wrapping or saturating INT32 accumulation.
// Owns public entry points and their validation/dispatch policy.
// Integer DOT semantics follow rocjitsu codegen/execute/packed.py.

#include "common/internal.h"
#include "common/simd.h"
#include "dot/operand_halves.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Accumulate packed integer dot products for the selected lanes. Reject invalid controls
// before changing outputs.
template <int Bits, bool Unsigned>
int dot(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known = rocjitsu::arith::kDotHalfSelection | RJ_ARITH_DOT_CLAMP |
                         (Unsigned ? 0 : RJ_ARITH_DOT_SIGNED_A | RJ_ARITH_DOT_SIGNED_B);
  if (int error = rocjitsu::arith::validate(flags, instruction_flags & ~known)) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
  if (instruction_flags & rocjitsu::arith::kDotHalfSelection) {
    return rocjitsu::arith::with_dot_halves(
        instruction_flags, a, b,
        [&](uint32_t selected, const uint32_t *const *av, const uint32_t *const *bv) {
          return dot<Bits, Unsigned>(flags, exec_mask, selected, d, av, bv, c);
        });
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::integer_dot_x86_64_v3(Bits, Unsigned, exec_mask, instruction_flags, d[0], a[0],
                                           b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  const int sign_a = instruction_flags & RJ_ARITH_DOT_SIGNED_A ? 1 << (Bits - 1) : 0;
  const int sign_b = instruction_flags & RJ_ARITH_DOT_SIGNED_B ? 1 << (Bits - 1) : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    int64_t sum = c[0][lane];
    if (!Unsigned && (c[0][lane] & 0x80000000)) {
      sum -= INT64_C(1) << 32;
    }
    for (int shift = 0; shift < 32; shift += Bits) {
      int x = ((a[0][lane] >> shift) & ((1 << Bits) - 1));
      int y = ((b[0][lane] >> shift) & ((1 << Bits) - 1));
      sum += ((x ^ sign_a) - sign_a) * ((y ^ sign_b) - sign_b);
    }
    if (instruction_flags & RJ_ARITH_DOT_CLAMP) {
      sum = rocjitsu::arith::clamp(sum, Unsigned ? INT64_C(0) : int64_t(INT32_MIN),
                                   Unsigned ? int64_t(UINT32_MAX) : int64_t(INT32_MAX));
    }
    result[lane] = uint32_t(sum);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Accumulate packed integer dot products for Wave64. Reject invalid controls before changing
// outputs.
template <int Bits, bool Unsigned>
int dot_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }

  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error =
          dot<Bits, Unsigned>(flags, 0, instruction_flags, nullptr, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source_a = a[0] + half * 32, *source_b = b[0] + half * 32,
                   *source_c = c[0] + half * 32;
    if (int error =
            dot<Bits, Unsigned>(flags, uint32_t(exec_mask >> (half * 32)), instruction_flags, &dest,
                                &source_a, &source_b, &source_c)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Accumulate four packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot4_i32_iu8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
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
  return dot<8, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate four packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot4_u32_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
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
  return dot<8, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate eight packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot8_i32_iu4_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
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
  return dot<4, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate eight packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot8_u32_u4_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
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
  return dot<4, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate four packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot4_i32_iu8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot_wave64<8, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate four packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot4_u32_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot_wave64<8, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate eight packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot8_i32_iu4_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot_wave64<4, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate eight packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot8_u32_u4_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot_wave64<4, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}
