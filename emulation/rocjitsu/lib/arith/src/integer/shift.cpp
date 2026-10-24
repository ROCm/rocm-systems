// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// 32/64-bit logical and arithmetic shifts, including reversed operand forms.
// Owns public entry points and their validation/dispatch policy.

#include "integer/shift.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Shift integer words for the selected lanes. Reject invalid controls before changing outputs.
template <int Bits, rocjitsu::arith::Shift Op>
int shift(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    if constexpr (Bits == 32) {
      return rocjitsu::arith::execute_dpp(
          flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
            return shift<Bits, Op>(flags, exec_mask, uint32_t(mode), d, source, b);
          });
    } else {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }

  if (int error = rocjitsu::arith::validate(flags, mode & ~RJ_ARITH_ALU_CLAMP, true)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::shift_x86_64_v3<Bits, Op>(exec_mask, d, a[0], b);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[Bits / 32][32];

  // Baseline SSE variable shifts can use FP conversion to construct powers of
  // two, raising invalid for a shift by 31. Keep this fallback integer-only;
  // the explicit x86-64-v3 path above retains SIMD variable shifts.
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
  for (int lane = 0; lane < 32; ++lane) {
    unsigned count = a[0][lane] & (Bits - 1);
    uint64_t value = b[0][lane];
    if constexpr (Bits == 64) {
      value |= uint64_t(b[1][lane]) << 32;
    }
    if constexpr (Op == rocjitsu::arith::Shift::Left) {
      value <<= count;
    }
    if constexpr (Op == rocjitsu::arith::Shift::LogicalRight) {
      value >>= count;
    }
    if constexpr (Op == rocjitsu::arith::Shift::ArithmeticRight) {
      const uint64_t width_mask = Bits == 32 ? UINT32_MAX : UINT64_MAX;
      const uint64_t sign = value >> (Bits - 1) ? width_mask : 0;

      // Complement negative values around a logical shift to supply sign bits
      // without implementation-defined signed right shifts.
      value = ((value ^ sign) >> count) ^ sign;
    }
    result[0][lane] = uint32_t(value);
    if constexpr (Bits == 64) {
      result[1][lane] = uint32_t(value >> 32);
    }
  }
  for (int reg = 0; reg < Bits / 32; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[reg][lane] = result[reg][lane];
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Shift integer words for Wave64. Reject invalid controls before changing outputs.
template <int Bits, rocjitsu::arith::Shift Op>
int shift_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = shift<Bits, Op>(flags, 0, mode, nullptr, nullptr, nullptr)) {
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
    const uint32_t *counts = a[0] + half * 32, *values[Bits / 32];
    uint32_t *dest[Bits / 32];
    for (int word = 0; word < Bits / 32; ++word) {
      values[word] = b[word] + half * 32;
      dest[word] = d[word] + half * 32;
    }
    if (int error = shift<Bits, Op>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, dest,
                                    &counts, values)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Shift B left by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshlrev_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return shift<32, rocjitsu::arith::Shift::Left>(flags, exec_mask, mode, d, a, b);
}

// Shift B right logically by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshrrev_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return shift<32, rocjitsu::arith::Shift::LogicalRight>(flags, exec_mask, mode, d, a, b);
}

// Shift B right arithmetically by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_ashrrev_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return shift<32, rocjitsu::arith::Shift::ArithmeticRight>(flags, exec_mask, mode, d, a, b);
}

// Shift B left by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshlrev_b64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return shift<64, rocjitsu::arith::Shift::Left>(flags, exec_mask, mode, d, a, b);
}

// Shift B right logically by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshrrev_b64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return shift<64, rocjitsu::arith::Shift::LogicalRight>(flags, exec_mask, mode, d, a, b);
}

// Shift B right arithmetically by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_ashrrev_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return shift<64, rocjitsu::arith::Shift::ArithmeticRight>(flags, exec_mask, mode, d, a, b);
}

// Shift B left by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshlrev_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return shift_wave64<32, rocjitsu::arith::Shift::Left>(flags, exec_mask, mode, d, a, b);
}

// Shift B right logically by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshrrev_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return shift_wave64<32, rocjitsu::arith::Shift::LogicalRight>(flags, exec_mask, mode, d, a, b);
}

// Shift B right arithmetically by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_ashrrev_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return shift_wave64<32, rocjitsu::arith::Shift::ArithmeticRight>(flags, exec_mask, mode, d, a, b);
}

// Shift B left by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshlrev_b64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return shift_wave64<64, rocjitsu::arith::Shift::Left>(flags, exec_mask, mode, d, a, b);
}

// Shift B right logically by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_lshrrev_b64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return shift_wave64<64, rocjitsu::arith::Shift::LogicalRight>(flags, exec_mask, mode, d, a, b);
}

// Shift B right arithmetically by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_ashrrev_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return shift_wave64<64, rocjitsu::arith::Shift::ArithmeticRight>(flags, exec_mask, mode, d, a, b);
}
