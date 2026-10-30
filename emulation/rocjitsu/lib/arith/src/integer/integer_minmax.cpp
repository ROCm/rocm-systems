// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Signed/unsigned two-input and packed 16-bit min/max.
// Owns public entry points and their validation/dispatch policy.

#include "integer/integer_minmax.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Select integer minima or maxima for the selected lanes. Reject invalid controls before
// changing outputs.
template <rocjitsu::arith::IntegerMinmax Op, bool Signed>
int minmax(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
           const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return minmax<Op, Signed>(flags, exec_mask, uint32_t(mode), d, source, b, c);
        });
  }

  if (int error = rocjitsu::arith::validate(flags, mode & ~uint64_t(RJ_ARITH_ALU_CLAMP))) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
  constexpr bool binary =
      Op == rocjitsu::arith::IntegerMinmax::Min || Op == rocjitsu::arith::IntegerMinmax::Max;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::integer_minmax_x86_64_v4<Op, Signed>(exec_mask, d[0], a[0], b[0],
                                                          binary ? nullptr : c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::integer_minmax_x86_64_v3<Op, Signed>(exec_mask, d[0], a[0], b[0],
                                                          binary ? nullptr : c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];

  // Bias signed values into unsigned ordering, avoiding implementation-defined
  // uint32_t-to-int32_t conversions while preserving the selected input bits.
  const uint32_t bias = Signed ? 0x80000000U : 0;
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t x = a[0][lane] ^ bias, y = b[0][lane] ^ bias, z = 0, value;
    if constexpr (!binary) {
      z = c[0][lane] ^ bias;
    }
    if constexpr (Op == rocjitsu::arith::IntegerMinmax::Min) {
      value = rocjitsu::arith::minimum(x, y);
    }
    if constexpr (Op == rocjitsu::arith::IntegerMinmax::Max) {
      value = rocjitsu::arith::maximum(x, y);
    }
    if constexpr (Op == rocjitsu::arith::IntegerMinmax::Min3) {
      value = rocjitsu::arith::minimum(rocjitsu::arith::minimum(x, y), z);
    }
    if constexpr (Op == rocjitsu::arith::IntegerMinmax::Max3) {
      value = rocjitsu::arith::maximum(rocjitsu::arith::maximum(x, y), z);
    }

    // Mixed operations combine A/B first, as specified by the RDNA4 ISA.
    if constexpr (Op == rocjitsu::arith::IntegerMinmax::Minmax) {
      value = rocjitsu::arith::maximum(rocjitsu::arith::minimum(x, y), z);
    }
    if constexpr (Op == rocjitsu::arith::IntegerMinmax::Maxmin) {
      value = rocjitsu::arith::minimum(rocjitsu::arith::maximum(x, y), z);
    }

    // The same median selection network is used by rocjitsu vector_alu.py.
    if constexpr (Op == rocjitsu::arith::IntegerMinmax::Median) {
      value = rocjitsu::arith::maximum(rocjitsu::arith::minimum(rocjitsu::arith::maximum(x, y), z),
                                       rocjitsu::arith::minimum(x, y));
    }
    result[lane] = value ^ bias;
  }
  for (int lane = 0; lane < 32; ++lane) {
    if (exec_mask >> lane & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Select integer minima or maxima for Wave64. Reject invalid controls before changing outputs.
template <rocjitsu::arith::IntegerMinmax Op, bool Signed>
int minmax_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = minmax<Op, Signed>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
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
    const uint32_t *source_c = nullptr;
    if constexpr (Op != rocjitsu::arith::IntegerMinmax::Min &&
                  Op != rocjitsu::arith::IntegerMinmax::Max) {
      source_c = c[0] + half * 32;
    }
    if (int error =
            minmax<Op, Signed>(flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags,
                               &dest, &source_a, &source_b, &source_c)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Select the minimum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax<rocjitsu::arith::IntegerMinmax::Min, true>(flags, exec_mask, mode, d, a, b,
                                                           nullptr);
}

// Select the maximum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax<rocjitsu::arith::IntegerMinmax::Max, true>(flags, exec_mask, mode, d, a, b,
                                                           nullptr);
}

// Select the minimum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min3_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax<rocjitsu::arith::IntegerMinmax::Min3, true>(flags, exec_mask, mode, d, a, b, c);
}

// Select the maximum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max3_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax<rocjitsu::arith::IntegerMinmax::Max3, true>(flags, exec_mask, mode, d, a, b, c);
}

// Combine minimum then maximum selections for selected lanes (V_MINMAX_I32).
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_minmax_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax<rocjitsu::arith::IntegerMinmax::Minmax, true>(flags, exec_mask, mode, d, a, b, c);
}

// Combine maximum then minimum selections for selected lanes (V_MAXMIN_I32).
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_maxmin_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax<rocjitsu::arith::IntegerMinmax::Maxmin, true>(flags, exec_mask, mode, d, a, b, c);
}

// Select the median of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_med3_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax<rocjitsu::arith::IntegerMinmax::Median, true>(flags, exec_mask, mode, d, a, b, c);
}

// Select the minimum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax<rocjitsu::arith::IntegerMinmax::Min, false>(flags, exec_mask, mode, d, a, b,
                                                            nullptr);
}

// Select the maximum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax<rocjitsu::arith::IntegerMinmax::Max, false>(flags, exec_mask, mode, d, a, b,
                                                            nullptr);
}

// Select the minimum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min3_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax<rocjitsu::arith::IntegerMinmax::Min3, false>(flags, exec_mask, mode, d, a, b, c);
}

// Select the maximum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max3_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax<rocjitsu::arith::IntegerMinmax::Max3, false>(flags, exec_mask, mode, d, a, b, c);
}

// Combine minimum then maximum selections for selected lanes (V_MINMAX_U32).
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_minmax_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax<rocjitsu::arith::IntegerMinmax::Minmax, false>(flags, exec_mask, mode, d, a, b, c);
}

// Combine maximum then minimum selections for selected lanes (V_MAXMIN_U32).
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_maxmin_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax<rocjitsu::arith::IntegerMinmax::Maxmin, false>(flags, exec_mask, mode, d, a, b, c);
}

// Select the median of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_med3_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax<rocjitsu::arith::IntegerMinmax::Median, false>(flags, exec_mask, mode, d, a, b, c);
}

// Select the minimum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax_wave64<rocjitsu::arith::IntegerMinmax::Min, true>(flags, exec_mask, mode, d, a, b,
                                                                  nullptr);
}

// Select the maximum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax_wave64<rocjitsu::arith::IntegerMinmax::Max, true>(flags, exec_mask, mode, d, a, b,
                                                                  nullptr);
}

// Select the minimum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min3_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax_wave64<rocjitsu::arith::IntegerMinmax::Min3, true>(flags, exec_mask, mode, d, a, b,
                                                                   c);
}

// Select the maximum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max3_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax_wave64<rocjitsu::arith::IntegerMinmax::Max3, true>(flags, exec_mask, mode, d, a, b,
                                                                   c);
}

// Combine minimum then maximum selections for selected lanes (V_MINMAX_I32).
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_minmax_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax_wave64<rocjitsu::arith::IntegerMinmax::Minmax, true>(flags, exec_mask, mode, d, a,
                                                                     b, c);
}

// Combine maximum then minimum selections for selected lanes (V_MAXMIN_I32).
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_maxmin_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax_wave64<rocjitsu::arith::IntegerMinmax::Maxmin, true>(flags, exec_mask, mode, d, a,
                                                                     b, c);
}

// Select the median of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_med3_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax_wave64<rocjitsu::arith::IntegerMinmax::Median, true>(flags, exec_mask, mode, d, a,
                                                                     b, c);
}

// Select the minimum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax_wave64<rocjitsu::arith::IntegerMinmax::Min, false>(flags, exec_mask, mode, d, a, b,
                                                                   nullptr);
}

// Select the maximum for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax_wave64<rocjitsu::arith::IntegerMinmax::Max, false>(flags, exec_mask, mode, d, a, b,
                                                                   nullptr);
}

// Select the minimum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_min3_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax_wave64<rocjitsu::arith::IntegerMinmax::Min3, false>(flags, exec_mask, mode, d, a, b,
                                                                    c);
}

// Select the maximum of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_max3_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax_wave64<rocjitsu::arith::IntegerMinmax::Max3, false>(flags, exec_mask, mode, d, a, b,
                                                                    c);
}

// Combine minimum then maximum selections for selected lanes (V_MINMAX_U32).
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_minmax_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax_wave64<rocjitsu::arith::IntegerMinmax::Minmax, false>(flags, exec_mask, mode, d, a,
                                                                      b, c);
}

// Combine maximum then minimum selections for selected lanes (V_MAXMIN_U32).
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_maxmin_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax_wave64<rocjitsu::arith::IntegerMinmax::Maxmin, false>(flags, exec_mask, mode, d, a,
                                                                      b, c);
}

// Select the median of three values for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_med3_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return minmax_wave64<rocjitsu::arith::IntegerMinmax::Median, false>(flags, exec_mask, mode, d, a,
                                                                      b, c);
}
