// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 three-input minimum, maximum, median, and combined selections.
// Owns public entry points and their validation/dispatch policy.

#include "fp/half_minmax.h"
#include "common/alu.h"
#include "common/float_formats.h"
#include "common/internal.h"
#include "fp/float_output.h"
#include "fp/minmax_impl.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Select among three FP16 values for the selected lanes. Reject invalid controls before
// changing outputs.
template <rocjitsu::arith::Minmax3 Op>
int minmax3(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  using Policy = rocjitsu::arith::MinmaxPolicy<Op>;
  constexpr bool FirstMaximum = Policy::first_maximum;
  constexpr bool SecondMaximum = Policy::second_maximum;
  constexpr bool PropagateNaN = Policy::propagate;
  constexpr bool Rdna3 = Policy::rdna3;
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return minmax3<Op>(flags, exec_mask, uint32_t(mode), d, source, b, c);
        });
  }
  constexpr uint64_t fp_flags = RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE |
                                RJ_ARITH_FP_ROUND_MASK | (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0);
  if (int error = rocjitsu::arith::validate(flags, mode & ~0x1fffU, fp_flags)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) > RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if constexpr (Rdna3) {
    mode = rocjitsu::arith::rdna3_output_modifiers(flags, mode);
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::half_minmax3_x86_64_v3<Op>(flags, exec_mask, mode, d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  int a_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  int b_shift = mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  int c_shift = mode & RJ_ARITH_ALU_HIGH_C ? 16 : 0;
  int d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  // Stage results before stores, preserving source aliases and the other half of D.
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t a_bits = uint16_t(a[0][lane] >> a_shift);
    uint16_t b_bits = uint16_t(b[0][lane] >> b_shift);
    uint16_t c_bits = uint16_t(c[0][lane] >> c_shift);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      if ((a_bits & 0x7fff) < 0x0400) {
        a_bits &= 0x8000;
      }
      if ((b_bits & 0x7fff) < 0x0400) {
        b_bits &= 0x8000;
      }
      if ((c_bits & 0x7fff) < 0x0400) {
        c_bits &= 0x8000;
      }
    }
    float x = rocjitsu::arith::alu_input(
        rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(a_bits)), mode);
    float y = rocjitsu::arith::alu_input(
        rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(b_bits)), mode >> 1);
    float z = rocjitsu::arith::alu_input(
        rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(c_bits)), mode >> 2);
    float value;
    if constexpr (Rdna3) {
      value = rocjitsu::arith::minmax3_rdna3<FirstMaximum, SecondMaximum, Policy::median>(
          x, y, z, flags & RJ_ARITH_FP_IEEE_MODE);
    } else {
      value =
          rocjitsu::arith::minmax3_value<FirstMaximum, SecondMaximum, PropagateNaN, Policy::median>(
              x, y, z);
    }
    result[lane] = uint16_t(rocjitsu::arith::selection_output<16>(
        rocjitsu::arith::float_to_f16(value, false), uint32_t(mode), flags));
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Select among three FP16 values for Wave64. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::Minmax3 Op>
int minmax3_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = minmax3<Op>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
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
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32, *z = c[0] + half * 32;
    if (int error = minmax3<Op>(flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags,
                                &dest, &x, &y, &z)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Select the minimum of three values with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_min3_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3<rocjitsu::arith::Minmax3::Min3Num>(flags, exec_mask, mode, d, a, b, c);
}

// Select the maximum of three values with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_max3_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3<rocjitsu::arith::Minmax3::Max3Num>(flags, exec_mask, mode, d, a, b, c);
}

// Combine minimum then maximum selections with numeric-preferred NaN selection for selected
// lanes (V_MINMAX_NUM_F16).
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_minmax_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3<rocjitsu::arith::Minmax3::MinmaxNum>(flags, exec_mask, mode, d, a, b, c);
}

// Combine maximum then minimum selections with numeric-preferred NaN selection for selected
// lanes (V_MAXMIN_NUM_F16).
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_maxmin_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3<rocjitsu::arith::Minmax3::MaxminNum>(flags, exec_mask, mode, d, a, b, c);
}

// Select the NaN-propagating minimum of three values for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_minimum3_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3<rocjitsu::arith::Minmax3::Minimum3>(flags, exec_mask, mode, d, a, b, c);
}

// Select the NaN-propagating maximum of three values for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_maximum3_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3<rocjitsu::arith::Minmax3::Maximum3>(flags, exec_mask, mode, d, a, b, c);
}

// Combine NaN-propagating minimum then maximum selections for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_minimummaximum_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c,
                                      [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3<rocjitsu::arith::Minmax3::MinimumMaximum>(flags, exec_mask, mode, d, a, b, c);
}

// Combine NaN-propagating maximum then minimum selections for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_maximumminimum_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c,
                                      [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3<rocjitsu::arith::Minmax3::MaximumMinimum>(flags, exec_mask, mode, d, a, b, c);
}

// Select the median of three values with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_med3_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3<rocjitsu::arith::Minmax3::MedianNum>(flags, exec_mask, mode, d, a, b, c);
}

// Select the minimum of three values for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_min3_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3<rocjitsu::arith::Minmax3::Min3Legacy>(flags, exec_mask, mode, d, a, b, c);
}

// Select the maximum of three values for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_max3_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3<rocjitsu::arith::Minmax3::Max3Legacy>(flags, exec_mask, mode, d, a, b, c);
}

// Combine minimum then maximum selections for selected lanes (V_MINMAX_F16).
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_minmax_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3<rocjitsu::arith::Minmax3::MinmaxLegacy>(flags, exec_mask, mode, d, a, b, c);
}

// Combine maximum then minimum selections for selected lanes (V_MAXMIN_F16).
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_maxmin_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3<rocjitsu::arith::Minmax3::MaxminLegacy>(flags, exec_mask, mode, d, a, b, c);
}

// Select the median of three values for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_med3_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3<rocjitsu::arith::Minmax3::MedianLegacy>(flags, exec_mask, mode, d, a, b, c);
}

// Select the minimum of three values with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_min3_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3_wave64<rocjitsu::arith::Minmax3::Min3Num>(flags, exec_mask, mode, d, a, b, c);
}

// Select the maximum of three values with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_max3_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3_wave64<rocjitsu::arith::Minmax3::Max3Num>(flags, exec_mask, mode, d, a, b, c);
}

// Combine minimum then maximum selections with numeric-preferred NaN selection for selected
// lanes (V_MINMAX_NUM_F16).
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_minmax_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3_wave64<rocjitsu::arith::Minmax3::MinmaxNum>(flags, exec_mask, mode, d, a, b, c);
}

// Combine maximum then minimum selections with numeric-preferred NaN selection for selected
// lanes (V_MAXMIN_NUM_F16).
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_maxmin_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3_wave64<rocjitsu::arith::Minmax3::MaxminNum>(flags, exec_mask, mode, d, a, b, c);
}

// Select the NaN-propagating minimum of three values for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_minimum3_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3_wave64<rocjitsu::arith::Minmax3::Minimum3>(flags, exec_mask, mode, d, a, b, c);
}

// Select the NaN-propagating maximum of three values for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_maximum3_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3_wave64<rocjitsu::arith::Minmax3::Maximum3>(flags, exec_mask, mode, d, a, b, c);
}

// Combine NaN-propagating minimum then maximum selections for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_minimummaximum_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c,
                                      [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3_wave64<rocjitsu::arith::Minmax3::MinimumMaximum>(flags, exec_mask, mode, d, a, b,
                                                                  c);
}

// Combine NaN-propagating maximum then minimum selections for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_maximumminimum_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c,
                                      [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3_wave64<rocjitsu::arith::Minmax3::MaximumMinimum>(flags, exec_mask, mode, d, a, b,
                                                                  c);
}

// Select the median of three values with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_med3_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3_wave64<rocjitsu::arith::Minmax3::MedianNum>(flags, exec_mask, mode, d, a, b, c);
}

// Select the minimum of three values for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_min3_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3_wave64<rocjitsu::arith::Minmax3::Min3Legacy>(flags, exec_mask, mode, d, a, b, c);
}

// Select the maximum of three values for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_max3_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3_wave64<rocjitsu::arith::Minmax3::Max3Legacy>(flags, exec_mask, mode, d, a, b, c);
}

// Combine minimum then maximum selections for selected lanes (V_MINMAX_F16).
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_minmax_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3_wave64<rocjitsu::arith::Minmax3::MinmaxLegacy>(flags, exec_mask, mode, d, a, b, c);
}

// Combine maximum then minimum selections for selected lanes (V_MAXMIN_F16).
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_maxmin_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3_wave64<rocjitsu::arith::Minmax3::MaxminLegacy>(flags, exec_mask, mode, d, a, b, c);
}

// Select the median of three values for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_med3_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return minmax3_wave64<rocjitsu::arith::Minmax3::MedianLegacy>(flags, exec_mask, mode, d, a, b, c);
}
