// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Always-true and always-false comparisons, including CMPX forms.
// Owns public entry points and their validation/dispatch policy.

// RDNA3 constant predicates follow rocjitsu execute_v_cmp_[ft]_* for results.
// Floating exception reporting needs a separate RDNA3 model.

#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Produce constant comparison masks for the selected lanes. Reject invalid controls before
// changing outputs.
template <typename Mask>
int compare(uint64_t flags, Mask exec_mask, uint64_t mode, Mask *d, bool value, unsigned bits,
            bool floating = false, [[maybe_unused]] uint32_t *trapsts = nullptr) {
  if (floating && rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    if (bits == 64 || !(mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16)) ||
        !rocjitsu::arith::valid_dpp(mode)) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }
  const uint32_t known = floating ? RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B |
                                        RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B |
                                        (bits == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0)
                                  : 0;
  const uint64_t fp = floating ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                     RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE
                               : 0;
  if (int error = rocjitsu::arith::validate(flags, uint32_t(mode) & ~known, true, fp)) {
    return error;
  }
  if (value && exec_mask && (mode & RJ_ARITH_DPP16)) {
    // Operand values are irrelevant, but DPP row/bank/boundary filtering is not.
    uint32_t zero[32] = {}, ignored[32];
    Mask filtered = 0;
    const unsigned rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
    for (int half = 0; half < int(sizeof(Mask) / sizeof(uint32_t)); ++half) {
      // DPP routes within 16-lane rows; Wave64 has four independent row bits.
      const uint64_t half_mode = (mode & ~RJ_ARITH_DPP_ROW_MASK) |
                                 (uint64_t((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
      filtered |= Mask(rocjitsu::arith::dpp16_source(flags, uint32_t(exec_mask >> (half * 32)),
                                                     half_mode, ignored, zero))
                  << (half * 32);
    }
    exec_mask = filtered;
  }
  *d = value ? exec_mask : 0;
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Write the Wave32 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_f_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

// Write the Wave32 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_t_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

// Write the Wave32 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_f_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

// Write the Wave32 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_t_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

// Write the Wave32 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_f_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

// Write the Wave32 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_t_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

// Write the Wave32 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_f_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

// Write the Wave32 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_t_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

// Write the Wave32 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_f_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

// Write the Wave32 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_t_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

// Write the Wave32 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_f_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

// Write the Wave32 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_t_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

// Write the Wave32 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_f_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

// Write the Wave32 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_t_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

// Write the Wave32 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_f_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

// Write the Wave32 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_t_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

// Write the Wave32 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_f_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *, const uint32_t *const *,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 16, true, trapsts);
}

// Write the Wave32 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_t_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *, const uint32_t *const *,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 16, true, trapsts);
}

// Write the Wave32 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_f_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *, const uint32_t *const *,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 16, true, trapsts);
}

// Write the Wave32 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_t_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *, const uint32_t *const *,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 16, true, trapsts);
}

// Write the Wave32 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_f_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *, const uint32_t *const *,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 32, true, trapsts);
}

// Write the Wave32 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_t_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *, const uint32_t *const *,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 32, true, trapsts);
}

// Write the Wave32 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_f_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *, const uint32_t *const *,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 32, true, trapsts);
}

// Write the Wave32 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_t_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *, const uint32_t *const *,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 32, true, trapsts);
}

// Write the Wave32 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_f_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *, const uint32_t *const *,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 64, true, trapsts);
}

// Write the Wave32 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_t_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *, const uint32_t *const *,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 64, true, trapsts);
}

// Write the Wave32 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_f_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *, const uint32_t *const *,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 64, true, trapsts);
}

// Write the Wave32 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_t_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *, const uint32_t *const *,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 64, true, trapsts);
}

// Write the Wave64 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_f_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

// Write the Wave64 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_t_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

// Write the Wave64 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_f_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

// Write the Wave64 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_t_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

// Write the Wave64 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_f_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

// Write the Wave64 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_t_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

// Write the Wave64 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_f_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

// Write the Wave64 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_t_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

// Write the Wave64 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_f_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

// Write the Wave64 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_t_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

// Write the Wave64 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_f_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

// Write the Wave64 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_t_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

// Write the Wave64 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_f_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

// Write the Wave64 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_t_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

// Write the Wave64 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_f_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

// Write the Wave64 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_t_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

// Write the Wave64 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_f_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 16, true, trapsts);
}

// Write the Wave64 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_t_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 16, true, trapsts);
}

// Write the Wave64 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_f_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 16, true, trapsts);
}

// Write the Wave64 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_t_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 16, true, trapsts);
}

// Write the Wave64 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_f_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 32, true, trapsts);
}

// Write the Wave64 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_t_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 32, true, trapsts);
}

// Write the Wave64 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_f_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 32, true, trapsts);
}

// Write the Wave64 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_t_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 32, true, trapsts);
}

// Write the Wave64 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_f_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 64, true, trapsts);
}

// Write the Wave64 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_t_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 64, true, trapsts);
}

// Write the Wave64 mask for false; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_f_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, false, 64, true, trapsts);
}

// Write the Wave64 mask for true; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_t_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare(flags, exec_mask, instruction_flags, d, true, 64, true, trapsts);
}
