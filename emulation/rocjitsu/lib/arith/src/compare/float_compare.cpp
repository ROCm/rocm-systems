// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Ordered/unordered floating-point predicates and optional FP exception reporting.
// Owns public entry points and their validation/dispatch policy.

// Raw ordering, signed-zero normalization and predicate inversion follow
// rocjitsu's shared/comparison.h, with guest input flushing before ordering.

#include "common/internal.h"
#include "compare/float_compare_impl.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Evaluate ordered/unordered FP predicates for the selected lanes. Reject invalid controls
// before changing outputs.
template <int Bits, unsigned Predicate>
int compare_wave32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                   const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (mode >> 32) {
    if constexpr (Bits == 64) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
    if (!(mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16)) || !rocjitsu::arith::valid_dpp(mode)) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }
  const uint32_t known = RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B |
                         RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B |
                         (Bits == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0);
  if (int error = rocjitsu::arith::validate(flags, uint32_t(mode) & ~known,
                                            RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                                                RJ_ARITH_FP_IEEE_MODE)) {
    return error;
  }
  if (!exec_mask) {
    *d = 0;
    return RJ_ARITH_SUCCESS;
  }
  if constexpr (Bits != 64) {
    if (mode >> 32) {
      uint32_t permuted[32];
      const uint32_t *source = permuted;
      if (mode & RJ_ARITH_DPP8) {
        rocjitsu::arith::dpp8_source(flags, exec_mask, mode, permuted, a[0]);
      } else {
        exec_mask = rocjitsu::arith::dpp16_source(flags, exec_mask, mode, permuted, a[0]);
      }
      return compare_wave32<Bits, Predicate>(flags, exec_mask, uint32_t(mode), d, &source, b,
                                             excp_flag_user);
    }
  }
  bool flush = flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  const bool report = rocjitsu::arith::reports_fp_status(flags);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (!report && rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    *d = rocjitsu::arith::float_compare_x86_64_v4<Bits, Predicate>(mode, flush, a, b) & exec_mask;
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (!report && rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    *d = rocjitsu::arith::float_compare_x86_64_v3<Bits, Predicate>(mode, flush, a, b) & exec_mask;
    return RJ_ARITH_SUCCESS;
  }
#endif
  constexpr uint64_t sign = 1ULL << (Bits - 1), magnitude = sign - 1;
  constexpr uint64_t infinity = Bits == 16   ? 0x7c00
                                : Bits == 32 ? 0x7f800000
                                             : 0x7ff0000000000000ULL;
  uint32_t less = 0, equal = 0, ordered = 0, exceptions = 0;

  // Keep baseline variable shifts from introducing host FP exceptions.
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
  for (int lane = 0; lane < 32; ++lane) {
    uint64_t av = a[0][lane], bv = b[0][lane];
    if constexpr (Bits == 64) {
      av |= uint64_t(a[1][lane]) << 32;
      bv |= uint64_t(b[1][lane]) << 32;
    }
    if constexpr (Bits == 16) {
      av = (av >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)) & 65535;
      bv = (bv >> (mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0)) & 65535;
    }
    if (report) {
      exceptions |= rocjitsu::arith::float_compare_exceptions<Bits>(av, bv, flush,
                                                                    mode & RJ_ARITH_ALU_CLAMP) *
                    ((exec_mask >> lane) & 1);
    }
    if (mode & RJ_ARITH_ALU_ABS_A) {
      av &= magnitude;
    }
    if (mode & RJ_ARITH_ALU_ABS_B) {
      bv &= magnitude;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      av ^= sign;
    }
    if (mode & RJ_ARITH_ALU_NEG_B) {
      bv ^= sign;
    }
    if (flush) {
      if (!(av & infinity)) {
        av &= sign;
      }
      if (!(bv & infinity)) {
        bv &= sign;
      }
    }
    bool ord = (av & magnitude) <= infinity && (bv & magnitude) <= infinity;
    if (!(av & magnitude)) {
      av = 0;
    }
    if (!(bv & magnitude)) {
      bv = 0;
    }
    av ^= av & sign ? (sign | magnitude) : sign;
    bv ^= bv & sign ? (sign | magnitude) : sign;
    less |= uint32_t(av < bv) << lane;
    equal |= uint32_t(av == bv) << lane;
    ordered |= uint32_t(ord) << lane;
  }
  *d = rocjitsu::arith::float_compare_result<Predicate>(less, equal, ordered) & exec_mask;
  if (report) {
    *excp_flag_user |= exceptions;
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate ordered/unordered FP predicates for Wave64. Reject invalid controls before changing
// outputs.
template <int Bits, unsigned Predicate>
int compare_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                   const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  uint32_t masks[2] = {}, exceptions = 0;

  // Validate before source accesses or writes. Stage both halves so result or
  // exception storage may overlap sources without changing the second half.
  if (int error =
          compare_wave32<Bits, Predicate>(flags, 0, mode, masks, nullptr, nullptr, nullptr)) {
    return error;
  }
  const bool report = rocjitsu::arith::reports_fp_status(flags);
  if (exec_mask) {
    for (int half = 0; half < 2; ++half) {
      uint64_t half_instruction_flags = mode;
      if (mode & RJ_ARITH_DPP16) {
        uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
        half_instruction_flags = (mode & ~RJ_ARITH_DPP_ROW_MASK) |
                                 (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
      }
      constexpr int words = Bits == 64 ? 2 : 1;
      const uint32_t *source_a[words], *source_b[words];
      for (int word = 0; word < words; ++word) {
        source_a[word] = a[word] + half * 32;
        source_b[word] = b[word] + half * 32;
      }
      if (int error = compare_wave32<Bits, Predicate>(
              flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags, masks + half,
              source_a, source_b, report ? &exceptions : nullptr)) {
        return error;
      }
    }
  }
  uint64_t result = masks[0] | (uint64_t(masks[1]) << 32);

  // The mask may start at an unaligned byte within aliased VGPR storage.
  // Erase the pointer alignment before the byte copy (also for UBSan).
  __builtin_memcpy(static_cast<void *>(d), &result, sizeof result);
  if (report) {
    *excp_flag_user |= exceptions;
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_lt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_eq_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_le_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_gt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_lg_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_ge_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_o_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_u_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nge_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nlg_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_ngt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nle_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_neq_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nlt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_lt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_eq_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_le_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_gt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_lg_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_ge_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_o_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_u_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nge_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nlg_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_ngt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nle_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_neq_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nlt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_lt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_eq_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_le_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_gt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_lg_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_ge_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_o_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_u_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nge_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nlg_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_ngt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nle_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_neq_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nlt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_lt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_eq_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_le_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_gt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_lg_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_ge_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_o_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_u_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nge_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nlg_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_ngt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nle_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_neq_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nlt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_lt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_eq_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_le_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_gt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_lg_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_ge_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_o_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_u_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nge_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nlg_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_ngt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nle_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_neq_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nlt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_lt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_eq_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_le_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_gt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_lg_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_ge_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_o_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_u_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nge_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nlg_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_ngt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nle_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_neq_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nlt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_lt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_eq_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_le_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_gt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_lg_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_ge_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_o_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_u_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nge_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nlg_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_ngt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nle_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_neq_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nlt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_lt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_eq_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_le_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_gt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_lg_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_ge_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_o_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_u_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nge_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nlg_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_ngt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nle_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_neq_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nlt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_lt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_eq_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_le_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_gt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_lg_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_ge_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_o_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_u_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nge_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nlg_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_ngt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nle_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_neq_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nlt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_lt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_eq_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_le_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_gt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_lg_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_ge_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_o_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_u_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nge_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nlg_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_ngt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nle_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_neq_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nlt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_lt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_eq_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_le_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_gt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_lg_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_ge_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_o_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_u_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nge_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nlg_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_ngt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nle_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_neq_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmp_nlt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_lt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_eq_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_le_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_gt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_lg_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_ge_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_o_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_u_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nge_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nlg_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_ngt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nle_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_neq_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave64 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs. FP causes accumulate only when resolved semantics request
// reporting.
int rj_arith_v_cmpx_nlt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_lt_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_eq_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_le_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_gt_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lg_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_lg_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_ge_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_o_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_o_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_u_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_u_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nge_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nge_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nlg_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nlg_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ngt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_ngt_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nle_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nle_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_neq_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_neq_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nlt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nlt_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_lt_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_eq_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_le_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_gt_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lg_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_lg_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_ge_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_o_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_o_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_u_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_u_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nge_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nge_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nlg_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nlg_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ngt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_ngt_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nle_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nle_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_neq_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_neq_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nlt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nlt_f16_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_lt_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_eq_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_le_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_gt_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lg_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_lg_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_ge_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_o_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_o_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_u_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_u_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nge_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nge_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nlg_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nlg_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ngt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_ngt_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nle_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nle_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_neq_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_neq_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nlt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nlt_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_lt_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_eq_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_le_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_gt_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lg_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_lg_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_ge_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_o_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_o_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_u_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_u_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nge_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nge_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nlg_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nlg_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ngt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_ngt_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nle_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nle_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_neq_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_neq_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nlt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nlt_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_lt_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_eq_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_le_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_gt_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lg_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_lg_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_ge_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_o_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_o_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_u_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_u_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nge_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nge_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nlg_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nlg_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ngt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_ngt_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nle_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nle_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_neq_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_neq_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nlt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nlt_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_lt_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_eq_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_le_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_gt_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lg_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_lg_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_ge_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_o_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_o_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_u_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_u_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nge_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nge_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nlg_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nlg_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ngt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_ngt_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nle_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nle_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_neq_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_neq_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave32 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nlt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nlt_f64_w32_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_lt_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_eq_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_le_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_gt_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lg_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_lg_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_ge_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_o_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_o_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_u_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_u_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nge_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nge_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nlg_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nlg_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ngt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_ngt_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nle_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nle_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_neq_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_neq_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nlt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nlt_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_lt_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_eq_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_le_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_gt_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lg_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_lg_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_ge_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_o_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_o_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_u_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_u_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nge_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nge_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nlg_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nlg_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ngt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_ngt_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nle_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nle_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_neq_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_neq_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nlt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nlt_f16_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_lt_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_eq_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_le_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_gt_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lg_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_lg_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_ge_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_o_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_o_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_u_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_u_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nge_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nge_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nlg_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nlg_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ngt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_ngt_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nle_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nle_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_neq_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_neq_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nlt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nlt_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_lt_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_eq_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_le_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_gt_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lg_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_lg_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_ge_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_o_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_o_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_u_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_u_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nge_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nge_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nlg_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nlg_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ngt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_ngt_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nle_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nle_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_neq_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_neq_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nlt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nlt_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_lt_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_eq_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_le_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_gt_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lg_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_lg_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_ge_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_o_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_o_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_u_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_u_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nge_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nge_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nlg_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nlg_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ngt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_ngt_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nle_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nle_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_neq_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_neq_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_nlt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmp_nlt_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_lt_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_eq_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_le_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_gt_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for ordered and a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lg_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_lg_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_ge_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for both operands ordered; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_o_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_o_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for either operand NaN; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_u_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_u_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a >= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nge_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nge_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for unordered or a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nlg_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nlg_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a > b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ngt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_ngt_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a <= b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nle_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nle_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for unordered or a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_neq_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_neq_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}

// Write the Wave64 mask for not (a < b); inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_nlt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cmpx_nlt_f64_w64_gfx12(flags, exec_mask, mode, d, a, b, nullptr);
}
