// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Signed and unsigned integer comparisons producing lane masks.
// Owns public entry points and their validation/dispatch policy.

// Comparisons follow rocjitsu's execute_v_cmp_* integer models.

#include "compare/integer_compare.h"
#include "common/validation.h"
#include "common/wave_alignment.h"
#include "compare/predicate.h"
#include "register/dpp.h"
#include "rj_arith/detail/rj_arith_compare.h"

#include <stdint.h>

namespace {

// Compare signed/unsigned integers for the selected lanes. Reject invalid controls before
// changing outputs.
template <int Bits, bool Signed, unsigned Predicate>
int compare_wave32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                   const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    if constexpr (Bits == 64) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
    if (!(mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16)) || !rocjitsu::arith::valid_dpp(mode)) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }
  const uint32_t known = Bits == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0;
  if (int error = rocjitsu::arith::validate(flags, uint32_t(mode) & ~known)) {
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
      return compare_wave32<Bits, Signed, Predicate>(flags, exec_mask, uint32_t(mode), d, &source,
                                                     b);
    }
  }
  [[maybe_unused]] const auto level =
      rocjitsu::arith::wave_cpu_level(flags, Bits != 64 ||
                                                 (Predicate != rocjitsu::arith::CompareEqual &&
                                                  Predicate != rocjitsu::arith::CompareNotEqual) ||
                                                 rocjitsu::arith::aligned_for_512(a[0]));
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (level >= RJ_ARITH_CPU_X86_64_V4) {
    *d = rocjitsu::arith::integer_compare_x86_64_v4<Bits, Signed, Predicate>(32, mode, a, b) &
         exec_mask;
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (level >= RJ_ARITH_CPU_X86_64_V3) {
    *d =
        rocjitsu::arith::integer_compare_x86_64_v3<Bits, Signed, Predicate>(mode, a, b) & exec_mask;
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t less = 0, equal = 0;

  // Baseline x86 vectorized variable shifts can raise host FP exceptions.
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
  for (int lane = 0; lane < 32; ++lane) {
    uint64_t av = a[0][lane], bv = b[0][lane];
    if constexpr (Bits == 16) {
      av = (av >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)) & 65535;
      bv = (bv >> (mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0)) & 65535;
    }
    if constexpr (Bits == 64) {
      av |= uint64_t(a[1][lane]) << 32;
      bv |= uint64_t(b[1][lane]) << 32;
    }
    if constexpr (Signed) {
      av ^= 1ULL << (Bits - 1);
      bv ^= 1ULL << (Bits - 1);
    }
    less |= uint32_t(av < bv) << lane;
    equal |= uint32_t(av == bv) << lane;
  }
  *d = rocjitsu::arith::integer_compare_result<Predicate>(less, equal) & exec_mask;
  return RJ_ARITH_SUCCESS;
}

// Accepted DPP controls are row-local. Compute both halves before publishing
// the scalar mask, since its storage may overlap either half of either source.
template <int Bits, bool Signed, unsigned Predicate>
int compare_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                   const uint32_t *const *a, const uint32_t *const *b) {
  uint32_t masks[2] = {};

  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error =
          compare_wave32<Bits, Signed, Predicate>(flags, 0, mode, masks, nullptr, nullptr)) {
    return error;
  }
  if constexpr (Bits == 64) {
    if (exec_mask && !(exec_mask & (exec_mask - 1))) {
      int lane = __builtin_ctzll(exec_mask);
      uint64_t x = a[0][lane] | (uint64_t(a[1][lane]) << 32);
      uint64_t y = b[0][lane] | (uint64_t(b[1][lane]) << 32);
      if constexpr (Signed) {
        x ^= uint64_t(1) << 63;
        y ^= uint64_t(1) << 63;
      }
      bool selected = rocjitsu::arith::integer_compare_result<Predicate>(x < y, x == y) & 1;
      uint64_t result = selected ? exec_mask : 0;
      __builtin_memcpy(static_cast<void *>(d), &result, sizeof result);
      return RJ_ARITH_SUCCESS;
    }
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (exec_mask && !(mode >> 32) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    uint64_t result = rocjitsu::arith::integer_compare_x86_64_v4<Bits, Signed, Predicate>(
                          64, uint32_t(mode), a, b) &
                      exec_mask;
    __builtin_memcpy(static_cast<void *>(d), &result, sizeof result);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  // The wave controls are already validated. Skip empty halves and reuse the first
  // half's VGPR pointers without a second public dispatch.
  if (exec_mask && !(mode >> 32) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    if (uint32_t(exec_mask)) {
      masks[0] = rocjitsu::arith::integer_compare_x86_64_v3<Bits, Signed, Predicate>(mode, a, b);
    }
    if (exec_mask >> 32) {
      constexpr int words = Bits == 64 ? 2 : 1;
      const uint32_t *high_a[words], *high_b[words];
      for (int word = 0; word < words; ++word) {
        high_a[word] = a[word] + 32;
        high_b[word] = b[word] + 32;
      }
      masks[1] =
          rocjitsu::arith::integer_compare_x86_64_v3<Bits, Signed, Predicate>(mode, high_a, high_b);
    }
    uint64_t result = (uint64_t(masks[0]) | (uint64_t(masks[1]) << 32)) & exec_mask;
    __builtin_memcpy(static_cast<void *>(d), &result, sizeof result);
    return RJ_ARITH_SUCCESS;
  }
#endif
  if (exec_mask) {
    for (int half = 0; half < 2; ++half) {
      uint32_t half_mask = uint32_t(exec_mask >> (half * 32));
      if (!half_mask) {
        continue;
      }
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
      if (int error = compare_wave32<Bits, Signed, Predicate>(
              flags, half_mask, half_instruction_flags, masks + half, source_a, source_b)) {
        return error;
      }
    }
  }
  uint64_t result = masks[0] | (uint64_t(masks[1]) << 32);

  // The mask may start at an unaligned byte within aliased VGPR storage.
  // Erase the pointer alignment before the byte copy (also for UBSan).
  __builtin_memcpy(static_cast<void *>(d), &result, sizeof result);
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, true, rocjitsu::arith::CompareLess>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, true, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, true, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, true, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Write the Wave32 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ne_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, true, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, true, rocjitsu::arith::CompareGreaterEqual>(flags, exec_mask,
                                                                        instruction_flags, d, a, b);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, true, rocjitsu::arith::CompareLess>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, true, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, true, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, true, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Write the Wave32 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ne_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, true, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, true, rocjitsu::arith::CompareGreaterEqual>(flags, exec_mask,
                                                                        instruction_flags, d, a, b);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, false, rocjitsu::arith::CompareLess>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, false, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, false, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, false, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave32 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ne_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, false, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, false, rocjitsu::arith::CompareGreaterEqual>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, false, rocjitsu::arith::CompareLess>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, false, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, false, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, false, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave32 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ne_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, false, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<16, false, rocjitsu::arith::CompareGreaterEqual>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, true, rocjitsu::arith::CompareLess>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, true, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, true, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, true, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Write the Wave32 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ne_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, true, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, true, rocjitsu::arith::CompareGreaterEqual>(flags, exec_mask,
                                                                        instruction_flags, d, a, b);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, true, rocjitsu::arith::CompareLess>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, true, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, true, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, true, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Write the Wave32 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ne_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, true, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, true, rocjitsu::arith::CompareGreaterEqual>(flags, exec_mask,
                                                                        instruction_flags, d, a, b);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, false, rocjitsu::arith::CompareLess>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, false, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, false, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, false, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave32 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ne_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, false, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, false, rocjitsu::arith::CompareGreaterEqual>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, false, rocjitsu::arith::CompareLess>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, false, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, false, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, false, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave32 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ne_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, false, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<32, false, rocjitsu::arith::CompareGreaterEqual>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, true, rocjitsu::arith::CompareLess>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, true, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, true, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, true, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Write the Wave32 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ne_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, true, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, true, rocjitsu::arith::CompareGreaterEqual>(flags, exec_mask,
                                                                        instruction_flags, d, a, b);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, true, rocjitsu::arith::CompareLess>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, true, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, true, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, true, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Write the Wave32 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ne_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, true, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, true, rocjitsu::arith::CompareGreaterEqual>(flags, exec_mask,
                                                                        instruction_flags, d, a, b);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, false, rocjitsu::arith::CompareLess>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, false, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, false, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, false, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave32 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ne_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, false, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, false, rocjitsu::arith::CompareGreaterEqual>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave32 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, false, rocjitsu::arith::CompareLess>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave32 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, false, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Write the Wave32 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, false, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Write the Wave32 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, false, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave32 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ne_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, false, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave32 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave32<64, false, rocjitsu::arith::CompareGreaterEqual>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, true, rocjitsu::arith::CompareLess>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, true, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, true, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, true, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Write the Wave64 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ne_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, true, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, true, rocjitsu::arith::CompareGreaterEqual>(flags, exec_mask,
                                                                        instruction_flags, d, a, b);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, true, rocjitsu::arith::CompareLess>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, true, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, true, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, true, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Write the Wave64 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ne_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, true, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, true, rocjitsu::arith::CompareGreaterEqual>(flags, exec_mask,
                                                                        instruction_flags, d, a, b);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, false, rocjitsu::arith::CompareLess>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, false, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, false, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, false, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave64 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ne_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, false, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, false, rocjitsu::arith::CompareGreaterEqual>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, false, rocjitsu::arith::CompareLess>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, false, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, false, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, false, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave64 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ne_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, false, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<16, false, rocjitsu::arith::CompareGreaterEqual>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, true, rocjitsu::arith::CompareLess>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, true, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, true, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, true, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Write the Wave64 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ne_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, true, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, true, rocjitsu::arith::CompareGreaterEqual>(flags, exec_mask,
                                                                        instruction_flags, d, a, b);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, true, rocjitsu::arith::CompareLess>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, true, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, true, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, true, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Write the Wave64 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ne_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, true, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, true, rocjitsu::arith::CompareGreaterEqual>(flags, exec_mask,
                                                                        instruction_flags, d, a, b);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, false, rocjitsu::arith::CompareLess>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, false, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, false, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, false, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave64 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ne_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, false, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, false, rocjitsu::arith::CompareGreaterEqual>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, false, rocjitsu::arith::CompareLess>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, false, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, false, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, false, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave64 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ne_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, false, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<32, false, rocjitsu::arith::CompareGreaterEqual>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, true, rocjitsu::arith::CompareLess>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, true, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, true, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, true, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Write the Wave64 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ne_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, true, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, true, rocjitsu::arith::CompareGreaterEqual>(flags, exec_mask,
                                                                        instruction_flags, d, a, b);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, true, rocjitsu::arith::CompareLess>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, true, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, true, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, true, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Write the Wave64 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ne_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, true, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, true, rocjitsu::arith::CompareGreaterEqual>(flags, exec_mask,
                                                                        instruction_flags, d, a, b);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_lt_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, false, rocjitsu::arith::CompareLess>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_eq_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, false, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_le_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, false, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_gt_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, false, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave64 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ne_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, false, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmp_ge_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, false, rocjitsu::arith::CompareGreaterEqual>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave64 mask for a < b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_lt_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, false, rocjitsu::arith::CompareLess>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Write the Wave64 mask for a == b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_eq_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, false, rocjitsu::arith::CompareEqual>(flags, exec_mask,
                                                                  instruction_flags, d, a, b);
}

// Write the Wave64 mask for a <= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_le_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, false, rocjitsu::arith::CompareLessEqual>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Write the Wave64 mask for a > b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_gt_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, false, rocjitsu::arith::CompareGreater>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Write the Wave64 mask for a != b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ne_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, false, rocjitsu::arith::CompareNotEqual>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Write the Wave64 mask for a >= b; inactive lane bits are zero.
// Errors preserve outputs.
int rj_arith_v_cmpx_ge_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return compare_wave64<64, false, rocjitsu::arith::CompareGreaterEqual>(
      flags, exec_mask, instruction_flags, d, a, b);
}
