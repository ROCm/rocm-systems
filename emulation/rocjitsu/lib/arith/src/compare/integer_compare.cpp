// SPDX-License-Identifier: MIT

// Signed and unsigned integer comparisons producing lane masks.
// Owns public entry points and their validation/dispatch policy.

// Comparisons follow rocjitsu's execute_v_cmp_* integer models.

#include "compare/integer_compare.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <cstring>
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
  if (int error = rocjitsu::arith::validate(flags, uint32_t(mode) & ~known, true)) {
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
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    *d =
        rocjitsu::arith::integer_compare_x86_64_v4<Bits, Signed, Predicate>(mode, a, b) & exec_mask;
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
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
  if (exec_mask) {
    for (int half = 0; half < 2; ++half) {
      uint64_t half_mode = mode;
      if (mode & RJ_ARITH_DPP16) {
        uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
        half_mode = (mode & ~RJ_ARITH_DPP_ROW_MASK) |
                    (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
      }
      constexpr int words = Bits == 64 ? 2 : 1;
      const uint32_t *source_a[words], *source_b[words];
      for (int word = 0; word < words; ++word) {
        source_a[word] = a[word] + half * 32;
        source_b[word] = b[word] + half * 32;
      }
      if (int error = compare_wave32<Bits, Signed, Predicate>(
              flags, uint32_t(exec_mask >> (half * 32)), half_mode, masks + half, source_a,
              source_b)) {
        return error;
      }
    }
  }
  uint64_t result = masks[0] | (uint64_t(masks[1]) << 32);

  // The mask may start at an unaligned byte within aliased VGPR storage.
  // Erase the pointer alignment before the byte copy (also for UBSan).
  std::memcpy(static_cast<void *>(d), &result, sizeof result);
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
  return compare_wave32<16, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<16, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<32, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave32<64, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<16, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<32, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
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
  return compare_wave64<64, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
}
