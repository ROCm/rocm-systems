// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Gfx1150 floating masks and the inherited DX10_CLAMP-gated INVALID policy.
// Numeric predicates and raw NaN classes reuse scalar/AVX2/AVX-512 backends.

#include "common/validation.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// CLASS consumes one class-mask word per lane; bit 0 selects signaling NaNs.
constexpr auto signaling_nan_classes = [] {
  struct Classes {
    uint32_t words[64];
  } result{};
  for (int lane = 0; lane < 64; ++lane) {
    result.words[lane] = 1;
  }
  return result;
}();

// Numeric comparison signature for a fixed wave-sized result mask.
template <typename Mask>
using NumericCompare = int (*)(uint64_t, Mask, uint64_t, Mask *, const uint32_t *const *,
                               const uint32_t *const *, uint32_t *);

// Evaluate Compatibility with DX10_CLAMP disabled, staging all reads before stores.
// Keeping its scratch storage here leaves numeric-only calls free of this frame.
template <int Bits, int Lanes, auto Unordered, auto Classify, typename Mask>
[[gnu::noinline]] int compare_gfx1150_status(uint64_t flags, Mask exec_mask, uint64_t mode, Mask *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             uint32_t *trapsts, NumericCompare<Mask> numeric) {
  flags = (flags & ~(RJ_ARITH_FP_DX10_CLAMP_OFF | RJ_ARITH_SEMANTICS_MASK)) |
          RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
  // Classification is fixed by format; keep the numeric predicate an argument
  // so all predicates share one status implementation per format and wave size.
  Mask result = 0, invalid = 0;
  if (int error = numeric(flags, exec_mask, mode, &result, a, b, nullptr)) {
    return error;
  }
  if (exec_mask) {
    if (mode & RJ_ARITH_ALU_CLAMP) {
      // Unordered masks identify either NaN without introducing a scalar scan.
      if (int error = Unordered(flags, exec_mask, mode, &invalid, a, b, nullptr)) {
        return error;
      }
    } else {
      const uint32_t *classes = signaling_nan_classes.words;
      uint32_t permuted[Lanes];
      const uint32_t *selected = nullptr;
      uint64_t class_mode = mode & RJ_ARITH_ALU_HIGH_A;
      const uint32_t *const *class_a = a;
      // DPP transports A and can suppress participating lanes. Apply that once
      // before classifying either operand; B must not itself be permuted.
      if constexpr (Bits != 64) {
        if (mode >> 32) {
          Mask class_exec_mask = 0;
          for (int base = 0; base < Lanes; base += 32) {
            uint32_t half_exec_mask = uint32_t(exec_mask >> base);
            if (mode & RJ_ARITH_DPP8) {
              rocjitsu::arith::dpp8_source(flags, half_exec_mask, mode, permuted + base,
                                           a[0] + base);
            } else {
              const uint64_t routing =
                  rocjitsu::arith::dpp_half_wave_instruction_flags(mode, base / 32);
              half_exec_mask = rocjitsu::arith::dpp16_source(flags, half_exec_mask, routing,
                                                             permuted + base, a[0] + base);
            }
            class_exec_mask |= Mask(half_exec_mask) << base;
          }
          exec_mask = class_exec_mask;
          selected = permuted;
          class_a = &selected;
        }
      }
      Mask first = 0, second = 0;
      if (int error = Classify(flags, exec_mask, class_mode, &first, class_a, &classes)) {
        return error;
      }
      class_mode = mode & RJ_ARITH_ALU_HIGH_B ? RJ_ARITH_ALU_HIGH_A : 0;
      if (int error = Classify(flags, exec_mask, class_mode, &second, b, &classes)) {
        return error;
      }
      invalid = first | second;
    }
  }
  __builtin_memcpy(static_cast<void *>(d), &result, sizeof result);
  if (invalid) {
    *trapsts |= RJ_ARITH_EXCEPTION_INVALID;
  }
  return RJ_ARITH_SUCCESS;
}

// Numeric-only calls delegate validation and dispatch directly to their comparator.
// Inline this small policy choice so the status helper's scratch frame stays off
// Loose and DX10_CLAMP-enabled paths, including their tail calls.
template <int Bits, int Lanes, auto Unordered, auto Classify, typename Mask>
[[gnu::always_inline]] inline int
compare_gfx1150(uint64_t flags, Mask exec_mask, uint64_t mode, Mask *d, const uint32_t *const *a,
                const uint32_t *const *b, uint32_t *trapsts, NumericCompare<Mask> numeric) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY &&
      (flags & RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return compare_gfx1150_status<Bits, Lanes, Unordered, Classify>(flags, exec_mask, mode, d, a, b,
                                                                    trapsts, numeric);
  }
  return numeric(flags & ~RJ_ARITH_FP_DX10_CLAMP_OFF, exec_mask, mode, d, a, b, trapsts);
}

} // namespace

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_f_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_f_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_f_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_f_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_lt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_lt_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_lt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_lt_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_eq_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_eq_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_eq_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_eq_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_le_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_le_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_le_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_le_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_gt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_gt_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_gt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_gt_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_lg_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_lg_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_lg_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_lg_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_ge_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_ge_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_ge_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_ge_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_o_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_o_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_o_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_o_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_u_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_u_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_u_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nge_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nge_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nge_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nge_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nlg_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nlg_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nlg_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nlg_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_ngt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_ngt_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_ngt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_ngt_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nle_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nle_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nle_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nle_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_neq_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_neq_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_neq_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_neq_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nlt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nlt_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nlt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nlt_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_t_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_t_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_t_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_t_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_f_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_f_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_f_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_f_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_lt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_lt_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_lt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_lt_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_eq_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_eq_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_eq_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_eq_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_le_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_le_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_le_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_le_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_gt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_gt_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_gt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_gt_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_lg_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_lg_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_lg_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_lg_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_ge_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_ge_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_ge_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_ge_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_o_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_o_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_o_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_o_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_u_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_u_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_u_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_u_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nge_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nge_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nge_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nge_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nlg_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nlg_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nlg_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nlg_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_ngt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_ngt_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_ngt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_ngt_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nle_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nle_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nle_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nle_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_neq_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_neq_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_neq_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_neq_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nlt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nlt_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nlt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nlt_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_t_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 32, rj_arith_v_cmp_u_f16_w32, rj_arith_v_cmp_class_f16_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_t_f16_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_t_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<16, 64, rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_class_f16_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_t_f16_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_f_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_f_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_f_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_f_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_lt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_lt_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_lt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_lt_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_eq_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_eq_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_eq_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_eq_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_le_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_le_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_le_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_le_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_gt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_gt_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_gt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_gt_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_lg_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_lg_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_lg_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_lg_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_ge_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_ge_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_ge_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_ge_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_o_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_o_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_o_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_o_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_u_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_u_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nge_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nge_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nge_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nge_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nlg_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nlg_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nlg_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nlg_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_ngt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_ngt_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_ngt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_ngt_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nle_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nle_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nle_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nle_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_neq_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_neq_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_neq_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_neq_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nlt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nlt_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nlt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nlt_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_t_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_t_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_t_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_t_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_f_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_f_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_f_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_f_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_lt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_lt_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_lt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_lt_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_eq_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_eq_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_eq_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_eq_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_le_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_le_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_le_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_le_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_gt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_gt_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_gt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_gt_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_lg_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_lg_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_lg_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_lg_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_ge_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_ge_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_ge_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_ge_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_o_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_o_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_o_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_o_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_u_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_u_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_u_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_u_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nge_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nge_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nge_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nge_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nlg_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nlg_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nlg_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nlg_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_ngt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_ngt_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_ngt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_ngt_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nle_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nle_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nle_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nle_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_neq_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_neq_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_neq_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_neq_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nlt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nlt_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nlt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nlt_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_t_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 32, rj_arith_v_cmp_u_f32_w32, rj_arith_v_cmp_class_f32_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_t_f32_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_t_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<32, 64, rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_class_f32_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_t_f32_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_f_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_f_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_f_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_f_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_lt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_lt_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_lt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_lt_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_eq_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_eq_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_eq_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_eq_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_le_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_le_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_le_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_le_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_gt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_gt_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_gt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_gt_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_lg_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_lg_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_lg_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_lg_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_ge_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_ge_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_ge_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_ge_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_o_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_o_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_o_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_o_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_u_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_u_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nge_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nge_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nge_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nge_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nlg_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nlg_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nlg_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nlg_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_ngt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_ngt_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_ngt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_ngt_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nle_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nle_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nle_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nle_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_neq_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_neq_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_neq_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_neq_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nlt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nlt_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_nlt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_nlt_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_t_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_t_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmp_t_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_t_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_f_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_f_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_f_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_f_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_lt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_lt_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_lt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_lt_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_eq_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_eq_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_eq_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_eq_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_le_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_le_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_le_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_le_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_gt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_gt_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_gt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_gt_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_lg_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_lg_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_lg_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_lg_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_ge_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_ge_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_ge_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_ge_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_o_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_o_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_o_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_o_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_u_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_u_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_u_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_u_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nge_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nge_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nge_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nge_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nlg_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nlg_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nlg_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nlg_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_ngt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_ngt_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_ngt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_ngt_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nle_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nle_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nle_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nle_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_neq_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_neq_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_neq_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_neq_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nlt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nlt_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_nlt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_nlt_f64_w64);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_t_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 32, rj_arith_v_cmp_u_f64_w32, rj_arith_v_cmp_class_f64_w32>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_t_f64_w32);
}

// Write the predicate mask and accumulate the inherited Compatibility causes.
int rj_arith_v_cmpx_t_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx1150<64, 64, rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_class_f64_w64>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_t_f64_w64);
}
