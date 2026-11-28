// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1251 packed FP64 validation and composition of existing CDNA arithmetic paths.
#include "common/validation.h"
#include "fp/fp64.h"
#include "fp/packed_f64.h"
#include "rj_arith/detail/rj_arith_fp.h"
namespace {
// Canonicalize two-NaN selection results and guest-flush subnormals without FP operations.
void publish_scalar(bool selection, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                    uint32_t *const *d, uint32_t *const *result, const uint32_t *const *a,
                    const uint32_t *const *b) {
  if (selection || (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    for (int reg = 0; reg < 4; reg += 2) {
      for (int lane = 0; lane < 32; ++lane) {
        uint64_t bits = uint64_t(result[reg][lane]) | (uint64_t(result[reg + 1][lane]) << 32);
        if (selection && !(mode & RJ_ARITH_PK_CLAMP)) {
          uint64_t x = uint64_t(a[reg][lane]) | (uint64_t(a[reg + 1][lane]) << 32);
          uint64_t y = uint64_t(b[reg][lane]) | (uint64_t(b[reg + 1][lane]) << 32);
          if ((x & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL &&
              (y & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL) {
            bits = 0x7ff8000000000000ULL;
          }
        }
        if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && !(bits & 0x7ff0000000000000ULL)) {
          bits &= 0x8000000000000000ULL;
        }
        result[reg][lane] = uint32_t(bits);
        result[reg + 1][lane] = uint32_t(bits >> 32);
      }
    }
    // Finish every selection fixup before a store can overwrite any original input.
  }
  for (int reg = 0; reg < 4; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      if (exec_mask >> lane & 1) {
        d[reg][lane] = result[reg][lane];
      }
    }
  }
}
// Validate the whole packed instruction, reuse each scalar/CPU backend, then commit once.
template <rocjitsu::arith::Fp64 Op>
int packed(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
           const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  constexpr bool fma = Op == Fp64::Fma, selection = Op == Fp64::MinNum || Op == Fp64::MaxNum;
  constexpr uint64_t supported = RJ_ARITH_PK_NEG_LO_A | RJ_ARITH_PK_NEG_LO_B |
                                 RJ_ARITH_PK_NEG_HI_A | RJ_ARITH_PK_NEG_HI_B | RJ_ARITH_PK_CLAMP |
                                 (fma ? RJ_ARITH_PK_NEG_LO_C | RJ_ARITH_PK_NEG_HI_C : 0);
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error =
          validate(flags, mode & ~supported,
                   RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_IEEE_MODE |
                       RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if (reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  // The inherited model is not newly empirically qualified; mode1 uses its allowed Loose fallback.
  flags = (flags & ~RJ_ARITH_SEMANTICS_MASK) | RJ_ARITH_SEMANTICS_LOOSE;
  // Preserve the composed arithmetic's inherited NaN policy in Compatibility.
  const uint64_t numeric_flags =
      compatibility ? (flags & ~RJ_ARITH_SEMANTICS_MASK) | RJ_ARITH_SEMANTICS_COMPATIBILITY : flags;
  uint32_t words[4][32];
  uint32_t *result[] = {words[0], words[1], words[2], words[3]};
  uint32_t ignored = 0;
  for (int element = 0; element < 2; ++element) {
    const int reg = element * 2;
    const uint32_t modifiers = uint32_t((mode >> (element * 3)) & 7) |
                               ((mode & RJ_ARITH_PK_CLAMP) ? RJ_ARITH_ALU_CLAMP : 0);
    // Evaluate all lanes into scratch, keeping work independent of mask density and avoiding
    // uninitialized inactive scratch. The public result is still strictly EXEC-masked.
    int status;
    if constexpr (Op == Fp64::Add) {
      status = rj_arith_v_add_f64_w32_gfx12(numeric_flags, ~0U, modifiers, result + reg, a + reg,
                                            b + reg, &ignored);
    } else if constexpr (Op == Fp64::Mul) {
      status = rj_arith_v_mul_f64_w32_gfx1250(numeric_flags, ~0U, modifiers, result + reg, a + reg,
                                              b + reg, &ignored);
    } else if constexpr (fma) {
      status = rj_arith_v_fma_f64_w32_gfx1250(numeric_flags, ~0U, modifiers, result + reg, a + reg,
                                              b + reg, c + reg, &ignored);
    } else if constexpr (Op == Fp64::MinNum) {
      status = rj_arith_v_min_num_f64_w32(numeric_flags, ~0U, modifiers, result + reg, a + reg,
                                          b + reg, &ignored);
    } else {
      status = rj_arith_v_max_num_f64_w32(numeric_flags, ~0U, modifiers, result + reg, a + reg,
                                          b + reg, &ignored);
    }
    if (status != RJ_ARITH_SUCCESS) {
      return status;
    }
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    publish_packed_f64_x86_64_v4(selection, flags, exec_mask, uint32_t(mode), d, result, a, b);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    publish_packed_f64_x86_64_v3(selection, flags, exec_mask, uint32_t(mode), d, result, a, b);
    return RJ_ARITH_SUCCESS;
  }
#endif
  publish_scalar(selection, flags, exec_mask, uint32_t(mode), d, result, a, b);
  return RJ_ARITH_SUCCESS;
}
} // namespace
// Evaluate two packed FP64 elements, retaining the inherited guest arithmetic policy.
int rj_arith_v_pk_add_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, [[maybe_unused]] uint32_t *excp_flag_user) {
  return packed<rocjitsu::arith::Fp64::Add>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}
// Evaluate two packed FP64 elements, retaining the inherited guest arithmetic policy.
int rj_arith_v_pk_mul_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, [[maybe_unused]] uint32_t *excp_flag_user) {
  return packed<rocjitsu::arith::Fp64::Mul>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}
// Evaluate two packed FP64 elements, retaining the inherited guest arithmetic policy.
int rj_arith_v_pk_fma_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              [[maybe_unused]] uint32_t *excp_flag_user) {
  return packed<rocjitsu::arith::Fp64::Fma>(flags, exec_mask, instruction_flags, d, a, b, c);
}
// Evaluate two packed FP64 elements, retaining the inherited guest arithmetic policy.
int rj_arith_v_pk_min_num_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  return packed<rocjitsu::arith::Fp64::MinNum>(flags, exec_mask, instruction_flags, d, a, b,
                                               nullptr);
}
// Evaluate two packed FP64 elements, retaining the inherited guest arithmetic policy.
int rj_arith_v_pk_max_num_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  return packed<rocjitsu::arith::Fp64::MaxNum>(flags, exec_mask, instruction_flags, d, a, b,
                                               nullptr);
}
