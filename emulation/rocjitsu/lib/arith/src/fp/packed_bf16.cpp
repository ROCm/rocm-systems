// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed BF16 arithmetic and numeric selection reuse FP32/mixed BF16 SIMD paths.
#include "common/packed_alu.h"
#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include <stdint.h>

namespace {
enum class Operation { Add, Mul, Fma, Min, Max };

// Decode one selected half, applying packed sign negation without FP arithmetic.
uint32_t operand(uint32_t word, uint32_t mode) {
  return ((word >> ((mode & RJ_ARITH_ALU_HIGH_A) ? 16 : 0)) << 16) ^
         ((mode & RJ_ARITH_ALU_NEG_A) ? 0x80000000U : 0);
}

// Preserve both destination halves until all inputs have been consumed. Packed
// BF16 ignores guest rounding/denormal controls, inheriting nearest-even arithmetic.
template <Operation Op>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  constexpr uint32_t c_flags =
      RJ_ARITH_PK_NEG_LO_C | RJ_ARITH_PK_NEG_HI_C | RJ_ARITH_PK_LO_C_HIGH | RJ_ARITH_PK_HI_C_LOW;
  constexpr uint32_t all = RJ_ARITH_PK_NEG_LO_A | RJ_ARITH_PK_NEG_LO_B | c_flags |
                           RJ_ARITH_PK_NEG_HI_A | RJ_ARITH_PK_NEG_HI_B | RJ_ARITH_PK_CLAMP |
                           RJ_ARITH_PK_LO_A_HIGH | RJ_ARITH_PK_LO_B_HIGH | RJ_ARITH_PK_HI_A_LOW |
                           RJ_ARITH_PK_HI_B_LOW;
  constexpr uint32_t known = Op == Operation::Fma ? all : all & ~c_flags;
  if (int error = rocjitsu::arith::validate(flags, mode & ~uint64_t(known),
                                            rocjitsu::arith::kFmaGuestFpFlags)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) > RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  constexpr bool selection = Op == Operation::Min || Op == Operation::Max;
  const bool saturate = !selection && (flags & RJ_ARITH_FP16_OVFL);
  flags &= ~(RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
             RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP16_OVFL);
  uint32_t packed[32] = {};
  for (int half = 0; half < 2; ++half) {
    const uint32_t selected = rocjitsu::arith::packed_half_instruction_flags(uint32_t(mode), half);
    uint32_t x[32], y[32], z[32], result[32] = {};
    for (int lane = 0; lane < 32; ++lane) {
      x[lane] = operand(a[0][lane], selected);
      y[lane] = operand(b[0][lane], selected >> 1);
      if constexpr (Op == Operation::Fma) {
        z[lane] = operand(c[0][lane], selected >> 2);
      } else if constexpr (Op == Operation::Add) {
        z[lane] = y[lane];
        y[lane] = 0x3f800000U;
      } else if constexpr (selection) {
        z[lane] = 0;
      } else {
        z[lane] = (x[lane] ^ y[lane]) & 0x80000000U;
      }
    }
    const uint32_t *pa = x, *pb = y, *pc = z;
    uint32_t *pd = result;
    int error;
    if constexpr (selection) {
      auto function =
          Op == Operation::Min ? rj_arith_v_min_num_f32_w32 : rj_arith_v_max_num_f32_w32;
      error = function(flags, exec_mask, 0, &pd, &pa, &pb, nullptr);
      for (int lane = 0; lane < 32; ++lane) {
        result[lane] >>= 16;
      }
    } else {
      error = rj_arith_v_fma_mixlo_bf16_w32(flags, exec_mask, 0, &pd, &pa, &pb, &pc, nullptr);
    }
    if (error) {
      return error;
    }
    for (int lane = 0; lane < 32; ++lane) {
      uint32_t value = result[lane] & 0xffffU;
      if (saturate && (value & 0x7fffU) == 0x7f80U && (x[lane] & 0x7fffffffU) < 0x7f800000U &&
          (y[lane] & 0x7fffffffU) < 0x7f800000U && (z[lane] & 0x7fffffffU) < 0x7f800000U) {
        value = (value & 0x8000U) | 0x7f7fU;
      }
      if (mode & RJ_ARITH_PK_CLAMP) {
        value = (value & 0x8000U) || value > 0x7f80U ? 0 : value > 0x3f80U ? 0x3f80U : value;
      }
      packed[lane] |= value << (16 * half);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = packed[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate paired BF16 results with packed selectors and unchanged inactive lanes.
int rj_arith_v_pk_fma_bf16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  return run<Operation::Fma>(flags, exec_mask, mode, d, a, b, c);
}

// Evaluate paired BF16 sums with packed selectors and unchanged inactive lanes.
int rj_arith_v_pk_add_bf16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  return run<Operation::Add>(flags, exec_mask, mode, d, a, b, nullptr);
}

// Evaluate paired BF16 products with packed selectors and unchanged inactive lanes.
int rj_arith_v_pk_mul_bf16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  return run<Operation::Mul>(flags, exec_mask, mode, d, a, b, nullptr);
}

// Select paired BF16 numbers, preferring numeric operands over NaNs.
int rj_arith_v_pk_min_num_bf16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  return run<Operation::Min>(flags, exec_mask, mode, d, a, b, nullptr);
}

// Select paired BF16 numbers, preferring numeric operands over NaNs.
int rj_arith_v_pk_max_num_bf16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  return run<Operation::Max>(flags, exec_mask, mode, d, a, b, nullptr);
}
