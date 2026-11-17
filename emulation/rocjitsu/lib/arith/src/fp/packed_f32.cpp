// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed FP32 arithmetic over two VGPRs per operand. Selectors address whole
// FP32 words; both results are staged before publishing potentially aliased outputs.

#include "common/packed_alu.h"
#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <stdint.h>

namespace {

enum class Operation { Add, Mul, Fma };

// Evaluate both selected FP32 results without changing host FP controls. Compatibility
// retains guest rounding and the inherited per-operation denormal/NaN policy.
template <int Lanes, bool OldCdna, Operation Op = Operation::Fma>
int packed_arithmetic(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  constexpr uint64_t guest =
      rocjitsu::arith::kFmaGuestFpFlags | (OldCdna ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0);
  constexpr uint32_t all_modifiers =
      RJ_ARITH_PK_NEG_LO_A | RJ_ARITH_PK_NEG_LO_B | RJ_ARITH_PK_NEG_LO_C | RJ_ARITH_PK_NEG_HI_A |
      RJ_ARITH_PK_NEG_HI_B | RJ_ARITH_PK_NEG_HI_C | RJ_ARITH_PK_CLAMP | RJ_ARITH_PK_LO_A_HIGH |
      RJ_ARITH_PK_LO_B_HIGH | RJ_ARITH_PK_LO_C_HIGH | RJ_ARITH_PK_HI_A_LOW | RJ_ARITH_PK_HI_B_LOW |
      RJ_ARITH_PK_HI_C_LOW;
  constexpr uint32_t c_modifiers =
      RJ_ARITH_PK_NEG_LO_C | RJ_ARITH_PK_NEG_HI_C | RJ_ARITH_PK_LO_C_HIGH | RJ_ARITH_PK_HI_C_LOW;
  constexpr uint32_t known = Op == Operation::Fma ? all_modifiers : all_modifiers & ~c_modifiers;
  if (int error = rocjitsu::arith::validate(flags, mode & ~uint64_t(known), guest)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  flags = (flags & ~RJ_ARITH_SEMANTICS_MASK) |
          (compatibility ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                         : RJ_ARITH_SEMANTICS_LOOSE);
  const bool flush_output = flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  if constexpr (Op != Operation::Fma) {
    // Packed ADD/MUL flush after FP32 rounding, unlike FMA's intermediate
    // tininess test. Native ADD/MUL also always quiet signaling NaNs.
    flags = (flags & ~RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) | RJ_ARITH_FP_IEEE_MODE;
  }
  uint32_t result[2][Lanes];
  for (int word = 0; word < 2; ++word) {
    const uint32_t selected = word ? (~(mode >> 10) & 7) : ((mode >> 7) & 7);
    const uint32_t arithmetic =
        rocjitsu::arith::packed_half_instruction_flags(uint32_t(mode), word) &
        ~(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_C);
    for (int first = 0; first < Lanes; first += 32) {
      const uint32_t block_exec_mask = uint32_t(exec_mask >> first);
      if (!block_exec_mask) {
        continue;
      }
      uint32_t *out[] = {result[word] + first};
      const uint32_t *x[] = {a[selected & 1] + first};
      const uint32_t *y[] = {b[(selected >> 1) & 1] + first};
      int error;
      if constexpr (Op == Operation::Fma) {
        const uint32_t *z[] = {c[(selected >> 2) & 1] + first};
        error = OldCdna ? rj_arith_v_fma_f32_w32(flags, block_exec_mask, arithmetic, out, x, y, z,
                                                 nullptr)
                        : rj_arith_v_fma_f32_w32_gfx12(flags, block_exec_mask, arithmetic, out, x,
                                                       y, z, nullptr);
      } else if constexpr (Op == Operation::Add) {
        error = OldCdna
                    ? rj_arith_v_add_f32_w32(flags, block_exec_mask, arithmetic, out, x, y, nullptr)
                    : rj_arith_v_add_f32_w32_gfx12(flags, block_exec_mask, arithmetic, out, x, y,
                                                   nullptr);
      } else {
        error = OldCdna
                    ? rj_arith_v_mul_f32_w32(flags, block_exec_mask, arithmetic, out, x, y, nullptr)
                    : rj_arith_v_mul_f32_w32_gfx12(flags, block_exec_mask, arithmetic, out, x, y,
                                                   nullptr);
      }
      if (error) {
        return error;
      }
    }
  }
  // Staging both words also preserves cross-word selectors when D overlaps a source.
  for (int word = 0; word < 2; ++word) {
    for (int lane = 0; lane < Lanes; ++lane) {
      if ((exec_mask >> lane) & 1) {
        uint32_t value = result[word][lane];
        if constexpr (Op != Operation::Fma) {
          if (flush_output && !(value & 0x7f800000U)) {
            value &= 0x80000000U;
          }
        }
        d[word][lane] = value;
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Evaluate selected packed FP32 results, preserving inactive lanes and input aliases.
int rj_arith_v_pk_fma_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c,
                                     [[maybe_unused]] uint32_t *trapsts) {
  return packed_arithmetic<32, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate selected packed FP32 results, preserving inactive lanes and input aliases.
int rj_arith_v_pk_fma_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c,
                                     [[maybe_unused]] uint32_t *trapsts) {
  return packed_arithmetic<64, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate selected packed FP32 results, preserving inactive lanes and input aliases.
int rj_arith_v_pk_fma_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c,
                                      [[maybe_unused]] uint32_t *excp_flag_user) {
  return packed_arithmetic<32, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate paired FP32 results, preserving inactive lanes and source aliases.
int rj_arith_v_pk_add_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  return packed_arithmetic<32, true, Operation::Add>(flags, exec_mask, instruction_flags, d, a, b,
                                                     nullptr);
}

// Evaluate paired FP32 results, preserving inactive lanes and source aliases.
int rj_arith_v_pk_add_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  return packed_arithmetic<64, true, Operation::Add>(flags, exec_mask, instruction_flags, d, a, b,
                                                     nullptr);
}

// Evaluate paired FP32 results, preserving inactive lanes and source aliases.
int rj_arith_v_pk_add_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      [[maybe_unused]] uint32_t *excp_flag_user) {
  return packed_arithmetic<32, false, Operation::Add>(flags, exec_mask, instruction_flags, d, a, b,
                                                      nullptr);
}

// Evaluate paired FP32 results, preserving inactive lanes and source aliases.
int rj_arith_v_pk_mul_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  return packed_arithmetic<32, true, Operation::Mul>(flags, exec_mask, instruction_flags, d, a, b,
                                                     nullptr);
}

// Evaluate paired FP32 results, preserving inactive lanes and source aliases.
int rj_arith_v_pk_mul_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  return packed_arithmetic<64, true, Operation::Mul>(flags, exec_mask, instruction_flags, d, a, b,
                                                     nullptr);
}

// Evaluate paired FP32 results, preserving inactive lanes and source aliases.
int rj_arith_v_pk_mul_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      [[maybe_unused]] uint32_t *excp_flag_user) {
  return packed_arithmetic<32, false, Operation::Mul>(flags, exec_mask, instruction_flags, d, a, b,
                                                      nullptr);
}
