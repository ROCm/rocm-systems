// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Whole register-pair moves and packed word selections with EXEC masking.
#include "register/pair_move.h"
#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_register.h"
namespace {
// Resolve controls before selecting register pointers or writing either result.
template <int Lanes, bool Packed>
int move(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  constexpr uint64_t known = Packed ? RJ_ARITH_PK_LO_A_HIGH | RJ_ARITH_PK_LO_B_HIGH : 0;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  const uint32_t *low = a[Packed && (mode & RJ_ARITH_PK_LO_A_HIGH) ? 1 : 0];
  const uint32_t *high;
  if constexpr (Packed) {
    high = b[(mode & RJ_ARITH_PK_LO_B_HIGH) ? 1 : 0];
  } else {
    high = a[1];
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::pair_move_x86_64_v4(Lanes, exec_mask, d, low, high);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::pair_move_x86_64_v3(Lanes, exec_mask, d, low, high);
    return RJ_ARITH_SUCCESS;
  }
#endif
  for (int lane = 0; lane < Lanes; ++lane) {
    uint32_t lo = low[lane], hi = high[lane];
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = lo;
      d[1][lane] = hi;
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Copy the selected pair for active lanes, preserving all input bits.
int rj_arith_v_mov_b64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a) {
  return move<32, false>(flags, exec_mask, mode, d, a, nullptr);
}

// Copy the selected pair for active lanes, preserving all input bits.
int rj_arith_v_pk_mov_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b) {
  return move<32, true>(flags, exec_mask, mode, d, a, b);
}

// Copy the selected pair for active lanes, preserving all input bits.
int rj_arith_v_mov_b64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a) {
  return move<64, false>(flags, exec_mask, mode, d, a, nullptr);
}

// Copy the selected pair for active lanes, preserving all input bits.
int rj_arith_v_pk_mov_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b) {
  return move<64, true>(flags, exec_mask, mode, d, a, b);
}
