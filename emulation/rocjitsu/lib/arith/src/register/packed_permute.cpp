// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed B4/B6/B8 table permutations, including elements crossing word boundaries.

#include "register/packed_permute.h"
#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_register.h"

namespace {
// Snapshot each lane's table and selectors before publishing any output word.
template <int Bits, int Lanes>
int permute(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, uint32_t(mode))) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::packed_permute_x86_64_v4(Bits, Lanes, exec_mask, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::packed_permute_x86_64_v3(Bits, Lanes, exec_mask, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  constexpr int Words = Bits / 2, LowWords = Bits == 4 ? 1 : 2;
  for (int lane = 0; lane < Lanes; ++lane) {
    uint32_t table[Words + 1] = {}, result[Words] = {};
    for (int word = 0; word < Words; ++word) {
      table[word] = word < LowWords ? b[word][lane] : a[word - LowWords][lane];
    }
    uint64_t selectors = uint64_t(c[0][lane]) | (uint64_t(c[1][lane]) << 32);
    for (int element = 0; element < 16; ++element) {
      int bit = int((selectors >> (element * 4)) & 15) * Bits;
      uint64_t pair = uint64_t(table[bit / 32]) | (uint64_t(table[bit / 32 + 1]) << 32);
      uint32_t value = uint32_t(pair >> (bit % 32)) & ((1U << Bits) - 1);
      int output = element * Bits;
      result[output / 32] |= value << (output % 32);
      if (output % 32 + Bits > 32) {
        result[output / 32 + 1] |= value >> (32 - output % 32);
      }
    }
    if ((exec_mask >> lane) & 1) {
      for (int word = 0; word < Words; ++word) {
        d[word][lane] = result[word];
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Select packed elements and preserve inactive destination lanes.
int rj_arith_v_perm_pk16_b4_u4_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return permute<4, 32>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select packed elements and preserve inactive destination lanes.
int rj_arith_v_perm_pk16_b4_u4_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return permute<4, 64>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select packed elements and preserve inactive destination lanes.
int rj_arith_v_perm_pk16_b6_u4_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return permute<6, 32>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select packed elements and preserve inactive destination lanes.
int rj_arith_v_perm_pk16_b6_u4_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return permute<6, 64>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select packed elements and preserve inactive destination lanes.
int rj_arith_v_perm_pk16_b8_u4_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return permute<8, 32>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select packed elements and preserve inactive destination lanes.
int rj_arith_v_perm_pk16_b8_u4_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return permute<8, 64>(flags, exec_mask, instruction_flags, d, a, b, c);
}
