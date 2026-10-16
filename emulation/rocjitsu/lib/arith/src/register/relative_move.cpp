// SPDX-License-Identifier: MIT

// VGPR moves with register indices offset by M0.
// Owns public entry points and their validation/dispatch policy.

// Indexing/routing follow rocjitsu. The 255 offset limit follows section
// 3.3.2.2 of both AMD ISA manuals (rocjitsu currently allows 1023).

#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Move M0-relative VGPRs for the selected lanes. Requires validated controls; preserve
// inactive destination lanes.
template <int Lanes>
int move(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, uint32_t d_count, uint32_t a_count, uint32_t d_base,
         uint32_t a_base, uint32_t dst_offset, uint32_t src_offset) {
  constexpr auto move_instruction = [] {
    if constexpr (Lanes == 32) {
      return rj_arith_v_mov_b32_w32;
    } else {
      return rj_arith_v_mov_b32_w64;
    }
  }();

  // Validate even if EXEC is zero or the destination index is out of range.
  if (int error = move_instruction(flags, 0, mode, nullptr, nullptr)) {
    return error;
  }
  uint64_t dst = uint64_t(d_base) + dst_offset;
  uint64_t src = uint64_t(a_base) + src_offset;
  if (!exec_mask || dst_offset > 255 || dst >= d_count) {
    return RJ_ARITH_SUCCESS;
  }
  uint32_t zero[Lanes] = {};
  const uint32_t *source = a_count ? a[src_offset <= 255 && src < a_count ? src : 0] : zero;
  return move_instruction(flags, exec_mask, mode, d + dst, &source);
}

} // namespace

// Read an M0-relative source register for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_movrels_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                               uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move<32>(flags, exec_mask, instruction_flags, d, a, d_count, a_count, d_base, a_base, 0,
                  m0);
}

// Write an M0-relative destination register for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_movreld_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                               uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move<32>(flags, exec_mask, instruction_flags, d, a, d_count, a_count, d_base, a_base, m0,
                  0);
}

// Move between M0-relative registers for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_movrelsd_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                                uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move<32>(flags, exec_mask, instruction_flags, d, a, d_count, a_count, d_base, a_base, m0,
                  m0);
}

// Move between M0-relative registers for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_movrelsd_2_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                                  uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move<32>(flags, exec_mask, instruction_flags, d, a, d_count, a_count, d_base, a_base,
                  (m0 >> 16) & 1023, m0 & 1023);
}

// Read an M0-relative source register for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_movrels_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                               uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move<64>(flags, exec_mask, instruction_flags, d, a, d_count, a_count, d_base, a_base, 0,
                  m0);
}

// Write an M0-relative destination register for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_movreld_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                               uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move<64>(flags, exec_mask, instruction_flags, d, a, d_count, a_count, d_base, a_base, m0,
                  0);
}

// Move between M0-relative registers for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_movrelsd_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                                uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move<64>(flags, exec_mask, instruction_flags, d, a, d_count, a_count, d_base, a_base, m0,
                  m0);
}

// Move between M0-relative registers for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_movrelsd_2_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                                  uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move<64>(flags, exec_mask, instruction_flags, d, a, d_count, a_count, d_base, a_base,
                  (m0 >> 16) & 1023, m0 & 1023);
}
