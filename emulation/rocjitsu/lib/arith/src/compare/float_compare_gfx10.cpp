// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Gfx10 native-precision predicates and inherited CLAMP-only INVALID policy.
// Both wave sizes share the existing scalar/AVX2/AVX-512 numeric comparators.

#include "compare/signaling_clamp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_f_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_f_f32_w32,
      rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_f_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_f_f32_w64,
      rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_lt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_lt_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_lt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_lt_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_eq_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_eq_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_eq_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_eq_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_le_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_le_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_le_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_le_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_gt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_gt_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_gt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_gt_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_lg_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_lg_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_lg_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_lg_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_ge_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_ge_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_ge_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_ge_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_o_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_o_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_o_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_o_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_u_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_u_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_u_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_u_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nge_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nge_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nge_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nge_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nlg_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nlg_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nlg_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nlg_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_ngt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_ngt_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_ngt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_ngt_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nle_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nle_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nle_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nle_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_neq_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_neq_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_neq_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_neq_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nlt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nlt_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nlt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nlt_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_t_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_t_f32_w32,
      rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_t_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_t_f32_w64,
      rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_f_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_f_f32_w32,
      rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_f_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_f_f32_w64,
      rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_lt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_lt_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_lt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_lt_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_eq_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_eq_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_eq_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_eq_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_le_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_le_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_le_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_le_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_gt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_gt_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_gt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_gt_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_lg_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_lg_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_lg_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_lg_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_ge_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_ge_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_ge_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_ge_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_o_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_o_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_o_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_o_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_u_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_u_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_u_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_u_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nge_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nge_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nge_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nge_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nlg_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nlg_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nlg_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nlg_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_ngt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_ngt_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_ngt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_ngt_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nle_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nle_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nle_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nle_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_neq_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_neq_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_neq_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_neq_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nlt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nlt_f32_w32,
                                                       rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nlt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nlt_f32_w64,
                                                       rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_t_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_t_f32_w32,
      rj_arith_v_cmp_u_f32_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_t_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_t_f32_w64,
      rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_f_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_f_f64_w32,
      rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_f_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_f_f64_w64,
      rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_lt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_lt_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_lt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_lt_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_eq_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_eq_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_eq_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_eq_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_le_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_le_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_le_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_le_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_gt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_gt_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_gt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_gt_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_lg_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_lg_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_lg_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_lg_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_ge_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_ge_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_ge_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_ge_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_o_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_o_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_o_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_o_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_u_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_u_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_u_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_u_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nge_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nge_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nge_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nge_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nlg_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nlg_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nlg_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nlg_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_ngt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_ngt_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_ngt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_ngt_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nle_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nle_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nle_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nle_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_neq_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_neq_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_neq_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_neq_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nlt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nlt_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_nlt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmp_nlt_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_t_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_t_f64_w32,
      rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmp_t_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmp_t_f64_w64,
      rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_f_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_f_f64_w32,
      rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_f_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_f_f64_w64,
      rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_lt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_lt_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_lt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_lt_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_eq_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_eq_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_eq_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_eq_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_le_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_le_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_le_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_le_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_gt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_gt_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_gt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_gt_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_lg_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_lg_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_lg_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_lg_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_ge_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_ge_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_ge_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_ge_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_o_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_o_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_o_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_o_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_u_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_u_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_u_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_u_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nge_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nge_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nge_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nge_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nlg_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nlg_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nlg_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nlg_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_ngt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_ngt_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_ngt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_ngt_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nle_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nle_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nle_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nle_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_neq_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_neq_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_neq_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_neq_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nlt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nlt_f64_w32,
                                                       rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_nlt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp(flags, exec_mask, instruction_flags, d, a, b,
                                                       trapsts, rj_arith_v_cmpx_nlt_f64_w64,
                                                       rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_t_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_t_f64_w32,
      rj_arith_v_cmp_u_f64_w32);
}

// Write the predicate mask and preserve gfx10's Compatibility signaling policy.
int rj_arith_v_cmpx_t_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return rocjitsu::arith::compare_with_signaling_clamp<true>(
      flags, exec_mask, instruction_flags, d, a, b, trapsts, rj_arith_v_cmpx_t_f64_w64,
      rj_arith_v_cmp_u_f64_w64);
}
