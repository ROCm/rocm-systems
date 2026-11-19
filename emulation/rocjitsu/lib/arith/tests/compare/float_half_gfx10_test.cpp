// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Gfx10 FP16 masks and NaN-only Compatibility status across both waves.
#include "compare/signaling_clamp_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {
using namespace rocjitsu::arith::test;

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpFF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32, true>(rj_arith_v_cmp_f_f16_w32_gfx10,
                                              rj_arith_v_cmp_f_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpFF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64, true>(rj_arith_v_cmp_f_f16_w64_gfx10,
                                              rj_arith_v_cmp_f_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLtF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmp_lt_f16_w32_gfx10, rj_arith_v_cmp_lt_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLtF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_lt_f16_w64_gfx10, rj_arith_v_cmp_lt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpEqF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmp_eq_f16_w32_gfx10, rj_arith_v_cmp_eq_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpEqF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_eq_f16_w64_gfx10, rj_arith_v_cmp_eq_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLeF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmp_le_f16_w32_gfx10, rj_arith_v_cmp_le_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLeF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_le_f16_w64_gfx10, rj_arith_v_cmp_le_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGtF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmp_gt_f16_w32_gfx10, rj_arith_v_cmp_gt_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGtF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_gt_f16_w64_gfx10, rj_arith_v_cmp_gt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLgF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmp_lg_f16_w32_gfx10, rj_arith_v_cmp_lg_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLgF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_lg_f16_w64_gfx10, rj_arith_v_cmp_lg_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGeF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmp_ge_f16_w32_gfx10, rj_arith_v_cmp_ge_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGeF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_ge_f16_w64_gfx10, rj_arith_v_cmp_ge_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpOF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmp_o_f16_w32_gfx10, rj_arith_v_cmp_o_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpOF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_o_f16_w64_gfx10, rj_arith_v_cmp_o_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpUF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmp_u_f16_w32_gfx10, rj_arith_v_cmp_u_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpUF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_u_f16_w64_gfx10, rj_arith_v_cmp_u_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgeF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmp_nge_f16_w32_gfx10,
                                        rj_arith_v_cmp_nge_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgeF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_nge_f16_w64_gfx10,
                                        rj_arith_v_cmp_nge_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNlgF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmp_nlg_f16_w32_gfx10,
                                        rj_arith_v_cmp_nlg_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNlgF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_nlg_f16_w64_gfx10,
                                        rj_arith_v_cmp_nlg_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgtF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmp_ngt_f16_w32_gfx10,
                                        rj_arith_v_cmp_ngt_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgtF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_ngt_f16_w64_gfx10,
                                        rj_arith_v_cmp_ngt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNleF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmp_nle_f16_w32_gfx10,
                                        rj_arith_v_cmp_nle_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNleF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_nle_f16_w64_gfx10,
                                        rj_arith_v_cmp_nle_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNeqF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmp_neq_f16_w32_gfx10,
                                        rj_arith_v_cmp_neq_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNeqF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_neq_f16_w64_gfx10,
                                        rj_arith_v_cmp_neq_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNltF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmp_nlt_f16_w32_gfx10,
                                        rj_arith_v_cmp_nlt_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNltF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_nlt_f16_w64_gfx10,
                                        rj_arith_v_cmp_nlt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpTF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32, true>(rj_arith_v_cmp_t_f16_w32_gfx10,
                                              rj_arith_v_cmp_t_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpTF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64, true>(rj_arith_v_cmp_t_f16_w64_gfx10,
                                              rj_arith_v_cmp_t_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxFF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32, true>(rj_arith_v_cmpx_f_f16_w32_gfx10,
                                              rj_arith_v_cmpx_f_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxFF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64, true>(rj_arith_v_cmpx_f_f16_w64_gfx10,
                                              rj_arith_v_cmpx_f_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLtF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmpx_lt_f16_w32_gfx10,
                                        rj_arith_v_cmpx_lt_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLtF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_lt_f16_w64_gfx10,
                                        rj_arith_v_cmpx_lt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxEqF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmpx_eq_f16_w32_gfx10,
                                        rj_arith_v_cmpx_eq_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxEqF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_eq_f16_w64_gfx10,
                                        rj_arith_v_cmpx_eq_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLeF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmpx_le_f16_w32_gfx10,
                                        rj_arith_v_cmpx_le_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLeF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_le_f16_w64_gfx10,
                                        rj_arith_v_cmpx_le_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGtF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmpx_gt_f16_w32_gfx10,
                                        rj_arith_v_cmpx_gt_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGtF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_gt_f16_w64_gfx10,
                                        rj_arith_v_cmpx_gt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLgF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmpx_lg_f16_w32_gfx10,
                                        rj_arith_v_cmpx_lg_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLgF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_lg_f16_w64_gfx10,
                                        rj_arith_v_cmpx_lg_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGeF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmpx_ge_f16_w32_gfx10,
                                        rj_arith_v_cmpx_ge_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGeF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_ge_f16_w64_gfx10,
                                        rj_arith_v_cmpx_ge_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxOF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmpx_o_f16_w32_gfx10, rj_arith_v_cmpx_o_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxOF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_o_f16_w64_gfx10, rj_arith_v_cmpx_o_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxUF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmpx_u_f16_w32_gfx10, rj_arith_v_cmpx_u_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxUF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_u_f16_w64_gfx10, rj_arith_v_cmpx_u_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgeF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmpx_nge_f16_w32_gfx10,
                                        rj_arith_v_cmpx_nge_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgeF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_nge_f16_w64_gfx10,
                                        rj_arith_v_cmpx_nge_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNlgF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmpx_nlg_f16_w32_gfx10,
                                        rj_arith_v_cmpx_nlg_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNlgF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_nlg_f16_w64_gfx10,
                                        rj_arith_v_cmpx_nlg_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgtF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmpx_ngt_f16_w32_gfx10,
                                        rj_arith_v_cmpx_ngt_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgtF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_ngt_f16_w64_gfx10,
                                        rj_arith_v_cmpx_ngt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNleF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmpx_nle_f16_w32_gfx10,
                                        rj_arith_v_cmpx_nle_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNleF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_nle_f16_w64_gfx10,
                                        rj_arith_v_cmpx_nle_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNeqF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmpx_neq_f16_w32_gfx10,
                                        rj_arith_v_cmpx_neq_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNeqF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_neq_f16_w64_gfx10,
                                        rj_arith_v_cmpx_neq_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNltF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32>(rj_arith_v_cmpx_nlt_f16_w32_gfx10,
                                        rj_arith_v_cmpx_nlt_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNltF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_nlt_f16_w64_gfx10,
                                        rj_arith_v_cmpx_nlt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxTF16W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 32, true>(rj_arith_v_cmpx_t_f16_w32_gfx10,
                                              rj_arith_v_cmpx_t_f16_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxTF16W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<16, 64, true>(rj_arith_v_cmpx_t_f16_w64_gfx10,
                                              rj_arith_v_cmpx_t_f16_w64);
}

} // namespace
