// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Gfx10 FP32/FP64 masks and NaN-only Compatibility status across both waves.
#include "common/registers.h"
#include "compare/signaling_clamp_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {
using namespace rocjitsu::arith::test;

// Check constant masks and signaling FP status.
TEST(CompareCmpFF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32, true>(rj_arith_v_cmp_f_f32_w32_gfx10,
                                              rj_arith_v_cmp_f_f32_w32);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpFF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64, true>(rj_arith_v_cmp_f_f32_w64_gfx10,
                                              rj_arith_v_cmp_f_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLtF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmp_lt_f32_w32_gfx10, rj_arith_v_cmp_lt_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLtF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_lt_f32_w64_gfx10, rj_arith_v_cmp_lt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpEqF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmp_eq_f32_w32_gfx10, rj_arith_v_cmp_eq_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpEqF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_eq_f32_w64_gfx10, rj_arith_v_cmp_eq_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLeF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmp_le_f32_w32_gfx10, rj_arith_v_cmp_le_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLeF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_le_f32_w64_gfx10, rj_arith_v_cmp_le_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGtF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmp_gt_f32_w32_gfx10, rj_arith_v_cmp_gt_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGtF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_gt_f32_w64_gfx10, rj_arith_v_cmp_gt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLgF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmp_lg_f32_w32_gfx10, rj_arith_v_cmp_lg_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLgF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_lg_f32_w64_gfx10, rj_arith_v_cmp_lg_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGeF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmp_ge_f32_w32_gfx10, rj_arith_v_cmp_ge_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGeF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_ge_f32_w64_gfx10, rj_arith_v_cmp_ge_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpOF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmp_o_f32_w32_gfx10, rj_arith_v_cmp_o_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpOF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_o_f32_w64_gfx10, rj_arith_v_cmp_o_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpUF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmp_u_f32_w32_gfx10, rj_arith_v_cmp_u_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpUF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_u_f32_w64_gfx10, rj_arith_v_cmp_u_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgeF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmp_nge_f32_w32_gfx10,
                                        rj_arith_v_cmp_nge_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgeF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_nge_f32_w64_gfx10,
                                        rj_arith_v_cmp_nge_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNlgF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmp_nlg_f32_w32_gfx10,
                                        rj_arith_v_cmp_nlg_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNlgF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_nlg_f32_w64_gfx10,
                                        rj_arith_v_cmp_nlg_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgtF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmp_ngt_f32_w32_gfx10,
                                        rj_arith_v_cmp_ngt_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgtF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_ngt_f32_w64_gfx10,
                                        rj_arith_v_cmp_ngt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNleF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmp_nle_f32_w32_gfx10,
                                        rj_arith_v_cmp_nle_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNleF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_nle_f32_w64_gfx10,
                                        rj_arith_v_cmp_nle_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNeqF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmp_neq_f32_w32_gfx10,
                                        rj_arith_v_cmp_neq_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNeqF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_neq_f32_w64_gfx10,
                                        rj_arith_v_cmp_neq_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNltF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmp_nlt_f32_w32_gfx10,
                                        rj_arith_v_cmp_nlt_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNltF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_nlt_f32_w64_gfx10,
                                        rj_arith_v_cmp_nlt_f32_w64);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpTF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32, true>(rj_arith_v_cmp_t_f32_w32_gfx10,
                                              rj_arith_v_cmp_t_f32_w32);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpTF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64, true>(rj_arith_v_cmp_t_f32_w64_gfx10,
                                              rj_arith_v_cmp_t_f32_w64);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpxFF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32, true>(rj_arith_v_cmpx_f_f32_w32_gfx10,
                                              rj_arith_v_cmpx_f_f32_w32);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpxFF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64, true>(rj_arith_v_cmpx_f_f32_w64_gfx10,
                                              rj_arith_v_cmpx_f_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLtF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmpx_lt_f32_w32_gfx10,
                                        rj_arith_v_cmpx_lt_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLtF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_lt_f32_w64_gfx10,
                                        rj_arith_v_cmpx_lt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxEqF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmpx_eq_f32_w32_gfx10,
                                        rj_arith_v_cmpx_eq_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxEqF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_eq_f32_w64_gfx10,
                                        rj_arith_v_cmpx_eq_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLeF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmpx_le_f32_w32_gfx10,
                                        rj_arith_v_cmpx_le_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLeF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_le_f32_w64_gfx10,
                                        rj_arith_v_cmpx_le_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGtF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmpx_gt_f32_w32_gfx10,
                                        rj_arith_v_cmpx_gt_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGtF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_gt_f32_w64_gfx10,
                                        rj_arith_v_cmpx_gt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLgF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmpx_lg_f32_w32_gfx10,
                                        rj_arith_v_cmpx_lg_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLgF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_lg_f32_w64_gfx10,
                                        rj_arith_v_cmpx_lg_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGeF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmpx_ge_f32_w32_gfx10,
                                        rj_arith_v_cmpx_ge_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGeF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_ge_f32_w64_gfx10,
                                        rj_arith_v_cmpx_ge_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxOF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmpx_o_f32_w32_gfx10, rj_arith_v_cmpx_o_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxOF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_o_f32_w64_gfx10, rj_arith_v_cmpx_o_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxUF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmpx_u_f32_w32_gfx10, rj_arith_v_cmpx_u_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxUF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_u_f32_w64_gfx10, rj_arith_v_cmpx_u_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgeF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmpx_nge_f32_w32_gfx10,
                                        rj_arith_v_cmpx_nge_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgeF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_nge_f32_w64_gfx10,
                                        rj_arith_v_cmpx_nge_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNlgF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmpx_nlg_f32_w32_gfx10,
                                        rj_arith_v_cmpx_nlg_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNlgF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_nlg_f32_w64_gfx10,
                                        rj_arith_v_cmpx_nlg_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgtF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmpx_ngt_f32_w32_gfx10,
                                        rj_arith_v_cmpx_ngt_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgtF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_ngt_f32_w64_gfx10,
                                        rj_arith_v_cmpx_ngt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNleF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmpx_nle_f32_w32_gfx10,
                                        rj_arith_v_cmpx_nle_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNleF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_nle_f32_w64_gfx10,
                                        rj_arith_v_cmpx_nle_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNeqF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmpx_neq_f32_w32_gfx10,
                                        rj_arith_v_cmpx_neq_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNeqF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_neq_f32_w64_gfx10,
                                        rj_arith_v_cmpx_neq_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNltF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32>(rj_arith_v_cmpx_nlt_f32_w32_gfx10,
                                        rj_arith_v_cmpx_nlt_f32_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNltF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_nlt_f32_w64_gfx10,
                                        rj_arith_v_cmpx_nlt_f32_w64);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpxTF32W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 32, true>(rj_arith_v_cmpx_t_f32_w32_gfx10,
                                              rj_arith_v_cmpx_t_f32_w32);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpxTF32W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<32, 64, true>(rj_arith_v_cmpx_t_f32_w64_gfx10,
                                              rj_arith_v_cmpx_t_f32_w64);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpFF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32, true>(rj_arith_v_cmp_f_f64_w32_gfx10,
                                              rj_arith_v_cmp_f_f64_w32);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpFF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64, true>(rj_arith_v_cmp_f_f64_w64_gfx10,
                                              rj_arith_v_cmp_f_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLtF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmp_lt_f64_w32_gfx10, rj_arith_v_cmp_lt_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLtF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_lt_f64_w64_gfx10, rj_arith_v_cmp_lt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpEqF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmp_eq_f64_w32_gfx10, rj_arith_v_cmp_eq_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpEqF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_eq_f64_w64_gfx10, rj_arith_v_cmp_eq_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLeF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmp_le_f64_w32_gfx10, rj_arith_v_cmp_le_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLeF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_le_f64_w64_gfx10, rj_arith_v_cmp_le_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGtF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmp_gt_f64_w32_gfx10, rj_arith_v_cmp_gt_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGtF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_gt_f64_w64_gfx10, rj_arith_v_cmp_gt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLgF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmp_lg_f64_w32_gfx10, rj_arith_v_cmp_lg_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLgF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_lg_f64_w64_gfx10, rj_arith_v_cmp_lg_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGeF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmp_ge_f64_w32_gfx10, rj_arith_v_cmp_ge_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGeF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_ge_f64_w64_gfx10, rj_arith_v_cmp_ge_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpOF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmp_o_f64_w32_gfx10, rj_arith_v_cmp_o_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpOF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_o_f64_w64_gfx10, rj_arith_v_cmp_o_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpUF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmp_u_f64_w32_gfx10, rj_arith_v_cmp_u_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpUF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_u_f64_w64_gfx10, rj_arith_v_cmp_u_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgeF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmp_nge_f64_w32_gfx10,
                                        rj_arith_v_cmp_nge_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgeF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_nge_f64_w64_gfx10,
                                        rj_arith_v_cmp_nge_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNlgF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmp_nlg_f64_w32_gfx10,
                                        rj_arith_v_cmp_nlg_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNlgF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_nlg_f64_w64_gfx10,
                                        rj_arith_v_cmp_nlg_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgtF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmp_ngt_f64_w32_gfx10,
                                        rj_arith_v_cmp_ngt_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgtF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_ngt_f64_w64_gfx10,
                                        rj_arith_v_cmp_ngt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNleF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmp_nle_f64_w32_gfx10,
                                        rj_arith_v_cmp_nle_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNleF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_nle_f64_w64_gfx10,
                                        rj_arith_v_cmp_nle_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNeqF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmp_neq_f64_w32_gfx10,
                                        rj_arith_v_cmp_neq_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNeqF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_neq_f64_w64_gfx10,
                                        rj_arith_v_cmp_neq_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNltF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmp_nlt_f64_w32_gfx10,
                                        rj_arith_v_cmp_nlt_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNltF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_nlt_f64_w64_gfx10,
                                        rj_arith_v_cmp_nlt_f64_w64);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpTF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32, true>(rj_arith_v_cmp_t_f64_w32_gfx10,
                                              rj_arith_v_cmp_t_f64_w32);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpTF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64, true>(rj_arith_v_cmp_t_f64_w64_gfx10,
                                              rj_arith_v_cmp_t_f64_w64);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpxFF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32, true>(rj_arith_v_cmpx_f_f64_w32_gfx10,
                                              rj_arith_v_cmpx_f_f64_w32);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpxFF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64, true>(rj_arith_v_cmpx_f_f64_w64_gfx10,
                                              rj_arith_v_cmpx_f_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLtF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmpx_lt_f64_w32_gfx10,
                                        rj_arith_v_cmpx_lt_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLtF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_lt_f64_w64_gfx10,
                                        rj_arith_v_cmpx_lt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxEqF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmpx_eq_f64_w32_gfx10,
                                        rj_arith_v_cmpx_eq_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxEqF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_eq_f64_w64_gfx10,
                                        rj_arith_v_cmpx_eq_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLeF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmpx_le_f64_w32_gfx10,
                                        rj_arith_v_cmpx_le_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLeF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_le_f64_w64_gfx10,
                                        rj_arith_v_cmpx_le_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGtF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmpx_gt_f64_w32_gfx10,
                                        rj_arith_v_cmpx_gt_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGtF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_gt_f64_w64_gfx10,
                                        rj_arith_v_cmpx_gt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLgF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmpx_lg_f64_w32_gfx10,
                                        rj_arith_v_cmpx_lg_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLgF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_lg_f64_w64_gfx10,
                                        rj_arith_v_cmpx_lg_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGeF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmpx_ge_f64_w32_gfx10,
                                        rj_arith_v_cmpx_ge_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGeF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_ge_f64_w64_gfx10,
                                        rj_arith_v_cmpx_ge_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxOF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmpx_o_f64_w32_gfx10, rj_arith_v_cmpx_o_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxOF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_o_f64_w64_gfx10, rj_arith_v_cmpx_o_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxUF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmpx_u_f64_w32_gfx10, rj_arith_v_cmpx_u_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxUF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_u_f64_w64_gfx10, rj_arith_v_cmpx_u_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgeF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmpx_nge_f64_w32_gfx10,
                                        rj_arith_v_cmpx_nge_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgeF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_nge_f64_w64_gfx10,
                                        rj_arith_v_cmpx_nge_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNlgF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmpx_nlg_f64_w32_gfx10,
                                        rj_arith_v_cmpx_nlg_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNlgF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_nlg_f64_w64_gfx10,
                                        rj_arith_v_cmpx_nlg_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgtF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmpx_ngt_f64_w32_gfx10,
                                        rj_arith_v_cmpx_ngt_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgtF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_ngt_f64_w64_gfx10,
                                        rj_arith_v_cmpx_ngt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNleF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmpx_nle_f64_w32_gfx10,
                                        rj_arith_v_cmpx_nle_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNleF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_nle_f64_w64_gfx10,
                                        rj_arith_v_cmpx_nle_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNeqF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmpx_neq_f64_w32_gfx10,
                                        rj_arith_v_cmpx_neq_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNeqF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_neq_f64_w64_gfx10,
                                        rj_arith_v_cmpx_neq_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNltF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32>(rj_arith_v_cmpx_nlt_f64_w32_gfx10,
                                        rj_arith_v_cmpx_nlt_f64_w32);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNltF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_nlt_f64_w64_gfx10,
                                        rj_arith_v_cmpx_nlt_f64_w64);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpxTF64W32Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 32, true>(rj_arith_v_cmpx_t_f64_w32_gfx10,
                                              rj_arith_v_cmpx_t_f64_w32);
}

// Check constant masks and signaling FP status.
TEST(CompareCmpxTF64W64Gfx10, Arithmetic) {
  check_signaling_clamp_compare<64, 64, true>(rj_arith_v_cmpx_t_f64_w64_gfx10,
                                              rj_arith_v_cmpx_t_f64_w64);
}

// Publishing status must not change a still-needed source lane.
TEST(CompareGfx10StatusAliasing, Arithmetic) {
  Registers<4, 32> r;
  for (int lane = 0; lane < 32; ++lane) {
    r.v[0][lane] = 0x3f800000;
    r.v[2][lane] = 0x3f800000;
  }
  r.v[0][31] = 0x7fc00001;
  uint32_t result = 0;
  ASSERT_EQ(rj_arith_v_cmp_eq_f32_w32_gfx10(RJ_ARITH_SEMANTICS_COMPATIBILITY, (1U << 31) | 1U,
                                            RJ_ARITH_ALU_CLAMP, &result, r.v, r.v + 2, r.v[2]),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(result, 1U);
  EXPECT_EQ(r.v[2][0], 0x3f800001U);
}

} // namespace
