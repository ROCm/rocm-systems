// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Gfx908 wrappers retain the validated shared predicate masks while adding
// Compatibility's CLAMP-only INVALID policy. Include constants and aliases.

#include "common/registers.h"
#include "compare/signaling_clamp_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {
using namespace rocjitsu::arith::test;

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpFF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_f_f16_w64_gfx908, rj_arith_v_cmp_f_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLtF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_lt_f16_w64_gfx908,
                                        rj_arith_v_cmp_lt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpEqF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_eq_f16_w64_gfx908,
                                        rj_arith_v_cmp_eq_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLeF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_le_f16_w64_gfx908,
                                        rj_arith_v_cmp_le_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGtF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_gt_f16_w64_gfx908,
                                        rj_arith_v_cmp_gt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLgF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_lg_f16_w64_gfx908,
                                        rj_arith_v_cmp_lg_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGeF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_ge_f16_w64_gfx908,
                                        rj_arith_v_cmp_ge_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpOF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_o_f16_w64_gfx908, rj_arith_v_cmp_o_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpUF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_u_f16_w64_gfx908, rj_arith_v_cmp_u_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgeF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_nge_f16_w64_gfx908,
                                        rj_arith_v_cmp_nge_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNlgF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_nlg_f16_w64_gfx908,
                                        rj_arith_v_cmp_nlg_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgtF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_ngt_f16_w64_gfx908,
                                        rj_arith_v_cmp_ngt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNleF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_nle_f16_w64_gfx908,
                                        rj_arith_v_cmp_nle_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNeqF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_neq_f16_w64_gfx908,
                                        rj_arith_v_cmp_neq_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNltF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_nlt_f16_w64_gfx908,
                                        rj_arith_v_cmp_nlt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpTF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmp_t_f16_w64_gfx908, rj_arith_v_cmp_t_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxFF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_f_f16_w64_gfx908,
                                        rj_arith_v_cmpx_f_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLtF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_lt_f16_w64_gfx908,
                                        rj_arith_v_cmpx_lt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxEqF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_eq_f16_w64_gfx908,
                                        rj_arith_v_cmpx_eq_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLeF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_le_f16_w64_gfx908,
                                        rj_arith_v_cmpx_le_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGtF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_gt_f16_w64_gfx908,
                                        rj_arith_v_cmpx_gt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLgF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_lg_f16_w64_gfx908,
                                        rj_arith_v_cmpx_lg_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGeF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_ge_f16_w64_gfx908,
                                        rj_arith_v_cmpx_ge_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxOF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_o_f16_w64_gfx908,
                                        rj_arith_v_cmpx_o_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxUF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_u_f16_w64_gfx908,
                                        rj_arith_v_cmpx_u_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgeF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_nge_f16_w64_gfx908,
                                        rj_arith_v_cmpx_nge_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNlgF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_nlg_f16_w64_gfx908,
                                        rj_arith_v_cmpx_nlg_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgtF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_ngt_f16_w64_gfx908,
                                        rj_arith_v_cmpx_ngt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNleF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_nle_f16_w64_gfx908,
                                        rj_arith_v_cmpx_nle_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNeqF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_neq_f16_w64_gfx908,
                                        rj_arith_v_cmpx_neq_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNltF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_nlt_f16_w64_gfx908,
                                        rj_arith_v_cmpx_nlt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxTF16W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<16, 64>(rj_arith_v_cmpx_t_f16_w64_gfx908,
                                        rj_arith_v_cmpx_t_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpFF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_f_f32_w64_gfx908, rj_arith_v_cmp_f_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLtF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_lt_f32_w64_gfx908,
                                        rj_arith_v_cmp_lt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpEqF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_eq_f32_w64_gfx908,
                                        rj_arith_v_cmp_eq_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLeF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_le_f32_w64_gfx908,
                                        rj_arith_v_cmp_le_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGtF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_gt_f32_w64_gfx908,
                                        rj_arith_v_cmp_gt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLgF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_lg_f32_w64_gfx908,
                                        rj_arith_v_cmp_lg_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGeF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_ge_f32_w64_gfx908,
                                        rj_arith_v_cmp_ge_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpOF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_o_f32_w64_gfx908, rj_arith_v_cmp_o_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpUF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_u_f32_w64_gfx908, rj_arith_v_cmp_u_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgeF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_nge_f32_w64_gfx908,
                                        rj_arith_v_cmp_nge_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNlgF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_nlg_f32_w64_gfx908,
                                        rj_arith_v_cmp_nlg_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgtF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_ngt_f32_w64_gfx908,
                                        rj_arith_v_cmp_ngt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNleF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_nle_f32_w64_gfx908,
                                        rj_arith_v_cmp_nle_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNeqF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_neq_f32_w64_gfx908,
                                        rj_arith_v_cmp_neq_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNltF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_nlt_f32_w64_gfx908,
                                        rj_arith_v_cmp_nlt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpTF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmp_t_f32_w64_gfx908, rj_arith_v_cmp_t_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxFF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_f_f32_w64_gfx908,
                                        rj_arith_v_cmpx_f_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLtF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_lt_f32_w64_gfx908,
                                        rj_arith_v_cmpx_lt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxEqF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_eq_f32_w64_gfx908,
                                        rj_arith_v_cmpx_eq_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLeF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_le_f32_w64_gfx908,
                                        rj_arith_v_cmpx_le_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGtF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_gt_f32_w64_gfx908,
                                        rj_arith_v_cmpx_gt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLgF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_lg_f32_w64_gfx908,
                                        rj_arith_v_cmpx_lg_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGeF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_ge_f32_w64_gfx908,
                                        rj_arith_v_cmpx_ge_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxOF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_o_f32_w64_gfx908,
                                        rj_arith_v_cmpx_o_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxUF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_u_f32_w64_gfx908,
                                        rj_arith_v_cmpx_u_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgeF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_nge_f32_w64_gfx908,
                                        rj_arith_v_cmpx_nge_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNlgF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_nlg_f32_w64_gfx908,
                                        rj_arith_v_cmpx_nlg_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgtF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_ngt_f32_w64_gfx908,
                                        rj_arith_v_cmpx_ngt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNleF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_nle_f32_w64_gfx908,
                                        rj_arith_v_cmpx_nle_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNeqF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_neq_f32_w64_gfx908,
                                        rj_arith_v_cmpx_neq_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNltF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_nlt_f32_w64_gfx908,
                                        rj_arith_v_cmpx_nlt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxTF32W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<32, 64>(rj_arith_v_cmpx_t_f32_w64_gfx908,
                                        rj_arith_v_cmpx_t_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpFF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_f_f64_w64_gfx908, rj_arith_v_cmp_f_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLtF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_lt_f64_w64_gfx908,
                                        rj_arith_v_cmp_lt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpEqF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_eq_f64_w64_gfx908,
                                        rj_arith_v_cmp_eq_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLeF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_le_f64_w64_gfx908,
                                        rj_arith_v_cmp_le_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGtF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_gt_f64_w64_gfx908,
                                        rj_arith_v_cmp_gt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLgF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_lg_f64_w64_gfx908,
                                        rj_arith_v_cmp_lg_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGeF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_ge_f64_w64_gfx908,
                                        rj_arith_v_cmp_ge_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpOF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_o_f64_w64_gfx908, rj_arith_v_cmp_o_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpUF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_u_f64_w64_gfx908, rj_arith_v_cmp_u_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgeF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_nge_f64_w64_gfx908,
                                        rj_arith_v_cmp_nge_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNlgF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_nlg_f64_w64_gfx908,
                                        rj_arith_v_cmp_nlg_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgtF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_ngt_f64_w64_gfx908,
                                        rj_arith_v_cmp_ngt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNleF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_nle_f64_w64_gfx908,
                                        rj_arith_v_cmp_nle_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNeqF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_neq_f64_w64_gfx908,
                                        rj_arith_v_cmp_neq_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNltF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_nlt_f64_w64_gfx908,
                                        rj_arith_v_cmp_nlt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpTF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmp_t_f64_w64_gfx908, rj_arith_v_cmp_t_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxFF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_f_f64_w64_gfx908,
                                        rj_arith_v_cmpx_f_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLtF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_lt_f64_w64_gfx908,
                                        rj_arith_v_cmpx_lt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxEqF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_eq_f64_w64_gfx908,
                                        rj_arith_v_cmpx_eq_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLeF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_le_f64_w64_gfx908,
                                        rj_arith_v_cmpx_le_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGtF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_gt_f64_w64_gfx908,
                                        rj_arith_v_cmpx_gt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLgF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_lg_f64_w64_gfx908,
                                        rj_arith_v_cmpx_lg_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGeF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_ge_f64_w64_gfx908,
                                        rj_arith_v_cmpx_ge_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxOF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_o_f64_w64_gfx908,
                                        rj_arith_v_cmpx_o_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxUF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_u_f64_w64_gfx908,
                                        rj_arith_v_cmpx_u_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgeF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_nge_f64_w64_gfx908,
                                        rj_arith_v_cmpx_nge_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNlgF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_nlg_f64_w64_gfx908,
                                        rj_arith_v_cmpx_nlg_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgtF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_ngt_f64_w64_gfx908,
                                        rj_arith_v_cmpx_ngt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNleF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_nle_f64_w64_gfx908,
                                        rj_arith_v_cmpx_nle_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNeqF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_neq_f64_w64_gfx908,
                                        rj_arith_v_cmpx_neq_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNltF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_nlt_f64_w64_gfx908,
                                        rj_arith_v_cmpx_nlt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxTF64W64Gfx908, Arithmetic) {
  check_signaling_clamp_compare<64, 64>(rj_arith_v_cmpx_t_f64_w64_gfx908,
                                        rj_arith_v_cmpx_t_f64_w64);
}

// Classification and predicate evaluation precede an aliased status store.
TEST(CompareGfx908StatusAliasing, Arithmetic) {
  Registers<4, 64> r;
  for (int lane = 0; lane < 64; ++lane) {
    r.v[0][lane] = 0x3f800000;
    r.v[2][lane] = 0x3f800000;
  }
  r.v[0][63] = 0x7fc00001;
  uint64_t result = 0;
  ASSERT_EQ(rj_arith_v_cmp_eq_f32_w64_gfx908(RJ_ARITH_SEMANTICS_COMPATIBILITY,
                                             (1ULL << 32) | (1ULL << 63), RJ_ARITH_ALU_CLAMP,
                                             &result, r.v, r.v + 2, r.v[2] + 32),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(result, 1ULL << 32);
  EXPECT_EQ(r.v[2][32], 0x3f800001U);
}

} // namespace
