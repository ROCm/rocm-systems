// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Gfx908 wrappers retain the validated shared predicate masks while adding
// Compatibility's CLAMP-only INVALID policy. Include constants and aliases.

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/semantics.h"
#include "compare/floating_point.h"
#include "compare/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {
using namespace rocjitsu::arith::test;
using Compare = int (*)(uint64_t, uint64_t, uint64_t, uint64_t *, const uint32_t *const *,
                        const uint32_t *const *, uint32_t *);

// Check the wrapper's numeric binding against the inherited comparator, with
// literal status rules independent of the implementation's unordered-mask call.
template <int Bits> void check_gfx908_compare(Compare call, Compare numeric) {
  auto input = make_compare_float_inputs<Bits, 64>();
  for (int half = 0; half < (Bits == 16 ? 2 : 1); ++half) {
    write_compare_operand(input.v, 63, Bits,
                          compare_infinity<Bits>() | compare_quiet_bit<Bits>() | 1, half);
    write_compare_operand(input.v + 2, 63, Bits, 0, half);
    write_compare_operand(input.v, 5, Bits, 1, half);
    // A preserved subnormal/finite pair must not add INPUT_DENORM.
    write_compare_operand(input.v + 2, 5, Bits, compare_class_value<Bits>(8), half);
  }
  constexpr uint64_t modifiers = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_A |
                                 RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_CLAMP |
                                 (Bits == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0);
  expect_alu_invalid_flags<Bits != 64>(
      [&](uint64_t flags, uint64_t mode) {
        uint64_t result = ~0ULL;
        uint32_t status = 0x100;
        int error = call(flags, 0, mode, &result, nullptr, nullptr, &status);
        EXPECT_EQ(result, ~0ULL);
        EXPECT_EQ(status, 0x100U);
        return error;
      },
      modifiers);
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 4; ++scenario) {
        const uint64_t exec_masks[] = {~0ULL, 1ULL << 63, 1ULL << 5, 0};
        const uint64_t exec_mask = exec_masks[scenario];
        const uint64_t mode = scenario
                                  ? RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_B |
                                        (Bits == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0)
                                  : 0;
        const uint64_t flags =
            cpu | semantics |
            (scenario & 1 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE : 0);
        const bool rejected = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
        uint64_t expected = ~0ULL;
        if (!rejected) {
          ASSERT_EQ(numeric(flags, exec_mask, mode, &expected, input.v, input.v + 2, nullptr),
                    RJ_ARITH_SUCCESS);
        }
        uint32_t expected_status = 0x100;
        if (!rejected && semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY && scenario) {
          for (int lane = 0; lane < 64; ++lane) {
            if ((exec_mask >> lane) & 1) {
              for (int operand : {0, 2}) {
                const uint64_t value = compare_operand(input.v + operand, lane, Bits, Bits == 16);
                if ((value & ((1ULL << (Bits - 1)) - 1)) > compare_infinity<Bits>()) {
                  expected_status |= RJ_ARITH_EXCEPTION_INVALID;
                }
              }
            }
          }
        }
        for (int aliasing_register : {-1, 0, 2}) {
          expect_comparison(
              input, expected, rejected ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS : RJ_ARITH_SUCCESS,
              aliasing_register, [&](auto &r, uint64_t *result) {
                uint32_t status = 0x100;
                int error = call(flags, exec_mask, mode, result, exec_mask ? r.v : nullptr,
                                 exec_mask ? r.v + 2 : nullptr, &status);
                EXPECT_EQ(status, expected_status);
                return error;
              });
        }
      }
    }
  }
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpFF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_f_f16_w64_gfx908, rj_arith_v_cmp_f_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLtF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_lt_f16_w64_gfx908, rj_arith_v_cmp_lt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpEqF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_eq_f16_w64_gfx908, rj_arith_v_cmp_eq_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLeF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_le_f16_w64_gfx908, rj_arith_v_cmp_le_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGtF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_gt_f16_w64_gfx908, rj_arith_v_cmp_gt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLgF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_lg_f16_w64_gfx908, rj_arith_v_cmp_lg_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGeF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_ge_f16_w64_gfx908, rj_arith_v_cmp_ge_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpOF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_o_f16_w64_gfx908, rj_arith_v_cmp_o_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpUF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_u_f16_w64_gfx908, rj_arith_v_cmp_u_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgeF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_nge_f16_w64_gfx908, rj_arith_v_cmp_nge_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNlgF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_nlg_f16_w64_gfx908, rj_arith_v_cmp_nlg_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgtF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_ngt_f16_w64_gfx908, rj_arith_v_cmp_ngt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNleF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_nle_f16_w64_gfx908, rj_arith_v_cmp_nle_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNeqF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_neq_f16_w64_gfx908, rj_arith_v_cmp_neq_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNltF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_nlt_f16_w64_gfx908, rj_arith_v_cmp_nlt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpTF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmp_t_f16_w64_gfx908, rj_arith_v_cmp_t_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxFF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_f_f16_w64_gfx908, rj_arith_v_cmpx_f_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLtF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_lt_f16_w64_gfx908, rj_arith_v_cmpx_lt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxEqF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_eq_f16_w64_gfx908, rj_arith_v_cmpx_eq_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLeF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_le_f16_w64_gfx908, rj_arith_v_cmpx_le_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGtF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_gt_f16_w64_gfx908, rj_arith_v_cmpx_gt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLgF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_lg_f16_w64_gfx908, rj_arith_v_cmpx_lg_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGeF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_ge_f16_w64_gfx908, rj_arith_v_cmpx_ge_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxOF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_o_f16_w64_gfx908, rj_arith_v_cmpx_o_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxUF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_u_f16_w64_gfx908, rj_arith_v_cmpx_u_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgeF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_nge_f16_w64_gfx908, rj_arith_v_cmpx_nge_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNlgF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_nlg_f16_w64_gfx908, rj_arith_v_cmpx_nlg_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgtF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_ngt_f16_w64_gfx908, rj_arith_v_cmpx_ngt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNleF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_nle_f16_w64_gfx908, rj_arith_v_cmpx_nle_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNeqF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_neq_f16_w64_gfx908, rj_arith_v_cmpx_neq_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNltF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_nlt_f16_w64_gfx908, rj_arith_v_cmpx_nlt_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxTF16W64Gfx908, Arithmetic) {
  check_gfx908_compare<16>(rj_arith_v_cmpx_t_f16_w64_gfx908, rj_arith_v_cmpx_t_f16_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpFF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_f_f32_w64_gfx908, rj_arith_v_cmp_f_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLtF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_lt_f32_w64_gfx908, rj_arith_v_cmp_lt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpEqF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_eq_f32_w64_gfx908, rj_arith_v_cmp_eq_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLeF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_le_f32_w64_gfx908, rj_arith_v_cmp_le_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGtF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_gt_f32_w64_gfx908, rj_arith_v_cmp_gt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLgF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_lg_f32_w64_gfx908, rj_arith_v_cmp_lg_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGeF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_ge_f32_w64_gfx908, rj_arith_v_cmp_ge_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpOF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_o_f32_w64_gfx908, rj_arith_v_cmp_o_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpUF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_u_f32_w64_gfx908, rj_arith_v_cmp_u_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgeF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_nge_f32_w64_gfx908, rj_arith_v_cmp_nge_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNlgF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_nlg_f32_w64_gfx908, rj_arith_v_cmp_nlg_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgtF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_ngt_f32_w64_gfx908, rj_arith_v_cmp_ngt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNleF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_nle_f32_w64_gfx908, rj_arith_v_cmp_nle_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNeqF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_neq_f32_w64_gfx908, rj_arith_v_cmp_neq_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNltF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_nlt_f32_w64_gfx908, rj_arith_v_cmp_nlt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpTF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmp_t_f32_w64_gfx908, rj_arith_v_cmp_t_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxFF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_f_f32_w64_gfx908, rj_arith_v_cmpx_f_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLtF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_lt_f32_w64_gfx908, rj_arith_v_cmpx_lt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxEqF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_eq_f32_w64_gfx908, rj_arith_v_cmpx_eq_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLeF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_le_f32_w64_gfx908, rj_arith_v_cmpx_le_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGtF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_gt_f32_w64_gfx908, rj_arith_v_cmpx_gt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLgF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_lg_f32_w64_gfx908, rj_arith_v_cmpx_lg_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGeF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_ge_f32_w64_gfx908, rj_arith_v_cmpx_ge_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxOF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_o_f32_w64_gfx908, rj_arith_v_cmpx_o_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxUF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_u_f32_w64_gfx908, rj_arith_v_cmpx_u_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgeF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_nge_f32_w64_gfx908, rj_arith_v_cmpx_nge_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNlgF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_nlg_f32_w64_gfx908, rj_arith_v_cmpx_nlg_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgtF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_ngt_f32_w64_gfx908, rj_arith_v_cmpx_ngt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNleF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_nle_f32_w64_gfx908, rj_arith_v_cmpx_nle_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNeqF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_neq_f32_w64_gfx908, rj_arith_v_cmpx_neq_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNltF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_nlt_f32_w64_gfx908, rj_arith_v_cmpx_nlt_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxTF32W64Gfx908, Arithmetic) {
  check_gfx908_compare<32>(rj_arith_v_cmpx_t_f32_w64_gfx908, rj_arith_v_cmpx_t_f32_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpFF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_f_f64_w64_gfx908, rj_arith_v_cmp_f_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLtF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_lt_f64_w64_gfx908, rj_arith_v_cmp_lt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpEqF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_eq_f64_w64_gfx908, rj_arith_v_cmp_eq_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLeF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_le_f64_w64_gfx908, rj_arith_v_cmp_le_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGtF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_gt_f64_w64_gfx908, rj_arith_v_cmp_gt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpLgF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_lg_f64_w64_gfx908, rj_arith_v_cmp_lg_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpGeF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_ge_f64_w64_gfx908, rj_arith_v_cmp_ge_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpOF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_o_f64_w64_gfx908, rj_arith_v_cmp_o_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpUF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_u_f64_w64_gfx908, rj_arith_v_cmp_u_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgeF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_nge_f64_w64_gfx908, rj_arith_v_cmp_nge_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNlgF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_nlg_f64_w64_gfx908, rj_arith_v_cmp_nlg_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNgtF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_ngt_f64_w64_gfx908, rj_arith_v_cmp_ngt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNleF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_nle_f64_w64_gfx908, rj_arith_v_cmp_nle_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNeqF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_neq_f64_w64_gfx908, rj_arith_v_cmp_neq_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpNltF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_nlt_f64_w64_gfx908, rj_arith_v_cmp_nlt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpTF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmp_t_f64_w64_gfx908, rj_arith_v_cmp_t_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxFF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_f_f64_w64_gfx908, rj_arith_v_cmpx_f_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLtF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_lt_f64_w64_gfx908, rj_arith_v_cmpx_lt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxEqF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_eq_f64_w64_gfx908, rj_arith_v_cmpx_eq_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLeF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_le_f64_w64_gfx908, rj_arith_v_cmpx_le_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGtF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_gt_f64_w64_gfx908, rj_arith_v_cmpx_gt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxLgF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_lg_f64_w64_gfx908, rj_arith_v_cmpx_lg_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxGeF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_ge_f64_w64_gfx908, rj_arith_v_cmpx_ge_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxOF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_o_f64_w64_gfx908, rj_arith_v_cmpx_o_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxUF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_u_f64_w64_gfx908, rj_arith_v_cmpx_u_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgeF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_nge_f64_w64_gfx908, rj_arith_v_cmpx_nge_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNlgF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_nlg_f64_w64_gfx908, rj_arith_v_cmpx_nlg_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNgtF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_ngt_f64_w64_gfx908, rj_arith_v_cmpx_ngt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNleF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_nle_f64_w64_gfx908, rj_arith_v_cmpx_nle_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNeqF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_neq_f64_w64_gfx908, rj_arith_v_cmpx_neq_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxNltF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_nlt_f64_w64_gfx908, rj_arith_v_cmpx_nlt_f64_w64);
}

// Check the numeric mask and Compatibility-only signaling contract.
TEST(CompareCmpxTF64W64Gfx908, Arithmetic) {
  check_gfx908_compare<64>(rj_arith_v_cmpx_t_f64_w64_gfx908, rj_arith_v_cmpx_t_f64_w64);
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
