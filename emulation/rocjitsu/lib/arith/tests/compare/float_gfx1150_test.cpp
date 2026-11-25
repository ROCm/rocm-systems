// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Gfx1150 masks and DX10_CLAMP-dependent INVALID across predicates and waves.
// Numeric references inherit validated comparators; status checks inspect raw bits.
#include "common/contracts.h"
#include "common/cpu.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "compare/floating_point.h"
#include "compare/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {
using namespace rocjitsu::arith::test;

// Exercise numeric routing and selected NaN causes without duplicating arithmetic.
template <int Bits, int Lanes, typename Call> void check_gfx1150_compare(Call call, Call numeric) {
  using Mask = WaveMask<Lanes>;
  auto input = make_compare_float_inputs<Bits, Lanes>();
  for (int half = 0; half < (Bits == 16 ? 2 : 1); ++half) {
    write_compare_operand(input.v, 0, Bits, compare_class_value<Bits>(half ? 1 : 0), half);
    write_compare_operand(input.v, 1, Bits, compare_class_value<Bits>(half ? 0 : 1), half);
    write_compare_operand(input.v + 2, 0, Bits, compare_class_value<Bits>(8), half);
    write_compare_operand(input.v + 2, 1, Bits, compare_class_value<Bits>(8), half);
  }
  const uint64_t accepted = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_A |
                            RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_CLAMP |
                            (Bits == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0);
  expect_alu_invalid_flags<Bits != 64>(
      [&](uint64_t flags, uint64_t mode) {
        Mask result = ~Mask(0);
        uint32_t status = 0x100;
        int error = call(flags, 0, mode, &result, nullptr, nullptr, &status);
        EXPECT_EQ(result, ~Mask(0));
        EXPECT_EQ(status, 0x100U);
        return error;
      },
      accepted);
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < (Bits == 64 ? 6 : 8); ++scenario) {
        Mask exec_mask = scenario == 2 || scenario == 3 ? Mask(2)
                         : scenario == 4                ? Mask(1)
                         : scenario == 5                ? Mask(0)
                                                        : ~Mask(0);
        uint64_t mode = scenario == 3 || scenario == 5 ? RJ_ARITH_ALU_CLAMP : 0;
        mode |= RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_B;
        if ((scenario == 1 || scenario == 7) && Bits == 16) {
          mode |= RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B;
        }
        if (scenario == 6) {
          mode |= dpp_modes[0];
        }
        if (scenario == 7) {
          mode |= dpp_modes[6];
        }
        const uint64_t flags =
            cpu | semantics | (scenario ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0) |
            ((scenario & 1) ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE : 0);
        const bool rejected = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
        Mask expected = ~Mask(0);
        if (!rejected) {
          ASSERT_EQ(numeric(flags & ~RJ_ARITH_FP_DX10_CLAMP_OFF, exec_mask, mode, &expected,
                            input.v, input.v + 2, nullptr),
                    RJ_ARITH_SUCCESS);
        }
        uint32_t expected_status = 0x100;
        if (!rejected && semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY && scenario) {
          const Mask invalid = comparison_mask<Lanes>(mode, exec_mask, [&](int source, int lane) {
            const uint64_t av = compare_operand(input.v, source, Bits, mode & RJ_ARITH_ALU_HIGH_A);
            const uint64_t bv =
                compare_operand(input.v + 2, lane, Bits, mode & RJ_ARITH_ALU_HIGH_B);
            const auto raises = [&](uint64_t value) {
              return (value & ((1ULL << (Bits - 1)) - 1)) > compare_infinity<Bits>() &&
                     ((mode & RJ_ARITH_ALU_CLAMP) || !(value & compare_quiet_bit<Bits>()));
            };
            return raises(av) || raises(bv);
          });
          if (invalid) {
            expected_status |= RJ_ARITH_EXCEPTION_INVALID;
          }
        }
        for (int aliasing_register : {-1, 0, 2}) {
          expect_comparison(
              input, expected, rejected ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS : RJ_ARITH_SUCCESS,
              aliasing_register, [&](auto &registers, Mask *result) {
                uint32_t status = 0x100;
                int error = call(flags, exec_mask, mode, result, exec_mask ? registers.v : nullptr,
                                 exec_mask ? registers.v + 2 : nullptr, &status);
                EXPECT_EQ(status, expected_status);
                return error;
              });
        }
      }
    }
  }
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpFF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_f_f16_w32_gfx1150, rj_arith_v_cmp_f_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpFF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_f_f16_w64_gfx1150, rj_arith_v_cmp_f_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLtF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_lt_f16_w32_gfx1150, rj_arith_v_cmp_lt_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLtF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_lt_f16_w64_gfx1150, rj_arith_v_cmp_lt_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpEqF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_eq_f16_w32_gfx1150, rj_arith_v_cmp_eq_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpEqF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_eq_f16_w64_gfx1150, rj_arith_v_cmp_eq_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLeF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_le_f16_w32_gfx1150, rj_arith_v_cmp_le_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLeF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_le_f16_w64_gfx1150, rj_arith_v_cmp_le_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpGtF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_gt_f16_w32_gfx1150, rj_arith_v_cmp_gt_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpGtF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_gt_f16_w64_gfx1150, rj_arith_v_cmp_gt_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLgF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_lg_f16_w32_gfx1150, rj_arith_v_cmp_lg_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLgF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_lg_f16_w64_gfx1150, rj_arith_v_cmp_lg_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpGeF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_ge_f16_w32_gfx1150, rj_arith_v_cmp_ge_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpGeF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_ge_f16_w64_gfx1150, rj_arith_v_cmp_ge_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpOF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_o_f16_w32_gfx1150, rj_arith_v_cmp_o_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpOF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_o_f16_w64_gfx1150, rj_arith_v_cmp_o_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpUF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_u_f16_w32_gfx1150, rj_arith_v_cmp_u_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpUF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_u_f16_w64_gfx1150, rj_arith_v_cmp_u_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNgeF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_nge_f16_w32_gfx1150, rj_arith_v_cmp_nge_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNgeF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_nge_f16_w64_gfx1150, rj_arith_v_cmp_nge_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNlgF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_nlg_f16_w32_gfx1150, rj_arith_v_cmp_nlg_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNlgF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_nlg_f16_w64_gfx1150, rj_arith_v_cmp_nlg_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNgtF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_ngt_f16_w32_gfx1150, rj_arith_v_cmp_ngt_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNgtF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_ngt_f16_w64_gfx1150, rj_arith_v_cmp_ngt_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNleF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_nle_f16_w32_gfx1150, rj_arith_v_cmp_nle_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNleF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_nle_f16_w64_gfx1150, rj_arith_v_cmp_nle_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNeqF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_neq_f16_w32_gfx1150, rj_arith_v_cmp_neq_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNeqF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_neq_f16_w64_gfx1150, rj_arith_v_cmp_neq_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNltF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_nlt_f16_w32_gfx1150, rj_arith_v_cmp_nlt_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNltF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_nlt_f16_w64_gfx1150, rj_arith_v_cmp_nlt_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpTF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmp_t_f16_w32_gfx1150, rj_arith_v_cmp_t_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpTF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmp_t_f16_w64_gfx1150, rj_arith_v_cmp_t_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxFF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_f_f16_w32_gfx1150, rj_arith_v_cmpx_f_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxFF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_f_f16_w64_gfx1150, rj_arith_v_cmpx_f_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLtF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_lt_f16_w32_gfx1150, rj_arith_v_cmpx_lt_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLtF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_lt_f16_w64_gfx1150, rj_arith_v_cmpx_lt_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxEqF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_eq_f16_w32_gfx1150, rj_arith_v_cmpx_eq_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxEqF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_eq_f16_w64_gfx1150, rj_arith_v_cmpx_eq_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLeF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_le_f16_w32_gfx1150, rj_arith_v_cmpx_le_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLeF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_le_f16_w64_gfx1150, rj_arith_v_cmpx_le_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxGtF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_gt_f16_w32_gfx1150, rj_arith_v_cmpx_gt_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxGtF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_gt_f16_w64_gfx1150, rj_arith_v_cmpx_gt_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLgF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_lg_f16_w32_gfx1150, rj_arith_v_cmpx_lg_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLgF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_lg_f16_w64_gfx1150, rj_arith_v_cmpx_lg_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxGeF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_ge_f16_w32_gfx1150, rj_arith_v_cmpx_ge_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxGeF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_ge_f16_w64_gfx1150, rj_arith_v_cmpx_ge_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxOF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_o_f16_w32_gfx1150, rj_arith_v_cmpx_o_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxOF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_o_f16_w64_gfx1150, rj_arith_v_cmpx_o_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxUF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_u_f16_w32_gfx1150, rj_arith_v_cmpx_u_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxUF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_u_f16_w64_gfx1150, rj_arith_v_cmpx_u_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNgeF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_nge_f16_w32_gfx1150, rj_arith_v_cmpx_nge_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNgeF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_nge_f16_w64_gfx1150, rj_arith_v_cmpx_nge_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNlgF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_nlg_f16_w32_gfx1150, rj_arith_v_cmpx_nlg_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNlgF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_nlg_f16_w64_gfx1150, rj_arith_v_cmpx_nlg_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNgtF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_ngt_f16_w32_gfx1150, rj_arith_v_cmpx_ngt_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNgtF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_ngt_f16_w64_gfx1150, rj_arith_v_cmpx_ngt_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNleF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_nle_f16_w32_gfx1150, rj_arith_v_cmpx_nle_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNleF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_nle_f16_w64_gfx1150, rj_arith_v_cmpx_nle_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNeqF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_neq_f16_w32_gfx1150, rj_arith_v_cmpx_neq_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNeqF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_neq_f16_w64_gfx1150, rj_arith_v_cmpx_neq_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNltF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_nlt_f16_w32_gfx1150, rj_arith_v_cmpx_nlt_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNltF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_nlt_f16_w64_gfx1150, rj_arith_v_cmpx_nlt_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxTF16W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 32>(rj_arith_v_cmpx_t_f16_w32_gfx1150, rj_arith_v_cmpx_t_f16_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxTF16W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<16, 64>(rj_arith_v_cmpx_t_f16_w64_gfx1150, rj_arith_v_cmpx_t_f16_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpFF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_f_f32_w32_gfx1150, rj_arith_v_cmp_f_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpFF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_f_f32_w64_gfx1150, rj_arith_v_cmp_f_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLtF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_lt_f32_w32_gfx1150, rj_arith_v_cmp_lt_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLtF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_lt_f32_w64_gfx1150, rj_arith_v_cmp_lt_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpEqF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_eq_f32_w32_gfx1150, rj_arith_v_cmp_eq_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpEqF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_eq_f32_w64_gfx1150, rj_arith_v_cmp_eq_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLeF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_le_f32_w32_gfx1150, rj_arith_v_cmp_le_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLeF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_le_f32_w64_gfx1150, rj_arith_v_cmp_le_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpGtF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_gt_f32_w32_gfx1150, rj_arith_v_cmp_gt_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpGtF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_gt_f32_w64_gfx1150, rj_arith_v_cmp_gt_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLgF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_lg_f32_w32_gfx1150, rj_arith_v_cmp_lg_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLgF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_lg_f32_w64_gfx1150, rj_arith_v_cmp_lg_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpGeF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_ge_f32_w32_gfx1150, rj_arith_v_cmp_ge_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpGeF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_ge_f32_w64_gfx1150, rj_arith_v_cmp_ge_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpOF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_o_f32_w32_gfx1150, rj_arith_v_cmp_o_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpOF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_o_f32_w64_gfx1150, rj_arith_v_cmp_o_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpUF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_u_f32_w32_gfx1150, rj_arith_v_cmp_u_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpUF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_u_f32_w64_gfx1150, rj_arith_v_cmp_u_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNgeF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_nge_f32_w32_gfx1150, rj_arith_v_cmp_nge_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNgeF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_nge_f32_w64_gfx1150, rj_arith_v_cmp_nge_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNlgF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_nlg_f32_w32_gfx1150, rj_arith_v_cmp_nlg_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNlgF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_nlg_f32_w64_gfx1150, rj_arith_v_cmp_nlg_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNgtF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_ngt_f32_w32_gfx1150, rj_arith_v_cmp_ngt_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNgtF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_ngt_f32_w64_gfx1150, rj_arith_v_cmp_ngt_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNleF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_nle_f32_w32_gfx1150, rj_arith_v_cmp_nle_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNleF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_nle_f32_w64_gfx1150, rj_arith_v_cmp_nle_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNeqF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_neq_f32_w32_gfx1150, rj_arith_v_cmp_neq_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNeqF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_neq_f32_w64_gfx1150, rj_arith_v_cmp_neq_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNltF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_nlt_f32_w32_gfx1150, rj_arith_v_cmp_nlt_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNltF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_nlt_f32_w64_gfx1150, rj_arith_v_cmp_nlt_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpTF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmp_t_f32_w32_gfx1150, rj_arith_v_cmp_t_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpTF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmp_t_f32_w64_gfx1150, rj_arith_v_cmp_t_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxFF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_f_f32_w32_gfx1150, rj_arith_v_cmpx_f_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxFF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_f_f32_w64_gfx1150, rj_arith_v_cmpx_f_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLtF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_lt_f32_w32_gfx1150, rj_arith_v_cmpx_lt_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLtF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_lt_f32_w64_gfx1150, rj_arith_v_cmpx_lt_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxEqF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_eq_f32_w32_gfx1150, rj_arith_v_cmpx_eq_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxEqF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_eq_f32_w64_gfx1150, rj_arith_v_cmpx_eq_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLeF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_le_f32_w32_gfx1150, rj_arith_v_cmpx_le_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLeF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_le_f32_w64_gfx1150, rj_arith_v_cmpx_le_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxGtF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_gt_f32_w32_gfx1150, rj_arith_v_cmpx_gt_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxGtF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_gt_f32_w64_gfx1150, rj_arith_v_cmpx_gt_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLgF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_lg_f32_w32_gfx1150, rj_arith_v_cmpx_lg_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLgF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_lg_f32_w64_gfx1150, rj_arith_v_cmpx_lg_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxGeF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_ge_f32_w32_gfx1150, rj_arith_v_cmpx_ge_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxGeF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_ge_f32_w64_gfx1150, rj_arith_v_cmpx_ge_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxOF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_o_f32_w32_gfx1150, rj_arith_v_cmpx_o_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxOF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_o_f32_w64_gfx1150, rj_arith_v_cmpx_o_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxUF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_u_f32_w32_gfx1150, rj_arith_v_cmpx_u_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxUF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_u_f32_w64_gfx1150, rj_arith_v_cmpx_u_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNgeF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_nge_f32_w32_gfx1150, rj_arith_v_cmpx_nge_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNgeF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_nge_f32_w64_gfx1150, rj_arith_v_cmpx_nge_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNlgF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_nlg_f32_w32_gfx1150, rj_arith_v_cmpx_nlg_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNlgF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_nlg_f32_w64_gfx1150, rj_arith_v_cmpx_nlg_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNgtF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_ngt_f32_w32_gfx1150, rj_arith_v_cmpx_ngt_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNgtF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_ngt_f32_w64_gfx1150, rj_arith_v_cmpx_ngt_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNleF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_nle_f32_w32_gfx1150, rj_arith_v_cmpx_nle_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNleF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_nle_f32_w64_gfx1150, rj_arith_v_cmpx_nle_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNeqF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_neq_f32_w32_gfx1150, rj_arith_v_cmpx_neq_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNeqF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_neq_f32_w64_gfx1150, rj_arith_v_cmpx_neq_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNltF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_nlt_f32_w32_gfx1150, rj_arith_v_cmpx_nlt_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNltF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_nlt_f32_w64_gfx1150, rj_arith_v_cmpx_nlt_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxTF32W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 32>(rj_arith_v_cmpx_t_f32_w32_gfx1150, rj_arith_v_cmpx_t_f32_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxTF32W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<32, 64>(rj_arith_v_cmpx_t_f32_w64_gfx1150, rj_arith_v_cmpx_t_f32_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpFF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_f_f64_w32_gfx1150, rj_arith_v_cmp_f_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpFF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_f_f64_w64_gfx1150, rj_arith_v_cmp_f_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLtF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_lt_f64_w32_gfx1150, rj_arith_v_cmp_lt_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLtF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_lt_f64_w64_gfx1150, rj_arith_v_cmp_lt_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpEqF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_eq_f64_w32_gfx1150, rj_arith_v_cmp_eq_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpEqF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_eq_f64_w64_gfx1150, rj_arith_v_cmp_eq_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLeF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_le_f64_w32_gfx1150, rj_arith_v_cmp_le_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLeF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_le_f64_w64_gfx1150, rj_arith_v_cmp_le_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpGtF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_gt_f64_w32_gfx1150, rj_arith_v_cmp_gt_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpGtF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_gt_f64_w64_gfx1150, rj_arith_v_cmp_gt_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLgF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_lg_f64_w32_gfx1150, rj_arith_v_cmp_lg_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpLgF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_lg_f64_w64_gfx1150, rj_arith_v_cmp_lg_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpGeF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_ge_f64_w32_gfx1150, rj_arith_v_cmp_ge_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpGeF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_ge_f64_w64_gfx1150, rj_arith_v_cmp_ge_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpOF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_o_f64_w32_gfx1150, rj_arith_v_cmp_o_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpOF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_o_f64_w64_gfx1150, rj_arith_v_cmp_o_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpUF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_u_f64_w32_gfx1150, rj_arith_v_cmp_u_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpUF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_u_f64_w64_gfx1150, rj_arith_v_cmp_u_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNgeF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_nge_f64_w32_gfx1150, rj_arith_v_cmp_nge_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNgeF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_nge_f64_w64_gfx1150, rj_arith_v_cmp_nge_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNlgF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_nlg_f64_w32_gfx1150, rj_arith_v_cmp_nlg_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNlgF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_nlg_f64_w64_gfx1150, rj_arith_v_cmp_nlg_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNgtF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_ngt_f64_w32_gfx1150, rj_arith_v_cmp_ngt_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNgtF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_ngt_f64_w64_gfx1150, rj_arith_v_cmp_ngt_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNleF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_nle_f64_w32_gfx1150, rj_arith_v_cmp_nle_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNleF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_nle_f64_w64_gfx1150, rj_arith_v_cmp_nle_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNeqF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_neq_f64_w32_gfx1150, rj_arith_v_cmp_neq_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNeqF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_neq_f64_w64_gfx1150, rj_arith_v_cmp_neq_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNltF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_nlt_f64_w32_gfx1150, rj_arith_v_cmp_nlt_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpNltF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_nlt_f64_w64_gfx1150, rj_arith_v_cmp_nlt_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpTF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmp_t_f64_w32_gfx1150, rj_arith_v_cmp_t_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpTF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmp_t_f64_w64_gfx1150, rj_arith_v_cmp_t_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxFF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_f_f64_w32_gfx1150, rj_arith_v_cmpx_f_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxFF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_f_f64_w64_gfx1150, rj_arith_v_cmpx_f_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLtF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_lt_f64_w32_gfx1150, rj_arith_v_cmpx_lt_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLtF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_lt_f64_w64_gfx1150, rj_arith_v_cmpx_lt_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxEqF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_eq_f64_w32_gfx1150, rj_arith_v_cmpx_eq_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxEqF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_eq_f64_w64_gfx1150, rj_arith_v_cmpx_eq_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLeF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_le_f64_w32_gfx1150, rj_arith_v_cmpx_le_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLeF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_le_f64_w64_gfx1150, rj_arith_v_cmpx_le_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxGtF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_gt_f64_w32_gfx1150, rj_arith_v_cmpx_gt_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxGtF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_gt_f64_w64_gfx1150, rj_arith_v_cmpx_gt_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLgF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_lg_f64_w32_gfx1150, rj_arith_v_cmpx_lg_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxLgF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_lg_f64_w64_gfx1150, rj_arith_v_cmpx_lg_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxGeF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_ge_f64_w32_gfx1150, rj_arith_v_cmpx_ge_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxGeF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_ge_f64_w64_gfx1150, rj_arith_v_cmpx_ge_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxOF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_o_f64_w32_gfx1150, rj_arith_v_cmpx_o_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxOF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_o_f64_w64_gfx1150, rj_arith_v_cmpx_o_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxUF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_u_f64_w32_gfx1150, rj_arith_v_cmpx_u_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxUF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_u_f64_w64_gfx1150, rj_arith_v_cmpx_u_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNgeF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_nge_f64_w32_gfx1150, rj_arith_v_cmpx_nge_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNgeF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_nge_f64_w64_gfx1150, rj_arith_v_cmpx_nge_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNlgF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_nlg_f64_w32_gfx1150, rj_arith_v_cmpx_nlg_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNlgF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_nlg_f64_w64_gfx1150, rj_arith_v_cmpx_nlg_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNgtF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_ngt_f64_w32_gfx1150, rj_arith_v_cmpx_ngt_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNgtF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_ngt_f64_w64_gfx1150, rj_arith_v_cmpx_ngt_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNleF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_nle_f64_w32_gfx1150, rj_arith_v_cmpx_nle_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNleF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_nle_f64_w64_gfx1150, rj_arith_v_cmpx_nle_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNeqF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_neq_f64_w32_gfx1150, rj_arith_v_cmpx_neq_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNeqF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_neq_f64_w64_gfx1150, rj_arith_v_cmpx_neq_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNltF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_nlt_f64_w32_gfx1150, rj_arith_v_cmpx_nlt_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxNltF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_nlt_f64_w64_gfx1150, rj_arith_v_cmpx_nlt_f64_w64);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxTF64W32Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 32>(rj_arith_v_cmpx_t_f64_w32_gfx1150, rj_arith_v_cmpx_t_f64_w32);
}

// Check this predicate's binding, numeric mask and Compatibility causes.
TEST(CompareCmpxTF64W64Gfx1150, Arithmetic) {
  check_gfx1150_compare<64, 64>(rj_arith_v_cmpx_t_f64_w64_gfx1150, rj_arith_v_cmpx_t_f64_w64);
}

// A status output overlapping B must not change any later source reads.
TEST(CompareGfx1150StatusAliasing, Arithmetic) {
  for (uint64_t cpu : supported_cpu_levels()) {
    Registers<4, 64> r;
    for (int lane = 0; lane < 64; ++lane) {
      r.v[0][lane] = 0x3f800000;
      r.v[2][lane] = 0x3f800000;
    }
    r.v[0][63] = 0x7f800001;
    uint64_t result = 0;
    ASSERT_EQ(rj_arith_v_cmp_eq_f32_w64_gfx1150(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY |
                                                    RJ_ARITH_FP_DX10_CLAMP_OFF,
                                                ~0ULL, 0, &result, r.v, r.v + 2, r.v[2]),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(result, ~(1ULL << 63));
    EXPECT_EQ(r.v[2][0], 0x3f800001U);
  }
}

// Literal upper-half NaNs independently check the new full-wave SIMD mask width.
TEST(CompareGfx1150UpperNaN, Arithmetic) {
  using Call = decltype(&rj_arith_v_cmp_u_f16_w64_gfx1150);
  struct Format {
    int bits;
    Call call;
    uint64_t one, quiet;
  };
  const Format formats[] = {
      {16, rj_arith_v_cmp_u_f16_w64_gfx1150, 0x3c00, 0x7e01},
      {32, rj_arith_v_cmp_u_f32_w64_gfx1150, 0x3f800000, 0x7fc00001},
      {64, rj_arith_v_cmp_u_f64_w64_gfx1150, 0x3ff0000000000000ULL, 0x7ff8000000000001ULL},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &format : formats) {
      Registers<4, 64> input;
      for (int lane = 0; lane < 64; ++lane) {
        write_compare_operand(input.v, lane, format.bits, format.one);
        write_compare_operand(input.v + 2, lane, format.bits, format.one);
      }
      write_compare_operand(input.v, 63, format.bits, format.quiet);
      for (uint64_t mode : {uint64_t(0), uint64_t(RJ_ARITH_ALU_CLAMP)}) {
        uint64_t result = 0;
        uint32_t status = 0x100;
        ASSERT_EQ(format.call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_DX10_CLAMP_OFF,
                              ~0ULL, mode, &result, input.v, input.v + 2, &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(result, 1ULL << 63);
        EXPECT_EQ(status, mode ? 0x101U : 0x100U);
      }
    }
  }
}

} // namespace
