// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1150 dual wrappers retain ordinary arithmetic and their inherited DX9 output stage.
#include "dual/helpers.h"
#include "rj_arith/detail/rj_arith_dual.h"
#include "rj_arith/detail/rj_arith_fp.h"

// Ordinary gfx1150 arithmetic already validates the numerical/status model.
TEST(DualGfx1150, Add) {
  rocjitsu::arith::test::check_dual(rj_arith_v_dual_add_f32_w32_gfx1150,
                                    rj_arith_v_add_f32_w32_gfx1150, 0x40000000U, 0x40400000U,
                                    0x40800000U, 0x40a00000U);
}

// Ordinary gfx1150 arithmetic already validates the numerical/status model.
TEST(DualGfx1150, Sub) {
  rocjitsu::arith::test::check_dual(rj_arith_v_dual_sub_f32_w32_gfx1150,
                                    rj_arith_v_sub_f32_w32_gfx1150, 0x40000000U, 0x40400000U,
                                    0x40800000U, 0xbf800000U);
}

// Ordinary gfx1150 arithmetic already validates the numerical/status model.
TEST(DualGfx1150, Subrev) {
  rocjitsu::arith::test::check_dual(rj_arith_v_dual_subrev_f32_w32_gfx1150,
                                    rj_arith_v_subrev_f32_w32_gfx1150, 0x40000000U, 0x40400000U,
                                    0x40800000U, 0x3f800000U);
}

// Ordinary gfx1150 arithmetic already validates the numerical/status model.
TEST(DualGfx1150, Mul) {
  rocjitsu::arith::test::check_dual(rj_arith_v_dual_mul_f32_w32_gfx1150,
                                    rj_arith_v_mul_f32_w32_gfx1150, 0x40000000U, 0x40400000U,
                                    0x40800000U, 0x40c00000U);
}

// Exercise accumulator/literal positions without duplicating the FMA reference.
TEST(DualGfx1150, Fmac) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_dual_fmac_f32_w32_gfx1150(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_fma_f32_w32_gfx1150(f, e, m, d, a, b, d, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x41200000U);
}

// Exercise accumulator/literal positions without duplicating the FMA reference.
TEST(DualGfx1150, Fmaak) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_dual_fmaak_f32_w32_gfx1150(f, e, m, d, a, b, 0x40800000U, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        uint32_t values[32];
        std::fill_n(values, 32, 0x40800000U);
        const uint32_t *literal = values;
        return rj_arith_v_fma_f32_w32_gfx1150(f, e, m, d, a, b, &literal, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x41200000U);
}

// Exercise accumulator/literal positions without duplicating the FMA reference.
TEST(DualGfx1150, Fmamk) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_dual_fmamk_f32_w32_gfx1150(f, e, m, d, a, 0x40800000U, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        uint32_t values[32];
        std::fill_n(values, 32, 0x40800000U);
        const uint32_t *literal = values;
        return rj_arith_v_fma_f32_w32_gfx1150(f, e, m, d, a, &literal, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x41300000U);
}

// Legacy native multiplication flushes after rounding, so a rounded minimum normal survives.
TEST(DualGfx1150, Dx9PostRoundFlushing) {
  struct Case {
    uint32_t a, b, expected[4];
  };
  const Case cases[] = {
      {0x00800000, 0x3f7fffff, {0x00800000, 0x00800000, 0, 0}},
      {0x80800000, 0x3f7fffff, {0x80800000, 0x80000000, 0x80800000, 0x80000000}},
      {0x80000000, 0x7f800123, {0, 0, 0, 0}},
      {0x7f800123, 0x3f800000, {0x7fc00123, 0x7fc00123, 0x7fc00123, 0x7fc00123}},
  };
  for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                             RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
    uint32_t status = 0x100;
    EXPECT_EQ(rj_arith_v_dual_mul_dx9_zero_f32_w32_gfx1150(semantics, 0, 0, nullptr, nullptr,
                                                           nullptr, &status),
              RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(status, 0x100U);
  }
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (uint32_t rounding = 0; rounding < 4; ++rounding) {
      for (const auto &test : cases) {
        for (uint32_t exec_mask : {0U, 0x80000001U, UINT32_MAX}) {
          for (int aliasing_offset : {0, 1, 3}) {
            rocjitsu::arith::test::Registers<4, 32> input;
            for (int lane = 0; lane < 32; ++lane) {
              input.v[0][lane] = test.a;
              input.v[1][lane] = test.b;
            }
            auto before = input;
            uint32_t status = 0x100;
            ASSERT_EQ(rj_arith_v_dual_mul_dx9_zero_f32_w32_gfx1150(
                          cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY |
                              RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | (uint64_t(rounding) << 24),
                          exec_mask, 0, input.v + aliasing_offset, input.v, input.v + 1, &status),
                      RJ_ARITH_SUCCESS);
            for (int lane = 0; lane < 32; ++lane) {
              EXPECT_EQ(input.v[aliasing_offset][lane], (exec_mask & (1U << lane))
                                                            ? test.expected[rounding]
                                                            : before.v[aliasing_offset][lane]);
            }
            input.expect_preserved(before, aliasing_offset, 1);
            EXPECT_EQ(status, 0x100U);
          }
        }
      }
    }
  }
}
