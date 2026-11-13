// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for division operand rescaling and the resulting condition mask.

#include "fp/div_scale_test.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <type_traits>

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<32, 32>(rj_arith_v_div_scale_f32_w32_gfx12);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<32, 32>(rj_arith_v_div_scale_f32_w32);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<32, 64>(rj_arith_v_div_scale_f32_w64_gfx12);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<32, 64>(rj_arith_v_div_scale_f32_w64);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<64, 32>(rj_arith_v_div_scale_f64_w32_gfx12);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<64, 32>(rj_arith_v_div_scale_f64_w32);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<64, 64>(rj_arith_v_div_scale_f64_w64_gfx12);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<64, 64>(rj_arith_v_div_scale_f64_w64);
}

// gfx1100/W7900 captures supplied in PR #13952: selected sNaNs stay signaling
// regardless of IEEE; A equals the NaN denominator or numerator.
template <int Width, int Lanes, typename Call> static void check_rdna3_nan(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  constexpr uint64_t one = Width == 32 ? 0x3f800000ULL : 0x3ff0000000000000ULL;
  constexpr uint64_t snan = Width == 32 ? 0x7f800001ULL : 0x7ff0000000000001ULL;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
        for (int operand : {1, 2}) {
          Registers<8, Lanes> r;
          for (int i = 0; i < 3; ++i) {
            write_fp(r.v + 2 * i, Lanes - 1, Width, i == 0 || i == operand ? snan : one);
          }
          Mask condition = ~Mask(0);
          ASSERT_EQ(call(cpu | semantics | ieee, Mask(1) << (Lanes - 1), 0, r.v + 6, &condition,
                         r.v, r.v + 2, r.v + 4),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(condition & ~(Mask(1) << (Lanes - 1)), Mask(0));
          EXPECT_EQ(read_fp(r.v + 6, Lanes - 1, Width), snan);
        }
      }
    }
  }
}

TEST(FpDivScaleGfx11, SignalingNan) {
  check_rdna3_nan<32, 32>(rj_arith_v_div_scale_f32_w32);
  check_rdna3_nan<32, 64>(rj_arith_v_div_scale_f32_w64);
  check_rdna3_nan<64, 32>(rj_arith_v_div_scale_f64_w32);
  check_rdna3_nan<64, 64>(rj_arith_v_div_scale_f64_w64);
}

// The CDNA model shares gfx12's validated scale core, with output modifiers disabled.
template <int Width, int Lanes, typename Call>
static void check_cdna_scale(Call call, Call inherited) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  auto input = make_div_scale_inputs<Width, Lanes>();
  constexpr uint64_t snan = Width == 32 ? 0x7f800001ULL : 0x7ff0000000000001ULL;
  constexpr uint64_t quiet = Width == 32 ? 0x400000ULL : 0x8000000000000ULL;
  write_fp(input.v, 0, Width, snan);
  write_fp(input.v + 2, 0, Width, snan);
  for (uint64_t cpu : supported_cpu_levels()) {
    for (int sample = 0; sample < 64; ++sample) {
      const uint64_t fp = (uint64_t(sample & 3) << 24) |
                          (sample & 4 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0) |
                          (sample & 8 ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                          (sample & 16 ? RJ_ARITH_FP_IEEE_MODE : 0);
      const uint64_t signs = sample & 32 ? RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B : 0;
      const uint64_t ignored = uint64_t(sample & 3) * RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP;
      const Mask exec_mask = Mask(0xaaaaaaaaaaaaaaaaULL) | 1;
      const int destination = sample & 1 ? 0 : 6;
      auto expected = input;
      Mask expected_condition = 0;
      ASSERT_EQ(inherited(fp | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, exec_mask,
                          signs, expected.v + destination, &expected_condition, expected.v,
                          expected.v + 2, expected.v + 4),
                0);
      for (uint64_t semantics :
           {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
            RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
        auto actual = input;
        Mask condition = ~Mask(0);
        ASSERT_EQ(call(cpu | fp | semantics, exec_mask, signs | ignored, actual.v + destination,
                       &condition, actual.v, actual.v + 2, actual.v + 4),
                  0);
        EXPECT_EQ(condition, expected_condition);
        actual.expect_preserved(input, destination, Width / 32);
        for (int reg = 0; reg < Width / 32; ++reg) {
          for (int lane = 0; lane < Lanes; ++lane) {
            EXPECT_EQ(actual.v[destination + reg][lane], expected.v[destination + reg][lane]);
          }
        }
        EXPECT_EQ(read_fp(actual.v + destination, 0, Width) & ~(1ULL << (Width - 1)), snan | quiet);
      }
    }
  }
  Mask condition = 7;
  EXPECT_EQ(call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, Mask(0), 0, nullptr, &condition, nullptr,
                 nullptr, nullptr),
            RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
  EXPECT_EQ(condition, Mask(7));
  EXPECT_EQ(call(0, Mask(0), 0, nullptr, &condition, nullptr, nullptr, nullptr), 0);
  EXPECT_EQ(condition, Mask(0));
}

TEST(CdnaDivScale, Arithmetic) {
  check_cdna_scale<32, 32>(rj_arith_v_div_scale_f32_w32_gfx942, rj_arith_v_div_scale_f32_w32_gfx12);
  check_cdna_scale<32, 64>(rj_arith_v_div_scale_f32_w64_gfx942, rj_arith_v_div_scale_f32_w64_gfx12);
  check_cdna_scale<64, 32>(rj_arith_v_div_scale_f64_w32_gfx942, rj_arith_v_div_scale_f64_w32_gfx12);
  check_cdna_scale<64, 64>(rj_arith_v_div_scale_f64_w64_gfx942, rj_arith_v_div_scale_f64_w64_gfx12);
}
