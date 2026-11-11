// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP add/subtract and reversed-subtract entry points.

#include "fp/binary_arithmetic_test.h"
#include "fp/captured_gfx11_test.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_add_f16_w32_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_add_f16_w32, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_add_f16_w64_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_add_f16_w64, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_add_f32_w32_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_add_f32_w32, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_add_f32_w64_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_add_f32_w64, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 32>(
      rj_arith_v_add_f64_w32_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 32>(
      rj_arith_v_add_f64_w32, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 64>(
      rj_arith_v_add_f64_w64_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 64>(
      rj_arith_v_add_f64_w64, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_sub_f16_w32_gfx12, [](double x, double y) { return x - y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_sub_f16_w32, [](double x, double y) { return x - y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_sub_f16_w64_gfx12, [](double x, double y) { return x - y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_sub_f16_w64, [](double x, double y) { return x - y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_sub_f32_w32_gfx12, [](double x, double y) { return x - y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_sub_f32_w32, [](double x, double y) { return x - y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_sub_f32_w64_gfx12, [](double x, double y) { return x - y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_sub_f32_w64, [](double x, double y) { return x - y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_subrev_f16_w32_gfx12, [](double x, double y) { return y - x; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_subrev_f16_w32, [](double x, double y) { return y - x; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_subrev_f16_w64_gfx12, [](double x, double y) { return y - x; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_subrev_f16_w64, [](double x, double y) { return y - x; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_subrev_f32_w32_gfx12, [](double x, double y) { return y - x; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_subrev_f32_w32, [](double x, double y) { return y - x; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_subrev_f32_w64_gfx12, [](double x, double y) { return y - x; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_subrev_f32_w64, [](double x, double y) { return y - x; }, /*gfx12=*/false);
}

// gfx1100/gfx1201 subtraction captures select NaNs in original source order.
TEST(FpSubtractF32, NaNSignAndPriority) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_sub_f32_w32, rj_arith_v_sub_f32_w32_gfx12}) {
      Registers<3, 32> r;
      r.v[0][0] = 0x3f800000U;
      r.v[1][0] = 0x7fc00001U;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 1, 0, r.v + 2, r.v,
                     r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[2][0], 0xffc00001U);
    }
    for (auto call : {rj_arith_v_subrev_f32_w32, rj_arith_v_subrev_f32_w32_gfx12}) {
      Registers<3, 32> r;
      r.v[0][0] = 0x7fc00001U;
      r.v[1][0] = 0xffc00002U;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 1, 0, r.v + 2, r.v,
                     r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[2][0], 0xffc00001U);
    }
    // RDNA3 FMA's inherited IEEE-clear model preserves the selected signaling NaN.
    for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
      Registers<3, 32> r;
      r.v[0][0] = 0x3f800000U;
      r.v[1][0] = 0x7f800001U;
      ASSERT_EQ(
          rj_arith_v_sub_f32_w32(cpu | ieee | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                                 1, 0, r.v + 2, r.v, r.v + 1, nullptr),
          RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[2][0], ieee ? 0xffc00001U : 0xff800001U);
    }
  }
}

// DX10_CLAMP_OFF is a shared guest control, including for ordinary RDNA3 binary arithmetic.
TEST(FpBinaryF32, NaNClampControl) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (auto call : {rj_arith_v_add_f32_w32, rj_arith_v_sub_f32_w32, rj_arith_v_subrev_f32_w32,
                        rj_arith_v_mul_f32_w32, rj_arith_v_mul_dx9_zero_f32_w32}) {
        Registers<3, 32> r;
        r.v[0][0] = 0x7fc00001U;
        r.v[1][0] = 0x3f800000U;
        ASSERT_EQ(call(cpu | semantics | RJ_ARITH_FP_DX10_CLAMP_OFF, 1, RJ_ARITH_ALU_CLAMP, r.v + 2,
                       r.v, r.v + 1, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_GT(r.v[2][0] & 0x7fffffffU, 0x7f800000U);
      }
    }
  }
}

// Subtraction negates the subtrahend, including NaNs, without swapping source priority.
TEST(FpSubF16, CompatibilityNanSigns) {
  using namespace rocjitsu::arith::test;
  constexpr uint16_t inputs[][4] = {
      {0x8000, 0x7e00, 0xfe00, 0x7e00},
      {0x0000, 0x7c01, 0xfe01, 0x7e01},
      {0x7e00, 0x7e00, 0x7e00, 0xfe00},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (bool reverse : {false, true}) {
      auto call = reverse ? rj_arith_v_subrev_f16_w32_gfx12 : rj_arith_v_sub_f16_w32_gfx12;
      for (const auto &sample : inputs) {
        Registers<3, 32> r;
        r.v[0][0] = sample[0];
        r.v[1][0] = sample[1];
        uint32_t status = 0;
        ASSERT_EQ(
            call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, 1, 0, r.v + 2, r.v, r.v + 1, &status),
            RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[2][0] & 0xffffU, sample[reverse ? 3 : 2]);
      }
    }
  }
}

// Compatibility quiets signaling NaNs on gfx11 only when IEEE is set; with IEEE clear the
// source is kept, as gfx1100 does.
TEST(FpBinaryF16, CompatibilitySignalingNan) {
  using namespace rocjitsu::arith::test;
  const auto calls = {rj_arith_v_add_f16_w32, rj_arith_v_sub_f16_w32, rj_arith_v_mul_f16_w32};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : calls) {
      for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
        for (uint64_t rounding : {uint64_t(0), RJ_ARITH_FP_ROUND_UP}) {
          for (uint64_t omod : {uint64_t(0), uint64_t(RJ_ARITH_ALU_OMOD_2)}) {
            Registers<3, 32> r;
            r.v[0][0] = 0x3c00U;
            r.v[1][0] = 0x7c01U;
            // gfx11 OMOD is enabled by output flushing and IEEE clear.
            uint64_t flags = cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | ieee | rounding |
                             (omod ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0);
            ASSERT_EQ(call(flags, 1, omod, r.v + 2, r.v, r.v + 1, nullptr), RJ_ARITH_SUCCESS);
            const uint32_t quiet = ieee ? 0x0200U : 0;
            EXPECT_EQ(r.v[2][0] & 0xffffU,
                      (call == rj_arith_v_sub_f16_w32 ? 0xfc01U : 0x7c01U) | quiet);
          }
        }
      }
    }
  }
}

TEST(FpAddF64, GuestRounding) {
  using rocjitsu::arith::test::check_binary64_rounding;
  const struct {
    uint64_t a, b, expected[4];
  } cases[] = {{0x3ff0000000000000ULL,
                0x3ca8000000000000ULL,
                {0x3ff0000000000001ULL, 0x3ff0000000000001ULL, 0x3ff0000000000000ULL,
                 0x3ff0000000000000ULL}},
               {0xbff0000000000000ULL,
                0xbca8000000000000ULL,
                {0xbff0000000000001ULL, 0xbff0000000000000ULL, 0xbff0000000000001ULL,
                 0xbff0000000000000ULL}},
               {0x3ff0000000000000ULL, 0xbff0000000000000ULL, {0, 0, 0x8000000000000000ULL, 0}},
               {0x7fefffffffffffffULL,
                0x7fefffffffffffffULL,
                {0x7ff0000000000000ULL, 0x7ff0000000000000ULL, 0x7fefffffffffffffULL,
                 0x7fefffffffffffffULL}}};
  for (const auto &sample : cases) {
    for (auto call : {rj_arith_v_add_f64_w32, rj_arith_v_add_f64_w32_gfx12}) {
      check_binary64_rounding<32>(call, sample.a, sample.b, sample.expected);
    }
    for (auto call : {rj_arith_v_add_f64_w64, rj_arith_v_add_f64_w64_gfx12}) {
      check_binary64_rounding<64>(call, sample.a, sample.b, sample.expected);
    }
  }
}

// gfx1201 captures: FP64 ADD returns the first NaN source, quieted, in operand order under every
// guest rounding mode.
TEST(FpAddF64, NanPriority) {
  using rocjitsu::arith::test::check_binary64_rounding;
  const uint64_t cases[][3] = {
      {0x7ff0000000000001ULL, 0x7ff8000000000000ULL, 0x7ff8000000000001ULL},
      {0xfff8000000000000ULL, 0x7ff8000000000000ULL, 0xfff8000000000000ULL},
      {0x7ff4000012345678ULL, 0x7ff8000000000000ULL, 0x7ffc000012345678ULL},
      {0x7ff8000000000000ULL, 0x7ff0000000000001ULL, 0x7ff8000000000000ULL},
      {0xfff8000000000000ULL, 0x7ff0000000000001ULL, 0xfff8000000000000ULL},
  };
  for (const auto &sample : cases) {
    const uint64_t expected[] = {sample[2], sample[2], sample[2], sample[2]};
    check_binary64_rounding<32>(rj_arith_v_add_f64_w32_gfx12, sample[0], sample[1], expected);
    check_binary64_rounding<64>(rj_arith_v_add_f64_w64_gfx12, sample[0], sample[1], expected);
  }
}

namespace {

using rocjitsu::arith::test::CapturedGfx11;

// gfx1100 lanes from the MODE.IEEE-clear corpus (hw1100-noieee): a signaling NaN source is
// returned unquieted, after NEG and the SUB sign flip, also under directed rounding and OMOD.
const CapturedGfx11 add_f16_lanes[] = {
    {"E32Mode00", 0, 0x100, 0x63377c01, 0x12680000, 0, 0xa5a57c01},
    {"E32RoundDown", 0, 0x15a, 0xaabb0000, 0xd5d77c01, 0, 0xa5a57c01},
    {"NegB", RJ_ARITH_ALU_NEG_B, 0x1f0, 0x2d450000, 0x452b7c01, 0, 0xa5a5fc01},
    {"Mul2", RJ_ARITH_ALU_OMOD_2, 0x100, 0xc9457c01, 0xba620000, 0, 0xa5a57c01},
    {"HighHalves", RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D, 0x1f0,
     0xaf2ec100, 0xfd498000, 0, 0xfd49a5a5},
};
const CapturedGfx11 sub_f16_lanes[] = {
    {"E32Mode00", 0, 0x100, 0xbd0b7c01, 0x78430000, 0, 0xa5a57c01},
    {"E32RoundDownSubtrahend", 0, 0x15a, 0x7a990000, 0x0e867c01, 0, 0xa5a5fc01},
    {"NegB", RJ_ARITH_ALU_NEG_B, 0x1f0, 0xadab0000, 0xb5ff7c01, 0, 0xa5a57c01},
    {"Mul2", RJ_ARITH_ALU_OMOD_2, 0x100, 0x76637c01, 0xb7bd0000, 0, 0xa5a57c01},
    {"HighHalves", RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D, 0x1f0,
     0xa6ec83ff, 0xfce20000, 0, 0x7ce2a5a5},
};
const CapturedGfx11 subrev_f16_lanes[] = {
    {"E32Mode00", 0, 0x100, 0xa4e67c01, 0x42b40000, 0, 0xa5a5fc01},
    {"E32RoundDown", 0, 0x15a, 0x940d0000, 0x9faf7c01, 0, 0xa5a57c01},
    {"NegB", RJ_ARITH_ALU_NEG_B, 0x1f0, 0x1b0d0000, 0x0e5b7c01, 0, 0xa5a5fc01},
    {"Mul2", RJ_ARITH_ALU_OMOD_2, 0x100, 0xc29d7c01, 0x569b0000, 0, 0xa5a5fc01},
    {"HighHalves", RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D, 0x1f0,
     0x4d62fe00, 0x7d338000, 0, 0x7d33a5a5},
};

// gfx1100 (W7900) CLAMP lanes with a signaling-NaN source. DX10_CLAMP clear lets the NaN
// through CLAMP, unquieted with IEEE clear and quieted with IEEE set; DX10_CLAMP set gives 0.
const CapturedGfx11 add_f16_clamp_lanes[] = {
    {"Dx10ClampOff", RJ_ARITH_ALU_CLAMP, 0x0f0, 0xa5a57c01, 0xa5a53c00, 0, 0xa5a57c01},
    {"Dx10ClampOffFlush", RJ_ARITH_ALU_CLAMP, 0x000, 0xa5a57c01, 0xa5a53c00, 0, 0xa5a57c01},
    {"Dx10ClampOn", RJ_ARITH_ALU_CLAMP, 0x1f0, 0xa5a57c01, 0xa5a53c00, 0, 0xa5a50000},
    {"IeeeSet", RJ_ARITH_ALU_CLAMP, 0x2f0, 0xa5a57c01, 0xa5a53c00, 0, 0xa5a57e01},
};
const CapturedGfx11 sub_f16_clamp_lanes[] = {
    {"Dx10ClampOff", RJ_ARITH_ALU_CLAMP, 0x0f0, 0xa5a53c00, 0xa5a57c01, 0, 0xa5a5fc01},
    {"Dx10ClampOn", RJ_ARITH_ALU_CLAMP, 0x1f0, 0xa5a53c00, 0xa5a57c01, 0, 0xa5a50000},
    {"IeeeSet", RJ_ARITH_ALU_CLAMP, 0x2f0, 0xa5a53c00, 0xa5a57c01, 0, 0xa5a5fe01},
};
const CapturedGfx11 subrev_f16_clamp_lanes[] = {
    {"Dx10ClampOff", RJ_ARITH_ALU_CLAMP, 0x0f0, 0xa5a57c01, 0xa5a53c00, 0, 0xa5a5fc01},
    {"Dx10ClampOn", RJ_ARITH_ALU_CLAMP, 0x1f0, 0xa5a57c01, 0xa5a53c00, 0, 0xa5a50000},
    {"IeeeSet", RJ_ARITH_ALU_CLAMP, 0x2f0, 0xa5a57c01, 0xa5a53c00, 0, 0xa5a5fe01},
};

// gfx1100 ADD_F64 returns the first NaN source with its modifiers applied, quieted only when
// MODE.IEEE is set. IEEE-set lanes are from the gfx1100 v2 capture (hw2-gfx1100).
const CapturedGfx11 add_f64_lanes[] = {
    {"SignalingFirst", 0, 0x1f0, 0x7ff0000000000001, 0x7ff8000000000000, 0, 0x7ff0000000000001},
    {"NegativeQuietFirst", 0, 0x1f0, 0xfff8000000000000, 0x7ff8000000000000, 0, 0xfff8000000000000},
    {"PayloadFirst", 0, 0x1f0, 0x7ff8000012345678, 0x7ff8000000000000, 0, 0x7ff8000012345678},
    {"SignalingSecondRoundZero", 0, 0x1ff, 0x0, 0x7ff0000000000001, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0x1f0, 0x7ff8000000000000, 0x7ff8000000000000, 0,
     0xfff8000000000000},
    {"Mul2", RJ_ARITH_ALU_OMOD_2, 0x100, 0x7ff0000000000001, 0x0, 0, 0x7ff0000000000001},
    {"IeeeSetSignalingFirst", 0, 0x3f0, 0x7ff0000000000001, 0x7ff8000000000000, 0,
     0x7ff8000000000001},
    {"IeeeSetNegativeQuietFirst", 0, 0x3f0, 0xfff8000000000000, 0x7ff8000000000000, 0,
     0xfff8000000000000},
};

} // namespace

// Compatibility reproduces captured gfx1100 FP16 ADD/SUB/SUBREV lanes.
TEST(FpAddSubF16Gfx11, CapturedSignalingNans) {
  using rocjitsu::arith::test::check_captured_gfx11;
  check_captured_gfx11<32, 16>(rj_arith_v_add_f16_w32, add_f16_lanes);
  check_captured_gfx11<64, 16>(rj_arith_v_add_f16_w64, add_f16_lanes);
  check_captured_gfx11<32, 16>(rj_arith_v_sub_f16_w32, sub_f16_lanes);
  check_captured_gfx11<64, 16>(rj_arith_v_sub_f16_w64, sub_f16_lanes);
  check_captured_gfx11<32, 16>(rj_arith_v_subrev_f16_w32, subrev_f16_lanes);
  check_captured_gfx11<64, 16>(rj_arith_v_subrev_f16_w64, subrev_f16_lanes);
}

// Compatibility reproduces captured gfx1100 clamped FP16 ADD/SUB/SUBREV NaN lanes.
TEST(FpAddSubF16Gfx11, CapturedClampedNans) {
  using rocjitsu::arith::test::check_captured_gfx11;
  check_captured_gfx11<32, 16>(rj_arith_v_add_f16_w32, add_f16_clamp_lanes);
  check_captured_gfx11<64, 16>(rj_arith_v_add_f16_w64, add_f16_clamp_lanes);
  check_captured_gfx11<32, 16>(rj_arith_v_sub_f16_w32, sub_f16_clamp_lanes);
  check_captured_gfx11<64, 16>(rj_arith_v_sub_f16_w64, sub_f16_clamp_lanes);
  check_captured_gfx11<32, 16>(rj_arith_v_subrev_f16_w32, subrev_f16_clamp_lanes);
  check_captured_gfx11<64, 16>(rj_arith_v_subrev_f16_w64, subrev_f16_clamp_lanes);
}

// Compatibility reproduces captured gfx1100 FP64 ADD NaN lanes.
TEST(FpAddF64Gfx11, CapturedNans) {
  using rocjitsu::arith::test::check_captured_gfx11;
  check_captured_gfx11<32, 64>(rj_arith_v_add_f64_w32, add_f64_lanes);
  check_captured_gfx11<64, 64>(rj_arith_v_add_f64_w64, add_f64_lanes);
}
