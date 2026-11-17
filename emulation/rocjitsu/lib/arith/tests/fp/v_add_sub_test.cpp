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
      rj_arith_v_add_f32_w32_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true,
      /*fp_status_supported=*/false, /*compatibility_status=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_add_f32_w32, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_add_f32_w64_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true,
      /*fp_status_supported=*/false, /*compatibility_status=*/true);
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
      rj_arith_v_sub_f32_w32_gfx12, [](double x, double y) { return x - y; }, /*gfx12=*/true,
      /*fp_status_supported=*/false, /*compatibility_status=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_sub_f32_w32, [](double x, double y) { return x - y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_sub_f32_w64_gfx12, [](double x, double y) { return x - y; }, /*gfx12=*/true,
      /*fp_status_supported=*/false, /*compatibility_status=*/true);
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

// gfx1250 keeps the gfx12 model but refuses strict exact until its captures are reviewed.
TEST(FpAddF16W32Gfx1250, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_add_f16_w32_gfx1250, [](double x, double y) { return x + y; },
      /*gfx12=*/true, /*fp_status_supported=*/false,
      /*compatibility_status=*/false, /*half_exact=*/false);
}

// RDNA2 keeps the RDNA3 model but refuses strict exact until its captures are reviewed.
TEST(FpAddF16W32Gfx10, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_add_f16_w32_gfx10, [](double x, double y) { return x + y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      /*compatibility_status=*/false, /*half_exact=*/false);
}

// gfx1250 keeps the gfx12 model but refuses strict exact until its captures are reviewed.
TEST(FpAddF16W64Gfx1250, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_add_f16_w64_gfx1250, [](double x, double y) { return x + y; },
      /*gfx12=*/true, /*fp_status_supported=*/false,
      /*compatibility_status=*/false, /*half_exact=*/false);
}

// RDNA2 keeps the RDNA3 model but refuses strict exact until its captures are reviewed.
TEST(FpAddF16W64Gfx10, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_add_f16_w64_gfx10, [](double x, double y) { return x + y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      /*compatibility_status=*/false, /*half_exact=*/false);
}

// gfx1250 keeps the gfx12 model but refuses strict exact until its captures are reviewed.
TEST(FpSubF16W32Gfx1250, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_sub_f16_w32_gfx1250, [](double x, double y) { return x - y; },
      /*gfx12=*/true, /*fp_status_supported=*/false,
      /*compatibility_status=*/false, /*half_exact=*/false);
}

// RDNA2 keeps the RDNA3 model but refuses strict exact until its captures are reviewed.
TEST(FpSubF16W32Gfx10, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_sub_f16_w32_gfx10, [](double x, double y) { return x - y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      /*compatibility_status=*/false, /*half_exact=*/false);
}

// gfx1250 keeps the gfx12 model but refuses strict exact until its captures are reviewed.
TEST(FpSubF16W64Gfx1250, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_sub_f16_w64_gfx1250, [](double x, double y) { return x - y; },
      /*gfx12=*/true, /*fp_status_supported=*/false,
      /*compatibility_status=*/false, /*half_exact=*/false);
}

// RDNA2 keeps the RDNA3 model but refuses strict exact until its captures are reviewed.
TEST(FpSubF16W64Gfx10, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_sub_f16_w64_gfx10, [](double x, double y) { return x - y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      /*compatibility_status=*/false, /*half_exact=*/false);
}

// gfx1250 keeps the gfx12 model but refuses strict exact until its captures are reviewed.
TEST(FpSubrevF16W32Gfx1250, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_subrev_f16_w32_gfx1250, [](double x, double y) { return y - x; },
      /*gfx12=*/true, /*fp_status_supported=*/false,
      /*compatibility_status=*/false, /*half_exact=*/false);
}

// RDNA2 keeps the RDNA3 model but refuses strict exact until its captures are reviewed.
TEST(FpSubrevF16W32Gfx10, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_subrev_f16_w32_gfx10, [](double x, double y) { return y - x; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      /*compatibility_status=*/false, /*half_exact=*/false);
}

// gfx1250 keeps the gfx12 model but refuses strict exact until its captures are reviewed.
TEST(FpSubrevF16W64Gfx1250, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_subrev_f16_w64_gfx1250, [](double x, double y) { return y - x; },
      /*gfx12=*/true, /*fp_status_supported=*/false,
      /*compatibility_status=*/false, /*half_exact=*/false);
}

// RDNA2 keeps the RDNA3 model but refuses strict exact until its captures are reviewed.
TEST(FpSubrevF16W64Gfx10, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_subrev_f16_w64_gfx10, [](double x, double y) { return y - x; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      /*compatibility_status=*/false, /*half_exact=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_subrev_f32_w32_gfx12, [](double x, double y) { return y - x; }, /*gfx12=*/true,
      /*fp_status_supported=*/false, /*compatibility_status=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_subrev_f32_w32, [](double x, double y) { return y - x; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_subrev_f32_w64_gfx12, [](double x, double y) { return y - x; }, /*gfx12=*/true,
      /*fp_status_supported=*/false, /*compatibility_status=*/true);
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

// Subtraction negates the subtrahend, including NaNs, without swapping source priority, in
// Compatibility and strict exact.
TEST(FpSubF16, NanSigns) {
  using namespace rocjitsu::arith::test;
  constexpr uint16_t inputs[][4] = {
      {0x8000, 0x7e00, 0xfe00, 0x7e00},
      {0x0000, 0x7c01, 0xfe01, 0x7e01},
      {0x7e00, 0x7e00, 0x7e00, 0xfe00},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : hardware_semantics()) {
      for (bool reverse : {false, true}) {
        auto call = reverse ? rj_arith_v_subrev_f16_w32_gfx12 : rj_arith_v_sub_f16_w32_gfx12;
        for (const auto &sample : inputs) {
          Registers<3, 32> r;
          r.v[0][0] = sample[0];
          r.v[1][0] = sample[1];
          uint32_t status = 0;
          ASSERT_EQ(call(cpu | semantics, 1, 0, r.v + 2, r.v, r.v + 1, &status), RJ_ARITH_SUCCESS);
          EXPECT_EQ(r.v[2][0] & 0xffffU, sample[reverse ? 3 : 2]);
        }
      }
    }
  }
}

// gfx11 quiets signaling NaNs only when IEEE is set; with IEEE clear the source is kept, as
// gfx1100 does, in Compatibility and strict exact alike.
TEST(FpBinaryF16, SignalingNan) {
  using namespace rocjitsu::arith::test;
  const auto calls = {rj_arith_v_add_f16_w32, rj_arith_v_sub_f16_w32, rj_arith_v_mul_f16_w32};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : hardware_semantics()) {
      for (auto call : calls) {
        for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
          for (uint64_t rounding : {uint64_t(0), RJ_ARITH_FP_ROUND_UP}) {
            for (uint64_t omod : {uint64_t(0), uint64_t(RJ_ARITH_ALU_OMOD_2)}) {
              Registers<3, 32> r;
              r.v[0][0] = 0x3c00U;
              r.v[1][0] = 0x7c01U;
              // gfx11 OMOD is enabled by output flushing and IEEE clear.
              uint64_t flags = cpu | semantics | ieee | rounding |
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
    check_binary64_rounding<32>(rj_arith_v_add_f64_w32, sample.a, sample.b, sample.expected, 0, 0,
                                false);
    check_binary64_rounding<32>(rj_arith_v_add_f64_w32_gfx12, sample.a, sample.b, sample.expected);
    check_binary64_rounding<64>(rj_arith_v_add_f64_w64, sample.a, sample.b, sample.expected, 0, 0,
                                false);
    check_binary64_rounding<64>(rj_arith_v_add_f64_w64_gfx12, sample.a, sample.b, sample.expected);
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

// gfx1100 lanes from the MODE.IEEE-clear corpus (hw1100-noieee), plus AbsA/AbsB from a W7900
// probe: a signaling NaN source is returned unquieted, after ABS, NEG and the SUB sign flip, also
// under directed rounding and OMOD.
const CapturedGfx11 add_f16_lanes[] = {
    {"E32Mode00", 0, 0x100, 0x63377c01, 0x12680000, 0, 0xa5a57c01},
    {"E32RoundDown", 0, 0x15a, 0xaabb0000, 0xd5d77c01, 0, 0xa5a57c01},
    {"NegB", RJ_ARITH_ALU_NEG_B, 0x1f0, 0x2d450000, 0x452b7c01, 0, 0xa5a5fc01},
    {"Mul2", RJ_ARITH_ALU_OMOD_2, 0x100, 0xc9457c01, 0xba620000, 0, 0xa5a57c01},
    {"AbsA", RJ_ARITH_ALU_ABS_A, 0x1f0, 0xa5a5fc01, 0xa5a53c00, 0, 0xa5a57c01},
    {"HighHalves", RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D, 0x1f0,
     0xaf2ec100, 0xfd498000, 0, 0xfd49a5a5},
};
const CapturedGfx11 sub_f16_lanes[] = {
    {"E32Mode00", 0, 0x100, 0xbd0b7c01, 0x78430000, 0, 0xa5a57c01},
    {"E32RoundDownSubtrahend", 0, 0x15a, 0x7a990000, 0x0e867c01, 0, 0xa5a5fc01},
    {"NegB", RJ_ARITH_ALU_NEG_B, 0x1f0, 0xadab0000, 0xb5ff7c01, 0, 0xa5a57c01},
    {"Mul2", RJ_ARITH_ALU_OMOD_2, 0x100, 0x76637c01, 0xb7bd0000, 0, 0xa5a57c01},
    {"AbsB", RJ_ARITH_ALU_ABS_B, 0x1f0, 0xa5a53c00, 0xa5a5fc01, 0, 0xa5a5fc01},
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

// Compatibility and strict exact reproduce captured gfx1100 FP16 ADD/SUB/SUBREV lanes.
TEST(FpAddSubF16Gfx11, CapturedSignalingNans) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 16>(rj_arith_v_add_f16_w32, add_f16_lanes);
  check_captured_gfx11_exact<64, 16>(rj_arith_v_add_f16_w64, add_f16_lanes);
  check_captured_gfx11_exact<32, 16>(rj_arith_v_sub_f16_w32, sub_f16_lanes);
  check_captured_gfx11_exact<64, 16>(rj_arith_v_sub_f16_w64, sub_f16_lanes);
  check_captured_gfx11_exact<32, 16>(rj_arith_v_subrev_f16_w32, subrev_f16_lanes);
  check_captured_gfx11_exact<64, 16>(rj_arith_v_subrev_f16_w64, subrev_f16_lanes);
}

// Compatibility and strict exact reproduce captured gfx1100 clamped FP16 ADD/SUB/SUBREV NaN lanes.
TEST(FpAddSubF16Gfx11, CapturedClampedNans) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 16>(rj_arith_v_add_f16_w32, add_f16_clamp_lanes);
  check_captured_gfx11_exact<64, 16>(rj_arith_v_add_f16_w64, add_f16_clamp_lanes);
  check_captured_gfx11_exact<32, 16>(rj_arith_v_sub_f16_w32, sub_f16_clamp_lanes);
  check_captured_gfx11_exact<64, 16>(rj_arith_v_sub_f16_w64, sub_f16_clamp_lanes);
  check_captured_gfx11_exact<32, 16>(rj_arith_v_subrev_f16_w32, subrev_f16_clamp_lanes);
  check_captured_gfx11_exact<64, 16>(rj_arith_v_subrev_f16_w64, subrev_f16_clamp_lanes);
}

namespace {

// gfx1100 (W7900, mkm2) Track C corpora, 2026-11-18, dispatch MODE with DX10_CLAMP set and
// FP16 denormals kept. hw1100-wide-ieee1: with IEEE set the signaling NaN source is quieted
// (the IEEE-clear rows above keep it). hw1100-wide: 1.0 + 2^-14 and -1.0 + 2^-14 round away
// from 1.0 only toward +infinity; a kernel that ignores the guest rounding mode fails two rows.
const CapturedGfx11 add_f16_mode_lanes[] = {
    {"IeeeSetSignalingNanQuieted", 0, 0x3f0, 0x4b957c01, 0x173b0000, 0, 0xa5a57e01},
    {"IeeeSetSignalingPayloadQuieted", 0, 0x3f0, 0xf8ae7d55, 0x1a8f0000, 0, 0xa5a57f55},
    {"OnePlusMinNormalRoundUp", 0, 0x1f5, 0x7e583c00, 0x51e60400, 0, 0xa5a53c01},
    {"OnePlusMinNormalRoundDown", 0, 0x1fa, 0x7e583c00, 0x51e60400, 0, 0xa5a53c00},
    {"MinusOnePlusMinNormalRoundUp", 0, 0x1f5, 0x3b1ebc00, 0xa5410400, 0, 0xa5a5bbff},
    {"MinusOnePlusMinNormalRoundDown", 0, 0x1fa, 0x3b1ebc00, 0xa5410400, 0, 0xa5a5bc00},
};

// gfx1201 (RX 9070 XT, rj-valu-audit-local hw, 2026-11-05; directed-rounding rows from the
// R9700 hw-r9700-wide corpus, 2026-11-18), FP16 denormals kept. The first NaN source wins and
// is quieted (gfx12 quiets regardless of IEEE); a second-source NaN is returned when the first
// is finite; SUB negates a NaN subtrahend and SUBREV negates a NaN first source. A kernel that
// lets the host pick the NaN, or that skips quieting, fails these rows.
const CapturedGfx11 add_f16_gfx12_lanes[] = {
    {"FirstNanWins", 0, 0xf0, 0x6aed7e55, 0x3a737e00, 0, 0xa5a57e55},
    {"FirstSignalingNanQuieted", 0, 0xf0, 0x515b7d55, 0xecbf7e00, 0, 0xa5a57f55},
    {"NegativeSignalingFirstQuieted", 0, 0xf0, 0x2c07fc01, 0xf0a77e00, 0, 0xa5a5fe01},
    {"SecondNan", 0, 0xf0, 0x3b4f0400, 0xa00c7e00, 0, 0xa5a57e00},
    {"SignalingNanQuieted", 0, 0xf0, 0x4b957c01, 0x173b0000, 0, 0xa5a57e01},
    {"OnePlusMinNormalRoundUp", 0, 0xf5, 0x7e583c00, 0x51e60400, 0, 0xa5a53c01},
    {"OnePlusMinNormalRoundDown", 0, 0xfa, 0x7e583c00, 0x51e60400, 0, 0xa5a53c00},
    {"MinusOnePlusMinNormalRoundUp", 0, 0xf5, 0x3b1ebc00, 0xa5410400, 0, 0xa5a5bbff},
    {"MinusOnePlusMinNormalRoundDown", 0, 0xfa, 0x3b1ebc00, 0xa5410400, 0, 0xa5a5bc00},
    {"ClampNan", RJ_ARITH_ALU_CLAMP, 0xf0, 0x3d3a7e00, 0xa5a90400, 0, 0xa5a50000},
    {"ClampSignalingNan", RJ_ARITH_ALU_CLAMP, 0xf0, 0xc0b87c01, 0x2dac0400, 0, 0xa5a50000},
};
const CapturedGfx11 sub_f16_gfx12_lanes[] = {
    {"NanSubtrahendNegated", 0, 0xf0, 0x91520400, 0xedac7e00, 0, 0xa5a5fe00},
    {"NanSubtrahendNegatedNearOne", 0, 0xf0, 0x3e393c00, 0xe9637e00, 0, 0xa5a5fe00},
};
const CapturedGfx11 subrev_f16_gfx12_lanes[] = {
    {"NanFirstSourceNegated", 0, 0xf0, 0xf6077e00, 0xd8700400, 0, 0xa5a5fe00},
    {"SignalingFirstSourceQuietedNegated", 0, 0xf0, 0x29f07c01, 0xb5880400, 0, 0xa5a5fe01},
};

} // namespace

// Compatibility and strict exact reproduce captured gfx1100 FP16 ADD lanes with IEEE set and
// under directed rounding.
TEST(FpAddF16Gfx11, CapturedIeeeAndDirectedRounding) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 16>(rj_arith_v_add_f16_w32, add_f16_mode_lanes);
  check_captured_gfx11_exact<64, 16>(rj_arith_v_add_f16_w64, add_f16_mode_lanes);
}

// Compatibility and strict exact reproduce captured gfx1201 FP16 ADD/SUB/SUBREV lanes.
TEST(FpAddSubF16Gfx12, CapturedNanPriorityAndRounding) {
  using rocjitsu::arith::test::check_captured_gfx12_exact;
  check_captured_gfx12_exact<32, 16>(rj_arith_v_add_f16_w32_gfx12, add_f16_gfx12_lanes);
  check_captured_gfx12_exact<64, 16>(rj_arith_v_add_f16_w64_gfx12, add_f16_gfx12_lanes);
  check_captured_gfx12_exact<32, 16>(rj_arith_v_sub_f16_w32_gfx12, sub_f16_gfx12_lanes);
  check_captured_gfx12_exact<64, 16>(rj_arith_v_sub_f16_w64_gfx12, sub_f16_gfx12_lanes);
  check_captured_gfx12_exact<32, 16>(rj_arith_v_subrev_f16_w32_gfx12, subrev_f16_gfx12_lanes);
  check_captured_gfx12_exact<64, 16>(rj_arith_v_subrev_f16_w64_gfx12, subrev_f16_gfx12_lanes);
}

// Compatibility and strict exact reproduce captured gfx1100 FP64 ADD NaN lanes.
TEST(FpAddF64Gfx11, CapturedNans) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 64>(rj_arith_v_add_f64_w32, add_f64_lanes);
  check_captured_gfx11_exact<64, 64>(rj_arith_v_add_f64_w64, add_f64_lanes);
}

namespace {

// gfx1100 (W7900, mkm2) Track C hw1100-wide corpus (2026-11-18), MODE.IEEE clear, denormals
// kept: min normal + -1.0 rounds to -1.0 only to nearest and toward -inf; toward +inf and zero
// give the predecessor. A kernel that ignores guest rounding fails two of the four rows.
const CapturedGfx11 add_f64_rounding_lanes[] = {
    {"MinNormalMinusOneNearest", 0, 0x1f0, 0x0010000000000000, 0xbff0000000000000, 0,
     0xbff0000000000000},
    {"MinNormalMinusOneRoundUp", 0, 0x1f5, 0x0010000000000000, 0xbff0000000000000, 0,
     0xbfefffffffffffff},
    {"MinNormalMinusOneRoundDown", 0, 0x1fa, 0x0010000000000000, 0xbff0000000000000, 0,
     0xbff0000000000000},
    {"MinNormalMinusOneRoundZero", 0, 0x1ff, 0x0010000000000000, 0xbff0000000000000, 0,
     0xbfefffffffffffff},
};

} // namespace

// Compatibility and strict exact reproduce captured gfx1100 FP64 ADD directed rounding.
TEST(FpAddF64Gfx11, CapturedDirectedRounding) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 64>(rj_arith_v_add_f64_w32, add_f64_rounding_lanes);
  check_captured_gfx11_exact<64, 64>(rj_arith_v_add_f64_w64, add_f64_rounding_lanes);
}

// ADD/SUB/SUBREV and selection retain the inherited bridge classifiers, as does RDNA3 MUL,
// which reports nothing for zero times infinity. gfx12 MUL reports the shared hardware
// classifier: gfx1201 (R9700, 2026-11-18) raises INVALID there, and raises nothing for a
// quiet NaN times a denormal or when input flushing turns the denormal factor into zero.
TEST(FpBinaryF32, CompatibilityStatus) {
  using namespace rocjitsu::arith::test;
  struct Case {
    decltype(&rj_arith_v_add_f32_w32) wave32;
    decltype(&rj_arith_v_add_f32_w64) wave64;
    uint32_t a, b, causes;
    uint64_t mode = 0, fp = 0;
  };
  const Case cases[] = {
      {rj_arith_v_add_f32_w32_gfx12, rj_arith_v_add_f32_w64_gfx12, 0x7f7fffff, 0x7f7fffff, 0x28},
      {rj_arith_v_add_f32_w32_gfx12, rj_arith_v_add_f32_w64_gfx12, 0x7f7fffff, 0x7f7fffff, 8,
       RJ_ARITH_ALU_OMOD_HALF},
      {rj_arith_v_sub_f32_w32_gfx12, rj_arith_v_sub_f32_w64_gfx12, 0x7f800000, 0x7f800000, 1},
      {rj_arith_v_subrev_f32_w32_gfx12, rj_arith_v_subrev_f32_w64_gfx12, 0x7f800000, 0x7f800000, 1},
      {rj_arith_v_mul_f32_w32_gfx12, rj_arith_v_mul_f32_w64_gfx12, 0, 0x7f800000, 1},
      {rj_arith_v_mul_f32_w32_gfx12, rj_arith_v_mul_f32_w64_gfx12, 1, 0x7fc00000, 0},
      {rj_arith_v_mul_f32_w32, rj_arith_v_mul_f32_w64, 0, 0x7f800000, 0},
      {rj_arith_v_mul_f32_w32_gfx12, rj_arith_v_mul_f32_w64_gfx12, 0x7f7fffff, 0x40000000, 8,
       RJ_ARITH_ALU_OMOD_HALF},
      {rj_arith_v_mul_f32_w32_gfx12, rj_arith_v_mul_f32_w64_gfx12, 1, 0x3f000000, 0x32},
      {rj_arith_v_mul_f32_w32_gfx12, rj_arith_v_mul_f32_w64_gfx12, 1, 0x3f000000, 0, 0,
       RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS},
      {rj_arith_v_min_num_f32_w32, rj_arith_v_min_num_f32_w64, 0x7f800000, 1, 2},
      {rj_arith_v_maximum_f32_w32, rj_arith_v_maximum_f32_w64, 0x7f800001, 1, 3},
      {rj_arith_v_min_num_f32_w32_gfx1250, rj_arith_v_min_num_f32_w64_gfx1250, 0x7f800001, 1, 0},
  };
  for (auto cpu : supported_cpu_levels()) {
    for (int lanes : {32, 64}) {
      for (const auto &test : cases) {
        for (uint64_t exec_mask : {uint64_t(0), uint64_t(1) << (lanes - 1)}) {
          Registers<3, 64> r;
          r.v[0][lanes - 1] = test.a;
          r.v[1][lanes - 1] = test.b;
          uint32_t status = 0x80;
          const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | test.fp;
          // Alias D with A to catch classification performed after the source is overwritten.
          const int error =
              lanes == 32
                  ? test.wave32(flags, uint32_t(exec_mask), test.mode, r.v, r.v, r.v + 1, &status)
                  : test.wave64(flags, exec_mask, test.mode, r.v, r.v, r.v + 1, &status);
          ASSERT_EQ(error, RJ_ARITH_SUCCESS);
          EXPECT_EQ(status, 0x80U | (exec_mask ? test.causes : 0));
        }
      }
    }
  }
}
