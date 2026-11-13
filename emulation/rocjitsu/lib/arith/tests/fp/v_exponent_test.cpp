// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for exponent extraction and power-of-two scaling entry points.

#include "fp/captured_gfx11_test.h"
#include "fp/frexp_exp_test.h"
#include "fp/ldexp_test.h"
#include "fp/loose_unary_f64.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include "common/cpu.h"
#include <algorithm>
#include <random>

#include <cmath>
#include <gtest/gtest.h>

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI16F16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<16, 32>(rj_arith_v_frexp_exp_i16_f16_w32_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI16F16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<16, 32>(rj_arith_v_frexp_exp_i16_f16_w32);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI16F16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<16, 64>(rj_arith_v_frexp_exp_i16_f16_w64_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI16F16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<16, 64>(rj_arith_v_frexp_exp_i16_f16_w64);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<32, 32>(rj_arith_v_frexp_exp_i32_f32_w32_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<32, 32>(rj_arith_v_frexp_exp_i32_f32_w32);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<32, 64>(rj_arith_v_frexp_exp_i32_f32_w64_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<32, 64>(rj_arith_v_frexp_exp_i32_f32_w64);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<64, 32>(rj_arith_v_frexp_exp_i32_f64_w32_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<64, 32>(rj_arith_v_frexp_exp_i32_f64_w32);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<64, 64>(rj_arith_v_frexp_exp_i32_f64_w64_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<64, 64>(rj_arith_v_frexp_exp_i32_f64_w64);
}

// Check the floating-point mantissa and operand contract.
TEST(FpFrexpMantF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_frexp_mant_f64_w32_gfx12,
      [](double x) {
        int exponent;
        double value = x;
        if (std::isfinite(x) && x != 0) {
          value = std::frexp(x, &exponent);
        }
        return value;
      },
      true);
}

// Check the floating-point mantissa and operand contract.
TEST(FpFrexpMantF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_frexp_mant_f64_w32,
      [](double x) {
        int exponent;
        double value = x;
        if (std::isfinite(x) && x != 0) {
          value = std::frexp(x, &exponent);
        }
        return value;
      },
      false);
}

// Check the floating-point mantissa and operand contract.
TEST(FpFrexpMantF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_frexp_mant_f64_w64_gfx12,
      [](double x) {
        int exponent;
        double value = x;
        if (std::isfinite(x) && x != 0) {
          value = std::frexp(x, &exponent);
        }
        return value;
      },
      true);
}

// Check the floating-point mantissa and operand contract.
TEST(FpFrexpMantF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_frexp_mant_f64_w64,
      [](double x) {
        int exponent;
        double value = x;
        if (std::isfinite(x) && x != 0) {
          value = std::frexp(x, &exponent);
        }
        return value;
      },
      false);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<16, 32>(rj_arith_v_ldexp_f16_w32_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<16, 32>(rj_arith_v_ldexp_f16_w32);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<16, 64>(rj_arith_v_ldexp_f16_w64_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<16, 64>(rj_arith_v_ldexp_f16_w64);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<32, 32>(rj_arith_v_ldexp_f32_w32_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<32, 32>(rj_arith_v_ldexp_f32_w32);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<32, 64>(rj_arith_v_ldexp_f32_w64_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<32, 64>(rj_arith_v_ldexp_f32_w64);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<64, 32>(rj_arith_v_ldexp_f64_w32_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<64, 32>(rj_arith_v_ldexp_f64_w32);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<64, 64>(rj_arith_v_ldexp_f64_w64_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<64, 64>(rj_arith_v_ldexp_f64_w64);
}

// Selected gfx1201/model boundaries distinguish tininess before rounding from
// classification of the rounded result. OMOD cannot rescue a tiny or overflowing intermediate.
TEST(FpLdexp, RoundingAndOutputStages) {
  using namespace rocjitsu::arith::test;
  struct Case {
    int width;
    uint64_t input;
    int exponent;
    uint32_t instruction_flags;
    uint64_t fp;
    uint64_t output;
  };
  const Case cases[] = {
      {16, 0x0400, -1, RJ_ARITH_ALU_OMOD_2, 0, 0},
      {16, 0x7bff, 1, RJ_ARITH_ALU_OMOD_HALF, 0, 0x7c00},
      {16, 0x6bff, -26, 0, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0},
      {32, 0x00ffffff, -1, 0, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0},
      {32, 0x00ffffff, -1, RJ_ARITH_ALU_OMOD_2, 0, 0},
      {32, 0x00ffffff, -1, 0, RJ_ARITH_FP_ROUND_UP | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0},
      {64, 0x001fffffffffffffULL, -1, 0, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0},
      {64, 0x001fffffffffffffULL, -1, RJ_ARITH_ALU_OMOD_2, 0, 0},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
          RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (const auto &test : cases) {
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << test.width << '/'
                                          << test.instruction_flags << '/' << test.fp);
        Registers<5, 32> r;
        write_fp(r.v, 0, test.width, test.input);
        r.v[2][0] = uint32_t(test.exponent);
        auto call = test.width == 16   ? rj_arith_v_ldexp_f16_w32_gfx12
                    : test.width == 32 ? rj_arith_v_ldexp_f32_w32_gfx12
                                       : rj_arith_v_ldexp_f64_w32_gfx12;
        ASSERT_EQ(call(cpu | semantics | test.fp, 1, test.instruction_flags, r.v + 3, r.v, r.v + 2,
                       nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(read_fp(r.v + 3, 0, test.width), test.output);
      }
    }
  }
}

// gfx1201 FREXP components describe the same input after guest flushing.
TEST(FpFrexpF64, PairedSubnormal) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t flush : {uint64_t(0), RJ_ARITH_FP_FLUSH_INPUT_DENORMALS}) {
      Registers<5, 32> r;
      write_fp(r.v, 0, 64, 1);
      uint64_t flags = cpu | RJ_ARITH_SEMANTICS_LOOSE | flush;
      ASSERT_EQ(rj_arith_v_frexp_mant_f64_w32_gfx12(flags, 1, 0, r.v + 2, r.v, nullptr),
                RJ_ARITH_SUCCESS);
      ASSERT_EQ(rj_arith_v_frexp_exp_i32_f64_w32_gfx12(flags, 1, 0, r.v + 4, r.v, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(read_fp(r.v + 2, 0, 64), flush ? 0ULL : 0x3fe0000000000000ULL);
      EXPECT_EQ(r.v[4][0], flush ? 0U : uint32_t(-1073));
    }
  }
}

// RDNA3 output scaling is enabled only with flushed outputs and IEEE mode clear.
TEST(FpLdexpF32, Rdna3OmodGate) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (uint64_t fp : {uint64_t(0), RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE}) {
        Registers<3, 32> r;
        r.v[0][0] = 0x3f800000U;
        r.v[1][0] = 0;
        ASSERT_EQ(rj_arith_v_ldexp_f32_w32(cpu | semantics | fp, 1, RJ_ARITH_ALU_OMOD_2, r.v + 2,
                                           r.v, r.v + 1, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[2][0], fp == RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS ? 0x40000000U : 0x3f800000U);
      }
    }
  }
}

// FREXP_MANT quiets signaling NaNs without losing their sign/payload, except on gfx11 with
// IEEE mode clear, which returns them unquieted.
TEST(FpFrexpMantF64, CompatibilitySignalingNan) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
      for (bool gfx12 : {false, true}) {
        auto call = gfx12 ? rj_arith_v_frexp_mant_f64_w32_gfx12 : rj_arith_v_frexp_mant_f64_w32;
        for (uint64_t bits : {0x7ff0000000000001ULL, 0xfff4000000000001ULL}) {
          Registers<4, 32> r;
          r.v[0][0] = uint32_t(bits);
          r.v[1][0] = uint32_t(bits >> 32);
          ASSERT_EQ(
              call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | ieee, 1, 0, r.v + 2, r.v, nullptr),
              RJ_ARITH_SUCCESS);
          EXPECT_EQ(uint64_t(r.v[2][0]) | (uint64_t(r.v[3][0]) << 32),
                    gfx12 || ieee ? bits | 0x0008000000000000ULL : bits);
        }
      }
    }
  }
}

namespace {

using rocjitsu::arith::test::CapturedGfx11;

// gfx1100 LDEXP lanes from the MODE.IEEE-clear corpus (hw1100-noieee): a signaling NaN source
// is returned unquieted, also under directed rounding and OMOD.
const CapturedGfx11 ldexp_f16_lanes[] = {
    {"E32Mode00", 0, 0x100, 0xcb4d7c01, 0x80fd0000, 0, 0xa5a57c01},
    {"E32RoundDown", 0, 0x15a, 0xcb4d7c01, 0x80fd0000, 0, 0xa5a57c01},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0x1f0, 0x24f27c01, 0x151c0000, 0, 0xa5a5fc01},
    {"Div2", RJ_ARITH_ALU_OMOD_HALF, 0x100, 0x5aa27c01, 0xa6020000, 0, 0xa5a57c01},
    {"HighHalves", RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D, 0x1f0,
     0xfcd07c01, 0xe94fffff, 0, 0xfcd0a5a5},
};
const CapturedGfx11 ldexp_f32_lanes[] = {
    {"Mode00", 0, 0x100, 0x7f800001, 0x0, 0, 0x7f800001},
    {"RoundDown", 0, 0x15a, 0x7f800001, 0x1, 0, 0x7f800001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0x1f0, 0x7f800001, 0x0, 0, 0xff800001},
    {"Mul2", RJ_ARITH_ALU_OMOD_2, 0x100, 0x7f800001, 0x0, 0, 0x7f800001},
};
// FREXP_MANT returns a signaling NaN source unquieted in the same corpus.
const CapturedGfx11 frexp_mant_f64_lanes[] = {
    {"E32Mode00", 0, 0x100, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"E32RoundDown", 0, 0x15a, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0x1f0, 0x7ff0000000000001, 0, 0, 0xfff0000000000001},
    {"Mul4", RJ_ARITH_ALU_OMOD_4, 0x100, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"NegativePayload", 0, 0x1f0, 0xfff027eaf1f97f65, 0, 0, 0xfff027eaf1f97f65},
};
const CapturedGfx11 ldexp_f64_lanes[] = {
    {"Mode00", 0, 0x100, 0x7ff0000000000001, 0x0, 0, 0x7ff0000000000001},
    {"RoundDown", 0, 0x15a, 0x7ff0000000000001, 0x1, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0x1f0, 0x7ff0000000000001, 0x0, 0, 0xfff0000000000001},
    {"Mul2", RJ_ARITH_ALU_OMOD_2, 0x100, 0x7ff0000000000001, 0x0, 0, 0x7ff0000000000001},
};

} // namespace

// Compatibility reproduces captured gfx1100 LDEXP lanes.
TEST(FpLdexpGfx11, CapturedSignalingNans) {
  using rocjitsu::arith::test::check_captured_gfx11;
  check_captured_gfx11<32, 16>(rj_arith_v_ldexp_f16_w32, ldexp_f16_lanes);
  check_captured_gfx11<64, 16>(rj_arith_v_ldexp_f16_w64, ldexp_f16_lanes);
  check_captured_gfx11<32, 32>(rj_arith_v_ldexp_f32_w32, ldexp_f32_lanes);
  check_captured_gfx11<64, 32>(rj_arith_v_ldexp_f32_w64, ldexp_f32_lanes);
  check_captured_gfx11<32, 64>(rj_arith_v_ldexp_f64_w32, ldexp_f64_lanes);
  check_captured_gfx11<64, 64>(rj_arith_v_ldexp_f64_w64, ldexp_f64_lanes);
}

// Compatibility reproduces captured gfx1100 FREXP_MANT lanes.
TEST(FpFrexpMantF64Gfx11, CapturedSignalingNans) {
  using rocjitsu::arith::test::check_captured_gfx11;
  check_captured_gfx11<32, 64>(rj_arith_v_frexp_mant_f64_w32, frexp_mant_f64_lanes);
  check_captured_gfx11<64, 64>(rj_arith_v_frexp_mant_f64_w64, frexp_mant_f64_lanes);
}

// SIMD must retain the inherited scalar model across guest modes and exceptional inputs.
TEST(FpLdexpF16, SimdMatchesInheritedRounding) {
  const auto levels = rocjitsu::arith::test::supported_cpu_levels();
  std::minstd_rand random;
  const uint16_t special[] = {0, 0x8000, 1, 0x83ff, 0x7bff, 0x7c00, 0xfc00, 0x7d23};
  for (auto call : {rj_arith_v_ldexp_f16_w32, rj_arith_v_ldexp_f16_w32_gfx12}) {
    for (unsigned rounding = 0; rounding < 4; ++rounding) {
      for (unsigned control = 0; control < 16; ++control) {
        const uint64_t flags =
            RJ_ARITH_SEMANTICS_COMPATIBILITY | (uint64_t(rounding) << 24) |
            (control & 1 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0) |
            (control & 2 ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0) |
            (control & 4 ? RJ_ARITH_FP16_OVFL : 0) |
            (control & 8 && call == rj_arith_v_ldexp_f16_w32 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0);
        for (unsigned variant = 0; variant < 8; ++variant) {
          const uint32_t modifiers =
              (variant % 4) * RJ_ARITH_ALU_OMOD_2 |
              (variant & 4 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D |
                                 RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_CLAMP
                           : 0);
          for (int batch = 0; batch < 16; ++batch) {
            uint32_t a[32], b[32], reference[32], actual[32];
            for (int lane = 0; lane < 32; ++lane) {
              uint16_t half = lane < 8 ? special[lane] : uint16_t(random() >> 15);
              uint16_t exponent = batch == 0   ? 0x8000
                                  : batch == 1 ? 0x7fff
                                               : uint16_t(int(random() >> 25) - 32);
              a[lane] = uint32_t(half) * 0x10001U;
              b[lane] = uint32_t(exponent) * 0x10001U;
              reference[lane] = actual[lane] = a[lane];
            }
            const uint32_t *x = a, *y = b;
            uint32_t *d = reference, status = 0x80;
            const uint32_t exec_mask = batch & 1 ? 0xa5555555U : 0xffffffffU;
            ASSERT_EQ(call(flags, exec_mask, modifiers, &d, &x, &y, &status), RJ_ARITH_SUCCESS);
            for (uint64_t cpu : levels) {
              std::copy(a, a + 32, actual);
              d = actual;
              x = actual; // Selected input aliases D, with the other half preserved.
              ASSERT_EQ(call(flags | cpu, exec_mask, modifiers, &d, &x, &y, &status),
                        RJ_ARITH_SUCCESS);
              for (int lane = 0; lane < 32; ++lane) {
                ASSERT_EQ(actual[lane], reference[lane])
                    << cpu << '/' << rounding << '/' << control << '/' << variant << '/' << batch
                    << '/' << lane;
              }
              EXPECT_EQ(status, 0x80U);
            }
          }
        }
      }
    }
  }
}
