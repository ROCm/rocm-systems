// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for lit-multiply special values and gfx11/gfx12 modifier behavior.

#include "fp/captured_gfx11_test.h"
#include "graphics/helpers.h"
#include "rj_arith/detail/rj_arith_graphics.h"

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsMullitF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_mullit_f32_w32, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Mullit, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsMullitF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_mullit_f32_w32_gfx12, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Mullit, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsMullitF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_mullit_f32_w64, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Mullit, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsMullitF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_mullit_f32_w64_gfx12, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Mullit, true);
}

// gfx1250 keeps the gfx12 model but refuses strict exact until its captures are reviewed.
TEST(GraphicsMullitF32W32Gfx1250, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_mullit_f32_w32_gfx1250, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Mullit, true,
                                            /*loose_only=*/true);
}

// gfx1250 keeps the gfx12 model but refuses strict exact until its captures are reviewed.
TEST(GraphicsMullitF32W64Gfx1250, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_mullit_f32_w64_gfx1250, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Mullit, true,
                                            /*loose_only=*/true);
}

// Selected MODE witnesses from rocjitsu's MULLIT regressions (#13669), in Compatibility and
// strict exact.
TEST(GraphicsMullit, GuestMode) {
  using namespace rocjitsu::arith::test;
  constexpr struct {
    uint32_t a, b, c, expected;
    uint64_t fp;
    uint32_t modifiers = 0;
  } cases[] = {
      {0x2d1d4000, 0x49032148, 0x44630000, 0x36a11862, RJ_ARITH_FP_ROUND_ZERO},
      {0x00800000, 0x3f7fffff, 0x3f800000, 0, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS},
      {0x3f800000, 0x3f800000, 1, 0xff7fffff, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS},
      {0x7f80002a, 0, 0x3f800000, 0, 0},
      // gfx1100 observations: flush before scaling, retaining negative underflow zero.
      {0x00c00000, 0x3f000000, 0x3f800000, 0, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
       RJ_ARITH_ALU_OMOD_2},
      {0x80800000, 0x3f800000, 0x3f800000, 0x80000000, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
       RJ_ARITH_ALU_OMOD_HALF},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : hardware_semantics()) {
      for (auto call : {rj_arith_v_mullit_f32_w32, rj_arith_v_mullit_f32_w32_gfx12}) {
        for (const auto &sample : cases) {
          Registers<4, 32> r;
          r.v[0][0] = sample.a;
          r.v[1][0] = sample.b;
          r.v[2][0] = sample.c;
          uint32_t status = 0x80000000U;
          ASSERT_EQ(call(cpu | semantics | sample.fp, 1, sample.modifiers, r.v + 3, r.v, r.v + 1,
                         r.v + 2, &status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(r.v[3][0], sample.expected);
          EXPECT_EQ(status, 0x80000000U);
        }
      }
    }
  }
}

namespace {

using rocjitsu::arith::test::CapturedGfx11;

// gfx1201 (RX 9070 XT, rj-valu-audit-local hw, 2026-11-05), dispatch MODE, FP32 denormals
// kept unless noted. A NaN or -infinity second source, or a nonpositive third source, gives
// -MAX_FLOAT; a zero factor gives +0. 2^-126 * 2^-126 rounds to the smallest denormal only
// toward +infinity. OMOD flushes a tiny product before scaling (+0, keeping the sign only for a
// value that became tiny by scaling), even with denormals kept. A kernel that lets the host
// round the product, or that scales before flushing, fails these rows.
const CapturedGfx11 mullit_gfx12_lanes[] = {
    {"NanSecondSourceGivesNegMax", 0, 0xf0, 0x0, 0x7fc00000, 0x0, 0xff7fffff},
    {"NegativeInfinitySecondSourceGivesNegMax", 0, 0xf0, 0x80000000, 0xff800000, 0x0, 0xff7fffff},
    {"NegativeZeroThirdSourceGivesNegMax", 0, 0xf0, 0x800000, 0x800000, 0x80000000, 0xff7fffff},
    {"ZeroFactor", 0, 0xf0, 0x80000000, 0x800000, 0x3f800000, 0x0},
    {"TinyProductRoundUp", 0, 0xf5, 0x800000, 0x800000, 0x3f800000, 0x1},
    {"TinyProductRoundDown", 0, 0xfa, 0x800000, 0x800000, 0x3f800000, 0x0},
    {"TinyProductNearest", 0, 0xf0, 0x800000, 0x800000, 0x3f800000, 0x0},
    {"MinNormalHalvedFlushes", RJ_ARITH_ALU_OMOD_HALF, 0xf0, 0x3f800000, 0x800000, 0x3f800000, 0x0},
    {"NegativeMinNormalHalvedKeepsSign", RJ_ARITH_ALU_OMOD_HALF, 0xf0, 0xbf800000, 0x800000,
     0x3f800000, 0x80000000},
    {"DenormalProductHalvedFlushes", RJ_ARITH_ALU_OMOD_HALF, 0xf0, 0x3f000000, 0x800000, 0x3f800000,
     0x0},
    {"DenormalDoubledFlushes", RJ_ARITH_ALU_OMOD_2, 0xf0, 0x1, 0x800000, 0x3f800000, 0x0},
    {"NegMaxClamped", RJ_ARITH_ALU_CLAMP, 0xf0, 0x7fc00000, 0x3f800000, 0x0, 0x0},
};

// gfx1100 (W7900, mkm2) rj-xarch-1100 hw1100-noieee corpus, 2026-11-12, dispatch MODE with
// DX10_CLAMP set and IEEE clear. Directed rounding matches gfx1201. gfx11 applies OMOD only
// with FP32 output denormals flushed: with denormals kept, 0.5 * 2^-126 div:2 returns the
// unscaled denormal product; with output flushing, 2^-126 div:2 flushes to +0 and -2^-126 div:2
// to -0. A kernel that applies OMOD regardless of the denormal mode fails the kept-denormal row.
const CapturedGfx11 mullit_gfx11_lanes[] = {
    {"TinyProductRoundUp", 0, 0x1f5, 0x800000, 0x800000, 0x3f800000, 0x1},
    {"TinyProductRoundDown", 0, 0x1fa, 0x800000, 0x800000, 0x3f800000, 0x0},
    {"HalvingIgnoredWithDenormalsKept", RJ_ARITH_ALU_OMOD_HALF, 0x1f0, 0x3f000000, 0x800000, 0x1,
     0x400000},
    {"MinNormalHalvedFlushes", RJ_ARITH_ALU_OMOD_HALF, 0x150, 0x3f800000, 0x800000, 0x3f800000,
     0x0},
    {"NegativeMinNormalHalvedKeepsSign", RJ_ARITH_ALU_OMOD_HALF, 0x150, 0xbf800000, 0x800000,
     0x3f800000, 0x80000000},
    {"DenormalDoubledWithOutputFlushGivesNegMax", RJ_ARITH_ALU_OMOD_2, 0x150, 0x1, 0x800000, 0x0,
     0xff800000},
    {"NegMaxClamped", RJ_ARITH_ALU_CLAMP, 0x1f0, 0x7fc00000, 0x3f800000, 0x0, 0x0},
};

} // namespace

// Compatibility and strict exact reproduce captured gfx1201 MULLIT lanes.
TEST(GraphicsMullitGfx12, CapturedSpecialsRoundingAndOmod) {
  using rocjitsu::arith::test::check_captured_gfx12_exact;
  check_captured_gfx12_exact<32, 32>(rj_arith_v_mullit_f32_w32_gfx12, mullit_gfx12_lanes);
  check_captured_gfx12_exact<64, 32>(rj_arith_v_mullit_f32_w64_gfx12, mullit_gfx12_lanes);
}

// Compatibility and strict exact reproduce captured gfx1100 MULLIT lanes.
TEST(GraphicsMullitGfx11, CapturedRoundingAndOmodGate) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 32>(rj_arith_v_mullit_f32_w32, mullit_gfx11_lanes);
  check_captured_gfx11_exact<64, 32>(rj_arith_v_mullit_f32_w64, mullit_gfx11_lanes);
}
