// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP16/FP32 interpolation stages and quad-lane operand routing.

#include "graphics/helpers.h"

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p10_f32_w32, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Interp32, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p2_f32_w32, 1,
                                            rocjitsu::arith::test::GraphicsFamily::Interp32, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10F16F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p10_f16_f32_w32, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2F16F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p2_f16_f32_w32, 1,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10RtzF16F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p10_rtz_f16_f32_w32, 2,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2RtzF16F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p2_rtz_f16_f32_w32, 3,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p10_f32_w64, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Interp32, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p2_f32_w64, 1,
                                            rocjitsu::arith::test::GraphicsFamily::Interp32, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10F16F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p10_f16_f32_w64, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2F16F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p2_f16_f32_w64, 1,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10RtzF16F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p10_rtz_f16_f32_w64, 2,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2RtzF16F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p2_rtz_f16_f32_w64, 3,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

namespace {

// gfx1201 results at MODE 0xf0 (denormals kept, nearest even), where Compatibility no longer
// takes the loose host FMA. Operands are the ones lane 0 reads: A[1] (P10) or A[2] (P2), B[0]
// and C[0]. An invalid product returns 0xffc00000 even when C is a NaN; otherwise the first NaN
// of A, B, C is returned quieted, after NEG, and an FP16 NaN keeps its payload when widened.
struct NanCase {
  int op;
  uint32_t modifiers, a, b, c, expected;
};

const NanCase nan_cases[] = {
    {0, 0, 0x7f800000, 0, 0x7fc00000, 0xffc00000},
    {0, 0, 0x7f800000, 0, 0x7f800001, 0xffc00000},
    {0, RJ_ARITH_ALU_NEG_A, 0x7fc00000, 0x7fc00000, 0, 0xffc00000},
    {0, RJ_ARITH_ALU_NEG_B, 0x7fc00000, 0x7fc00000, 0, 0x7fc00000},
    {1, 0, 0x7f800001, 0x7fc00000, 0, 0x7fc00001},
    {1, RJ_ARITH_ALU_NEG_A, 0x7f800001, 0x7fc00000, 0, 0xffc00001},
    {1, RJ_ARITH_ALU_NEG_C, 0xff800000, 0, 0x7f800001, 0xffc00000},
    {2, 0, 0x83ff7c00, 0, 0xb8007c01, 0xffc00000},
    {2, RJ_ARITH_ALU_HIGH_A, 0x7e553e00, 0x7fc00000, 0x80000000, 0x7fcaa000},
    {3, RJ_ARITH_ALU_HIGH_A, 0x7e553e00, 0x7fc00000, 0x80000000, 0x7fcaa000},
    {3, RJ_ARITH_ALU_NEG_C, 0xcae0fde7, 0x7faac328, 0x82a84d0c, 0xfffce000},
};

} // namespace

// Check Compatibility NaN results against the hardware without guest FP controls.
TEST(GraphicsInterpW32, CompatibilityNanMatchesGfx1201) {
  decltype(&rj_arith_v_interp_p10_f32_w32) calls[] = {
      rj_arith_v_interp_p10_f32_w32, rj_arith_v_interp_p2_f32_w32,
      rj_arith_v_interp_p10_f16_f32_w32, rj_arith_v_interp_p10_rtz_f16_f32_w32};
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (const NanCase &test : nan_cases) {
      SCOPED_TRACE(::testing::Message() << cpu << '/' << (&test - nan_cases));
      const bool p2 = test.op == 1;
      rocjitsu::arith::test::Registers<4, 32> r{};
      r.v[0][p2 ? 2 : 1] = test.a;
      r.v[1][0] = test.b;
      r.v[2][0] = test.c;
      ASSERT_EQ(calls[test.op](cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, 1, test.modifiers, r.v + 3,
                               r.v, r.v + 1, r.v + 2),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][0], test.expected);
    }
  }
}
