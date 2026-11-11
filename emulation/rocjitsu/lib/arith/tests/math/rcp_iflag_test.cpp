// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for reciprocal-with-flags numeric results and FP-status behavior.

#include "math/unary_test.h"
#include "rj_arith/detail/rj_arith_math.h"

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRcpIflagF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_rcp_iflag_f32_w32, 5, true);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRcpIflagF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_rcp_iflag_f32_w32_gfx12, 5, true);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRcpIflagF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_rcp_iflag_f32_w64, 5, true);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRcpIflagF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_rcp_iflag_f32_w64_gfx12, 5, true);
}

// gfx1100 IEEE=0 capture (hw1100-noieee): RCP_IFLAG OMOD applies under every captured MODE,
// including kept F32 output denormals (0xf0, 0xa0, 0xff), with identical results.
TEST(MathRcpIflagF32Gfx11, CapturedOmodWithIeeeClear) {
  using namespace rocjitsu::arith::test;
  const uint32_t inputs[] = {0x00800000, 0x3f800000, 0xff800000, 0x3fc00000};
  const struct {
    uint64_t omod;
    uint32_t expected[4];
  } cases[] = {{RJ_ARITH_ALU_OMOD_2, {0x7f000000, 0x40000000, 0, 0x3faaaaaa}},
               {RJ_ARITH_ALU_OMOD_4, {0x7f800000, 0x40800000, 0, 0x402aaaaa}},
               {RJ_ARITH_ALU_OMOD_HALF, {0x7e000000, 0x3f000000, 0, 0x3eaaaaaa}}};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (uint32_t m : {0xf0u, 0x00u, 0xffu, 0x50u, 0xa0u}) {
        for (const auto &sample : cases) {
          for (bool wide : {false, true}) {
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << std::hex << m
                                              << '/' << sample.omod << '/' << wide);
            uint32_t a[64], d[64], status = 0;
            for (int lane = 0; lane < 64; ++lane) {
              a[lane] = inputs[lane % 4];
            }
            uint32_t *dest[] = {d};
            const uint32_t *source[] = {a};
            // Translate the F32 denormal fields (the entry point takes no rounding control);
            // the capture leaves DX10_CLAMP and IEEE clear.
            const uint64_t flags = cpu | semantics | RJ_ARITH_FP_DX10_CLAMP_OFF |
                                   (m & 0x10 ? 0 : RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) |
                                   (m & 0x20 ? 0 : RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
            ASSERT_EQ(
                wide
                    ? rj_arith_v_rcp_iflag_f32_w64(flags, ~0ULL, sample.omod, dest, source, &status)
                    : rj_arith_v_rcp_iflag_f32_w32(flags, ~0U, sample.omod, dest, source, &status),
                RJ_ARITH_SUCCESS);
            for (int lane = 0; lane < (wide ? 64 : 32); ++lane) {
              EXPECT_EQ(d[lane], sample.expected[lane % 4]) << lane;
            }
          }
        }
      }
    }
  }
}
