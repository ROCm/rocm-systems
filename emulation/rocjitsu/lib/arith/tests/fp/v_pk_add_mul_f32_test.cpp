// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed FP32 binary arithmetic: guest rounding, post-round flushing, and aliases.

#include "common/cpu.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

// The product just below minimum normal distinguishes post-round output flushing
// from FMA's intermediate-tininess rule. Other words exercise ordinary rounding.
template <int Lanes, bool Multiply, typename Function> void check(Function function) {
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    // Loose and fallback-permitted modes retain selectors/signs while allowing
    // native rounding. Use exactly representable values to isolate that contract.
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE}) {
      uint32_t a_words[2][Lanes] = {}, b_words[2][Lanes] = {}, output[2][Lanes] = {};
      a_words[0][0] = 0x40000000U;
      a_words[1][0] = 0x40400000U;
      b_words[0][0] = 0x40800000U;
      b_words[1][0] = 0x40a00000U;
      const uint32_t *a[] = {a_words[0], a_words[1]};
      const uint32_t *b[] = {b_words[0], b_words[1]};
      uint32_t *d[] = {output[0], output[1]};
      const uint32_t mode = RJ_ARITH_PK_LO_A_HIGH | RJ_ARITH_PK_NEG_LO_B | RJ_ARITH_PK_HI_B_LOW;
      ASSERT_EQ(function(cpu | semantics, 1, mode, d, a, b, nullptr), RJ_ARITH_SUCCESS);
      EXPECT_EQ(output[0][0], Multiply ? 0xc1400000U : 0xbf800000U);
      EXPECT_EQ(output[1][0], Multiply ? 0x41400000U : 0x40e00000U);
    }
    for (int rounding = 0; rounding < 4; ++rounding) {
      for (bool flush : {false, true}) {
        uint32_t values[4][Lanes];
        for (int lane = 0; lane < Lanes; ++lane) {
          values[0][lane] = Multiply ? 0x3f800001U : 0x3f800000U;
          values[1][lane] = 0x00800000U;
          values[2][lane] = Multiply ? 0x3f800001U : 0x33800000U;
          values[3][lane] = Multiply ? 0x3f7fffffU : 0x80000001U;
        }
        const uint32_t *a[] = {values[0], values[1]};
        const uint32_t *b[] = {values[2], values[3]};
        uint32_t *d[] = {values[0], values[1]};
        const uint64_t exec_mask = 1ULL | (1ULL << (Lanes - 1));
        const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | (uint64_t(rounding) << 24) |
                               (flush ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0);
        uint32_t status = 123;
        ASSERT_EQ(function(flags, exec_mask, 0, d, a, b, &status), RJ_ARITH_SUCCESS);
        EXPECT_EQ(status, 123U);
        uint32_t high = Multiply && rounding <= 1 ? 0x00800000U : 0x007fffffU;
        if (flush && high < 0x00800000U) {
          high = 0;
        }
        for (int lane = 0; lane < Lanes; ++lane) {
          if ((exec_mask >> lane) & 1) {
            EXPECT_EQ(d[0][lane], (Multiply ? 0x3f800002U : 0x3f800000U) + (rounding == 1));
            EXPECT_EQ(d[1][lane], high);
          } else {
            EXPECT_EQ(d[0][lane], Multiply ? 0x3f800001U : 0x3f800000U);
            EXPECT_EQ(d[1][lane], 0x00800000U);
          }
        }
      }
    }
  }
  EXPECT_EQ(function(RJ_ARITH_SEMANTICS_COMPATIBILITY, 0, 0, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(function(RJ_ARITH_SEMANTICS_COMPATIBILITY, 0, RJ_ARITH_PK_NEG_LO_C, nullptr, nullptr,
                     nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}

} // namespace

TEST(PackedAddMulF32, AddGfx942Wave32) { check<32, false>(rj_arith_v_pk_add_f32_w32_gfx942); }
TEST(PackedAddMulF32, AddGfx942Wave64) { check<64, false>(rj_arith_v_pk_add_f32_w64_gfx942); }
TEST(PackedAddMulF32, AddGfx1250Wave32) { check<32, false>(rj_arith_v_pk_add_f32_w32_gfx1250); }
TEST(PackedAddMulF32, MulGfx942Wave32) { check<32, true>(rj_arith_v_pk_mul_f32_w32_gfx942); }
TEST(PackedAddMulF32, MulGfx942Wave64) { check<64, true>(rj_arith_v_pk_mul_f32_w64_gfx942); }
TEST(PackedAddMulF32, MulGfx1250Wave32) { check<32, true>(rj_arith_v_pk_mul_f32_w32_gfx1250); }
