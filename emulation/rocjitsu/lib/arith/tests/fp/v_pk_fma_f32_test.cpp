// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed FP32 FMA word selection, aliasing, guest rounding and inactive lanes.

#include "common/cpu.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

// Both words reuse exactly representable inputs except the half-ULP addend.
// The expected values expose independent selectors, negations and directed rounding.
template <int Lanes, typename Function> void check(Function function, bool old_cdna) {
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (int rounding = 0; rounding < 4; ++rounding) {
        for (bool alias : {false, true}) {
          uint32_t storage[8][Lanes];
          for (int lane = 0; lane < Lanes; ++lane) {
            storage[0][lane] = 0x3f800000U;
            storage[1][lane] = 0x40000000U;
            storage[2][lane] = 0x3f800000U;
            storage[3][lane] = 0x3f000000U;
            storage[4][lane] = 0x33800000U;
            storage[5][lane] = 0x3f800000U;
            storage[6][lane] = storage[7][lane] = 0xdeadbeefU;
          }
          const uint32_t *a[] = {storage[0], storage[1]};
          const uint32_t *b[] = {storage[2], storage[3]};
          const uint32_t *c[] = {storage[4], storage[5]};
          uint32_t *d[] = {storage[alias ? 0 : 6], storage[alias ? 1 : 7]};
          const uint64_t exec_mask = 1ULL | (1ULL << (Lanes - 1));
          // The upper result reads old A.low after the lower result would overwrite it.
          const uint32_t mode = RJ_ARITH_PK_HI_A_LOW | RJ_ARITH_PK_NEG_HI_C;
          const uint64_t flags = cpu | semantics | (uint64_t(rounding) << 24);
          uint32_t status = 0x12345678U;
          ASSERT_EQ(function(flags, exec_mask, mode, d, a, b, c, &status), RJ_ARITH_SUCCESS);
          EXPECT_EQ(status, 0x12345678U);
          for (int lane = 0; lane < Lanes; ++lane) {
            if ((exec_mask >> lane) & 1) {
              const uint32_t low = semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY && rounding == 1
                                       ? 0x3f800001U
                                       : 0x3f800000U;
              EXPECT_EQ(d[0][lane], low);
              EXPECT_EQ(d[1][lane], 0xbf000000U);
            } else {
              EXPECT_EQ(d[0][lane], alias ? 0x3f800000U : 0xdeadbeefU);
              EXPECT_EQ(d[1][lane], alias ? 0x40000000U : 0xdeadbeefU);
            }
          }
        }
      }
    }
  }
  // Inherited FMA policy: old CDNA preserves signaling payloads unless IEEE is set;
  // gfx1250 always quiets. A zero-times-infinity product overrides a NaN addend.
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (uint64_t ieee : {uint64_t(0), uint64_t(RJ_ARITH_FP_IEEE_MODE)}) {
      uint32_t a_words[2][Lanes] = {}, b_words[2][Lanes] = {}, c_words[2][Lanes] = {};
      uint32_t result[2][Lanes] = {};
      a_words[0][0] = 0x7f800123U;
      b_words[0][0] = 0x3f800000U;
      b_words[1][0] = 0x7f800000U;
      c_words[1][0] = 0x7fc00456U;
      const uint32_t *a[] = {a_words[0], a_words[1]};
      const uint32_t *b[] = {b_words[0], b_words[1]};
      const uint32_t *c[] = {c_words[0], c_words[1]};
      uint32_t *d[] = {result[0], result[1]};
      ASSERT_EQ(function(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | ieee, 1, 0, d, a, b, c, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(result[0][0], old_cdna && !ieee ? 0x7f800123U : 0x7fc00123U);
      EXPECT_EQ(result[1][0], 0xffc00000U);
    }
  }
  EXPECT_EQ(
      function(RJ_ARITH_SEMANTICS_COMPATIBILITY, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr),
      RJ_ARITH_SUCCESS);
  EXPECT_EQ(function(RJ_ARITH_SEMANTICS_COMPATIBILITY, 0, 1ULL << 32, nullptr, nullptr, nullptr,
                     nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
  for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                             RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
    EXPECT_EQ(function(semantics, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr),
              RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
  }
}

} // namespace

TEST(PackedFmaF32, Gfx942Wave32) { check<32>(rj_arith_v_pk_fma_f32_w32_gfx942, true); }
TEST(PackedFmaF32, Gfx942Wave64) { check<64>(rj_arith_v_pk_fma_f32_w64_gfx942, true); }
TEST(PackedFmaF32, Gfx1250Wave32) { check<32>(rj_arith_v_pk_fma_f32_w32_gfx1250, false); }
