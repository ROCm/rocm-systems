// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited legacy multiply: selected rounding witnesses and scalar/SIMD agreement.
#include "common/cpu.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <type_traits>

namespace {

// Exercise every backend with the same guest controls, aliases and sparse EXEC.
// Golden witnesses distinguish post-round flushing and host-style NaN quieting.
template <int Lanes, bool Older, typename Call> void check(Call call) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  const auto cpus = rocjitsu::arith::test::supported_cpu_levels();
  const uint64_t semantics[] = {RJ_ARITH_SEMANTICS_LOOSE,
                                RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                                RJ_ARITH_SEMANTICS_COMPATIBILITY};
  std::minstd_rand random;
  uint32_t a[Lanes], b[Lanes], expected[Lanes], actual[Lanes];
  const uint32_t *ap = a, *bp = b;
  for (int batch = 0; batch < 12; ++batch) {
    for (int lane = 0; lane < Lanes; ++lane) {
      a[lane] = (uint32_t(random()) << 1) | (uint32_t(random()) >> 30);
      b[lane] = (uint32_t(random()) << 1) | (uint32_t(random()) >> 30);
    }
    a[0] = 0;
    b[0] = 0x7f800000U;
    a[1] = 0x7f812345U;
    b[1] = 0x3f800000U;
    a[2] = 0x00800000U;
    b[2] = 0x3f7fffffU;
    a[3] = 0x7f7fffffU;
    b[3] = 0x40000000U;
    a[4] = 1;
    b[4] = 0x3f000000U;
    a[5] = 0x80000001U;
    b[5] = 0x3f000000U;
    a[6] = 0x3f800000U;
    b[6] = 0x3f800000U;
    for (unsigned rounding = 0; rounding < 4; ++rounding) {
      for (uint64_t fp :
           {uint64_t(0), RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
            RJ_ARITH_FP_IEEE_MODE, RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
            uint64_t(Older ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0)}) {
        for (uint32_t mode :
             {0U, uint32_t(RJ_ARITH_ALU_OMOD_HALF), uint32_t(RJ_ARITH_ALU_OMOD_2),
              uint32_t(RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A)}) {
          const uint64_t flags = RJ_ARITH_SEMANTICS_COMPATIBILITY | fp | (uint64_t(rounding) << 24);
          uint32_t *out = expected;
          ASSERT_EQ(call(flags, Mask(-1), mode, &out, &ap, &bp, nullptr), RJ_ARITH_SUCCESS);
          if (!rounding && !fp && !mode) {
            EXPECT_EQ(expected[0], 0U);
            EXPECT_EQ(expected[1], 0x7fc12345U);
            EXPECT_EQ(expected[2], 0x00800000U);
            EXPECT_EQ(expected[3], 0x7f800000U);
            EXPECT_EQ(expected[4], 0U);
            EXPECT_EQ(expected[5], 0x80000000U);
          }
          if (!rounding && fp == RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS && !mode) {
            EXPECT_EQ(expected[2], 0x00800000U);
          }
          if (mode == RJ_ARITH_ALU_OMOD_2) {
            const bool suppressed = Older && (!(fp & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) ||
                                              (fp & RJ_ARITH_FP_IEEE_MODE));
            EXPECT_EQ(expected[6], suppressed ? 0x3f800000U : 0x40000000U);
          }
          for (uint64_t cpu : cpus) {
            for (Mask exec_mask : {Mask(-1), Mask(0x8000000180000001ULL)}) {
              SCOPED_TRACE(::testing::Message()
                           << batch << '/' << rounding << '/' << fp << '/' << mode << '/' << cpu);
              for (int lane = 0; lane < Lanes; ++lane) {
                actual[lane] = a[lane];
              }
              out = actual;
              const uint32_t *alias = actual;
              uint32_t status = 0x12345678U;
              ASSERT_EQ(call((flags & ~RJ_ARITH_SEMANTICS_MASK) | cpu | semantics[batch % 3],
                             exec_mask, mode, &out, &alias, &bp, &status),
                        RJ_ARITH_SUCCESS);
              EXPECT_EQ(status, 0x12345678U);
              for (int lane = 0; lane < Lanes; ++lane) {
                EXPECT_EQ(actual[lane], (exec_mask >> lane) & 1 ? expected[lane] : a[lane]) << lane;
              }
            }
          }
        }
      }
    }
  }
  uint32_t *out = actual;
  for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                             RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
    EXPECT_EQ(call(semantics, Mask(-1), 0, &out, &ap, &bp, nullptr),
              RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
  }
}

} // namespace

// Cover both wave sizes of the shared gfx942/gfx950 policy.
TEST(LegacyMul, Cdna) {
  check<32, true>(rj_arith_v_mul_legacy_f32_w32);
  check<64, true>(rj_arith_v_mul_legacy_f32_w64);
}

// Cover gfx1250's unconditional output-modifier policy.
TEST(LegacyMul, Gfx1250) {
  check<32, false>(rj_arith_v_mul_dx9_zero_f32_w32_gfx1250);
  check<64, false>(rj_arith_v_mul_dx9_zero_f32_w64_gfx1250);
}
