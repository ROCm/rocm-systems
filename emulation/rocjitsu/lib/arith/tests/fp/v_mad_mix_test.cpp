// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Mixed MAD's FP32 intermediate, selected-half rounding, masks and aliases.
#include "common/cpu.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <type_traits>

namespace {

// Compare SIMD against inherited scalar arithmetic using deterministic raw inputs.
// Selected goldens separately pin the intermediate precision and half rounding.
template <int Lanes, int Half, typename Call> void check(Call call) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  const auto cpus = rocjitsu::arith::test::supported_cpu_levels();
  std::minstd_rand random;
  uint32_t a[Lanes], b[Lanes], c[Lanes], expected[Lanes], actual[Lanes];
  const uint32_t *ap = a, *bp = b, *cp = c;
  const uint64_t semantics[] = {RJ_ARITH_SEMANTICS_LOOSE,
                                RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                                RJ_ARITH_SEMANTICS_COMPATIBILITY};
  for (int batch = 0; batch < 8; ++batch) {
    for (int lane = 0; lane < Lanes; ++lane) {
      a[lane] = uint32_t(random()) << 1;
      b[lane] = uint32_t(random()) << 1;
      c[lane] = uint32_t(random()) << 1;
    }
    // 1 + 2^-11 is exactly halfway between neighboring half numbers.
    a[0] = 0x3f800000U;
    b[0] = 0x3f800000U;
    c[0] = 0x3a000000U;
    a[1] = 0x7fc12345U;
    b[1] = 0x3f800000U;
    c[1] = 0;
    a[2] = 0;
    b[2] = 0x7f800000U;
    c[2] = 0;
    // Fused cancellation retains a residual lost by an FP32 product round.
    a[3] = 0x47000400;
    b[3] = 0x3f7ff800;
    c[3] = 0xc7000000;
    for (unsigned rounding = 0; rounding < 4; ++rounding) {
      for (uint64_t fp : {uint64_t(0), RJ_ARITH_FP16_OVFL, RJ_ARITH_FP_DX10_CLAMP_OFF,
                          RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS}) {
        for (uint32_t mode :
             {0U, uint32_t(RJ_ARITH_MIX_F16_A | RJ_ARITH_MIX_F16_B | RJ_ARITH_MIX_F16_C),
              uint32_t(RJ_ARITH_MIX_F16_A | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_NEG_C),
              uint32_t(RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_A)}) {
          const uint64_t flags = fp | (uint64_t(rounding) << 24);
          for (int lane = 0; lane < Lanes; ++lane) {
            expected[lane] = c[lane];
          }
          uint32_t *out = expected;
          ASSERT_EQ(call(flags | RJ_ARITH_SEMANTICS_COMPATIBILITY, Mask(-1), mode, &out, &ap, &bp,
                         &cp, nullptr),
                    RJ_ARITH_SUCCESS);
          if (!fp && !mode) {
            if constexpr (Half < 0) {
              EXPECT_EQ(expected[3], 0xba000000U);
            } else {
              EXPECT_EQ((expected[3] >> (Half * 16)) & 0xffffU, 0x9000U);
            }
            if constexpr (Half < 0) {
              EXPECT_EQ(expected[0], 0x3f801000U);
            } else {
              EXPECT_EQ((expected[0] >> (Half * 16)) & 0xffffU, rounding == 1 ? 0x3c01U : 0x3c00U);
            }
          }
          for (uint64_t cpu : cpus) {
            for (Mask exec_mask : {Mask(-1), Mask(0x8000000180000001ULL)}) {
              SCOPED_TRACE(::testing::Message()
                           << batch << '/' << rounding << '/' << fp << '/' << mode << '/' << cpu);
              for (int lane = 0; lane < Lanes; ++lane) {
                actual[lane] = c[lane];
              }
              out = actual;
              const uint32_t *alias = actual;
              uint32_t status = 0x12345678U;
              ASSERT_EQ(call(flags | cpu | semantics[batch % 3], exec_mask, mode, &out, &ap, &bp,
                             &alias, &status),
                        RJ_ARITH_SUCCESS);
              EXPECT_EQ(status, 0x12345678U);
              for (int lane = 0; lane < Lanes; ++lane) {
                EXPECT_EQ(actual[lane], (exec_mask >> lane) & 1 ? expected[lane] : c[lane]) << lane;
              }
            }
          }
        }
      }
    }
  }
  if constexpr (Half >= 0) {
    // FP32 intermediate rounds to a half tie: direct single-round FMA gives e5a1.
    a[0] = 0x4782;
    b[0] = 0xda00;
    c[0] = 0x03e1;
    uint32_t *out = actual;
    const uint32_t mode = RJ_ARITH_MIX_F16_A | RJ_ARITH_MIX_F16_B | RJ_ARITH_MIX_F16_C;
    ASSERT_EQ(call(RJ_ARITH_SEMANTICS_COMPATIBILITY, Mask(1), mode, &out, &ap, &bp, &cp, nullptr),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ((actual[0] >> (Half * 16)) & 0xffffU, 0xe5a2U);
  }
}

} // namespace

// FP32 output ignores guest rounding while retaining source selection and modifiers.
TEST(MixedMad, Float) {
  check<32, -1>(rj_arith_v_mad_mix_f32_w32);
  check<64, -1>(rj_arith_v_mad_mix_f32_w64);
}

// Low and high halves share the same guest-rounded arithmetic and alias contracts.
TEST(MixedMad, Half) {
  check<32, 0>(rj_arith_v_mad_mixlo_f16_w32);
  check<64, 0>(rj_arith_v_mad_mixlo_f16_w64);
  check<32, 1>(rj_arith_v_mad_mixhi_f16_w32);
  check<64, 1>(rj_arith_v_mad_mixhi_f16_w64);
}
