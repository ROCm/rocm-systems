// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA vector dots retain host FP32 pair-sum arithmetic, modifiers and aliasing.
// The reference is the inherited expression, not a new hardware-exact model.

#include "common/floating_point.h"
#include "dot/helpers.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_dot.h"

#include <bit>
#include <gtest/gtest.h>

namespace {
using namespace rocjitsu::arith::test;

// Check supported modes against the inherited expression and reject unsupported
// requests before writes. Each CPU path sees the same deterministic operands.
template <int Lanes, bool Bf16, bool Accumulate, typename Call> void check(Call call) {
  using Mask = WaveMask<Lanes>;
  auto input = dot2_inputs<Lanes, Bf16, false>();
  const uint32_t one = Bf16 ? 0x3f80U : 0x3c00U;
  input.v[0][0] = input.v[1][0] = one * 0x10001U;
  input.v[2][0] = input.v[3][0] = 0x4b800000U;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (uint64_t mode : {0ULL, 1ULL, 0x40ULL, 0x7ffULL, 1ULL << 40}) {
        for (int destination : {0, 1, 2, 3}) {
          for (Mask exec_mask :
               {Mask(0), Mask(0x80000001ULL | (Lanes == 64 ? 1ULL << 63 : 0)), ~Mask(0)}) {
            auto regs = input;
            uint32_t status = 0x123U;
            const int expected_error =
                (mode & ~(Accumulate ? 0ULL : 0x7ffULL)) ? RJ_ARITH_ERROR_INVALID_FLAGS
                : semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                        semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                    ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS
                    : RJ_ARITH_SUCCESS;
            const uint64_t flags = cpu | semantics | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                   RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_ROUND_MASK |
                                   RJ_ARITH_FP16_OVFL;
            ASSERT_EQ((invoke_dot<Accumulate, true>(
                          call, flags, exec_mask, mode, exec_mask ? regs.v + destination : nullptr,
                          exec_mask ? regs.v : nullptr, exec_mask ? regs.v + 1 : nullptr,
                          exec_mask ? regs.v + 2 : nullptr, &status)),
                      expected_error);
            EXPECT_EQ(status, 0x123U);
            if (expected_error) {
              regs.expect_preserved(input, 0, 0);
              continue;
            }
            regs.expect_preserved(input, destination, 1);
            for (int lane = 0; lane < Lanes; ++lane) {
              if (!((exec_mask >> lane) & 1)) {
                EXPECT_EQ(regs.v[destination][lane], input.v[destination][lane]);
                continue;
              }
              auto factor = [&](int reg, bool high, bool negate) {
                uint16_t bits = input.v[reg][lane] >> (high ? 16 : 0);
                bits ^= negate ? 0x8000U : 0;
                return Bf16 ? bf16_value(bits) : f16_value(bits);
              };
              const float low =
                  factor(0, mode & RJ_ARITH_DOT_LO_A_HIGH, mode & RJ_ARITH_DOT_NEG_LO_A) *
                  factor(1, mode & RJ_ARITH_DOT_LO_B_HIGH, mode & RJ_ARITH_DOT_NEG_LO_B);
              const float high =
                  factor(0, !(mode & RJ_ARITH_DOT_HI_A_LOW), mode & RJ_ARITH_DOT_NEG_HI_A) *
                  factor(1, !(mode & RJ_ARITH_DOT_HI_B_LOW), mode & RJ_ARITH_DOT_NEG_HI_B);
              uint32_t acc = input.v[Accumulate ? destination : 2][lane];
              acc = (acc & (mode & RJ_ARITH_DOT_ABS_C ? 0x7fffffffU : ~0U)) ^
                    (mode & RJ_ARITH_DOT_NEG_C ? 0x80000000U : 0U);
              expect_f32(regs.v[destination][lane], (low + high) + std::bit_cast<float>(acc),
                         RJ_ARITH_SEMANTICS_LOOSE, /*atol=*/1e-6, /*rtol=*/1e-6);
              // Two products of one must be added before the large accumulator;
              // sequential FMAs would lose both units. CLAMP is ignored as before.
              if (lane == 0 && (mode == 0 || mode == RJ_ARITH_DOT_CLAMP) && destination >= 2) {
                EXPECT_EQ(regs.v[destination][lane], 0x4b800001U);
              }
            }
          }
        }
      }
    }
  }
}

} // namespace

// Check this instruction form with shared inputs and contract checks.
TEST(Dot2F32F16W32Gfx942, Arithmetic) {
  check<32, false, false>(rj_arith_v_dot2_f32_f16_w32_gfx942);
}

// Check this instruction form with shared inputs and contract checks.
TEST(Dot2F32F16W64Gfx942, Arithmetic) {
  check<64, false, false>(rj_arith_v_dot2_f32_f16_w64_gfx942);
}

// Check this instruction form with shared inputs and contract checks.
TEST(Dot2CF32F16W32Gfx942, Arithmetic) {
  check<32, false, true>(rj_arith_v_dot2c_f32_f16_w32_gfx942);
}

// Check this instruction form with shared inputs and contract checks.
TEST(Dot2CF32F16W64Gfx942, Arithmetic) {
  check<64, false, true>(rj_arith_v_dot2c_f32_f16_w64_gfx942);
}

// Check this instruction form with shared inputs and contract checks.
TEST(Dot2F32Bf16W32Gfx950, Arithmetic) {
  check<32, true, false>(rj_arith_v_dot2_f32_bf16_w32_gfx950);
}

// Check this instruction form with shared inputs and contract checks.
TEST(Dot2F32Bf16W64Gfx950, Arithmetic) {
  check<64, true, false>(rj_arith_v_dot2_f32_bf16_w64_gfx950);
}

// Check this instruction form with shared inputs and contract checks.
TEST(Dot2CF32Bf16W32Gfx950, Arithmetic) {
  check<32, true, true>(rj_arith_v_dot2c_f32_bf16_w32_gfx950);
}

// Check this instruction form with shared inputs and contract checks.
TEST(Dot2CF32Bf16W64Gfx950, Arithmetic) {
  check<64, true, true>(rj_arith_v_dot2c_f32_bf16_w64_gfx950);
}

TEST(Dot2F32F16Gfx950, ExactArithmeticAndContracts) {
  auto check_gfx950 = []<int Lanes, bool Accumulate>(auto call) {
    using Mask = WaveMask<Lanes>;
    auto input = dot2_inputs<Lanes, false, false>();
    input.v[0][0] = input.v[1][0] = 0x3c003c00U;
    input.v[2][0] = input.v[3][0] = 0x4b800000U;
    // Canonical -qNaN (0xffc00000) on payload NaN in A or C.
    input.v[0][1] = 0x7e123c00U;
    input.v[1][1] = 0x3c003c00U;
    input.v[2][1] = input.v[3][1] = 0U;
    // Physical MI350X preserves FP16 subnormal inputs under MODE.FP_DENORM=0 despite CDNA4 §6.4
    // (0 / 32,768 diffs between wave2_3f default and rne_flush_all on both DOT2 and DOT2C):
    // - lane 2: 2^-24 + 2^-24 = 2^-23 (0x34000000)
    // - lane 3 (wave2_3f row 4): 0x00000001 * 0xd08fdee0 + 0x40880000 -> 0x4087ffc9 (vs 0x40880000
    //   if flushed)
    // - lane 4 (wave2_3f row 118): 0x00800000 * 0x47c6cc38 + 0x89a0cbcd -> 0x3878bffc (vs
    //   0x89a0cbcd if flushed and 0x3878c000 in single IEEE / Loose)
    input.v[0][2] = 0x00010001U;
    input.v[1][2] = 0x3c003c00U;
    input.v[2][2] = input.v[3][2] = 0U;
    input.v[0][3] = 0x00000001U;
    input.v[1][3] = 0xd08fdee0U;
    input.v[2][3] = input.v[3][3] = 0x40880000U;
    input.v[0][4] = 0x00800000U;
    input.v[1][4] = 0x47c6cc38U;
    input.v[2][4] = input.v[3][4] = 0x89a0cbcdU;
    // Nearest-even normalization increment and 24-bit fixed-point C alignment:
    // - lane 5 (wave2_3f row 169): 0x3c01fbff * 0x793e1591 + 0x42260000 -> 0x4727ba6c (rup=0,
    //   single IEEE RNE, and Loose all give 0x4727ba6b)
    // - lane 6 (wave2_3f row 87): 0x3c003c00 * 0x3e2b6a24 + 0x3e633cdc -> 0x45449c38 (single IEEE
    //   RNE and Loose give 0x45449c39)
    // - lane 7: A=B=0x00003c00, C=0x34400000 -> 0x3f800002 (rup=0 gives 0x3f800001)
    // - lane 8: A=B=0x00003c00, C=0xa1800000 (-2^-60) -> 0x3f7fffff (single IEEE RNE and Loose give
    //   0x3f800000)
    input.v[0][5] = 0x3c01fbffU;
    input.v[1][5] = 0x793e1591U;
    input.v[2][5] = input.v[3][5] = 0x42260000U;
    input.v[0][6] = 0x3c003c00U;
    input.v[1][6] = 0x3e2b6a24U;
    input.v[2][6] = input.v[3][6] = 0x3e633cdcU;
    input.v[0][7] = input.v[1][7] = 0x00003c00U;
    input.v[2][7] = input.v[3][7] = 0x34400000U;
    input.v[0][8] = input.v[1][8] = 0x00003c00U;
    input.v[2][8] = input.v[3][8] = 0xa1800000U;
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t semantics : semantics_modes()) {
        for (uint64_t mode : {uint64_t(0), uint64_t(RJ_ARITH_DOT_CLAMP), uint64_t(1) << 40}) {
          auto regs = input;
          uint32_t status = 0x55aaU;
          const int expected_error = (mode & ~(Accumulate ? 0ULL : 0x7ffULL))
                                         ? RJ_ARITH_ERROR_INVALID_FLAGS
                                     : semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                                         ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                                         : RJ_ARITH_SUCCESS;
          const uint64_t flags = cpu | semantics | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                 RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_ROUND_MASK |
                                 RJ_ARITH_FP16_OVFL;
          ASSERT_EQ((invoke_dot<Accumulate, true>(call, flags, ~Mask(0), mode, regs.v + 3, regs.v,
                                                  regs.v + 1, regs.v + 2, &status)),
                    expected_error);
          EXPECT_EQ(status, 0x55aaU);
          if (expected_error) {
            regs.expect_preserved(input, 0, 0);
            continue;
          }
          EXPECT_EQ(regs.v[3][0], 0x4b800001U);
          if (semantics != RJ_ARITH_SEMANTICS_LOOSE &&
              semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY) {
            EXPECT_EQ(regs.v[3][1], 0xffc00000U);
            EXPECT_EQ(regs.v[3][2], 0x34000000U);
            EXPECT_EQ(regs.v[3][3], 0x4087ffc9U);
            EXPECT_EQ(regs.v[3][4], 0x3878bffcU);
            EXPECT_EQ(regs.v[3][5], 0x4727ba6cU);
            EXPECT_EQ(regs.v[3][6], 0x45449c38U);
            EXPECT_EQ(regs.v[3][7], 0x3f800002U);
            EXPECT_EQ(regs.v[3][8], 0x3f7fffffU);
          }
        }
      }
    }
  };
  check_gfx950.template operator()<32, false>(rj_arith_v_dot2_f32_f16_w32_gfx950);
  check_gfx950.template operator()<64, false>(rj_arith_v_dot2_f32_f16_w64_gfx950);
  check_gfx950.template operator()<32, true>(rj_arith_v_dot2c_f32_f16_w32_gfx950);
  check_gfx950.template operator()<64, true>(rj_arith_v_dot2c_f32_f16_w64_gfx950);
}
