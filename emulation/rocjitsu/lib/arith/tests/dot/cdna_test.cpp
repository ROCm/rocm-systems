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
