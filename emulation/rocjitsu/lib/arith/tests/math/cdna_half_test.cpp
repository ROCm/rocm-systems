// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA FP16 entry points retain the existing emulator models and reject new
// strict-exactness claims. Compare to their inherited implementations across CPU paths.

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <type_traits>

namespace {

// Check policy forwarding, masks, aliases and statuses without duplicating the model tests.
template <int Lanes, typename Call> void check_inherited(Call call, Call reference) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  Registers<2, Lanes> input;
  std::minstd_rand rng;
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[0][lane] = (uint32_t(rng() >> 15) << 16) | uint32_t(rng() >> 15);
  }
  input.v[0][0] = 0xfc017c01U;
  input.v[0][1] = 0xfc007c00U;
  input.v[0][2] = 0x80010001U;
  input.v[0][3] = 0x80000000U;
  const uint64_t modes[] = {0, RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D,
                            RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP,
                            RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_HIGH_D};
  const uint64_t fp_modes[] = {0, RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                      RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP16_OVFL};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (uint64_t fp : fp_modes) {
        for (uint64_t mode : modes) {
          for (int aliasing_offset : {0, 1}) {
            Registers<2, Lanes> actual = input, expected = input;
            uint32_t state = 0x80000000U, expected_state = state;
            const uint64_t flags = cpu | semantics | fp;
            const Mask exec_mask = Mask(0xaaaaaaaaaaaaaaaaULL) | 1;
            const int result =
                call(flags, exec_mask, mode, actual.v + aliasing_offset, actual.v, &state);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
              actual.expect_preserved(input, 0, 0);
              EXPECT_EQ(state, expected_state);
              EXPECT_EQ(call(flags, Mask(0), mode, nullptr, nullptr, nullptr), result);
              continue;
            }
            ASSERT_EQ(result, reference(flags, exec_mask, mode, expected.v + aliasing_offset,
                                        expected.v, &expected_state));
            EXPECT_EQ(state, expected_state);
            for (int reg = 0; reg < 2; ++reg) {
              for (int lane = 0; lane < Lanes; ++lane) {
                EXPECT_EQ(actual.v[reg][lane], expected.v[reg][lane])
                    << cpu << ':' << semantics << ':' << reg << ':' << lane;
              }
            }
            actual.expect_preserved(input, aliasing_offset, 1);
          }
        }
      }
    }
  }
}

} // namespace

TEST(CdnaHalfMath, Rcp) {
  check_inherited<32>(rj_arith_v_rcp_f16_w32_gfx942, rj_arith_v_rcp_f16_w32);
  check_inherited<64>(rj_arith_v_rcp_f16_w64_gfx942, rj_arith_v_rcp_f16_w64);
  check_inherited<32>(rj_arith_v_rcp_f16_w32_gfx1250, rj_arith_v_rcp_f16_w32_gfx12);
}

TEST(CdnaHalfMath, Rsq) {
  check_inherited<32>(rj_arith_v_rsq_f16_w32_gfx942, rj_arith_v_rsq_f16_w32);
  check_inherited<64>(rj_arith_v_rsq_f16_w64_gfx942, rj_arith_v_rsq_f16_w64);
  check_inherited<32>(rj_arith_v_rsq_f16_w32_gfx1250, rj_arith_v_rsq_f16_w32_gfx12);
}

TEST(CdnaHalfMath, Sqrt) {
  check_inherited<32>(rj_arith_v_sqrt_f16_w32_gfx942, rj_arith_v_sqrt_f16_w32);
  check_inherited<64>(rj_arith_v_sqrt_f16_w64_gfx942, rj_arith_v_sqrt_f16_w64);
  check_inherited<32>(rj_arith_v_sqrt_f16_w32_gfx1250, rj_arith_v_sqrt_f16_w32_gfx12);
}

TEST(CdnaHalfMath, Exp) {
  check_inherited<32>(rj_arith_v_exp_f16_w32_gfx942, rj_arith_v_exp_f16_w32);
  check_inherited<64>(rj_arith_v_exp_f16_w64_gfx942, rj_arith_v_exp_f16_w64);
  check_inherited<32>(rj_arith_v_exp_f16_w32_gfx1250, rj_arith_v_exp_f16_w32_gfx12);
}

TEST(CdnaHalfMath, Log) {
  check_inherited<32>(rj_arith_v_log_f16_w32_gfx942, rj_arith_v_log_f16_w32);
  check_inherited<64>(rj_arith_v_log_f16_w64_gfx942, rj_arith_v_log_f16_w64);
  check_inherited<32>(rj_arith_v_log_f16_w32_gfx1250, rj_arith_v_log_f16_w32_gfx12);
}

TEST(CdnaHalfMath, Sin) {
  check_inherited<32>(rj_arith_v_sin_f16_w32_gfx942, rj_arith_v_sin_f16_w32);
  check_inherited<64>(rj_arith_v_sin_f16_w64_gfx942, rj_arith_v_sin_f16_w64);
  check_inherited<32>(rj_arith_v_sin_f16_w32_gfx1250, rj_arith_v_sin_f16_w32_gfx12);
}

TEST(CdnaHalfMath, Cos) {
  check_inherited<32>(rj_arith_v_cos_f16_w32_gfx942, rj_arith_v_cos_f16_w32);
  check_inherited<64>(rj_arith_v_cos_f16_w64_gfx942, rj_arith_v_cos_f16_w64);
  check_inherited<32>(rj_arith_v_cos_f16_w32_gfx1250, rj_arith_v_cos_f16_w32_gfx12);
}
