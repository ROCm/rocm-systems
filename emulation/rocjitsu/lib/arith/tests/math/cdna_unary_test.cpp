// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA unary entry points retain the existing emulator models and reject new
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
  input.v[0][0] = 0x7f800001U;
  input.v[0][1] = 0x7f800000U;
  input.v[0][2] = 1;
  input.v[0][3] = 0x80000000U;
  const uint64_t modes[] = {0, RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP};
  const uint64_t fp_modes[] = {0, RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                      RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_ROUND_UP};
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

// Legacy CDNA RCP_IFLAG quiets NaNs and classifies only signed zero, even with CLAMP.
// These power-of-two results avoid inventing a new reference for the reciprocal approximation.
template <int Lanes, bool Gfx1250, typename Call> void check_rcp_iflag(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  const uint32_t inputs[] = {0,           0x80000000U, 1,           0x3f800000U,
                             0x7f800123U, 0xff800123U, 0x40000000U, 0xc0000000U};
  const uint32_t outputs[] = {0x7f800000U, 0xff800000U, 0x7f800000U, 0x3f800000U,
                              0x7fc00123U, 0xffc00123U, 0x3f000000U, 0xbf000000U};
  Registers<2, Lanes> input;
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[0][lane] = inputs[lane % 8];
  }
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (uint64_t fp : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                          RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS}) {
        for (uint64_t mode : {0U, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_ALU_CLAMP,
                              RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_CLAMP}) {
          // Exercise each cause independently, including the upper half of Wave64.
          for (Mask exec_mask : {Mask(0x1111111111111111ULL), Mask(0x2222222222222222ULL),
                                 Mask(0xccccccccccccccccULL)}) {
            for (int aliasing_offset : {0, 1}) {
              auto actual = input;
              uint32_t state = 0x80000000U;
              const int result = call(cpu | semantics | fp, exec_mask, mode,
                                      actual.v + aliasing_offset, actual.v, &state);
              if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                  semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
                EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
                actual.expect_preserved(input, 0, 0);
                EXPECT_EQ(state, 0x80000000U);
                continue;
              }
              ASSERT_EQ(result, RJ_ARITH_SUCCESS);
              const bool scale = (mode & RJ_ARITH_ALU_OMOD_2) &&
                                 (Gfx1250 || ((fp & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) &&
                                              !(fp & RJ_ARITH_FP_IEEE_MODE)));
              uint32_t causes = 0;
              for (int lane = 0; lane < Lanes; ++lane) {
                uint32_t expected = input.v[aliasing_offset][lane];
                if ((exec_mask >> lane) & 1) {
                  expected = outputs[lane % 8];
                  if (mode & RJ_ARITH_ALU_NEG_A) {
                    expected ^= 0x80000000U;
                  }
                  if (scale && (expected & 0x7fffffffU) < 0x7f800000U) {
                    expected += 0x800000U;
                  }
                  if (mode & RJ_ARITH_ALU_CLAMP) {
                    expected = (expected & 0x80000000U) || (expected & 0x7fffffffU) > 0x7f800000U
                                   ? 0
                               : expected > 0x3f800000U ? 0x3f800000U
                                                        : expected;
                  }
                  if (!(inputs[lane % 8] & 0x7fffffffU)) {
                    causes |= RJ_ARITH_EXCEPTION_INT_DIV0;
                  }
                }
                EXPECT_EQ(actual.v[aliasing_offset][lane], expected)
                    << cpu << ':' << semantics << ':' << fp << ':' << mode << ':' << lane;
              }
              EXPECT_EQ(state,
                        0x80000000U | (semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY ? causes : 0));
              actual.expect_preserved(input, aliasing_offset, 1);
            }
          }
        }
      }
    }
  }
}

} // namespace

TEST(CdnaMath, Rcp) {
  check_inherited<32>(rj_arith_v_rcp_f32_w32_gfx942, rj_arith_v_rcp_f32_w32);
  check_inherited<64>(rj_arith_v_rcp_f32_w64_gfx942, rj_arith_v_rcp_f32_w64);
  check_inherited<32>(rj_arith_v_rcp_f32_w32_gfx1250, rj_arith_v_rcp_f32_w32_gfx12);
}

TEST(CdnaMath, RcpIflag) {
  check_rcp_iflag<32, false>(rj_arith_v_rcp_iflag_f32_w32_gfx942);
  check_rcp_iflag<64, false>(rj_arith_v_rcp_iflag_f32_w64_gfx942);
  check_rcp_iflag<32, true>(rj_arith_v_rcp_iflag_f32_w32_gfx1250);
}

TEST(CdnaMath, Rsq) {
  check_inherited<32>(rj_arith_v_rsq_f32_w32_gfx942, rj_arith_v_rsq_f32_w32);
  check_inherited<64>(rj_arith_v_rsq_f32_w64_gfx942, rj_arith_v_rsq_f32_w64);
  check_inherited<32>(rj_arith_v_rsq_f32_w32_gfx1250, rj_arith_v_rsq_f32_w32_gfx12);
}

TEST(CdnaMath, Sqrt) {
  check_inherited<32>(rj_arith_v_sqrt_f32_w32_gfx942, rj_arith_v_sqrt_f32_w32);
  check_inherited<64>(rj_arith_v_sqrt_f32_w64_gfx942, rj_arith_v_sqrt_f32_w64);
  check_inherited<32>(rj_arith_v_sqrt_f32_w32_gfx1250, rj_arith_v_sqrt_f32_w32_gfx12);
}

TEST(CdnaMath, Exp) {
  check_inherited<32>(rj_arith_v_exp_f32_w32_gfx942, rj_arith_v_exp_f32_w32);
  check_inherited<64>(rj_arith_v_exp_f32_w64_gfx942, rj_arith_v_exp_f32_w64);
  check_inherited<32>(rj_arith_v_exp_f32_w32_gfx1250, rj_arith_v_exp_f32_w32_gfx12);
}

TEST(CdnaMath, Log) {
  check_inherited<32>(rj_arith_v_log_f32_w32_gfx942, rj_arith_v_log_f32_w32);
  check_inherited<64>(rj_arith_v_log_f32_w64_gfx942, rj_arith_v_log_f32_w64);
  check_inherited<32>(rj_arith_v_log_f32_w32_gfx1250, rj_arith_v_log_f32_w32_gfx12);
}

TEST(CdnaMath, Sin) {
  check_inherited<32>(rj_arith_v_sin_f32_w32_gfx942, rj_arith_v_sin_f32_w32);
  check_inherited<64>(rj_arith_v_sin_f32_w64_gfx942, rj_arith_v_sin_f32_w64);
  check_inherited<32>(rj_arith_v_sin_f32_w32_gfx1250, rj_arith_v_sin_f32_w32_gfx12);
}

TEST(CdnaMath, Cos) {
  check_inherited<32>(rj_arith_v_cos_f32_w32_gfx942, rj_arith_v_cos_f32_w32);
  check_inherited<64>(rj_arith_v_cos_f32_w64_gfx942, rj_arith_v_cos_f32_w64);
  check_inherited<32>(rj_arith_v_cos_f32_w32_gfx1250, rj_arith_v_cos_f32_w32_gfx12);
}

// Legacy spelling preserves the ordinary numeric and API policy.
TEST(CdnaMath, LegacyExp) {
  check_inherited<32>(rj_arith_v_exp_legacy_f32_w32_gfx942, rj_arith_v_exp_f32_w32_gfx942);
  check_inherited<64>(rj_arith_v_exp_legacy_f32_w64_gfx942, rj_arith_v_exp_f32_w64_gfx942);
}

// Legacy spelling preserves the ordinary numeric and API policy.
TEST(CdnaMath, LegacyLog) {
  check_inherited<32>(rj_arith_v_log_legacy_f32_w32_gfx942, rj_arith_v_log_f32_w32_gfx942);
  check_inherited<64>(rj_arith_v_log_legacy_f32_w64_gfx942, rj_arith_v_log_f32_w64_gfx942);
}
