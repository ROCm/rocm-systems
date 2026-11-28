// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA unary entry points retain the existing emulator models and reject new
// strict-exactness claims. Compare to their inherited implementations across CPU paths.

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
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
    const uint32_t high = uint32_t(rng() >> 15), low = uint32_t(rng() >> 15);
    input.v[0][lane] = (high << 16) | low;
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

namespace {
// Check each CDNA integral core's rounded output scaling, aliasing and selected stores.
template <int Bits, int Lanes, typename Call>
void check_integral_sdwa(Call call, uint32_t expected) {
  using namespace rocjitsu::arith::test;
  using Mask = WaveMask<Lanes>;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (int aliasing_offset : {0, 1}) {
      Registers<2, Lanes> input;
      for (int reg = 0; reg < 2; ++reg) {
        for (int lane = 0; lane < Lanes; ++lane) {
          input.v[reg][lane] = Bits == 16 ? 0x3e003e00U : 0x3fc00000U;
        }
      }
      Registers<2, Lanes> data = input;
      const uint64_t flags =
          cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
      const uint64_t mode = RJ_ARITH_ALU_SDWA | RJ_ARITH_ALU_OMOD_2 |
                            (Bits == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D : 0U);
      const Mask exec_mask = Mask(1) | (Mask(1) << (Lanes - 1));
      uint32_t status = 0x12345678U;
      ASSERT_EQ(call(flags, exec_mask, mode, data.v + aliasing_offset, data.v, &status), 0);
      for (int lane = 0; lane < Lanes; ++lane) {
        const uint32_t output = Bits == 16 ? (expected << 16) | 0x3e00U : expected;
        EXPECT_EQ(data.v[aliasing_offset][lane],
                  (exec_mask >> lane) & 1U ? output : input.v[aliasing_offset][lane]);
      }
      data.expect_preserved(input, aliasing_offset, 1);
      EXPECT_EQ(status, 0x12345678U);
      EXPECT_EQ(call(flags, Mask(0), mode, nullptr, nullptr, nullptr), 0);
      EXPECT_EQ(call(flags, Mask(0), mode | RJ_ARITH_DPP16, nullptr, nullptr, nullptr),
                RJ_ARITH_ERROR_INVALID_FLAGS);
    }
  }
}
} // namespace

// The same literal input distinguishes the six operations; no computed arithmetic oracle.
TEST(CdnaIntegral, SdwaOutput) {
  using Call32 = decltype(&rj_arith_v_trunc_f16_w32_gfx942);
  using Call64 = decltype(&rj_arith_v_trunc_f16_w64_gfx942);
  struct Case {
    Call32 f16_wave32;
    Call64 f16_wave64;
    Call32 f32_wave32;
    Call64 f32_wave64;
    uint32_t f16_output, f32_output;
  };
  const Case cases[] = {
      {rj_arith_v_trunc_f16_w32_gfx942, rj_arith_v_trunc_f16_w64_gfx942,
       rj_arith_v_trunc_f32_w32_gfx942, rj_arith_v_trunc_f32_w64_gfx942, 0x4000, 0x40000000U},
      {rj_arith_v_ceil_f16_w32_gfx942, rj_arith_v_ceil_f16_w64_gfx942,
       rj_arith_v_ceil_f32_w32_gfx942, rj_arith_v_ceil_f32_w64_gfx942, 0x4400, 0x40800000U},
      {rj_arith_v_floor_f16_w32_gfx942, rj_arith_v_floor_f16_w64_gfx942,
       rj_arith_v_floor_f32_w32_gfx942, rj_arith_v_floor_f32_w64_gfx942, 0x4000, 0x40000000U},
      {rj_arith_v_rndne_f16_w32_gfx942, rj_arith_v_rndne_f16_w64_gfx942,
       rj_arith_v_rndne_f32_w32_gfx942, rj_arith_v_rndne_f32_w64_gfx942, 0x4400, 0x40800000U},
      {rj_arith_v_frexp_mant_f16_w32_gfx942, rj_arith_v_frexp_mant_f16_w64_gfx942,
       rj_arith_v_frexp_mant_f32_w32_gfx942, rj_arith_v_frexp_mant_f32_w64_gfx942, 0x3e00,
       0x3fc00000U},
      {rj_arith_v_fract_f16_w32_gfx942, rj_arith_v_fract_f16_w64_gfx942,
       rj_arith_v_fract_f32_w32_gfx942, rj_arith_v_fract_f32_w64_gfx942, 0x3c00, 0x3f800000U},
  };
  for (const Case &sample : cases) {
    check_integral_sdwa<16, 32>(sample.f16_wave32, sample.f16_output);
    check_integral_sdwa<16, 64>(sample.f16_wave64, sample.f16_output);
    check_integral_sdwa<32, 32>(sample.f32_wave32, sample.f32_output);
    check_integral_sdwa<32, 64>(sample.f32_wave64, sample.f32_output);
  }
}

namespace {
// Check the output gate on exact power-of-two results without recreating native math models.
template <int Lanes, typename Call> void check_sdwa(Call call, uint32_t bits, uint32_t expected) {
  using namespace rocjitsu::arith::test;
  using Mask = WaveMask<Lanes>;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (int aliasing_offset : {0, 1}) {
        Registers<2, Lanes> input;
        for (int lane = 0; lane < Lanes; ++lane) {
          input.v[0][lane] = bits;
        }
        auto actual = input;
        const uint64_t flags = cpu | semantics | RJ_ARITH_FP_ROUND_UP;
        const uint64_t mode = RJ_ARITH_ALU_SDWA | RJ_ARITH_ALU_OMOD_2;
        const Mask exec_mask = Mask(1) | (Mask(1) << (Lanes - 1));
        uint32_t status = 0x80000000U;
        const int error =
            call(flags, exec_mask, mode, actual.v + aliasing_offset, actual.v, &status);
        EXPECT_EQ(status, 0x80000000U);
        if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
            semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
          EXPECT_EQ(error, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
          actual.expect_preserved(input, 0, 0);
          continue;
        }
        ASSERT_EQ(error, RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < Lanes; ++lane) {
          EXPECT_EQ(actual.v[aliasing_offset][lane],
                    (exec_mask >> lane) & 1U ? expected : input.v[aliasing_offset][lane]);
        }
        actual.expect_preserved(input, aliasing_offset, 1);
        EXPECT_EQ(call(flags, Mask(0), mode, nullptr, nullptr, nullptr), RJ_ARITH_SUCCESS);
        EXPECT_EQ(call(flags, Mask(0), mode | RJ_ARITH_DPP16, nullptr, nullptr, nullptr),
                  RJ_ARITH_ERROR_INVALID_FLAGS);
      }
    }
  }
}
} // namespace

// Native nearest output scales despite preserved guest denormals; SIN/IFLAG keep the MODE gate.
TEST(CdnaUnary, SdwaOutput) {
  struct Case {
    decltype(&rj_arith_v_rcp_f32_w32_gfx942) wave32;
    decltype(&rj_arith_v_rcp_f32_w64_gfx942) wave64;
    uint32_t input, expected;
  };
  const Case cases[] = {
      {rj_arith_v_rcp_f32_w32_gfx942, rj_arith_v_rcp_f32_w64_gfx942, 0x3f800000U, 0x40000000U},
      {rj_arith_v_rcp_iflag_f32_w32_gfx942, rj_arith_v_rcp_iflag_f32_w64_gfx942, 0x3f800000U,
       0x3f800000U},
      {rj_arith_v_rsq_f32_w32_gfx942, rj_arith_v_rsq_f32_w64_gfx942, 0x3f800000U, 0x40000000U},
      {rj_arith_v_sqrt_f32_w32_gfx942, rj_arith_v_sqrt_f32_w64_gfx942, 0x40800000U, 0x40800000U},
      {rj_arith_v_exp_f32_w32_gfx942, rj_arith_v_exp_f32_w64_gfx942, 0, 0x40000000U},
      {rj_arith_v_log_f32_w32_gfx942, rj_arith_v_log_f32_w64_gfx942, 0x40800000U, 0x40800000U},
      {rj_arith_v_sin_f32_w32_gfx942, rj_arith_v_sin_f32_w64_gfx942, 0x80000000U, 0x80000000U},
      {rj_arith_v_cos_f32_w32_gfx942, rj_arith_v_cos_f32_w64_gfx942, 0, 0x40000000U},
      {rj_arith_v_exp_legacy_f32_w32_gfx942, rj_arith_v_exp_legacy_f32_w64_gfx942, 0, 0x40000000U},
      {rj_arith_v_log_legacy_f32_w32_gfx942, rj_arith_v_log_legacy_f32_w64_gfx942, 0x40800000U,
       0x40800000U},
  };
  for (const Case &sample : cases) {
    check_sdwa<32>(sample.wave32, sample.input, sample.expected);
    check_sdwa<64>(sample.wave64, sample.input, sample.expected);
  }
  EXPECT_EQ(rj_arith_v_rcp_f32_w32_gfx1250(RJ_ARITH_SEMANTICS_LOOSE, 0, RJ_ARITH_ALU_SDWA, nullptr,
                                           nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}

namespace {

template <int Lanes, bool ActiveOmodCaptured, bool DenormGatesOmod, typename Call>
void check_gfx950(Call call, Call reference) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  Registers<2, Lanes> input;
  std::minstd_rand rng;
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[0][lane] = (uint32_t(rng() >> 15) << 16) | uint32_t(rng() >> 15);
  }
  input.v[0][0] = 0x7f800001U;
  input.v[0][1] = 0xbf800000U;
  input.v[0][2] = 0U;
  input.v[0][3] = 0x80000000U;
  const uint64_t modes[] = {0, RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP};
  const uint64_t fp_modes[] = {0, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                               RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                   RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (uint64_t fp : fp_modes) {
        for (uint64_t mode : modes) {
          Registers<2, Lanes> actual = input, expected = input;
          uint32_t state = 0x80000000U, expected_state = state;
          const uint64_t flags = cpu | semantics | fp;
          const Mask exec_mask = Mask(0xaaaaaaaaaaaaaaaaULL) | 3;
          const int result = call(flags, exec_mask, mode, actual.v + 1, actual.v, &state);
          if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
            EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
            actual.expect_preserved(input, 0, 0);
            EXPECT_EQ(state, expected_state);
            continue;
          }
          const bool active_omod = !ActiveOmodCaptured && (mode & RJ_ARITH_ALU_OMOD_MASK) &&
                                   !(fp & RJ_ARITH_FP_IEEE_MODE) &&
                                   (!DenormGatesOmod || (fp & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS));
          if (active_omod && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
            EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
            actual.expect_preserved(input, 0, 0);
            EXPECT_EQ(state, expected_state);
            continue;
          }
          const uint64_t ref_flags =
              active_omod && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE
                  ? (flags & ~RJ_ARITH_SEMANTICS_MASK) | RJ_ARITH_SEMANTICS_LOOSE
                  : flags;
          ASSERT_EQ(result, reference(ref_flags, exec_mask, mode, expected.v + 1, expected.v,
                                      &expected_state));
          EXPECT_EQ(state, expected_state);
          for (int lane = 0; lane < Lanes; ++lane) {
            EXPECT_EQ(actual.v[1][lane], expected.v[1][lane]);
          }
        }
      }
    }
  }
}

} // namespace

TEST(CdnaMath, Gfx950ExactAndQualification) {
  check_gfx950<32, true, false>(rj_arith_v_rcp_f32_w32_gfx950, rj_arith_v_rcp_f32_w32);
  check_gfx950<64, true, false>(rj_arith_v_rcp_f32_w64_gfx950, rj_arith_v_rcp_f32_w64);
  check_gfx950<32, false, false>(rj_arith_v_rsq_f32_w32_gfx950, rj_arith_v_rsq_f32_w32);
  check_gfx950<64, false, false>(rj_arith_v_rsq_f32_w64_gfx950, rj_arith_v_rsq_f32_w64);
  check_gfx950<32, false, false>(rj_arith_v_sqrt_f32_w32_gfx950, rj_arith_v_sqrt_f32_w32);
  check_gfx950<64, false, false>(rj_arith_v_sqrt_f32_w64_gfx950, rj_arith_v_sqrt_f32_w64);
  check_gfx950<32, true, false>(rj_arith_v_exp_f32_w32_gfx950, rj_arith_v_exp_f32_w32);
  check_gfx950<64, true, false>(rj_arith_v_exp_f32_w64_gfx950, rj_arith_v_exp_f32_w64);
  check_gfx950<32, false, false>(rj_arith_v_log_f32_w32_gfx950, rj_arith_v_log_f32_w32);
  check_gfx950<64, false, false>(rj_arith_v_log_f32_w64_gfx950, rj_arith_v_log_f32_w64);
  check_gfx950<32, false, true>(rj_arith_v_sin_f32_w32_gfx950, rj_arith_v_sin_f32_w32);
  check_gfx950<64, false, true>(rj_arith_v_sin_f32_w64_gfx950, rj_arith_v_sin_f32_w64);
  check_gfx950<32, false, false>(rj_arith_v_cos_f32_w32_gfx950, rj_arith_v_cos_f32_w32);
  check_gfx950<64, false, false>(rj_arith_v_cos_f32_w64_gfx950, rj_arith_v_cos_f32_w64);

  // Verify MI350X RCP_IFLAG IEEE-gated sNaN preservation, Compatibility INT_DIV0, and guards.
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (bool ieee : {false, true}) {
      rocjitsu::arith::test::Registers<2, 64> r;
      r.v[0][0] = 0x7f800123U;
      r.v[0][1] = 0U;
      uint32_t state = 0x10U;
      const uint64_t flags = cpu | (ieee ? RJ_ARITH_FP_IEEE_MODE : 0);
      ASSERT_EQ(rj_arith_v_rcp_iflag_f32_w64_gfx950(flags | RJ_ARITH_SEMANTICS_COMPATIBILITY, 3ULL,
                                                    0, r.v + 1, r.v, &state),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[1][0], ieee ? 0x7fc00123U : 0x7f800123U);
      EXPECT_EQ(r.v[1][1], 0x7f800000U);
      EXPECT_EQ(state, 0x10U | RJ_ARITH_EXCEPTION_INT_DIV0);

      uint32_t mode2_state = 0x10U;
      ASSERT_EQ(rj_arith_v_rcp_iflag_f32_w32_gfx950(
                    flags | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 3U, 0, r.v + 1,
                    r.v, &mode2_state),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(mode2_state, 0x10U);
      EXPECT_EQ(rj_arith_v_rcp_iflag_f32_w32_gfx950(flags | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL,
                                                    3U, 0, r.v + 1, r.v, &mode2_state),
                RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
    }
  }
}

// Compatibility retains guest-directed narrowing even though strict modes are unqualified.
TEST(CdnaIntegral, CompatibilityFractionRounding) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    Registers<2, 64> data;
    // -3*2^-12 gives a fraction halfway between 0x3bfe and 0x3bff: upward selects the latter.
    data.v[0][0] = data.v[0][63] = 0x9200U;
    const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_ROUND_UP;
    uint32_t status = 0x80000000U;
    ASSERT_EQ(rj_arith_v_fract_f16_w32_gfx942(flags, 1, 0, data.v + 1, data.v, &status), 0);
    EXPECT_EQ(data.v[1][0] & 0xffffU, 0x3bffU);
    ASSERT_EQ(rj_arith_v_fract_f16_w64_gfx942(flags, 1ULL << 63, 0, data.v + 1, data.v, &status),
              0);
    EXPECT_EQ(data.v[1][63] & 0xffffU, 0x3bffU);
    EXPECT_EQ(status, 0x80000000U);
  }
}
