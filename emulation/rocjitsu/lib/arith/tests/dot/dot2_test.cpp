// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP16/BF16 DOT2 arithmetic, modifiers, and exact reference cases.

#include "common/cpu.h"
#include "common/dpp_reference.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "dot/dot2_cases.h"
#include "dot/helpers.h"
#include "rj_arith/rj_arith.h"
#include <bit>

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

static const uint32_t dot2_modifiers =
    RJ_ARITH_DOT_NEG_LO_A | RJ_ARITH_DOT_NEG_LO_B | RJ_ARITH_DOT_NEG_C | RJ_ARITH_DOT_NEG_HI_A |
    RJ_ARITH_DOT_NEG_HI_B | RJ_ARITH_DOT_LO_A_HIGH | RJ_ARITH_DOT_LO_B_HIGH |
    RJ_ARITH_DOT_HI_A_LOW | RJ_ARITH_DOT_HI_B_LOW | RJ_ARITH_DOT_CLAMP | RJ_ARITH_DOT_ABS_C;

// Undo A sign modifiers so stored DOT2 cases exercise the requested instruction mode.
uint32_t undo_dot2_a(uint32_t a, uint32_t mode) {
  a ^= (mode & RJ_ARITH_DOT_NEG_LO_A ? 0x8000U : 0) |
       (mode & RJ_ARITH_DOT_NEG_HI_A ? 0x80000000U : 0);
  if ((mode & (RJ_ARITH_DOT_LO_A_HIGH | RJ_ARITH_DOT_HI_A_LOW)) ==
      (RJ_ARITH_DOT_LO_A_HIGH | RJ_ARITH_DOT_HI_A_LOW)) {
    a = (a << 16) | (a >> 16);
  }
  return a;
}

// Undo B sign modifiers so stored DOT2 cases exercise the requested instruction mode.
uint32_t undo_dot2_b(uint32_t b, uint32_t mode) {
  b ^= (mode & RJ_ARITH_DOT_NEG_LO_B ? 0x8000U : 0) |
       (mode & RJ_ARITH_DOT_NEG_HI_B ? 0x80000000U : 0);
  if ((mode & (RJ_ARITH_DOT_LO_B_HIGH | RJ_ARITH_DOT_HI_B_LOW)) ==
      (RJ_ARITH_DOT_LO_B_HIGH | RJ_ARITH_DOT_HI_B_LOW)) {
    b = (b << 16) | (b >> 16);
  }
  return b;
}

// Check stored exact DOT2 cases across CPU levels and modifier encodings for the supplied entry
// point.
template <int Lanes, bool Accumulate, bool Gfx12, typename Call>
void check_exact_dot2(Call call, const Dot2Case (&cases)[16]) {
  using Mask = WaveMask<Lanes>;
  const uint32_t modifiers[] = {0,
                                RJ_ARITH_DOT_NEG_LO_A,
                                RJ_ARITH_DOT_NEG_HI_B,
                                RJ_ARITH_DOT_NEG_C,
                                dot2_modifiers & ~RJ_ARITH_DOT_ABS_C,
                                RJ_ARITH_DOT_CLAMP,
                                RJ_ARITH_DOT_LO_A_HIGH,
                                RJ_ARITH_DOT_LO_B_HIGH,
                                RJ_ARITH_DOT_HI_A_LOW,
                                RJ_ARITH_DOT_HI_B_LOW};
  // Both routings are in-range permutations with FI enabled, so every active
  // lane receives a captured A. Choose its matching B/C in the destination lane.
  const uint64_t routes[] = {0, dpp_modes[1],
                             RJ_ARITH_DPP16 | RJ_ARITH_DPP_FI |
                                 (0x1bULL << RJ_ARITH_DPP_CTRL_SHIFT) | RJ_ARITH_DPP_ROW_MASK |
                                 RJ_ARITH_DPP_BANK_MASK};
  auto cpu_levels = supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : semantics_modes()) {
      if (semantics == RJ_ARITH_SEMANTICS_LOOSE) {
        continue;
      }
      for (int scenario = 0; scenario < 15; ++scenario) {
        uint32_t modifier = Accumulate ? 0 : scenario < 5 ? 0 : modifiers[scenario - 5];
        bool same_halves = !Accumulate && scenario >= 11;
        uint64_t mode = modifier | routes[scenario % 3];
        Mask exec_mask = exec_mask_for_case<Lanes>(scenario < 5 ? scenario : scenario % 2 ? 0 : 3);
        Registers<4, Lanes> input;
        uint32_t expected[Lanes];
        for (int lane = 0; lane < Lanes; ++lane) {
          input.v[0][lane] = undo_dot2_a(cases[same_halves ? 2 : lane % 16].a, modifier);
        }
        for (int lane = 0; lane < Lanes; ++lane) {
          int source = lane;
          if (mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16)) {
            ASSERT_TRUE(dpp_source(mode, ~Mask(0), lane, source));
            ASSERT_GE(source, 0);
          }
          auto f = cases[same_halves ? 2 : source % 16];
          input.v[1][lane] = undo_dot2_b(f.b, modifier);
          input.v[2][lane] = f.c ^ (modifier & RJ_ARITH_DOT_NEG_C ? 0x80000000U : 0);
          input.v[3][lane] = Accumulate ? input.v[2][lane] : 0xdeadbeefU;
          expected[lane] = f.expected;
        }
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << scenario);
        for (int aliasing_offset : {3, 0, 1, 2}) {
          auto r = input;
          uint32_t want[Lanes];
          for (int lane = 0; lane < Lanes; ++lane) {
            want[lane] = expected[lane];
          }
          uint64_t flags = cpu | semantics | (scenario % 2 ? RJ_ARITH_FP16_OVFL : 0);
          int error = !Gfx12 && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                          ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                          : RJ_ARITH_SUCCESS;
          if constexpr (Accumulate) {
            if (aliasing_offset < 2 && !error) {
              // Aliasing old D with A/B changes the accumulator. A disjoint
              // execution checks snapshot semantics only; numerical correctness
              // is independently checked against the captures for D=C/disjoint.
              auto separate = input;
              for (int lane = 0; lane < Lanes; ++lane) {
                separate.v[3][lane] = input.v[aliasing_offset][lane];
              }
              uint32_t status = 0x80000055U;
              ASSERT_EQ(
                  (invoke_dot<true, true>(call, flags, exec_mask, mode, separate.v + 3, separate.v,
                                          separate.v + 1, separate.v + 2, &status)),
                  RJ_ARITH_SUCCESS);
              for (int lane = 0; lane < Lanes; ++lane) {
                want[lane] = separate.v[3][lane];
              }
            }
          }
          uint32_t status = scenario % 2 ? 0xffffff80U : 0x80000055U, old_status = status;
          uint32_t *status_ptr = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL || scenario % 2
                                     ? &status
                                     : nullptr;
          ASSERT_EQ((invoke_dot<Accumulate, true>(
                        call, flags, exec_mask, mode, exec_mask ? r.v + aliasing_offset : nullptr,
                        exec_mask ? r.v : nullptr, exec_mask ? r.v + 1 : nullptr,
                        exec_mask ? r.v + 2 : nullptr, status_ptr)),
                    error);
          // The inherited gfx12 DOT2 model leaves EXCP_FLAG_USER unchanged,
          // including invalid products and overflow.
          EXPECT_EQ(status, old_status);
          if (error) {
            r.expect_preserved(input, 0, 0);
            continue;
          }
          r.expect_preserved(input, aliasing_offset, 1);
          for (int lane = 0; lane < Lanes; ++lane) {
            EXPECT_EQ(r.v[aliasing_offset][lane],
                      ((exec_mask >> lane) & 1) ? want[lane] : input.v[aliasing_offset][lane]);
          }
        }
      }
    }
  }
}

// Differential coverage of the new SIMD implementation against the unchanged
// inherited scalar model; the independent captured cases above remain the oracle.
template <int Lanes, bool Bf16, bool Accumulate, typename Call> void check_dot2_simd(Call call) {
  std::minstd_rand random;
  auto word = [&] { return uint32_t(random() >> 7) ^ (uint32_t(random() >> 7) << 16); };
  auto cpus = supported_cpu_levels();
  for (int scenario = 0; scenario < 128; ++scenario) {
    Registers<4, Lanes> input;
    for (int reg = 0; reg < 4; ++reg) {
      for (int lane = 0; lane < Lanes; ++lane) {
        input.v[reg][lane] = word();
      }
    }
    uint64_t mode = Accumulate ? 0 : word() & dot2_modifiers;
    if (scenario % 4 == 0) {
      mode |= dpp_modes[1];
    }
    auto exec_mask = exec_mask_for_case<Lanes>(scenario % 5);
    int aliasing_offset = scenario % 4;
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, RJ_ARITH_SEMANTICS_LOOSE}) {
      if (!Bf16 && semantics == RJ_ARITH_SEMANTICS_LOOSE) {
        continue;
      }
      auto reference = input;
      uint32_t status = 0x80000055U;
      ASSERT_EQ((invoke_dot<Accumulate, true>(call, semantics, exec_mask, mode,
                                              reference.v + aliasing_offset, reference.v,
                                              reference.v + 1, reference.v + 2, &status)),
                RJ_ARITH_SUCCESS);
      for (uint64_t cpu : cpus) {
        auto actual = input;
        SCOPED_TRACE(::testing::Message() << scenario << '/' << cpu);
        ASSERT_EQ((invoke_dot<Accumulate, true>(call, cpu | semantics, exec_mask, mode,
                                                actual.v + aliasing_offset, actual.v, actual.v + 1,
                                                actual.v + 2, &status)),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < Lanes; ++lane) {
          if (semantics == RJ_ARITH_SEMANTICS_LOOSE && ((exec_mask >> lane) & 1)) {
            expect_f32(actual.v[aliasing_offset][lane],
                       std::bit_cast<float>(reference.v[aliasing_offset][lane]), semantics,
                       /*atol=*/0, /*rtol=*/1e-6);
          } else {
            ASSERT_EQ(actual.v[aliasing_offset][lane], reference.v[aliasing_offset][lane]) << lane;
          }
        }
        actual.expect_preserved(input, aliasing_offset, 1);
        EXPECT_EQ(status, 0x80000055U);
      }
    }
  }
}

// Check DOT2 arithmetic, modifiers, aliases, and semantics support for the supplied entry point.
template <int Lanes, bool Bf16, bool Gfx12, bool Accumulate = false, typename Call>
void check_dot2(Call call) {
  check_dot2_simd<Lanes, Bf16, Accumulate>(call);
  check_loose_dot<Lanes, Accumulate, true>(
      call, dot2_inputs<Lanes, Bf16, false>(),
      [](uint32_t actual, uint32_t a, uint32_t b, uint32_t c, uint32_t, uint64_t, uint32_t mode) {
        c &= mode & RJ_ARITH_DOT_ABS_C ? 0x7fffffffU : ~0U;
        double sum = std::bit_cast<float>(c ^ (mode & RJ_ARITH_DOT_NEG_C ? 0x80000000U : 0));
        for (int term = 0; term < 2; ++term) {
          int sa = term ? (mode & RJ_ARITH_DOT_HI_A_LOW ? 0 : 16)
                        : (mode & RJ_ARITH_DOT_LO_A_HIGH ? 16 : 0);
          int sb = term ? (mode & RJ_ARITH_DOT_HI_B_LOW ? 0 : 16)
                        : (mode & RJ_ARITH_DOT_LO_B_HIGH ? 16 : 0);
          uint16_t x = uint16_t(a >> sa), y = uint16_t(b >> sb);
          if (mode & (term ? RJ_ARITH_DOT_NEG_HI_A : RJ_ARITH_DOT_NEG_LO_A)) {
            x ^= 0x8000;
          }
          if (mode & (term ? RJ_ARITH_DOT_NEG_HI_B : RJ_ARITH_DOT_NEG_LO_B)) {
            y ^= 0x8000;
          }
          sum += double(Bf16 ? bf16_value(x) : f16_value(x)) *
                 double(Bf16 ? bf16_value(y) : f16_value(y));
        }
        expect_f32(actual, float(sum), RJ_ARITH_SEMANTICS_LOOSE, /*atol=*/2e-6, /*rtol=*/2e-7);
      },
      Accumulate ? 0 : dot2_modifiers, true);
  if constexpr (Gfx12) {
    check_exact_dot2<Lanes, Accumulate, Gfx12>(call, Bf16 ? gfx12_bf16_cases : gfx12_f16_cases);
  } else {
    check_exact_dot2<Lanes, Accumulate, Gfx12>(call, Bf16 ? gfx11_bf16_cases : gfx11_f16_cases);
  }
}

} // namespace

// Check DOT2 arithmetic, modifiers, aliases, and semantics support.
TEST(Dot2F32F16W32Gfx12, Arithmetic) {
  check_dot2<32, false, true, false>(rj_arith_v_dot2_f32_f16_w32_gfx12);
}

// Check DOT2 arithmetic, modifiers, aliases, and semantics support.
TEST(Dot2F32Bf16W32Gfx12, Arithmetic) {
  check_dot2<32, true, true, false>(rj_arith_v_dot2_f32_bf16_w32_gfx12);
}

// Check DOT2 arithmetic, modifiers, aliases, and semantics support.
TEST(Dot2F32F16W32Gfx11, Arithmetic) {
  check_dot2<32, false, false, false>(rj_arith_v_dot2_f32_f16_w32);
}

// Check DOT2 arithmetic, modifiers, aliases, and semantics support.
TEST(Dot2F32Bf16W32Gfx11, Arithmetic) {
  check_dot2<32, true, false, false>(rj_arith_v_dot2_f32_bf16_w32);
}

// Check DOT2 arithmetic, modifiers, aliases, and semantics support.
TEST(Dot2AccF32F16W32, Arithmetic) {
  check_dot2<32, false, false, true>(rj_arith_v_dot2acc_f32_f16_w32);
}

// Check DOT2 arithmetic, modifiers, aliases, and semantics support.
TEST(Dot2F32F16W64Gfx12, Arithmetic) {
  check_dot2<64, false, true, false>(rj_arith_v_dot2_f32_f16_w64_gfx12);
}

// Check DOT2 arithmetic, modifiers, aliases, and semantics support.
TEST(Dot2F32Bf16W64Gfx12, Arithmetic) {
  check_dot2<64, true, true, false>(rj_arith_v_dot2_f32_bf16_w64_gfx12);
}

// Check DOT2 arithmetic, modifiers, aliases, and semantics support.
TEST(Dot2F32F16W64Gfx11, Arithmetic) {
  check_dot2<64, false, false, false>(rj_arith_v_dot2_f32_f16_w64);
}

// Check DOT2 arithmetic, modifiers, aliases, and semantics support.
TEST(Dot2F32Bf16W64Gfx11, Arithmetic) {
  check_dot2<64, true, false, false>(rj_arith_v_dot2_f32_bf16_w64);
}

// Check DOT2 arithmetic, modifiers, aliases, and semantics support.
TEST(Dot2AccF32F16W64, Arithmetic) {
  check_dot2<64, false, false, true>(rj_arith_v_dot2acc_f32_f16_w64);
}
