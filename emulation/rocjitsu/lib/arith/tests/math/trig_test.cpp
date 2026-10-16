// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for sine/cosine inputs, modifiers, and exact/loose result checks.

#include "common/capture_hash.h"
#include "common/contracts.h"
#include "common/cpu.h"
#include "common/dpp_reference.h"
#include "common/floating_point.h"
#include "common/half_reference.h"
#include "common/integer.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "math/trig_golden.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {
using namespace rocjitsu::arith::test;

// Check sine/cosine arithmetic, modifiers, and FP-status expectations for the supplied entry
// point.
template <int Lanes, bool Half, bool Gfx12, class Call> void check_trig(Call call, bool cosine) {
  // The inherited rocjitsu digest independently validates the complete half
  // domain. Retain its results as a compact runtime oracle for modifier/DPP
  // checks; never use an unchecked scalar rj-arith call as a numerical reference.
  std::vector<uint16_t> half_table;
  if constexpr (Half) {
    half_table.resize(65536);
    uint64_t hash = capture_hash_seed;
    Registers<2, Lanes> r;
    for (int base = 0; base < 65536; base += Lanes) {
      for (int lane = 0; lane < Lanes; ++lane) {
        r.v[0][lane] = base + lane;
      }
      ASSERT_EQ(call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | RJ_ARITH_FP_IEEE_MODE,
                     WaveMask<Lanes>(~WaveMask<Lanes>(0)), 0, r.v + 1, r.v, nullptr),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < Lanes; ++lane) {
        half_table[base + lane] = uint16_t(r.v[1][lane]);
        hash = capture_hash_word(hash, half_table[base + lane]);
      }
    }
    ASSERT_EQ(hash, cosine ? 0xc6c3112c531e4391ULL : 0xb6245e65a23f99f9ULL);
  }
  Registers<2, Lanes> random_input;
  std::minstd_rand random;
  for (int reg = 0; reg < 2; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      random_input.v[reg][lane] =
          Half ? random_word(random)
               : bit_cast<uint32_t>(float(int(random() >> 7) - int(1U << 23)) / float(1U << 23));
    }
  }
  for (auto cpu : supported_cpu_levels()) {
    for (auto semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 40; ++scenario) {
        for (int d : {0, 1}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << scenario << '/' << d);
          bool exact = semantics != RJ_ARITH_SEMANTICS_LOOSE;
          auto input = random_input;
          int golden_count = int(sizeof(trig_golden) / sizeof(trig_golden[0]));
          if constexpr (!Half) {
            if (exact) {
              for (int lane = 0; lane < Lanes; ++lane) {
                input.v[0][lane] = trig_golden[(lane + scenario) % golden_count][0];
              }
            }
          }
          uint64_t mode = ((uint64_t(scenario) & 3) << 6) | (scenario & 4 ? RJ_ARITH_ALU_CLAMP : 0);
          if (Half || !exact) {
            mode |=
                (scenario & 8 ? RJ_ARITH_ALU_NEG_A : 0) | (scenario & 16 ? RJ_ARITH_ALU_ABS_A : 0);
          }
          if (Half) {
            mode |=
                (scenario & 1 ? RJ_ARITH_ALU_HIGH_A : 0) | (scenario & 2 ? RJ_ARITH_ALU_HIGH_D : 0);
          }
          if (scenario >= 33) {
            mode |= dpp_modes[scenario - 33];
          }
          auto actual = input;
          auto exec_mask = exec_mask_for_case<Lanes>(scenario);
          uint32_t status = 0x80000000U;
          int wanted = !Gfx12 && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                           ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                           : RJ_ARITH_SUCCESS;
          ASSERT_EQ(call(cpu | semantics, exec_mask, mode, actual.v + d, actual.v, &status),
                    wanted);
          if (wanted) {
            actual.expect_preserved(input, 0, 0);
            EXPECT_EQ(status, 0x80000000U);
            continue;
          }
          EXPECT_EQ(status & 0xffffff80U, 0x80000000U);
          if (semantics != RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
            EXPECT_EQ(status, 0x80000000U);
          }
          for (int lane = 0; lane < Lanes; ++lane) {
            int source = lane;
            bool enabled = mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16)
                               ? dpp_source(mode, exec_mask, lane, source)
                               : bool((exec_mask >> lane) & 1);
            if (!enabled) {
              EXPECT_EQ(actual.v[d][lane], input.v[d][lane]);
              continue;
            }
            uint32_t raw = source < 0 ? 0 : input.v[0][source];
            int shift = Half && (mode & RJ_ARITH_ALU_HIGH_D) ? 16 : 0;
            double expected;
            if constexpr (Half) {
              uint16_t h = uint16_t(raw >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
              if (mode & RJ_ARITH_ALU_ABS_A) {
                h &= 0x7fffU;
              }
              if (mode & RJ_ARITH_ALU_NEG_A) {
                h ^= 0x8000U;
              }
              expected = half_value(half_table[h]);
            } else if (exact) {
              expected = bit_cast<float>(
                  source < 0 ? (cosine ? 0x3f800000U : 0)
                             : trig_golden[(source + scenario) % golden_count][cosine ? 2 : 1]);
            } else {
              double x = bit_cast<float>(raw);
              if (mode & RJ_ARITH_ALU_ABS_A) {
                x = std::abs(x);
              }
              if (mode & RJ_ARITH_ALU_NEG_A) {
                x = -x;
              }
              double angle = x * 0x1.921fb54442d18p+2;
              expected = cosine ? std::cos(angle) : std::sin(angle);
            }
            int omod = Gfx12 || !Half ? int((mode >> 6) & 3) : 0;
            if (omod) {
              if (std::abs(expected) < (Half ? 0x1p-14 : 0x1p-126)) {
                expected = 0;
              }
              expected = std::ldexp(expected, omod == 3 ? -1 : omod);
              if (std::abs(expected) < (Half ? 0x1p-14 : 0x1p-126)) {
                expected = 0;
              }
            }
            if (mode & RJ_ARITH_ALU_CLAMP) {
              expected = !(expected > 0) ? 0 : expected > 1 ? 1 : expected;
            }
            if constexpr (Half) {
              EXPECT_EQ(actual.v[d][lane] & ~(0xffffU << shift),
                        input.v[d][lane] & ~(0xffffU << shift));
              expected = half_value(half_bits(expected));
              expect_f32(bit_cast<uint32_t>(f16_value(uint16_t(actual.v[d][lane] >> shift))),
                         float(expected),
                         exact ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                               : RJ_ARITH_SEMANTICS_LOOSE,
                         /*atol=*/0.0005f, /*rtol=*/0.0f);
            } else {
              expect_f32(actual.v[d][lane], float(expected),
                         exact ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                               : RJ_ARITH_SEMANTICS_LOOSE,
                         /*atol=*/8e-7f, /*rtol=*/0.0f);
            }
          }
          actual.expect_preserved(input, d, 1);
        }
      }
    }
  }
  auto unchanged = random_input;
  expect_alu_invalid_flags<true>(
      [&](auto flags, auto mode) {
        uint32_t status = 0x80000000U;
        int error = call(flags, WaveMask<Lanes>(~WaveMask<Lanes>(0)), mode, unchanged.v + 1,
                         unchanged.v, &status);
        unchanged.expect_preserved(random_input, 0, 0);
        EXPECT_EQ(status, 0x80000000U);
        return error;
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP |
          (Half ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D : 0));
  if constexpr (Gfx12) {
    // Existing gfx1201 exception captures: full half domain or a fixed FP32
    // sign/exponent/mantissa corpus. Inactive signaling NaNs contribute nothing.
    uint64_t hash = capture_hash_seed;
    Registers<2, Lanes> r;
    for (int lane = 0; lane < Lanes; ++lane) {
      r.v[0][lane] = Half ? 0x7c01 : 0x7f800001;
    }
    for (int i = 0; i < 65536; ++i) {
      r.v[0][0] = Half ? uint32_t(i) : (uint32_t(i) << 16) | ((uint32_t(i) * 0x9e37U) & 0xffffU);
      uint32_t status = 0xffffff80U;
      ASSERT_EQ(call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, WaveMask<Lanes>(1), 0, r.v + 1, r.v,
                     &status),
                RJ_ARITH_SUCCESS);
      hash = capture_hash_word(hash, status & 127);
    }
    static const uint64_t hashes[] = {0x913b970847c228c5ULL, 0xe20e76a163b6a725ULL,
                                      0xf472e5e37b9f5087ULL, 0x07408432d8c13fa5ULL};
    EXPECT_EQ(hash, hashes[(Half ? 0 : 2) + int(cosine)]);
  }
}
} // namespace

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_sin_f32_w32, Arithmetic) { check_trig<32, false, false>(rj_arith_v_sin_f32_w32, false); }

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_cos_f32_w32, Arithmetic) { check_trig<32, false, false>(rj_arith_v_cos_f32_w32, true); }

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_sin_f32_w32_gfx12, Arithmetic) {
  check_trig<32, false, true>(rj_arith_v_sin_f32_w32_gfx12, false);
}

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_cos_f32_w32_gfx12, Arithmetic) {
  check_trig<32, false, true>(rj_arith_v_cos_f32_w32_gfx12, true);
}

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_sin_f16_w32, Arithmetic) { check_trig<32, true, false>(rj_arith_v_sin_f16_w32, false); }

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_cos_f16_w32, Arithmetic) { check_trig<32, true, false>(rj_arith_v_cos_f16_w32, true); }

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_sin_f16_w32_gfx12, Arithmetic) {
  check_trig<32, true, true>(rj_arith_v_sin_f16_w32_gfx12, false);
}

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_cos_f16_w32_gfx12, Arithmetic) {
  check_trig<32, true, true>(rj_arith_v_cos_f16_w32_gfx12, true);
}

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_sin_f32_w64, Arithmetic) { check_trig<64, false, false>(rj_arith_v_sin_f32_w64, false); }

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_cos_f32_w64, Arithmetic) { check_trig<64, false, false>(rj_arith_v_cos_f32_w64, true); }

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_sin_f32_w64_gfx12, Arithmetic) {
  check_trig<64, false, true>(rj_arith_v_sin_f32_w64_gfx12, false);
}

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_cos_f32_w64_gfx12, Arithmetic) {
  check_trig<64, false, true>(rj_arith_v_cos_f32_w64_gfx12, true);
}

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_sin_f16_w64, Arithmetic) { check_trig<64, true, false>(rj_arith_v_sin_f16_w64, false); }

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_cos_f16_w64, Arithmetic) { check_trig<64, true, false>(rj_arith_v_cos_f16_w64, true); }

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_sin_f16_w64_gfx12, Arithmetic) {
  check_trig<64, true, true>(rj_arith_v_sin_f16_w64_gfx12, false);
}

// Check sine/cosine arithmetic, modifiers, and FP-status expectations.
TEST(v_cos_f16_w64_gfx12, Arithmetic) {
  check_trig<64, true, true>(rj_arith_v_cos_f16_w64_gfx12, true);
}
