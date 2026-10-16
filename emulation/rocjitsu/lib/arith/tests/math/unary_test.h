// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared unary math test scenarios and reference selection.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/capture_hash.h"
#include "common/contracts.h"
#include "common/cpu.h"
#include "common/dpp_reference.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "math/half_sfu_checkpoints.h"
#include "math/half_unary_reference.h"
#include "math/unary_goldens.h"
#include "math/unary_reference.h"
#include "rj_arith/rj_arith.h"
#include <bit>

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Check unary math results and semantics/status support across CPU levels for the supplied entry
// point.
template <int Lanes, bool Half, bool Gfx12, typename Call>
void check_unary(Call call, int op, bool iflag = false) {
  if constexpr (Half) {
    if (op <= 6) {
      // Exhaust all half encodings against the inherited scalar implementation.
      // Rotate policies across waves; independent model checkpoints below
      // remain the model oracle rather than treating SIMD agreement as proof.
      auto cpus = supported_cpu_levels();
      for (int base = 0; base < 65536; base += Lanes) {
        Registers<2, Lanes> input;
        for (int lane = 0; lane < Lanes; ++lane) {
          input.v[0][lane] = uint32_t(base + lane) * 0x10001U;
          input.v[1][lane] = 0xdeadbeefU;
        }
        int scenario = base / Lanes;
        uint64_t flags = RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
        if (scenario & 1) {
          flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
        }
        if (scenario & 2) {
          flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
        }
        if (scenario & 4) {
          flags |= RJ_ARITH_FP16_OVFL;
        }
        if (scenario & 8) {
          flags |= RJ_ARITH_FP_IEEE_MODE;
        }
        if (!Gfx12 && (scenario & 16)) {
          flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
        }
        uint32_t mode = ((scenario >> 2) & 3) << 6;
        if (scenario & 32) {
          mode |= RJ_ARITH_ALU_CLAMP;
        }
        if (scenario & 64) {
          mode |= RJ_ARITH_ALU_NEG_A;
        }
        if (scenario & 128) {
          mode |= RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D;
        }
        int dest = scenario & 1;
        WaveMask<Lanes> exec_mask =
            scenario & 2 ? WaveMask<Lanes>(0xaaaaaaaaaaaaaaaaULL) : ~WaveMask<Lanes>(0);
        auto expected = input;
        ASSERT_EQ(call(flags, exec_mask, mode, expected.v + dest, expected.v, nullptr),
                  RJ_ARITH_SUCCESS);
        for (uint64_t cpu : cpus) {
          auto actual = input;
          ASSERT_EQ(call(flags | cpu, exec_mask, mode, actual.v + dest, actual.v, nullptr),
                    RJ_ARITH_SUCCESS);
          for (int lane = 0; lane < Lanes; ++lane) {
            ASSERT_EQ(actual.v[dest][lane], expected.v[dest][lane])
                << base << '/' << cpu << '/' << lane;
          }
          actual.expect_preserved(input, dest, 1);
        }
      }
    }
  }
  if constexpr (!Half) {
    if (op <= 6 || iflag) {
      // SIMD rounding/roots/reciprocal and the unchanged scalar model must agree beyond the
      // independent captured values below, including exponent extremes and
      // NaN payloads. Use high RNG bits, with deterministic guest-policy cases.
      std::minstd_rand engine;
      auto word = [&] { return uint32_t(engine() >> 7) ^ (uint32_t(engine() >> 7) << 16); };
      auto cpus = supported_cpu_levels();
      for (int scenario = 0;
           scenario < ((op == 4 || op == 6) ? 196 : ((op == 5 || iflag) ? 192 : 128)); ++scenario) {
        Registers<2, Lanes> input;
        for (int lane = 0; lane < Lanes; ++lane) {
          input.v[0][lane] = word();
          if ((op == 5 || iflag) && scenario >= 128 && scenario < 160) {
            // First/last correction keys from inherited buckets 0, 1, 128,
            // 254 and 255, plus neighbors. The last two keys belong to an
            // adjacent bucket and must not match despite being in the same
            // SIMD load. The unchanged scalar model supplies expectations.
            static const uint32_t keys[] = {0x2922,   0x75df,   0x87a4,   0xf3df,
                                            0x400000, 0x407b49, 0x7f0320, 0x7f7074,
                                            0x7f8388, 0x7ff486, 0x7a4,    0x7f0388};
            uint32_t mantissa = keys[(lane + scenario) % 12] + (lane / 12) % 3 - 1;
            input.v[0][lane] = 0x3f800000U | mantissa;
          }
          input.v[1][lane] = 0xdeadbeefU;
        }
        if ((op == 5 || iflag) && scenario >= 160) {
          // Cover every coefficient index, particularly across register-table
          // boundaries, with low/high polynomial fractions and midpoint neighbors.
          static const uint32_t fractions[] = {0,       1,       0x1ffff, 0x20000,
                                               0x20001, 0x3fffe, 0x3ffff, 0x12345};
          for (int lane = 0; lane < Lanes; ++lane) {
            input.v[0][lane] = 0x3f800000U | (uint32_t(scenario - 160) << 18) | fractions[lane % 8];
          }
        }
        if ((op == 4 || op == 6) && scenario >= 128) {
          // Both exponent parities and every inherited polynomial segment.
          int shift = op == 4 ? 19 : 18;
          int segment = (scenario - 128) % (op == 4 ? 32 : 64);
          uint32_t midpoint = 1U << (shift - 1);
          const uint32_t fractions[] = {
              0,      1, midpoint - 1, midpoint, midpoint + 1, 2 * midpoint - 2, 2 * midpoint - 1,
              0x12345};
          for (int lane = 0; lane < Lanes; ++lane) {
            input.v[0][lane] = 0x3f800000U + (uint32_t(segment) << shift) + fractions[lane % 8];
          }
        }
        if ((op == 4 || op == 6) && scenario >= 192) {
          // Sparse inherited correction keys, spanning both exponent parities.
          // Random inputs alone seldom land exactly on these residuals.
          static const uint32_t sqrt_indices[] = {0x339a8,  0x53c8d,  0x60e8d,  0x7cf825,
                                                  0x7cfebc, 0xffd59b, 0xffd5a3, 0xffe39e};
          static const uint32_t rsq_indices[] = {0x1bbc,   0x7dce,   0x7f8f01, 0x7fc498,
                                                 0x80000c, 0x807e9c, 0xffa020, 0xfffe59};
          for (int lane = 0; lane < Lanes; ++lane) {
            input.v[0][lane] =
                0x3f800000U + (op == 4 ? sqrt_indices[lane % 8] : rsq_indices[lane % 8]);
          }
        }
        if ((op == 4 || op == 6) && scenario >= 194) {
          // Entire special-value blocks must skip polynomial work without losing
          // negative-input causes, NaN quieting or signed-zero behavior.
          static const uint32_t special[] = {0xbf800000U, 0x80000001U, 0xff800000U, 0x7f800000U,
                                             0,           0x80000000U, 0x7f812345U, 0xffc54321U};
          for (int lane = 0; lane < Lanes; ++lane) {
            input.v[0][lane] = special[lane % 8];
          }
        }
        input.v[0][scenario % Lanes] = scenario % 3 == 0   ? 0xff812345U
                                       : scenario % 3 == 1 ? 0x7f800000U
                                                           : 1U;
        uint64_t flags = ((iflag || (op == 4 && (scenario & 1)))
                              ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                              : RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) |
                         (scenario & 1 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0) |
                         (scenario & 4 ? RJ_ARITH_FP_IEEE_MODE : 0) |
                         (scenario & 2 ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                         (!Gfx12 && (scenario & 8) ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0);
        uint64_t mode = ((uint64_t(scenario) & 3) << 6) | (scenario & 4 ? RJ_ARITH_ALU_CLAMP : 0) |
                        (scenario & 8 ? RJ_ARITH_ALU_ABS_A : 0) |
                        (scenario & 16 ? RJ_ARITH_ALU_NEG_A : 0);
        if (scenario >= 128) {
          mode = 0; // Keep residual corrections observable without CLAMP/OMOD.
        }
        auto exec_mask = exec_mask_for_case<Lanes>(scenario % 5);
        int destination = scenario % 2;
        auto expected = input;
        uint32_t expected_status = 0x80000000U;
        ASSERT_EQ(
            call(flags, exec_mask, mode, expected.v + destination, expected.v, &expected_status),
            RJ_ARITH_SUCCESS);
        for (auto cpu : cpus) {
          auto actual = input;
          uint32_t status = 0x80000000U;
          SCOPED_TRACE(::testing::Message() << scenario << '/' << cpu);
          ASSERT_EQ(call(flags | cpu, exec_mask, mode, actual.v + destination, actual.v, &status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(status, expected_status);
          for (int lane = 0; lane < Lanes; ++lane) {
            ASSERT_EQ(actual.v[destination][lane], expected.v[destination][lane]) << lane;
          }
          actual.expect_preserved(input, destination, 1);
        }
      }
    }
  }
  const bool exact = op != 9;
  const bool reporting = !Half && (op == 4 || iflag);
  Registers<2, Lanes> random_input;
  std::minstd_rand random;
  for (int reg = 0; reg < 2; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      random_input.v[reg][lane] =
          Half ? uint32_t(random_f16_bits(random)) | (uint32_t(random_f16_bits(random)) << 16)
               : random_f32_bits(random);
    }
  }
  random_input.v[0][0] = Half ? 0x7e017e01U : 0x7fc00001U;
  random_input.v[0][1] = Half ? 0x7c007c00U : 0x7f800000U;
  for (auto cpu : supported_cpu_levels()) {
    for (auto semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 40; ++scenario) {
        for (int d : {0, 1}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << scenario << '/' << d);
          bool precise = exact && semantics != RJ_ARITH_SEMANTICS_LOOSE;
          auto input = random_input;
          uint64_t mode =
              (scenario & 1 ? RJ_ARITH_ALU_NEG_A : 0) | (scenario & 2 ? RJ_ARITH_ALU_ABS_A : 0) |
              ((uint64_t(scenario >> 2) & 3) << 6) | (scenario & 16 ? RJ_ARITH_ALU_CLAMP : 0);
          if constexpr (Half) {
            mode |=
                (scenario & 1 ? RJ_ARITH_ALU_HIGH_A : 0) | (scenario & 2 ? RJ_ARITH_ALU_HIGH_D : 0);
          }
          if (precise) {
            if constexpr (!Half) {
              // Empirical SFU outputs come from goldens, not a libm oracle.
              for (int lane = 0; lane < Lanes; ++lane) {
                input.v[0][lane] = unary_inputs[lane % 32];
              }
              mode &= ~(RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
            } else if (op == 7 || op == 8) {
              // Exactly representable exp/log identities; random noninteger
              // inputs below are only checked with the loose reference.
              const uint16_t identities[] = {0, 0x3c00, 0x4000, 0x4400};
              for (int lane = 0; lane < Lanes; ++lane) {
                input.v[0][lane] = uint32_t(identities[lane % 4]) * 0x10001U;
              }
              mode &= ~(RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
            }
          }
          if (scenario >= 33) {
            mode |= dpp_modes[scenario - 33];
          }
          auto actual = input;
          auto exec_mask = exec_mask_for_case<Lanes>(scenario);
          uint32_t status = 0x80000040U;
          int wanted = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL && !reporting
                           ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                       : !exact && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                           ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS
                           : RJ_ARITH_SUCCESS;
          ASSERT_EQ(call(cpu | semantics, exec_mask, mode, actual.v + d, actual.v, &status),
                    wanted);
          if (wanted) {
            actual.expect_preserved(input, 0, 0);
            EXPECT_EQ(status, 0x80000040U);
            continue;
          }
          // SQRT status gets dedicated literal cases below; other semantics
          // must not access status even when supplied with valid storage.
          if (semantics != RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL &&
              !(reporting && semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY)) {
            EXPECT_EQ(status, 0x80000040U);
          }
          uint32_t effective = uint32_t(mode);
          if (!Gfx12 && (Half || iflag || op < 4 || op >= 9)) {
            effective &= ~RJ_ARITH_ALU_OMOD_HALF;
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
            if constexpr (Half) {
              uint16_t expected = half_unary_reference(op, raw, effective, false);
              int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
              EXPECT_EQ(actual.v[d][lane] & ~(0xffffU << shift),
                        input.v[d][lane] & ~(0xffffU << shift));
              expect_f32(std::bit_cast<uint32_t>(f16_value(uint16_t(actual.v[d][lane] >> shift))),
                         f16_value(expected),
                         precise ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                                 : RJ_ARITH_SEMANTICS_LOOSE,
                         /*atol=*/0x1p-24f, /*rtol=*/1.1e-3f);
            } else {
              float expected;
              if (precise) {
                expected = std::bit_cast<float>(unary_goldens[op][source < 0 ? 0 : source % 32]);
                expected = omod_f32_reference(expected, effective);
                if (effective & RJ_ARITH_ALU_CLAMP) {
                  expected = !(expected > 0) ? 0 : std::min(expected, 1.0f);
                }
              } else {
                expected = unary_reference(op, std::bit_cast<float>(raw), effective);
              }
              expect_f32(actual.v[d][lane], expected,
                         precise ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                                 : RJ_ARITH_SEMANTICS_LOOSE,
                         /*atol=*/0.0f, /*rtol=*/2.5e-7f);
            }
          }
          actual.expect_preserved(input, d, 1);
        }
      }
    }
  }
  auto unchanged = random_input;
  uint64_t known = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_HALF |
                   RJ_ARITH_ALU_CLAMP | (Half ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D : 0);
  expect_alu_invalid_flags<true>(
      [&](auto f, auto m) {
        uint32_t status = 0x80000040U;
        int error =
            call(f, WaveMask<Lanes>(~WaveMask<Lanes>(0)), m, unchanged.v + 1, unchanged.v, &status);
        unchanged.expect_preserved(random_input, 0, 0);
        EXPECT_EQ(status, 0x80000040U);
        return error;
      },
      known);
  if constexpr (Half) {
    if (op >= 4 && op <= 6) {
      // Compact inherited rocjitsu oracle: 16 hashes validate every half
      // encoding, including all NaN payloads. Run once, not per scenario.
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
          hash = capture_hash_word(hash, uint16_t(r.v[1][lane]));
        }
        if ((base + Lanes) % 4096 == 0) {
          EXPECT_EQ(hash, half_sfu_checkpoints[op == 5   ? 0
                                               : op == 6 ? 1
                                                         : 2][(base + Lanes) / 4096 - 1]);
        }
      }
    }
  } else if (reporting && !iflag) {
    // Inherited rocjitsu SQRT classification reports INVALID for negative
    // inputs; it does not report arithmetic INEXACT for sqrt(2).
    const struct {
      uint32_t input, causes;
    } cases[] = {{0xbf800000U, RJ_ARITH_EXCEPTION_INVALID}, {0x40000000U, 0}, {0x40800000U, 0}};
    for (const auto &sample : cases) {
      SCOPED_TRACE(sample.input);
      Registers<2, Lanes> r;
      for (int lane = 0; lane < Lanes; ++lane) {
        r.v[0][lane] = sample.input;
      }
      uint32_t status = 0x80000040U;
      ASSERT_EQ(call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, WaveMask<Lanes>(1), 0, r.v + 1, r.v,
                     &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(status, 0x80000040U | sample.causes);
    }
  }
  if (iflag) {
    // Existing capture model: zero/subnormal inputs set DIV0. gfx12 CLAMP
    // suppresses it; gfx11 keeps it. Only participating lanes contribute.
    for (auto cpu : supported_cpu_levels()) {
      for (auto semantics : semantics_modes()) {
        for (bool clamp : {false, true}) {
          for (int scenario = 0; scenario < 5; ++scenario) {
            Registers<2, Lanes> r;
            for (int lane = 0; lane < Lanes; ++lane) {
              r.v[0][lane] = 0x3f800000U;
            }
            r.v[0][0] = 0;
            r.v[0][Lanes - 1] = 1;
            auto exec_mask = exec_mask_for_case<Lanes>(scenario);
            uint32_t status = 0x80000000U;
            ASSERT_EQ(
                call(cpu | semantics, exec_mask, clamp ? RJ_ARITH_ALU_CLAMP : 0, r.v, r.v, &status),
                RJ_ARITH_SUCCESS);
            bool report = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL ||
                          semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY;
            bool exceptional = (exec_mask & 1) || (Gfx12 && ((exec_mask >> (Lanes - 1)) & 1));
            EXPECT_EQ(status,
                      0x80000000U | (report && exceptional && (!Gfx12 || !clamp) ? 0x40U : 0));
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
