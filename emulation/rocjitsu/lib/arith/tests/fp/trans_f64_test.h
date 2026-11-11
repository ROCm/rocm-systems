// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared FP64 RCP/RSQ/SQRT test scenarios and reference selection.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "fp/reference.h"
#include "fp/trans_f64_goldens.h"
#include "math/omod_reference.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Loose results follow the host's correctly rounded functions.
inline double trans_f64_reference(int op, double x) {
  return op == 0 ? 1 / x : op == 1 ? 1 / std::sqrt(x) : std::sqrt(x);
}

// Check FP64 RCP (op 0), RSQ (op 1) or SQRT (op 2) results and semantics/status support
// across CPU levels for the supplied entry point.
template <int Lanes, bool Gfx12, typename Call> void check_trans_f64(Call call, int op) {
  {
    // SIMD levels and the unchanged scalar model must agree beyond the
    // independent captured values below, including exponent extremes and NaN
    // payloads. Use high RNG bits, with deterministic guest-policy cases.
    std::minstd_rand engine;
    auto word = [&] {
      return uint64_t(engine() >> 7) ^ (uint64_t(engine() >> 7) << 24) ^
             (uint64_t(engine() >> 7) << 40);
    };
    auto cpus = supported_cpu_levels();
    // RSQ has a segment table per exponent parity; SQRT splits 32 segments the same way.
    const int bits = op == 2 ? 28 : 29;
    const int segments = op == 1 ? 64 : 32;
    for (int scenario = 0; scenario < 128 + segments + 2; ++scenario) {
      Registers<4, Lanes> input;
      for (int lane = 0; lane < Lanes; ++lane) {
        write_fp(input.v, lane, 64, word());
        write_fp(input.v + 2, lane, 64, word());
      }
      if (scenario >= 128 && scenario < 128 + segments) {
        // Every datapath segment at both exponent parities, with low/high
        // segment offsets and midpoint neighbors.
        const int index = scenario - 128;
        const uint64_t parity = op == 0 ? 0 : uint64_t(index) / (op == 1 ? 32 : 16);
        const uint64_t segment = op == 0 ? uint64_t(index) : uint64_t(index) % (op == 1 ? 32 : 16);
        static const uint64_t offsets[] = {0,        1,        0x7fffff, 0x800000,
                                           0x800001, 0xfffffe, 0xffffff, 0x12345};
        for (int lane = 0; lane < Lanes; ++lane) {
          const uint64_t significand = (segment << 24) | offsets[lane % 8];
          const uint64_t low = lane & 8 ? (uint64_t(1) << (52 - bits)) - 1 : 0;
          write_fp(input.v, lane, 64, ((1023 + parity) << 52) | (significand << (52 - bits)) | low);
        }
      }
      if (scenario == 128 + segments) {
        // Power-of-two neighbors: exact powers, fractions with only bits below
        // the top 29, every fraction bit, and subnormal powers whose reciprocal
        // overflows, across the exponent range.
        static const uint64_t neighbors[] = {
            0x3ff0000000000000, 0x3ff0000000000001, 0x3fefffffffffffff, 0x3fe0000000000001,
            0x3fdfffffffffffff, 0x3ff00000007fffff, 0x3ff0000000800000, 0x4000000000000001,
            0x7fe0000000000001, 0x7fefffffffffffff, 0x0010000000000001, 0x000fffffffffffff,
            0x0000000000000001, 0x0000000000000003, 0x0000010000000001, 0x0008000000000001};
        for (int lane = 0; lane < Lanes; ++lane) {
          write_fp(input.v, lane, 64, neighbors[lane % 16]);
        }
      }
      if (scenario == 128 + segments + 1) {
        // Entire special-value blocks must skip datapath work without losing
        // NaN quieting, negative-source NaNs or signed zeros.
        static const uint64_t special[] = {
            0xbff0000000000000, 0x8000000000000001, 0xfff0000000000000, 0x7ff0000000000000, 0,
            0x8000000000000000, 0x7ff0000012345678, 0xfff8000000054321};
        for (int lane = 0; lane < Lanes; ++lane) {
          // Negative finite sources are ordinary for RCP.
          write_fp(input.v, lane, 64, special[op == 0 ? 2 + lane % 6 : lane % 8]);
        }
      }
      // Keep deliberate datapath checkpoints intact. Mixed special values
      // belong to the random scenarios, not the finite checkpoint blocks.
      if (scenario < 128) {
        write_fp(input.v, scenario % Lanes, 64,
                 scenario % 3 == 0   ? 0xfff0000000012345ULL
                 : scenario % 3 == 1 ? 0x7ff0000000000000ULL
                                     : 1ULL);
      }
      uint64_t flags = RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS |
                       (scenario & 1 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0) |
                       (scenario & 4 ? RJ_ARITH_FP_IEEE_MODE : 0) |
                       (scenario & 2 ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                       (!Gfx12 && (scenario & 8) ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0) |
                       (uint64_t(scenario >> 5) & 3) << 24;
      uint64_t mode = ((uint64_t(scenario) & 3) << 6) | (scenario & 4 ? RJ_ARITH_ALU_CLAMP : 0) |
                      (scenario & 8 ? RJ_ARITH_ALU_ABS_A : 0) |
                      (scenario & 16 ? RJ_ARITH_ALU_NEG_A : 0);
      if (scenario >= 128) {
        mode = 0; // Keep the datapath result observable without CLAMP/OMOD.
      }
      auto exec_mask = exec_mask_for_case<Lanes>(scenario % 5);
      // A destination at 1 overlaps the source's high word with the result's low word.
      int destination = 1 + scenario % 2;
      auto expected = input;
      ASSERT_EQ(call(flags, exec_mask, mode, expected.v + destination, expected.v, nullptr),
                RJ_ARITH_SUCCESS);
      for (auto cpu : cpus) {
        auto actual = input;
        uint32_t status = 0x80000000U;
        SCOPED_TRACE(::testing::Message() << scenario << '/' << cpu);
        ASSERT_EQ(call(flags | cpu, exec_mask, mode, actual.v + destination, actual.v, &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(status, 0x80000000U);
        for (int lane = 0; lane < Lanes; ++lane) {
          ASSERT_EQ(read_fp(actual.v + destination, lane, 64),
                    read_fp(expected.v + destination, lane, 64))
              << lane;
        }
        actual.expect_preserved(input, destination, 2);
      }
    }

    // Every significand in one segment window, starting a segment whose linear
    // coefficient is odd: the linear product then takes every residue below its
    // rounding position, including single-residue rounding cases that random
    // sources rarely reach.
    const uint64_t window = op == 0 ? 3 : op == 1 ? 1 : 0;
    for (uint64_t base = 0; base < (uint64_t(1) << 21); base += Lanes) {
      Registers<4, Lanes> input;
      for (int lane = 0; lane < Lanes; ++lane) {
        write_fp(input.v, lane, 64,
                 0x3ff0000000000000ULL | (((window << 24) + base + uint64_t(lane)) << (52 - bits)));
      }
      const uint64_t flags = RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
      auto expected = input;
      ASSERT_EQ(call(flags, ~WaveMask<Lanes>(0), 0, expected.v + 2, expected.v, nullptr),
                RJ_ARITH_SUCCESS);
      for (auto cpu : cpus) {
        auto actual = input;
        ASSERT_EQ(call(flags | cpu, ~WaveMask<Lanes>(0), 0, actual.v + 2, actual.v, nullptr),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < Lanes; ++lane) {
          ASSERT_EQ(read_fp(actual.v + 2, lane, 64), read_fp(expected.v + 2, lane, 64))
              << base << '/' << cpu << '/' << lane;
        }
      }
    }
  }
  Registers<4, Lanes> random_input;
  std::minstd_rand random;
  for (int reg = 0; reg < 4; reg += 2) {
    for (int lane = 0; lane < Lanes; ++lane) {
      write_fp(random_input.v + reg, lane, 64, random_fp(random, 64));
    }
  }
  write_fp(random_input.v, 0, 64, 0x7ff8000000000001ULL);
  write_fp(random_input.v, 1, 64, 0x7ff0000000000000ULL);
  for (auto cpu : supported_cpu_levels()) {
    for (auto semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 32; ++scenario) {
        for (int d : {1, 2}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << scenario << '/' << d);
          bool precise = semantics != RJ_ARITH_SEMANTICS_LOOSE;
          auto input = random_input;
          uint64_t mode =
              (scenario & 1 ? RJ_ARITH_ALU_NEG_A : 0) | (scenario & 2 ? RJ_ARITH_ALU_ABS_A : 0) |
              ((uint64_t(scenario >> 2) & 3) << 6) | (scenario & 16 ? RJ_ARITH_ALU_CLAMP : 0);
          if (precise) {
            // Empirical math outputs come from goldens, not a libm oracle.
            for (int lane = 0; lane < Lanes; ++lane) {
              write_fp(input.v, lane, 64, trans_f64_inputs[lane % 32]);
            }
            mode &= ~(RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
          }
          auto actual = input;
          auto exec_mask = exec_mask_for_case<Lanes>(scenario);
          uint32_t status = 0x80000040U;
          int wanted = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                           ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                           : RJ_ARITH_SUCCESS;
          ASSERT_EQ(call(cpu | semantics, exec_mask, mode, actual.v + d, actual.v, &status),
                    wanted);
          // No semantics reports FP status; none may access the supplied storage.
          EXPECT_EQ(status, 0x80000040U);
          if (wanted) {
            actual.expect_preserved(input, 0, 0);
            continue;
          }
          // gfx11 disables OMOD when output denormals are preserved, as here.
          uint32_t effective = uint32_t(mode);
          if (!Gfx12) {
            effective &= ~RJ_ARITH_ALU_OMOD_MASK;
          }
          for (int lane = 0; lane < Lanes; ++lane) {
            if (!((exec_mask >> lane) & 1)) {
              EXPECT_EQ(read_fp(actual.v + d, lane, 64), read_fp(input.v + d, lane, 64));
              continue;
            }
            double expected;
            if (precise) {
              expected = fp_value(trans_f64_goldens[op][lane % 32], 64);
            } else {
              expected = trans_f64_reference(
                  op, modified_fp(fp_value(read_fp(input.v, lane, 64), 64), mode, 0));
            }
            expected = omod_f64_reference(expected, effective);
            if (effective & RJ_ARITH_ALU_CLAMP) {
              expected = !(expected > 0) ? 0 : std::min(expected, 1.0);
            }
            expect_fp(read_fp(actual.v + d, lane, 64), fp_bits(expected, 64), 64,
                      precise ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                              : RJ_ARITH_SEMANTICS_LOOSE,
                      /*atol=*/0, /*rtol=*/1e-15);
          }
          actual.expect_preserved(input, d, 2);
        }
      }
    }
  }
  auto unchanged = random_input;
  uint64_t known =
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP;
  expect_alu_invalid_flags<false>(
      [&](auto f, auto m) {
        uint32_t status = 0x80000040U;
        int error =
            call(f, WaveMask<Lanes>(~WaveMask<Lanes>(0)), m, unchanged.v + 1, unchanged.v, &status);
        unchanged.expect_preserved(random_input, 0, 0);
        EXPECT_EQ(status, 0x80000040U);
        return error;
      },
      known);
}

} // namespace rocjitsu::arith::test
