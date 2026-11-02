// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Mixed FP16/FP32 fused arithmetic and selected-halfword results.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/elementwise.h"
#include "fp/fma_dpp_test.h"
#include "fp/mixed_fma_reference.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith::test {

// Oracle result kind: 0 = FP32, 1 = low FP16, 2 = high FP16.
// Fill active destination elements from the independent arithmetic reference.
template <int Kind, int Lanes>
void compute_mixed_fma_expected(const rocjitsu::arith::test::Registers<10, Lanes> &input,
                                uint64_t exec_mask, uint64_t instruction_flags, int aliasing_offset,
                                rocjitsu::arith::test::Registers<10, Lanes> &expected_registers,
                                bool gfx12) {
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    // DPP8 reverses each group; FI controls reads from inactive sources.
    int source_lane = instruction_flags & RJ_ARITH_DPP8 ? (lane & ~7) + 7 - (lane & 7) : lane;
    uint64_t a = rocjitsu::arith::test::read_fp(input.v, source_lane, 32, 0);
    if ((instruction_flags & RJ_ARITH_DPP8) && !(instruction_flags & RJ_ARITH_DPP_FI) &&
        !((exec_mask >> source_lane) & 1)) {
      a = 0;
    }
    uint64_t b = rocjitsu::arith::test::read_fp(input.v + 2, lane, 32, 0);
    uint64_t c = rocjitsu::arith::test::read_fp(input.v + 4, lane, 32, 0);
    // Evaluate the test-side integer oracle without production arithmetic helpers.
    uint64_t expected = rocjitsu::arith::test::mixed_fma_reference::evaluate(
        Kind, uint32_t(a), uint32_t(b), uint32_t(c), input.v[aliasing_offset][lane],
        uint32_t(instruction_flags), false, false, 0, !gfx12);
    if constexpr (Kind != 0) {
      expected = uint16_t(expected >> (Kind == 2 ? 16 : 0));
    }
    rocjitsu::arith::test::write_fp(expected_registers.v + aliasing_offset, lane,
                                    Kind == 0 ? 32 : 16, expected, Kind == 2 ? 1 : 0);
  }
}

// Supplement the independent oracle with broad SIMD/scalar differential cases.
// The scalar implementation is the optimization reference, including
// directed rounding and NaN payloads that loose comparisons do not constrain.
template <int Kind, int Lanes, typename Call> void check_mixed_fma_simd(Call call, bool gfx12) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  const auto levels = supported_cpu_levels();
  std::minstd_rand random;
  const uint32_t boundaries[] = {0,           0x80000000U, 1,           0x007fffffU, 0x00800000U,
                                 0x7f7fffffU, 0x7f800000U, 0xff800000U, 0x7f800123U, 0x7fc12345U,
                                 0x3f800001U, 0xbf800000U, 0x387fe000U, 0x387ff000U, 0x38800000U,
                                 0x33000000U, 0x477fe000U, 0x477ff000U};
  for (int sample = 0; sample < 256; ++sample) {
    Registers<4, Lanes> input;
    uint32_t instruction_flags = uint32_t(random()) >> 15;
    instruction_flags &= RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C |
                         RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C |
                         RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B |
                         RJ_ARITH_ALU_HIGH_C;
    for (int operand = 0; operand < 3; ++operand) {
      if ((sample >> operand) & 1) {
        instruction_flags |= RJ_ARITH_MIX_F16_A << operand;
      }
    }
    for (int reg = 0; reg < 4; ++reg) {
      for (int lane = 0; lane < Lanes; ++lane) {
        uint32_t raw = (uint32_t(random()) >> 15) | ((uint32_t(random()) >> 15) << 16);
        uint32_t bits = raw;
        if (sample % 3 == 0) {
          bits = reg < 3 && (instruction_flags & (RJ_ARITH_MIX_F16_A << reg))
                     ? (raw & 0x83ff83ffU) | 0x38003800U
                     : (raw & 0x807fffffU) | 0x3f000000U;
        }
        if (sample % 3 == 1 && lane % 7 == 0) {
          bits = boundaries[(sample + reg + lane) % (sizeof(boundaries) / sizeof(*boundaries))];
        }
        input.v[reg][lane] = bits;
      }
    }
    if ((sample & 7) == 0) {
      // Halfway +/- tiny addends expose double rounding; cancellation and tiny
      // products distinguish exact zero from rounded underflow. All sources
      // here are FP32, so the same cases exercise both output precisions.
      const uint32_t triples[][3] = {
          {0x3f800000U, 0x3f801000U, 0x17800000U}, {0x3f800000U, 0x3f801000U, 0x97800000U},
          {0x3f800000U, 0x387ff000U, 0x17800000U}, {0x3f800000U, 0x387ff000U, 0x97800000U},
          {0x3f800001U, 0x3f7ffffeU, 0xbf800000U}, {0x00800000U, 0x00800000U, 0x00000000U},
          {0x7f7fffffU, 0x7f7fffffU, 0xff7fffffU}, {0x80000000U, 0x3f800000U, 0x80000000U}};
      for (int lane = 0; lane < 8; ++lane) {
        for (int reg = 0; reg < 3; ++reg) {
          input.v[reg][lane] = triples[lane][reg];
        }
      }
    }
    uint64_t flags =
        RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | (uint64_t((sample / 8) % 4) << 24);
    if (sample & 32) {
      flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
    }
    if (sample & 64) {
      flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
    }
    if (sample & 128) {
      flags |= RJ_ARITH_FP16_OVFL;
    }
    flags |= sample % 3 == 0   ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16
             : sample % 3 == 1 ? RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16
                               : 0;
    if (!gfx12 && sample % 5 == 0) {
      flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
    }
    if (sample % 7 == 0) {
      flags |= RJ_ARITH_FP_IEEE_MODE;
    }
    if (gfx12 && Kind != 0 && sample % 2) {
      flags = (flags & ~RJ_ARITH_SEMANTICS_MASK) | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
    }
    Mask exec_mask = sample % 3 == 0 ? ~Mask(0) : Mask(0xaaaaaaaaaaaaaaaaULL);
    int aliasing_offset = sample % 4;
    auto expected = input;
    uint32_t expected_status = 0x80000000U;
    ASSERT_EQ(call(flags, exec_mask, instruction_flags, expected.v + aliasing_offset, expected.v,
                   expected.v + 1, expected.v + 2, &expected_status),
              RJ_ARITH_SUCCESS);
    for (uint64_t cpu : levels) {
      if (cpu == RJ_ARITH_CPU_BASELINE) {
        continue;
      }
      SCOPED_TRACE(::testing::Message() << sample << '/' << cpu << '/' << instruction_flags);
      auto actual = input;
      uint32_t status = 0x80000000U;
      ASSERT_EQ(call(flags | cpu, exec_mask, instruction_flags, actual.v + aliasing_offset,
                     actual.v, actual.v + 1, actual.v + 2, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(status, expected_status);
      for (int lane = 0; lane < Lanes; ++lane) {
        ASSERT_EQ(actual.v[aliasing_offset][lane], expected.v[aliasing_offset][lane]) << lane;
      }
      actual.expect_preserved(input, aliasing_offset, 1);
    }
  }
}

// Exercise this instruction contract for either wave size.
template <int Kind, int Lanes, typename Call> void check_mixed_fma(Call call, bool gfx12) {
  check_mixed_fma_simd<Kind, Lanes>(call, gfx12);
  // 2^-24 (FP16 subnormal) times 2^24 is exactly one unless FP16 input flushes.
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (bool preserve : {false, true}) {
        Registers<4, Lanes> r;
        r.v[0][0] = 0x0001;
        r.v[1][0] = 0x4b800000;
        r.v[2][0] = 0;
        uint64_t guest =
            preserve ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16
                     : RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16;
        ASSERT_EQ(call(cpu | semantics | guest, 1, RJ_ARITH_MIX_F16_A, r.v + 3, r.v, r.v + 1,
                       r.v + 2, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(read_fp(r.v + 3, 0, Kind == 0 ? 32 : 16, Kind == 2 ? 1 : 0),
                  preserve ? (Kind == 0 ? 0x3f800000U : 0x3c00U) : 0U);
      }
    }
  }

  if constexpr (Kind == 0) {
    check_fma_dpp16<Lanes>(call);
  }
  static_assert(Kind >= 0 && Kind <= 2);
  static_assert(Lanes == 32 || Lanes == 64);
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = rocjitsu::arith::test::make_fp_inputs<32, Lanes>();

  rocjitsu::arith::test::expect_invalid_fma_requests(
      [&](uint64_t flags, uint64_t instruction_flags) {
        Registers r = input;
        int aliasing_offset = 6;
        Mask exec_mask = ~Mask(0);
        uint32_t fp_status = 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;

        int result = call(flags, exec_mask, instruction_flags, r.v + aliasing_offset, r.v, r.v + 2,
                          r.v + 4, &fp_status);
        r.expect_preserved(input, 0, 0);
        EXPECT_EQ(fp_status, 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0);
        EXPECT_EQ(call(flags, exec_mask, instruction_flags, nullptr, nullptr, nullptr, nullptr,
                       &fp_status),
                  result);
        return result;
      });

  // Default guest rounding and normal fractional inputs: the existing rocjitsu
  // arithmetic model reduces to IEEE arithmetic here. Special values are checked
  // by classification. No production helper or rj-arith result is a numeric oracle.
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t instruction_flags :
           {uint64_t(0), uint64_t(RJ_ARITH_ALU_NEG_A), uint64_t(RJ_ARITH_ALU_CLAMP),
            uint64_t(RJ_ARITH_MIX_F16_A | RJ_ARITH_ALU_HIGH_A), uint64_t(RJ_ARITH_MIX_F16_B),
            uint64_t(RJ_ARITH_MIX_F16_B | RJ_ARITH_ALU_HIGH_B), uint64_t(RJ_ARITH_MIX_F16_C),
            uint64_t(RJ_ARITH_MIX_F16_C | RJ_ARITH_ALU_HIGH_C),
            uint64_t(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT)),
            uint64_t(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT) | RJ_ARITH_DPP_FI |
                     RJ_ARITH_ALU_NEG_A)}) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : {6, 0, 2, 4}) {
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << instruction_flags
                                              << '/' << exec_mask << '/' << aliasing_offset);
            Registers r = input, expected_registers = input;
            uint64_t flags = cpu | semantics;
            uint32_t fp_status = 0x80000000U;

            int result = call(flags, exec_mask, instruction_flags, r.v + aliasing_offset, r.v,
                              r.v + 2, r.v + 4, &fp_status);
            // Reporting is available only for gfx12 half-output variants,
            // which produce no causes for these inputs.
            if ((!gfx12 || Kind == 0) && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            ASSERT_EQ(result, RJ_ARITH_SUCCESS);
            EXPECT_EQ(fp_status, 0x80000000U);
            compute_mixed_fma_expected<Kind>(input, exec_mask, instruction_flags, aliasing_offset,
                                             expected_registers, gfx12);

            rocjitsu::arith::test::expect_fp_result<Kind == 0 ? 32 : 16>(
                input, r, expected_registers, exec_mask, aliasing_offset,
                Kind == 2 ? RJ_ARITH_ALU_HIGH_D : 0, semantics, /*atol=*/1e-5,
                /*rtol=*/Kind == 0 ? 1e-6 : 5e-4);
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
