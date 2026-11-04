// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Division operand rescaling and the resulting condition mask.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/elementwise.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdint.h>
#include <type_traits>

#include <vector>

namespace rocjitsu::arith::test {

// Build valid DIV_SCALE operands with A equal to either B or C.
template <int Width, int Lanes>
rocjitsu::arith::test::Registers<10, Lanes> make_div_scale_inputs() {
  auto input = rocjitsu::arith::test::make_fp_inputs<Width, Lanes>();
  // DIV_SCALE requires A to be either denominator B or numerator C.
  for (int lane = 0; lane < Lanes; ++lane) {
    for (int reg = 0; reg < Width / 32; ++reg) {
      input.v[reg][lane] = input.v[2 + 2 * (lane % 2) + reg][lane];
    }
  }
  // Straddle both exponent-gap thresholds with finite powers of two. These
  // exercise rescaling and condition bits independently of the SIMD differential test.
  constexpr int threshold = Width == 32 ? 96 : 768;
  for (int i = 0; i < 4; ++i) {
    int gap = i < 2 ? threshold - i : -threshold + (i - 2);
    uint64_t denominator = fp_bits(std::ldexp(1.0, -gap), Width);
    uint64_t numerator = fp_bits(1.0, Width);
    write_fp(input.v + 2, 8 + i, Width, denominator);
    write_fp(input.v + 4, 8 + i, Width, numerator);
    write_fp(input.v, 8 + i, Width, i % 2 ? numerator : denominator);
  }
  return input;
}

// Fill active destination elements from the independent arithmetic reference.
// Return the required DIV_SCALE condition bits for participating lanes.
template <int Width, int Lanes>
uint64_t
compute_div_scale_expected(const rocjitsu::arith::test::Registers<10, Lanes> &input,
                           uint64_t exec_mask, uint64_t mode, int aliasing_offset,
                           rocjitsu::arith::test::Registers<10, Lanes> &expected_registers) {
  uint64_t expected_condition = 0;
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    uint64_t a = rocjitsu::arith::test::read_fp(input.v, lane, Width, 0);
    uint64_t b = rocjitsu::arith::test::read_fp(input.v + 2, lane, Width, 0);
    uint64_t c = rocjitsu::arith::test::read_fp(input.v + 4, lane, Width, 0);

    double x =
        rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(a, Width), mode, 0);
    double y =
        rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(b, Width), mode, 1);
    double z =
        rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(c, Width), mode, 2);
    // Borrowed DIV_SCALE exponent decision, evaluated numerically.
    // A denotes B or C; NEG_A changes that identity just as in the ISA.
    const int fraction_bits = Width == 32 ? 23 : 52;
    const int exponent_mask = Width == 32 ? 255 : 2047;
    const int threshold = Width == 32 ? 96 : 768;
    const int scale = Width == 32 ? 64 : 128;
    int de = int((b >> fraction_bits) & exponent_mask);
    int ne = int((c >> fraction_bits) & exponent_mask);
    int delta = ne - de, adjustment = 0;
    bool post_scale = false;
    if (delta >= threshold) {
      adjustment = x == y ? scale : 0;
      post_scale = true;
    } else if (!de) {
      adjustment = scale;
    } else if (de >= exponent_mask - 2) {
      post_scale = delta <= -threshold;
      adjustment = post_scale && x != y ? 0 : -scale;
    } else if (delta <= -threshold) {
      adjustment = x == y ? 0 : scale;
      post_scale = true;
    } else if (ne <= fraction_bits + 1) {
      adjustment = scale;
    }
    double value =
        y == 0 || z == 0 ? std::numeric_limits<double>::quiet_NaN() : std::ldexp(x, adjustment);
    expected_condition |= uint64_t(post_scale) << lane;
    uint64_t expected = rocjitsu::arith::test::fp_bits(value, Width);
    rocjitsu::arith::test::write_fp(expected_registers.v + aliasing_offset, lane, Width, expected,
                                    0);
  }
  return expected_condition;
}

// Exercise this instruction contract for either wave size.
template <int Width, int Lanes, typename Call> void check_div_scale(Call call) {
  static_assert(Width == 32 || Width == 64);
  static_assert(Lanes == 32 || Lanes == 64);
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = make_div_scale_inputs<Width, Lanes>();
  // Directed integer rounding must agree with the inherited scalar policy,
  // including gradual underflow, finite overflow, OMOD and the condition mask.
  std::minstd_rand engine;
  auto cpus = supported_cpu_levels();
  for (int sample = 0; sample < 128; ++sample) {
    Registers source;
    for (int reg = 0; reg < 10; ++reg) {
      for (int lane = 0; lane < Lanes; ++lane) {
        source.v[reg][lane] = uint32_t(engine() >> 7) ^ (uint32_t(engine() >> 7) << 16);
      }
    }
    for (int lane = 0; lane < Lanes; ++lane) {
      for (int reg = 0; reg < Width / 32; ++reg) {
        source.v[reg][lane] = source.v[2 + 2 * (lane % 2) + reg][lane];
      }
    }
    uint64_t flags =
        RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | (uint64_t(sample % 4) << 24);
    uint32_t mode = ((sample >> 2) & 3) << 6;
    if (sample & 16) {
      flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
    }
    if (sample & 32) {
      flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
    }
    if (sample & 64) {
      mode |= RJ_ARITH_ALU_CLAMP;
    }
    int dest = ((sample >> 3) & 3) * 2;
    Mask exec_mask = sample & 1 ? Mask(0xaaaaaaaaaaaaaaaaULL) : ~Mask(0);
    auto expected = source;
    Mask expected_condition = 0;
    int error = call(flags, exec_mask, mode, expected.v + dest, &expected_condition, expected.v,
                     expected.v + 2, expected.v + 4);
    for (uint64_t cpu : cpus) {
      SCOPED_TRACE(::testing::Message() << sample << '/' << cpu);
      auto actual = source;
      Mask condition = 0;
      ASSERT_EQ(call(flags | cpu, exec_mask, mode, actual.v + dest, &condition, actual.v,
                     actual.v + 2, actual.v + 4),
                error);
      EXPECT_EQ(condition, expected_condition);
      for (int reg = 0; reg < Width / 32; ++reg) {
        for (int lane = 0; lane < Lanes; ++lane) {
          ASSERT_EQ(actual.v[dest + reg][lane], expected.v[dest + reg][lane]);
        }
      }
      actual.expect_preserved(source, dest, Width / 32);
    }
  }

  rocjitsu::arith::test::expect_invalid_fp_requests([&](uint64_t flags, uint64_t mode) {
    Registers r = input;
    int aliasing_offset = 6;
    Mask exec_mask = ~Mask(0);
    Mask condition = ~Mask(0);
    int result =
        call(flags, exec_mask, mode, r.v + aliasing_offset, &condition, r.v, r.v + 2, r.v + 4);
    r.expect_preserved(input, 0, 0);
    EXPECT_EQ(condition, ~Mask(0));
    EXPECT_EQ(call(flags, exec_mask, mode, nullptr, &condition, nullptr, nullptr, nullptr), result);
    return result;
  });

  std::vector<int> aliasing_offsets = {6, 0, 2, 4};
  if constexpr (Width == 64) {
    aliasing_offsets.push_back(1);
  }

  // Default guest rounding and normal fractional inputs: the existing rocjitsu
  // arithmetic model reduces to IEEE arithmetic here. Special values are checked
  // by classification. No production helper or rj-arith result is a numeric oracle.
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : {uint64_t(0), uint64_t(RJ_ARITH_ALU_NEG_A)}) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : aliasing_offsets) {
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << mode << '/'
                                              << exec_mask << '/' << aliasing_offset);
            Registers r = input, expected_registers = input;
            uint64_t flags = cpu | semantics;
            Mask condition = ~Mask(0);
            int result = call(flags, exec_mask, mode, r.v + aliasing_offset, &condition, r.v,
                              r.v + 2, r.v + 4);
            ASSERT_EQ(result, RJ_ARITH_SUCCESS);
            uint64_t expected_condition = compute_div_scale_expected<Width>(
                input, exec_mask, mode, aliasing_offset, expected_registers);
            EXPECT_EQ(condition, expected_condition);
            rocjitsu::arith::test::expect_fp_result<Width>(
                input, r, expected_registers, exec_mask, aliasing_offset, 0, semantics,
                /*atol=*/Width == 64 ? 1e-14 : 1e-5, /*rtol=*/Width == 64 ? 1e-14 : 1e-6);
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
