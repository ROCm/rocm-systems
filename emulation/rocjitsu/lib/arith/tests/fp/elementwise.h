// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Deterministic elementwise FP inputs and whole-register result verification.

#pragma once

#include "common/registers.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Build a deterministic bank with two VGPR slots per operand, including one
// NaN and infinity per source. Width describes the encoded FP format, not a C++ type.
template <int Width, int Lanes> Registers<10, Lanes> make_fp_inputs(std::minstd_rand &random) {
  static_assert(Width == 16 || Width == 32 || Width == 64);
  Registers<10, Lanes> input;
  for (int operand = 0; operand < 5; ++operand) {
    for (int lane = 0; lane < Lanes; ++lane) {
      for (int half = 0; half < (Width == 16 ? 2 : 1); ++half) {
        write_fp(input.v + 2 * operand, lane, Width, random_fp(random, Width), half);
      }
    }
  }
  for (int operand = 0; operand < 3; ++operand) {
    write_fp(input.v + 2 * operand, 2 * operand, Width,
             fp_bits(std::numeric_limits<double>::quiet_NaN(), Width));
    write_fp(input.v + 2 * operand, 2 * operand + 1, Width,
             fp_bits(std::numeric_limits<double>::infinity(), Width));
  }
  return input;
}

// Use the standardized default seed and consume high engine bits.
template <int Width, int Lanes> Registers<10, Lanes> make_fp_inputs() {
  std::minstd_rand random;
  return make_fp_inputs<Width, Lanes>(random);
}

// Check active results and preservation of inactive lanes, other halves and guards.
// Semantics selects exact versus fuzzy comparison; tolerances belong to the caller.
template <int Width, bool Packed = false, int Lanes>
void expect_fp_result(const Registers<10, Lanes> &input, const Registers<10, Lanes> &result,
                      const Registers<10, Lanes> &expected, uint64_t exec_mask, int aliasing_offset,
                      uint64_t instruction_flags, uint64_t semantics, double atol, double rtol) {
  static_assert(Width == 16 || Width == 32 || Width == 64);
  static_assert(!Packed || Width == 16);
  constexpr int register_count = Width == 64 ? 2 : 1;
  result.expect_preserved(input, aliasing_offset, register_count);
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      for (int reg = 0; reg < register_count; ++reg) {
        EXPECT_EQ(result.v[aliasing_offset + reg][lane], input.v[aliasing_offset + reg][lane]);
      }
      continue;
    }
    int half = 0;
    if constexpr (Width == 16 && !Packed) {
      half = bool(instruction_flags & RJ_ARITH_ALU_HIGH_D);
      EXPECT_EQ(uint16_t(result.v[aliasing_offset][lane] >> (16 * (1 - half))),
                uint16_t(input.v[aliasing_offset][lane] >> (16 * (1 - half))));
    }
    for (int output_half = half; output_half < half + (Packed ? 2 : 1); ++output_half) {
      expect_fp(read_fp(result.v + aliasing_offset, lane, Width, output_half),
                read_fp(expected.v + aliasing_offset, lane, Width, output_half), Width, semantics,
                atol, rtol);
    }
  }
}

} // namespace rocjitsu::arith::test
