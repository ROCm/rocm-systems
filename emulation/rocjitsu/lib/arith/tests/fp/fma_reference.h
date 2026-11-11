// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Fused arithmetic across FP widths, aliases, and guest controls.

#pragma once

#include "common/registers.h"
#include "fp/half_fma_reference.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Raw operands and expected accumulated FP exception causes for one FMA witness.
struct StatusCase {
  uint64_t a, b, c;
  uint32_t causes;
};

// Fill active FMA results. FP16 uses the existing independent integer oracle,
// validated against inherited rocjitsu/captured cases. FP32 uses its established
// std::fma reference; FP64 uses std::fma only as a loose reference.
// Accumulating forms read C from D, including its selected half for FP16.
template <int Width, int Lanes>
void compute_fma_expected(const Registers<10, Lanes> &input, uint64_t exec_mask,
                          uint64_t instruction_flags, int aliasing_offset,
                          Registers<10, Lanes> &expected_registers, bool gfx12,
                          bool accumulate = false) {
  static_assert(Width == 16 || Width == 32 || Width == 64);
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    int a_half = Width == 16 && (instruction_flags & RJ_ARITH_ALU_HIGH_A);
    int b_half = Width == 16 && (instruction_flags & RJ_ARITH_ALU_HIGH_B);
    int c_half = Width == 16 &&
                 (instruction_flags & (accumulate ? RJ_ARITH_ALU_HIGH_D : RJ_ARITH_ALU_HIGH_C));
    int d_half = Width == 16 && (instruction_flags & RJ_ARITH_ALU_HIGH_D);
    // DPP8 reverses each group; FI controls reads from inactive sources.
    int source_lane = instruction_flags & RJ_ARITH_DPP8 ? (lane & ~7) + 7 - (lane & 7) : lane;
    uint64_t a = read_fp(input.v, source_lane, Width, a_half);
    if ((instruction_flags & RJ_ARITH_DPP8) && !(instruction_flags & RJ_ARITH_DPP_FI) &&
        !((exec_mask >> source_lane) & 1)) {
      a = 0;
    }
    uint64_t b = read_fp(input.v + 2, lane, Width, b_half);
    uint64_t c = read_fp(input.v + (accumulate ? aliasing_offset : 4), lane, Width, c_half);
    uint64_t expected;
    if constexpr (Width == 16) {
      constexpr uint32_t kReferenceModifiers =
          RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_ABS_A |
          RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP;
      uint32_t reference_instruction_flags = uint32_t(instruction_flags) & kReferenceModifiers;
      expected = half_fma_reference::evaluate(uint32_t(a), uint32_t(b), uint32_t(c),
                                              reference_instruction_flags, false, 0, !gfx12);
    } else {
      double x = modified_fp(fp_value(a, Width), instruction_flags, 0);
      double y = modified_fp(fp_value(b, Width), instruction_flags, 1);
      double z = modified_fp(fp_value(c, Width), instruction_flags, 2);
      double value;
      if constexpr (Width == 32) {
        value = double(std::fma(float(x), float(y), float(z)));
        // gfx11 ignores OMOD under the default guest FP mode.
        if (gfx12) {
          const double scales[] = {1, 2, 4, 0.5};
          value *= scales[(instruction_flags & RJ_ARITH_ALU_OMOD_MASK) >> RJ_ARITH_ALU_OMOD_SHIFT];
        }
      } else {
        value = std::fma(x, y, z);
      }
      if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
        value = !(value > 0) ? 0 : std::min(value, 1.0);
      }
      expected = fp_bits(value, Width);
    }
    write_fp(expected_registers.v + aliasing_offset, lane, Width, expected, d_half);
  }
}

} // namespace rocjitsu::arith::test
