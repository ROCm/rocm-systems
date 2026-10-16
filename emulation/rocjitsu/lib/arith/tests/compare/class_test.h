// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Floating-point class predicates and their lane-mask results.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/dpp_reference.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "compare/floating_point.h"
#include "compare/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace rocjitsu::arith::test {

// IEEE raw-encoding classification from the ISA and the old suite's
// class_reference.h (validated there against gfx1201 captures). Classification
// never quiets NaNs or flushes subnormals, regardless of guest FP controls.
template <int Width> uint32_t compare_class(uint64_t bits, uint64_t mode) {
  constexpr uint64_t sign = 1ULL << (Width - 1), inf = compare_infinity<Width>();
  if (mode & RJ_ARITH_ALU_ABS_A) {
    bits &= ~sign;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    bits ^= sign;
  }
  bool negative = bits & sign;
  uint64_t magnitude = bits & (sign - 1);
  if (magnitude > inf) {
    return bits & compare_quiet_bit<Width>() ? RJ_ARITH_CLASS_QNAN : RJ_ARITH_CLASS_SNAN;
  }
  if (magnitude == inf) {
    return negative ? RJ_ARITH_CLASS_NEG_INF : RJ_ARITH_CLASS_POS_INF;
  }
  if (magnitude == 0) {
    return negative ? RJ_ARITH_CLASS_NEG_ZERO : RJ_ARITH_CLASS_POS_ZERO;
  }
  if (!(magnitude & inf)) {
    return negative ? RJ_ARITH_CLASS_NEG_SUBNORMAL : RJ_ARITH_CLASS_POS_SUBNORMAL;
  }
  return negative ? RJ_ARITH_CLASS_NEG_NORMAL : RJ_ARITH_CLASS_POS_NORMAL;
}

// Check class predicates and result masks for the supplied entry point.
template <int Width, int Lanes, class Call> void check_class_compare(Call call) {
  using Mask = WaveMask<Lanes>;
  auto input = make_compare_float_inputs<Width, Lanes>();
  // First ten lanes have a matching one-hot class mask, the next ten a
  // deliberately mismatched one. Higher bits must be ignored in both halves.
  for (int lane = 0; lane < 20; ++lane) {
    for (int half = 0; half < (Width == 16 ? 2 : 1); ++half) {
      uint32_t classes = 0xfc00U | (1U << ((lane + half + (lane >= 10 ? 1 : 0)) % 10));
      write_compare_operand(input.v + 2, lane, Width == 16 ? 16 : 32, classes, half);
    }
  }
  input.v[2][20] = 0;
  input.v[2][21] = UINT32_MAX;
  const uint64_t low_modifiers = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_CLAMP |
                                 (Width == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0);
  expect_alu_invalid_flags<Width != 64>(
      [&](uint64_t flags, uint64_t mode) {
        Mask d = ~Mask(0);
        int error = call(flags, ~Mask(0), mode, &d, nullptr, nullptr);
        EXPECT_EQ(d, ~Mask(0));
        return error;
      },
      low_modifiers);
  std::vector<uint64_t> modes = {0, RJ_ARITH_ALU_ABS_A, RJ_ARITH_ALU_NEG_A,
                                 RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A, RJ_ARITH_ALU_CLAMP};
  if constexpr (Width == 16) {
    modes.insert(modes.end(), {RJ_ARITH_ALU_HIGH_A, RJ_ARITH_ALU_HIGH_B,
                               RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B});
  }
  if constexpr (Width != 64) {
    for (uint64_t descriptor : dpp_modes) {
      modes.push_back(descriptor | RJ_ARITH_ALU_NEG_A);
    }
  }
  const auto cpu_levels = supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < int(modes.size()) + 5; ++scenario) {
        uint64_t mode = scenario < 5 ? 0 : modes[scenario - 5];
        Mask exec_mask = exec_mask_for_case<Lanes>(scenario);
        uint64_t flags =
            cpu | semantics |
            (scenario % 2 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                                RJ_ARITH_FP_IEEE_MODE
                          : 0);
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << scenario);
        Mask expected = comparison_mask<Lanes>(mode, exec_mask, [&](int source, int lane) {
          uint64_t bits = compare_operand(input.v, source, Width, mode & RJ_ARITH_ALU_HIGH_A);
          uint32_t classes = uint32_t(compare_operand(input.v + 2, lane, Width == 16 ? 16 : 32,
                                                      mode & RJ_ARITH_ALU_HIGH_B));
          return bool(compare_class<Width>(bits, mode) & classes);
        });
        for (int aliasing_register : {-1, 0, 2}) {
          expect_comparison(input, expected, RJ_ARITH_SUCCESS, aliasing_register,
                            [&](auto &r, Mask *d) {
                              return call(flags, exec_mask, mode, d, exec_mask ? r.v : nullptr,
                                          exec_mask ? r.v + 2 : nullptr);
                            });
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
