// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Independent finite FP64 output-scaling witnesses and gfx11 gating checks.
#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "fp/elementwise.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"
#include <gtest/gtest.h>
#include <stdint.h>

namespace rocjitsu::arith::test {

// The adapter evaluates fixed inputs into D; expected is its unscaled result.
template <int Lanes, typename Evaluate>
void check_fp64_omod(Evaluate evaluate, double expected, bool gfx12) {
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t guest : {uint64_t(0), RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, RJ_ARITH_FP_IEEE_MODE,
                           RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE}) {
      for (auto modifier : {RJ_ARITH_ALU_OMOD_2, RJ_ARITH_ALU_OMOD_4, RJ_ARITH_ALU_OMOD_HALF}) {
        Registers<2, Lanes> output;
        ASSERT_EQ(evaluate(cpu | RJ_ARITH_SEMANTICS_LOOSE | guest, modifier, output.v),
                  RJ_ARITH_SUCCESS);
        double scale = modifier == RJ_ARITH_ALU_OMOD_2   ? 2
                       : modifier == RJ_ARITH_ALU_OMOD_4 ? 4
                                                         : 0.5;
        bool enabled = gfx12 || ((guest & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) &&
                                 !(guest & RJ_ARITH_FP_IEEE_MODE));
        EXPECT_EQ(read_fp(output.v, 0, 64), fp_bits(expected * (enabled ? scale : 1), 64));
      }
    }
  }
}

} // namespace rocjitsu::arith::test
