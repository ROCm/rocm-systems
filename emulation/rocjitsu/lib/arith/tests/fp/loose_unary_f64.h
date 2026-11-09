// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared FP64 loose-only unary checks and unsupported-semantics expectations.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/elementwise.h"
#include "fp/omod_f64_test.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith::test {

// Exercise the common loose-only FP64 unary contract, including rejected exact
// modes, modifiers, EXEC, aliasing, and untouched FP status. The caller supplies
// the entry point and an independent scalar reference expression.
template <int Lanes, typename Call, typename Reference>
void check_loose_unary_f64(Call call, Reference reference, bool gfx12) {
  static_assert(Lanes == 32 || Lanes == 64);
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = make_fp_inputs<64, Lanes>();
  Registers witness;
  write_fp(witness.v, 0, 64, 0x3ff8000000000000ULL); // A positive, nonintegral finite input.
  check_fp64_omod<Lanes>(
      [&](uint64_t flags, uint64_t modifier, uint32_t *const *d) {
        return call(flags, Mask(1), modifier, d, witness.v, nullptr);
      },
      reference(1.5), gfx12);

  auto invalid_request = [&](uint64_t flags, uint64_t mode) {
    Registers r = input;
    int aliasing_offset = 6;
    Mask exec_mask = ~Mask(0);
    uint32_t fp_status = 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;

    int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, &fp_status);
    r.expect_preserved(input, 0, 0);
    EXPECT_EQ(fp_status, 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0);
    EXPECT_EQ(call(flags, exec_mask, mode, nullptr, nullptr, &fp_status), result);
    return result;
  };
  expect_invalid_fp_requests(invalid_request);

  // This entry point is loose-only. Its reference is computed independently;
  // strict numeric semantics and required reporting are checked as error contracts.
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode :
           {uint64_t(0), uint64_t(RJ_ARITH_ALU_NEG_A), uint64_t(RJ_ARITH_ALU_CLAMP)}) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : {6, 0, 2, 1}) {
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << mode << '/'
                                              << exec_mask << '/' << aliasing_offset);
            Registers r = input, expected_registers = input;
            uint64_t flags = cpu | semantics;
            uint32_t fp_status = 0x80000000U;

            int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, &fp_status);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            ASSERT_EQ(result, RJ_ARITH_SUCCESS);
            EXPECT_EQ(fp_status, 0x80000000U);
            for (int lane = 0; lane < Lanes; ++lane) {
              if (!((exec_mask >> lane) & 1)) {
                continue;
              }
              double x = modified_fp(fp_value(read_fp(input.v, lane, 64), 64), mode, 0);
              double value = reference(x);
              if (mode & RJ_ARITH_ALU_CLAMP) {
                value = !(value > 0) ? 0 : std::min(value, 1.0);
              }
              write_fp(expected_registers.v + aliasing_offset, lane, 64, fp_bits(value, 64));
            }

            expect_fp_result<64>(input, r, expected_registers, exec_mask, aliasing_offset, mode,
                                 RJ_ARITH_SEMANTICS_LOOSE, /*atol=*/1e-14, /*rtol=*/1e-14);
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
