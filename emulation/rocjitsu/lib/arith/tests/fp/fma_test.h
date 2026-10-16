// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Fused arithmetic across FP widths, aliases, and guest controls.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/elementwise.h"
#include "fp/fma_reference.h"
#include "fp/fma_simd_test.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith::test {

// Exercise FP32/FP64 FMA arithmetic, modifiers, EXEC, aliasing and API errors.
// FP32's inherited model matches IEEE FMA for these normal fractional inputs
// with default guest rounding; special values are checked by classification.
// FP64's computed reference is used only for loose numeric comparisons.
template <int Width, int Lanes, bool Accumulate = false, class Call>
void check_fma(Call entry_point, bool gfx12) {
  static_assert(Width == 32 || Width == 64);
  static_assert(Lanes == 32 || Lanes == 64);
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  static_assert(!Accumulate || Width == 32);
  // FMAC reads its addend from D and has no explicit C operand.
  auto call = [&](uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                  uint32_t *fp_status) {
    if constexpr (Accumulate) {
      return entry_point(flags, exec_mask, mode, d, a, b, fp_status);
    } else {
      return entry_point(flags, exec_mask, mode, d, a, b, c, fp_status);
    }
  };
  if constexpr (Width == 32) {
    check_fma_simd<Width, Lanes>(call, {0, RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_B,
                                        RJ_ARITH_ALU_CLAMP, RJ_ARITH_ALU_OMOD_2,
                                        RJ_ARITH_ALU_OMOD_4, RJ_ARITH_ALU_OMOD_HALF});
  }
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = make_fp_inputs<Width, Lanes>();

  rocjitsu::arith::test::expect_invalid_fp_requests([&](uint64_t flags, uint64_t mode) {
    Registers r = input;
    int aliasing_offset = 6;
    Mask exec_mask = ~Mask(0);
    uint32_t fp_status = 0x80000000U;

    int result =
        call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 4, &fp_status);
    r.expect_preserved(input, 0, 0);
    EXPECT_EQ(fp_status, 0x80000000U);
    EXPECT_EQ(call(flags, exec_mask, mode, nullptr, nullptr, nullptr, nullptr, &fp_status), result);
    return result;
  });

  // FP32 exercises DPP8, OMOD and combined ABS/NEG in addition to the
  // basic modifiers common to both formats. gfx11 ignores OMOD under the
  // default guest FP mode; compute_fma_expected accounts for that difference.
  const uint64_t modes[] = {0,
                            RJ_ARITH_ALU_NEG_A,
                            RJ_ARITH_ALU_CLAMP,
                            RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT),
                            RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT) |
                                RJ_ARITH_DPP_FI | RJ_ARITH_ALU_NEG_A,
                            RJ_ARITH_ALU_OMOD_2,
                            RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_A};
  const int mode_count = Width == 32 ? 7 : 3;
  // The last offset tests partial overlap of the two-VGPR FP64 operands.
  const int aliasing_offsets[] = {6, 0, 2, 4, 1};
  const int aliasing_count = Accumulate ? 3 : Width == 64 ? 5 : 4;

  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (int mode_index = 0; mode_index < mode_count; ++mode_index) {
        uint64_t mode = modes[mode_index];
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_index = 0; aliasing_index < aliasing_count; ++aliasing_index) {
            int aliasing_offset = aliasing_offsets[aliasing_index];
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << mode << '/'
                                              << exec_mask << '/' << aliasing_offset);
            Registers r = input, expected_registers = input;
            uint64_t flags = cpu | semantics;
            uint32_t fp_status = 0x80000000U;

            int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 4,
                              &fp_status);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            // FP64 is loose-only; FP32 supports exact numeric results on
            // this domain. Neither format currently supports FP status.
            if (Width == 64 && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            ASSERT_EQ(result, RJ_ARITH_SUCCESS);
            EXPECT_EQ(fp_status, 0x80000000U);
            compute_fma_expected<Width>(input, exec_mask, mode, aliasing_offset, expected_registers,
                                        gfx12, Accumulate);

            expect_fp_result<Width>(input, r, expected_registers, exec_mask, aliasing_offset, mode,
                                    Width == 64 ? RJ_ARITH_SEMANTICS_LOOSE : semantics,
                                    /*atol=*/Width == 64 ? 1e-14 : 1e-5,
                                    /*rtol=*/Width == 64 ? 1e-14 : 1e-6);
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
