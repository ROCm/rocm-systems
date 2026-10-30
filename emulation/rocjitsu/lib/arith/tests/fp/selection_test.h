// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Two- and three-input FP selection with NaN and signed-zero policies.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/elementwise.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>
#include <vector>

namespace rocjitsu::arith::test {

// gfx1201 selection OMOD rounds overflow by MODE, as does the inherited model.
// Equal finite operands isolate output scaling from NaN/operand selection policy.
template <int Width, int Operands, typename Call> void check_selection_omod(Call call) {
  constexpr uint64_t infinity = Width == 16   ? 0x7c00ULL
                                : Width == 32 ? 0x7f800000ULL
                                              : 0x7ff0000000000000ULL;
  constexpr uint64_t sign_bit = uint64_t(1) << (Width - 1);
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t rounding :
         {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
      for (bool negative : {false, true}) {
        for (uint32_t mode : {RJ_ARITH_ALU_OMOD_2, RJ_ARITH_ALU_OMOD_4}) {
          Registers<8, 32> r;
          uint64_t bits = (negative ? sign_bit : 0) | (infinity - 1);
          for (int reg : {0, 2, 4}) {
            write_fp(r.v + reg, 0, Width, bits);
          }
          uint64_t flags = cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY |
                           RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | rounding;
          int result;
          if constexpr (Operands == 3) {
            result = call(flags, 1, mode, r.v + 6, r.v, r.v + 2, r.v + 4, nullptr);
          } else {
            result = call(flags, 1, mode, r.v + 6, r.v, r.v + 2, nullptr);
          }
          ASSERT_EQ(result, RJ_ARITH_SUCCESS);
          bool to_infinity = rounding == 0 ||
                             rounding == (negative ? RJ_ARITH_FP_ROUND_DOWN : RJ_ARITH_FP_ROUND_UP);
          EXPECT_EQ(read_fp(r.v + 6, 0, Width),
                    (negative ? sign_bit : 0) | (infinity - !to_infinity))
              << cpu << '/' << rounding << '/' << negative << '/' << mode;
        }
      }
    }
  }
}

// Output scaling preserves the selected NaN bits in Compatibility, under every guest rounding.
template <int Width, int Operands, typename Call> void check_selection_omod_nan(Call call) {
  constexpr uint64_t nan = Width == 16   ? 0x7c01ULL
                           : Width == 32 ? 0x7f800001ULL
                                         : 0x7ff0000000000001ULL;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t rounding :
         {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
      for (uint32_t instruction_flags :
           {RJ_ARITH_ALU_OMOD_2, RJ_ARITH_ALU_OMOD_4, RJ_ARITH_ALU_OMOD_HALF}) {
        Registers<8, 32> r;
        for (int reg : {0, 2, 4}) {
          write_fp(r.v + reg, 0, Width, nan);
        }
        uint64_t flags =
            cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | rounding;
        int result;
        if constexpr (Operands == 3) {
          result = call(flags, 1, instruction_flags, r.v + 6, r.v, r.v + 2, r.v + 4, nullptr);
        } else {
          result = call(flags, 1, instruction_flags, r.v + 6, r.v, r.v + 2, nullptr);
        }
        ASSERT_EQ(result, RJ_ARITH_SUCCESS);
        EXPECT_EQ(read_fp(r.v + 6, 0, Width), nan)
            << cpu << '/' << rounding << '/' << instruction_flags;
      }
    }
  }
}

// Test loose-only min/max/median selection, with the operation's independent
// scalar expression supplied by its named test. Packed instructions write both
// halves; ordinary FP16 instructions preserve the unselected half.
template <int Width, int Lanes, int Operands, bool Packed = false, typename Call,
          typename Reference>
void check_selection(Call call, Reference reference) {
  static_assert(Width == 16 || Width == 32 || Width == 64);
  static_assert(Lanes == 32 || Lanes == 64);
  static_assert(Operands == 2 || Operands == 3);
  static_assert(!Packed || (Width == 16 && Operands == 2));
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  using Bank = Registers<10, Lanes>;
  Bank input = make_fp_inputs<Width, Lanes>();
  // Include both-NaN and signaling-NaN/number pairs: these distinguish
  // propagating selection from number-preferring selection without large fixtures.
  constexpr uint64_t signaling_nan = Width == 16   ? 0x7c01ULL
                                     : Width == 32 ? 0x7f800001ULL
                                                   : 0x7ff0000000000001ULL;
  for (int half = 0; half < (Width == 16 ? 2 : 1); ++half) {
    write_fp(input.v, 8, Width, signaling_nan, half);
    write_fp(input.v + 2, 8, Width, fp_bits(1.0, Width), half);
    write_fp(input.v, 9, Width, fp_bits(1.0, Width), half);
    write_fp(input.v + 2, 9, Width, signaling_nan, half);
    write_fp(input.v, 10, Width, signaling_nan, half);
    write_fp(input.v + 2, 10, Width, signaling_nan, half);
  }

  // A null bank exercises early validation with all operand pointers null.
  auto invoke = [&](uint64_t flags, Mask exec_mask, uint64_t mode, Bank *bank, int aliasing_offset,
                    uint32_t *fp_status) {
    auto d = bank ? bank->v + aliasing_offset : nullptr;
    auto a = bank ? bank->v : nullptr;
    auto b = bank ? bank->v + 2 : nullptr;
    if constexpr (Operands == 2) {
      return call(flags, exec_mask, mode, d, a, b, fp_status);
    } else {
      return call(flags, exec_mask, mode, d, a, b, bank ? bank->v + 4 : nullptr, fp_status);
    }
  };
  expect_invalid_fp_requests([&](uint64_t flags, uint64_t mode) {
    Bank result = input;
    uint32_t fp_status = 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;
    int error = invoke(flags, ~Mask(0), mode, &result, 6, &fp_status);
    result.expect_preserved(input, 0, 0);
    EXPECT_EQ(fp_status, 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0);
    EXPECT_EQ(invoke(flags, ~Mask(0), mode, nullptr, 6, &fp_status), error);
    return error;
  });

  std::vector<uint64_t> modes = {0, RJ_ARITH_ALU_NEG_A,
                                 Packed ? RJ_ARITH_PK_CLAMP : RJ_ARITH_ALU_CLAMP};
  if constexpr (Width == 16 && !Packed) {
    modes.push_back(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D);
  }
  if constexpr (Width != 64 && !Packed) {
    modes.push_back(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT));
    modes.push_back(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT) | RJ_ARITH_DPP_FI |
                    RJ_ARITH_ALU_NEG_A);
  }
  std::vector<int> aliasing_offsets = {6, 0, 2};
  if constexpr (Operands == 3) {
    aliasing_offsets.push_back(4);
  }
  if constexpr (Width == 64) {
    aliasing_offsets.push_back(1);
  }

  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : aliasing_offsets) {
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << mode << '/'
                                              << exec_mask << '/' << aliasing_offset);
            Bank result = input, expected = input;
            uint32_t fp_status = 0x80000000U;
            int error =
                invoke(cpu | semantics, exec_mask, mode, &result, aliasing_offset, &fp_status);
            EXPECT_EQ(fp_status, 0x80000000U);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL ||
                semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
              EXPECT_EQ(error, semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                                   ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                                   : RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
              result.expect_preserved(input, 0, 0);
              continue;
            }
            ASSERT_EQ(error, RJ_ARITH_SUCCESS);
            for (int lane = 0; lane < Lanes; ++lane) {
              if (!((exec_mask >> lane) & 1)) {
                continue;
              }
              for (int half = 0; half < (Packed ? 2 : 1); ++half) {
                int a_half = Packed ? half : Width == 16 && (mode & RJ_ARITH_ALU_HIGH_A);
                int b_half = Packed ? half : Width == 16 && (mode & RJ_ARITH_ALU_HIGH_B);
                int c_half = Width == 16 && (mode & RJ_ARITH_ALU_HIGH_C);
                int d_half = Packed ? half : Width == 16 && (mode & RJ_ARITH_ALU_HIGH_D);
                // DPP8 reverses groups of eight; inactive sources become zero unless FI is set.
                int source_lane =
                    !Packed && (mode & RJ_ARITH_DPP8) ? (lane & ~7) + 7 - (lane & 7) : lane;
                uint64_t a = read_fp(input.v, source_lane, Width, a_half);
                if (!Packed && (mode & RJ_ARITH_DPP8) && !(mode & RJ_ARITH_DPP_FI) &&
                    !((exec_mask >> source_lane) & 1)) {
                  a = 0;
                }
                uint64_t b = read_fp(input.v + 2, lane, Width, b_half);
                uint64_t source_mode = Packed ? (mode >> (half ? 3 : 0)) & 7 : mode;
                double x = modified_fp(fp_value(a, Width), source_mode, 0);
                double y = modified_fp(fp_value(b, Width), source_mode, 1);
                double value;
                if constexpr (Operands == 2) {
                  value = reference(x, y);
                } else {
                  uint64_t c = read_fp(input.v + 4, lane, Width, c_half);
                  double z = modified_fp(fp_value(c, Width), source_mode, 2);
                  value = reference(x, y, z);
                }
                if (mode & (Packed ? RJ_ARITH_PK_CLAMP : RJ_ARITH_ALU_CLAMP)) {
                  value = !(value > 0) ? 0 : std::min(value, 1.0);
                }
                write_fp(expected.v + aliasing_offset, lane, Width, fp_bits(value, Width), d_half);
              }
            }
            expect_fp_result<Width, Packed>(input, result, expected, exec_mask, aliasing_offset,
                                            mode, RJ_ARITH_SEMANTICS_LOOSE,
                                            /*atol=*/Width == 64 ? 1e-14 : 1e-5,
                                            /*rtol=*/Width == 16 ? 5e-4
                                            : Width == 32        ? 1e-6
                                                                 : 1e-14);
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
