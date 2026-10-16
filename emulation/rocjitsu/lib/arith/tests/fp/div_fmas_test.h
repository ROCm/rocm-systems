// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Conditionally scaled division FMA and FP-status expectations.
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

// Fill active destination elements from the independent arithmetic reference.
template <int Width, int Lanes>
void compute_div_fmas_expected(const rocjitsu::arith::test::Registers<10, Lanes> &input,
                               uint64_t exec_mask, uint64_t mode, int aliasing_offset,
                               rocjitsu::arith::test::Registers<10, Lanes> &expected_registers) {
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
    double value;
    if constexpr (Width == 32) {
      value = double(std::fma(float(x), float(y), float(z)));
    } else {
      value = std::fma(x, y, z);
    }
    if (lane % 2) {
      value = std::ldexp(value, (std::abs(z) >= 2 ? 1 : -1) * (Width == 32 ? 64 : 128));
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = !(value > 0) ? 0 : std::min(value, 1.0);
    }
    uint64_t expected = rocjitsu::arith::test::fp_bits(value, Width);
    rocjitsu::arith::test::write_fp(expected_registers.v + aliasing_offset, lane, Width, expected,
                                    0);
  }
}

// Exercise this instruction contract for either wave size.
template <int Width, int Lanes, class Call, class Cases>
void check_div_fmas(Call call, bool gfx12, const Cases &status_cases) {
  static_assert(Width == 32 || Width == 64);
  static_assert(Lanes == 32 || Lanes == 64);
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = rocjitsu::arith::test::make_fp_inputs<Width, Lanes>();
  if (gfx12) {
    // The SIMD integer kernel now shares discarded bits with optional status
    // reporting. Compare the inherited scalar model over broad exponents,
    // post-scaling, modifiers, aliases and inactive exceptional lanes.
    std::minstd_rand engine;
    auto cpus = supported_cpu_levels();
    for (int sample = 0; sample < 256; ++sample) {
      Registers source;
      for (int reg = 0; reg < 10; ++reg) {
        for (int lane = 0; lane < Lanes; ++lane) {
          source.v[reg][lane] = uint32_t(engine() >> 7) ^ (uint32_t(engine() >> 7) << 16);
        }
      }
      uint32_t mode = ((sample >> 1) & 3) << 6;
      if (sample & 8) {
        mode |= RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_C;
      }
      if (sample & 16) {
        mode |= RJ_ARITH_ALU_CLAMP;
      }
      Mask condition = sample & 1 ? Mask(0xaaaaaaaaaaaaaaaaULL) : Mask(0);
      Mask exec_mask = sample & 32 ? Mask(0x5555555555555555ULL) : ~Mask(0);
      int dest = ((sample >> 6) & 3) * 2;
      for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                                 RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
        auto expected = source;
        uint32_t expected_status = 0x80000000U;
        ASSERT_EQ(call(semantics, exec_mask, mode, expected.v + dest, expected.v, expected.v + 2,
                       expected.v + 4, condition, &expected_status),
                  RJ_ARITH_SUCCESS);
        for (uint64_t cpu : cpus) {
          SCOPED_TRACE(::testing::Message() << sample << '/' << cpu << '/' << semantics);
          auto actual = source;
          uint32_t status = 0x80000000U;
          ASSERT_EQ(call(cpu | semantics, exec_mask, mode, actual.v + dest, actual.v, actual.v + 2,
                         actual.v + 4, condition, &status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(status, expected_status);
          for (int reg = 0; reg < Width / 32; ++reg) {
            for (int lane = 0; lane < Lanes; ++lane) {
              ASSERT_EQ(actual.v[dest + reg][lane], expected.v[dest + reg][lane]);
            }
          }
          actual.expect_preserved(source, dest, Width / 32);
        }
      }
    }
  }

  rocjitsu::arith::test::expect_invalid_fp_requests([&](uint64_t flags, uint64_t mode) {
    Registers r = input;
    int aliasing_offset = 6;
    Mask exec_mask = ~Mask(0);
    uint32_t fp_status = 0x80000000U;

    int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 4,
                      Mask(0xaaaaaaaaaaaaaaaaULL), &fp_status);
    r.expect_preserved(input, 0, 0);
    EXPECT_EQ(fp_status, 0x80000000U);
    EXPECT_EQ(call(flags, exec_mask, mode, nullptr, nullptr, nullptr, nullptr,
                   Mask(0xaaaaaaaaaaaaaaaaULL), &fp_status),
              result);
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
      for (uint64_t mode :
           {uint64_t(0), uint64_t(RJ_ARITH_ALU_NEG_A), uint64_t(RJ_ARITH_ALU_CLAMP)}) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : aliasing_offsets) {
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << mode << '/'
                                              << exec_mask << '/' << aliasing_offset);
            Registers r = input, expected_registers = input;
            uint64_t flags = cpu | semantics;
            uint32_t fp_status = 0x80000000U;

            int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 4,
                              Mask(0xaaaaaaaaaaaaaaaaULL), &fp_status);
            if (!gfx12 && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            ASSERT_EQ(result, RJ_ARITH_SUCCESS);
            bool reporting = gfx12 && (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
            if (!reporting || !exec_mask) {
              EXPECT_EQ(fp_status, 0x80000000U);
            } else {
              EXPECT_EQ(fp_status & 0x80000000U, 0x80000000U);
            }
            compute_div_fmas_expected<Width>(input, exec_mask, mode, aliasing_offset,
                                             expected_registers);

            rocjitsu::arith::test::expect_fp_result<Width>(
                input, r, expected_registers, exec_mask, aliasing_offset, 0, semantics,
                /*atol=*/Width == 64 ? 1e-14 : 1e-5, /*rtol=*/Width == 64 ? 1e-14 : 1e-6);
          }
        }
      }
    }
  }
  if (gfx12) {
    for (uint64_t cpu : cpu_levels) {
      for (const auto &test : status_cases) {
        for (uint64_t mode : {uint64_t(0), uint64_t(RJ_ARITH_ALU_CLAMP)}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << mode << '/' << test.a << '/' << test.b << '/' << test.c);
          Registers r = input;
          const int lane = Lanes - 1;
          for (int part = 0; part < 1; ++part) {
            rocjitsu::arith::test::write_fp(r.v, lane, Width, test.a, part);
            rocjitsu::arith::test::write_fp(r.v + 2, lane, Width, test.b, part);
            rocjitsu::arith::test::write_fp(r.v + 4, lane, Width, test.c, part);
          }
          Mask exec_mask = Mask(1) << lane;
          uint32_t fp_status = 0x80000000U;
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6,
                         r.v, r.v + 2, r.v + 4, 0, &fp_status),
                    RJ_ARITH_SUCCESS);
          uint32_t causes = mode ? 0 : test.causes;
          EXPECT_EQ(fp_status, 0x80000000U | causes);
          // Preserve previously accumulated causes as well as unrelated bits.
          fp_status = 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;
          for (int part = 0; part < 1; ++part) {
            rocjitsu::arith::test::write_fp(r.v + 4, lane, Width, test.c, part);
          }
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6,
                         r.v, r.v + 2, r.v + 4, 0, &fp_status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(fp_status, 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0 | causes);
          Registers before = r;
          exec_mask = 0;
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6,
                         r.v, r.v + 2, r.v + 4, 0, &fp_status),
                    RJ_ARITH_SUCCESS);
          r.expect_preserved(before, 0, 0);
          EXPECT_EQ(fp_status, 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0 | causes);
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
