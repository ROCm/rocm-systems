// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FMA with a literal multiplier or addend.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/elementwise.h"
#include "fp/fma_simd_test.h"
#include "fp/half_fma_reference.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith::test {

// Fill active destination elements from the independent arithmetic reference.
template <int Lanes, bool MultiplyConstant>
void compute_immediate_fma_f16_expected(
    const rocjitsu::arith::test::Registers<10, Lanes> &input, uint64_t exec_mask, uint64_t mode,
    int aliasing_offset, rocjitsu::arith::test::Registers<10, Lanes> &expected_registers,
    bool gfx12) {
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    const int a_half = bool(mode & RJ_ARITH_ALU_HIGH_A);
    const int b_half = bool(mode & RJ_ARITH_ALU_HIGH_B);
    const int c_half = bool(mode & RJ_ARITH_ALU_HIGH_C);
    const int d_half = bool(mode & RJ_ARITH_ALU_HIGH_D);
    uint64_t a = rocjitsu::arith::test::read_fp(input.v, lane, 16, a_half);
    uint64_t b = rocjitsu::arith::test::read_fp(input.v + 2, lane, 16, b_half);
    uint64_t c = rocjitsu::arith::test::read_fp(input.v + 4, lane, 16, c_half);
    if constexpr (MultiplyConstant) {
      c = b;
      b = 0x3555U;
    } else {
      c = 0x3555U;
    }
    // Existing independent integer FMA oracle, validated by the old
    // suite against the inherited rocjitsu model and captured cases.
    uint32_t reference_mode = uint32_t(mode) & 0x1ffU;
    uint64_t expected = rocjitsu::arith::test::half_fma_reference::evaluate(
        uint32_t(a), uint32_t(b), uint32_t(c), reference_mode, false, 0, !gfx12);
    rocjitsu::arith::test::write_fp(expected_registers.v + aliasing_offset, lane, 16, expected,
                                    d_half);
  }
}

// Exercise this instruction contract for either wave size.
template <int Lanes, bool MultiplyConstant, typename Call, typename Cases>
void check_immediate_fma_f16(Call entry_point, bool gfx12, const Cases &status_cases) {
  static_assert(Lanes == 32 || Lanes == 64);
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  // Normalize the two ISA operand orders while retaining the immediate's type.
  auto call = [&](uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint16_t immediate,
                  uint32_t *fp_status) {
    if constexpr (MultiplyConstant) {
      return entry_point(flags, exec_mask, mode, d, a, immediate, b, fp_status);
    } else {
      return entry_point(flags, exec_mask, mode, d, a, b, immediate, fp_status);
    }
  };
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = rocjitsu::arith::test::make_fp_inputs<16, Lanes>();
  check_fma_simd<16, Lanes>(
      [&](uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
          uint32_t *status) {
        return call(flags, exec_mask, mode, d, a, b, uint16_t(c[0][0]), status);
      },
      {0}, /*register_operands=*/false);

  rocjitsu::arith::test::expect_invalid_fp_requests([&](uint64_t flags, uint64_t mode) {
    Registers r = input;
    int aliasing_offset = 6;
    Mask exec_mask = ~Mask(0);
    uint32_t fp_status = 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;

    int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, uint16_t(0x3555),
                      &fp_status);
    r.expect_preserved(input, 0, 0);
    EXPECT_EQ(fp_status, 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0);
    EXPECT_EQ(call(flags, exec_mask, mode, nullptr, nullptr, nullptr, uint16_t(0x3555), &fp_status),
              result);
    return result;
  });

  // Default guest rounding and normal fractional inputs: the existing rocjitsu
  // arithmetic model reduces to IEEE arithmetic here. Special values are checked
  // by classification. No production helper or rj-arith result is a numeric oracle.
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : {uint64_t(0), uint64_t(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D)}) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : {6, 0, 2}) {
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << mode << '/'
                                              << exec_mask << '/' << aliasing_offset);
            Registers r = input, expected_registers = input;
            uint64_t flags = cpu | semantics;
            uint32_t fp_status = 0x80000000U;

            int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2,
                              uint16_t(0x3555), &fp_status);
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
            compute_immediate_fma_f16_expected<Lanes, MultiplyConstant>(
                input, exec_mask, mode, aliasing_offset, expected_registers, gfx12);

            rocjitsu::arith::test::expect_fp_result<16>(input, r, expected_registers, exec_mask,
                                                        aliasing_offset, mode, semantics,
                                                        /*atol=*/1e-5, /*rtol=*/5e-4);
          }
        }
      }
    }
  }
  if (gfx12) {
    for (uint64_t cpu : cpu_levels) {
      for (const auto &test : status_cases) {
        for (uint64_t mode : {uint64_t(0)}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << mode << '/' << test.a << '/' << test.b << '/' << test.c);
          Registers r = input;
          const int lane = Lanes - 1;
          for (int part = 0; part < 1; ++part) {
            rocjitsu::arith::test::write_fp(r.v, lane, 16, test.a, part);
            rocjitsu::arith::test::write_fp(r.v + 2, lane, 16, test.b, part);
            rocjitsu::arith::test::write_fp(r.v + 4, lane, 16, test.c, part);
          }
          Mask exec_mask = Mask(1) << lane;
          uint32_t fp_status = 0x80000000U;
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6,
                         r.v, r.v + 2, uint16_t(0x3c01), &fp_status),
                    RJ_ARITH_SUCCESS);
          uint32_t causes = test.causes;
          EXPECT_EQ(fp_status, 0x80000000U | causes);
          // Preserve previously accumulated causes as well as unrelated bits.
          fp_status = 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;
          for (int part = 0; part < 1; ++part) {
            rocjitsu::arith::test::write_fp(r.v + 4, lane, 16, test.c, part);
          }
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6,
                         r.v, r.v + 2, uint16_t(0x3c01), &fp_status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(fp_status, 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0 | causes);
          Registers before = r;
          exec_mask = 0;
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6,
                         r.v, r.v + 2, uint16_t(0x3c01), &fp_status),
                    RJ_ARITH_SUCCESS);
          r.expect_preserved(before, 0, 0);
          EXPECT_EQ(fp_status, 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0 | causes);
        }
      }
    }
  }
}

// Fill active destination elements from the independent arithmetic reference.
template <int Lanes, bool MultiplyConstant>
void compute_immediate_fma_f32_expected(
    const rocjitsu::arith::test::Registers<10, Lanes> &input, uint64_t exec_mask, uint64_t mode,
    int aliasing_offset, rocjitsu::arith::test::Registers<10, Lanes> &expected_registers) {
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    uint64_t a = rocjitsu::arith::test::read_fp(input.v, lane, 32, 0);
    uint64_t b = rocjitsu::arith::test::read_fp(input.v + 2, lane, 32, 0);
    uint64_t c = rocjitsu::arith::test::read_fp(input.v + 4, lane, 32, 0);
    if constexpr (MultiplyConstant) {
      c = b;
      b = 0x3eaaaaabU;
    } else {
      c = 0x3eaaaaabU;
    }

    double x = rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(a, 32), mode, 0);
    double y = rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(b, 32), mode, 1);
    double z = rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(c, 32), mode, 2);
    double value = double(std::fma(float(x), float(y), float(z)));
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = !(value > 0) ? 0 : std::min(value, 1.0);
    }
    uint64_t expected = rocjitsu::arith::test::fp_bits(value, 32);
    rocjitsu::arith::test::write_fp(expected_registers.v + aliasing_offset, lane, 32, expected, 0);
  }
}

// Exercise this instruction contract for either wave size.
template <int Lanes, bool MultiplyConstant, typename Call>
void check_immediate_fma_f32(Call entry_point, bool gfx12) {
  static_assert(Lanes == 32 || Lanes == 64);
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  // Normalize the two ISA operand orders while retaining the immediate's type.
  auto call = [&](uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint32_t immediate,
                  uint32_t *fp_status) {
    if constexpr (MultiplyConstant) {
      return entry_point(flags, exec_mask, mode, d, a, immediate, b, fp_status);
    } else {
      return entry_point(flags, exec_mask, mode, d, a, b, immediate, fp_status);
    }
  };
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = rocjitsu::arith::test::make_fp_inputs<32, Lanes>();
  // (1 + 2^-23)^2 - 1 lies just above 2^-22: only upward rounding
  // advances the last bit. Literal and register operand orders share this witness.
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t rounding :
         {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
      Registers r = input;
      r.v[0][0] = 0x3f800001U;
      r.v[2][0] = MultiplyConstant ? 0xbf800000U : 0x3f800001U;
      uint32_t status = 0x80000000U;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | rounding, Mask(1),
                     0, r.v + 6, r.v, r.v + 2, MultiplyConstant ? 0x3f800001U : 0xbf800000U,
                     &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[6][0], rounding == RJ_ARITH_FP_ROUND_UP ? 0x34800001U : 0x34800000U);
      EXPECT_EQ(status, 0x80000000U);
    }
  }
  // A guest MODE word may pass through gfx11 literal forms unchanged.
  for (uint64_t cpu : supported_cpu_levels()) {
    Registers r = input;
    r.v[0][0] = r.v[2][0] = 0x3f800000U;
    uint32_t status = 0;
    int result =
        call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | RJ_ARITH_FP_DX10_CLAMP_OFF,
             Mask(1), 0, r.v + 6, r.v, r.v + 2, 0x3f800000U, &status);
    ASSERT_EQ(result, gfx12 ? RJ_ARITH_ERROR_INVALID_FLAGS : RJ_ARITH_SUCCESS);
    if (!gfx12) {
      EXPECT_EQ(r.v[6][0], 0x40000000U);
    }
  }
  check_fma_simd<32, Lanes>(
      [&](uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
          uint32_t *status) {
        return call(flags, exec_mask, mode, d, a, b, uint32_t(c[0][0]), status);
      },
      {0}, /*register_operands=*/false);

  rocjitsu::arith::test::expect_invalid_fp_requests([&](uint64_t flags, uint64_t mode) {
    Registers r = input;
    int aliasing_offset = 6;
    Mask exec_mask = ~Mask(0);
    uint32_t fp_status = 0x80000000U;

    int result =
        call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, 0x3eaaaaabU, &fp_status);
    r.expect_preserved(input, 0, 0);
    EXPECT_EQ(fp_status, 0x80000000U);
    EXPECT_EQ(call(flags, exec_mask, mode, nullptr, nullptr, nullptr, 0x3eaaaaabU, &fp_status),
              result);
    return result;
  });

  // Default guest rounding and normal fractional inputs: the existing rocjitsu
  // arithmetic model reduces to IEEE arithmetic here. Special values are checked
  // by classification. No production helper or rj-arith result is a numeric oracle.
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : {uint64_t(0)}) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : {6, 0, 2}) {
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << mode << '/'
                                              << exec_mask << '/' << aliasing_offset);
            Registers r = input, expected_registers = input;
            uint64_t flags = cpu | semantics;
            uint32_t fp_status = 0x80000000U;

            int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2,
                              0x3eaaaaabU, &fp_status);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            ASSERT_EQ(result, RJ_ARITH_SUCCESS);
            EXPECT_EQ(fp_status, 0x80000000U);
            compute_immediate_fma_f32_expected<Lanes, MultiplyConstant>(
                input, exec_mask, mode, aliasing_offset, expected_registers);

            rocjitsu::arith::test::expect_fp_result<32>(input, r, expected_registers, exec_mask,
                                                        aliasing_offset, 0, semantics,
                                                        /*atol=*/1e-5, /*rtol=*/1e-6);
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
