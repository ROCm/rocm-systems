// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for packed FP16 selection operations.

#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <random>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_min_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_min_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaxF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_max_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaxF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_max_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinNumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_min_num_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinNumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_min_num_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaxNumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_max_num_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaxNumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_max_num_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinimumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_minimum_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinimumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_minimum_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaximumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_maximum_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaximumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_maximum_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Selection preserves subnormal result bits even when arithmetic output flushing is enabled.
TEST(FpPackedSelection, PreservesDenormalResults) {
  const struct {
    decltype(&rj_arith_v_pk_min_f16_w32) call;
    uint32_t b;
  } cases[] = {
      {rj_arith_v_pk_min_f16_w32, 0x3c003c00U},     {rj_arith_v_pk_max_f16_w32, 0xbc00bc00U},
      {rj_arith_v_pk_min_num_f16_w32, 0x3c003c00U}, {rj_arith_v_pk_max_num_f16_w32, 0xbc00bc00U},
      {rj_arith_v_pk_minimum_f16_w32, 0x3c003c00U}, {rj_arith_v_pk_maximum_f16_w32, 0xbc00bc00U}};
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (const auto &sample : cases) {
        rocjitsu::arith::test::Registers<3, 32> r;
        for (int lane = 0; lane < 32; ++lane) {
          r.v[0][lane] = 0x80010001U;
          r.v[1][lane] = sample.b;
        }
        ASSERT_EQ(sample.call(cpu | semantics | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0xffffffffU, 0,
                              r.v + 2, r.v, r.v + 1, nullptr),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 32; ++lane) {
          EXPECT_EQ(r.v[2][lane], 0x80010001U);
        }
      }
    }
  }
}

// gfx1201 observations support first-NaN selection and preserved positive
// subnormals under output flushing. Compatibility intentionally keeps this
// hardware-fidelity improvement over the interpreter's canonicalize/flush policy.
TEST(FpPackedSelection, FirstNaN) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (auto call : {rj_arith_v_pk_minimum_f16_w32, rj_arith_v_pk_maximum_f16_w32}) {
        Registers<3, 32> r;
        r.v[0][0] = 0x7c04fe01U;
        r.v[1][0] = 0xfe017c04U;
        ASSERT_EQ(call(cpu | semantics, 1, 0, r.v + 2, r.v, r.v + 1, nullptr), RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[2][0], 0x7e04fe01U);
      }
    }
  }
}

namespace {

// Reference selection uses numeric comparisons; inputs and expected special
// encodings follow the inherited packed_select3_f16 contract.
uint16_t packed_pick(uint16_t a, uint16_t b, bool maximum, bool propagate) {
  const bool an = (a & 0x7fffU) > 0x7c00U, bn = (b & 0x7fffU) > 0x7c00U;
  if (an || bn) {
    if (propagate) {
      return 0x7e00U;
    }
    return an ? (bn ? uint16_t(a | 0x200U) : b) : a;
  }
  double x = rocjitsu::arith::test::fp_value(a, 16);
  double y = rocjitsu::arith::test::fp_value(b, 16);
  if (x == y) {
    return maximum ? a & b : a | b;
  }
  return (maximum ? x > y : x < y) ? a : b;
}

// Vary packed selectors/signs, guest flushing, CLAMP, aliases, and both EXEC widths.
template <int Lanes, class Call>
void check_packed_ternary(Call call, bool maximum, bool propagate) {
  using namespace rocjitsu::arith::test;
  for (uint64_t semantics : semantics_modes()) {
    uint32_t status = 0x80000000U;
    const int expected = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                             ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                             : RJ_ARITH_SUCCESS;
    EXPECT_EQ(call(semantics, 0, 0, nullptr, nullptr, nullptr, nullptr, &status), expected);
    EXPECT_EQ(status, 0x80000000U);
  }
  Registers<4, Lanes> input;
  std::minstd_rand random;
  for (int reg = 0; reg < 3; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] = uint32_t(random() >> 15) | (uint32_t(random() >> 15) << 16);
    }
  }
  // Distinct signaling/quiet payloads, signed zeros, and selected denormals.
  for (int reg = 0; reg < 3; ++reg) {
    input.v[reg][0] = 0xfc010001U + reg;
    input.v[reg][1] = 0x7e230000U | (reg == 1 ? 0x8000U : 0U);
  }
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t state : {uint64_t(0), RJ_ARITH_FP_FLUSH_INPUT_DENORMALS,
                           RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, RJ_ARITH_FP_DX10_CLAMP_OFF}) {
      for (uint32_t mode : {0U, RJ_ARITH_PK_CLAMP,
                            RJ_ARITH_PK_LO_A_HIGH | RJ_ARITH_PK_HI_C_LOW | RJ_ARITH_PK_NEG_LO_B |
                                RJ_ARITH_PK_NEG_HI_A}) {
        for (int alias : {3, 0, 1, 2}) {
          Registers<4, Lanes> r = input;
          using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
          constexpr Mask exec_mask = Mask(0x5a5a5a5aa5a5a5a5ULL);
          ASSERT_EQ(call(cpu | state | RJ_ARITH_SEMANTICS_COMPATIBILITY, exec_mask, mode,
                         r.v + alias, r.v, r.v + 1, r.v + 2, nullptr),
                    RJ_ARITH_SUCCESS);
          for (int lane = 0; lane < Lanes; ++lane) {
            uint32_t expected = 0;
            for (int half = 0; half < 2; ++half) {
              uint16_t operands[3];
              for (int reg = 0; reg < 3; ++reg) {
                bool high = half ? !(mode & (1U << (10 + reg))) : mode & (1U << (7 + reg));
                uint16_t value = input.v[reg][lane] >> (high ? 16 : 0);
                if (mode & (1U << (3 * half + reg))) {
                  value ^= 0x8000U;
                }
                if ((state & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && !(value & 0x7c00U)) {
                  value &= 0x8000U;
                }
                operands[reg] = value;
              }
              uint16_t value =
                  packed_pick(packed_pick(operands[0], operands[1], maximum, propagate),
                              operands[2], maximum, propagate);
              bool nan = (value & 0x7fffU) > 0x7c00U;
              if ((mode & RJ_ARITH_PK_CLAMP) && !(nan && (state & RJ_ARITH_FP_DX10_CLAMP_OFF))) {
                value = nan || (value & 0x8000U) ? 0 : value > 0x3c00U ? 0x3c00U : value;
              }
              if (propagate && (state & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && !(value & 0x7c00U)) {
                value &= 0x8000U;
              }
              expected |= uint32_t(value) << (16 * half);
            }
            EXPECT_EQ(r.v[alias][lane], (exec_mask >> lane) & 1 ? expected : input.v[alias][lane]);
          }
          r.expect_preserved(input, alias, 1);
        }
      }
    }
  }
}
} // namespace

TEST(FpPkTernarySelection, Wave32) {
  check_packed_ternary<32>(rj_arith_v_pk_minimum3_f16_w32, false, true);
  check_packed_ternary<32>(rj_arith_v_pk_maximum3_f16_w32, true, true);
  check_packed_ternary<32>(rj_arith_v_pk_min3_num_f16_w32, false, false);
  check_packed_ternary<32>(rj_arith_v_pk_max3_num_f16_w32, true, false);
}
TEST(FpPkTernarySelection, Wave64) {
  check_packed_ternary<64>(rj_arith_v_pk_minimum3_f16_w64, false, true);
  check_packed_ternary<64>(rj_arith_v_pk_maximum3_f16_w64, true, true);
  check_packed_ternary<64>(rj_arith_v_pk_min3_num_f16_w64, false, false);
  check_packed_ternary<64>(rj_arith_v_pk_max3_num_f16_w64, true, false);
}
