// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for wordwise boolean operations and the stochastic-rounding LFSR step.

#include "integer/boolean_test.h"
#include "rj_arith/detail/rj_arith_integer.h"

#include <gtest/gtest.h>

// Check boolean operations and supported truth-table controls.
TEST(IntegerAndB32W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 2>(rj_arith_v_and_b32_w32,
                                              rocjitsu::arith::test::Boolean::And32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerAndB32W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 2>(rj_arith_v_and_b32_w64,
                                              rocjitsu::arith::test::Boolean::And32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerOrB32W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 2>(rj_arith_v_or_b32_w32,
                                              rocjitsu::arith::test::Boolean::Or32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerOrB32W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 2>(rj_arith_v_or_b32_w64,
                                              rocjitsu::arith::test::Boolean::Or32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerXorB32W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 2>(rj_arith_v_xor_b32_w32,
                                              rocjitsu::arith::test::Boolean::Xor32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerXorB32W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 2>(rj_arith_v_xor_b32_w64,
                                              rocjitsu::arith::test::Boolean::Xor32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerNotB32W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 1>(rj_arith_v_not_b32_w32,
                                              rocjitsu::arith::test::Boolean::Not32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerNotB32W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 1>(rj_arith_v_not_b32_w64,
                                              rocjitsu::arith::test::Boolean::Not32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerAndB16W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 2>(rj_arith_v_and_b16_w32,
                                              rocjitsu::arith::test::Boolean::And16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerAndB16W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 2>(rj_arith_v_and_b16_w64,
                                              rocjitsu::arith::test::Boolean::And16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerOrB16W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 2>(rj_arith_v_or_b16_w32,
                                              rocjitsu::arith::test::Boolean::Or16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerOrB16W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 2>(rj_arith_v_or_b16_w64,
                                              rocjitsu::arith::test::Boolean::Or16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerXorB16W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 2>(rj_arith_v_xor_b16_w32,
                                              rocjitsu::arith::test::Boolean::Xor16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerXorB16W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 2>(rj_arith_v_xor_b16_w64,
                                              rocjitsu::arith::test::Boolean::Xor16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerNotB16W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 1>(rj_arith_v_not_b16_w32,
                                              rocjitsu::arith::test::Boolean::Not16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerNotB16W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 1>(rj_arith_v_not_b16_w64,
                                              rocjitsu::arith::test::Boolean::Not16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerXnorB32W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 2>(rj_arith_v_xnor_b32_w32,
                                              rocjitsu::arith::test::Boolean::Xnor);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerXnorB32W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 2>(rj_arith_v_xnor_b32_w64,
                                              rocjitsu::arith::test::Boolean::Xnor);
}

// Reference polynomial division over GF(2), independent of the SIMD feedback selection.
template <int Lanes, typename Call> void check_prng(Call call) {
  rocjitsu::arith::test::check_integer_word<Lanes, 1, true>(
      call,
      [](uint32_t a, uint32_t, uint32_t, uint32_t, uint32_t, int) {
        uint64_t polynomial = uint64_t(a) * 2;
        if (polynomial >> 32) {
          polynomial ^= 0x1000000c5ULL;
        }
        return uint32_t(polynomial);
      },
      RJ_ARITH_ALU_CLAMP, false);
}

TEST(IntegerPrngB32W32, Arithmetic) { check_prng<32>(rj_arith_v_prng_b32_w32); }
TEST(IntegerPrngB32W64, Arithmetic) { check_prng<64>(rj_arith_v_prng_b32_w64); }

// Check truth-table basis bits and a combined parity table with selectors, masks and aliases.
template <int Lanes, bool Half, typename Call> void check_bitop3(Call call) {
  using namespace rocjitsu::arith::test;
  const uint64_t controls =
      RJ_ARITH_BITOP3_TABLE_MASK | RJ_ARITH_ALU_CLAMP |
      (Half ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_C | RJ_ARITH_ALU_HIGH_D
            : 0);
  for (uint32_t table : {0U, 0x96U}) {
    check_integer_word<Lanes, 3, true>(
        [=](uint64_t flags, auto exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b,
            const uint32_t *const *c) { return call(flags, exec_mask, mode | table, d, a, b, c); },
        [=](uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t mode, int) {
          mode |= table;
          int sa = Half && (mode & RJ_ARITH_ALU_HIGH_A) ? 16 : 0;
          int sb = Half && (mode & RJ_ARITH_ALU_HIGH_B) ? 16 : 0;
          int sc = Half && (mode & RJ_ARITH_ALU_HIGH_C) ? 16 : 0;
          int sd = Half && (mode & RJ_ARITH_ALU_HIGH_D) ? 16 : 0;
          uint32_t result = 0;
          for (int bit = 0; bit < (Half ? 16 : 32); ++bit) {
            int index = (((a >> (sa + bit)) & 1) << 2) | (((b >> (sb + bit)) & 1) << 1) |
                        ((c >> (sc + bit)) & 1);
            result |= ((mode >> index) & 1) << bit;
          }
          return Half ? (d & ~(0xffffU << sd)) | (result << sd) : result;
        },
        controls);
  }
}

TEST(IntegerBitop3B32W32, Arithmetic) { check_bitop3<32, false>(rj_arith_v_bitop3_b32_w32); }
TEST(IntegerBitop3B32W64, Arithmetic) { check_bitop3<64, false>(rj_arith_v_bitop3_b32_w64); }
TEST(IntegerBitop3B16W32, Arithmetic) { check_bitop3<32, true>(rj_arith_v_bitop3_b16_w32); }
TEST(IntegerBitop3B16W64, Arithmetic) { check_bitop3<64, true>(rj_arith_v_bitop3_b16_w64); }
