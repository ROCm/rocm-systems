// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar moves, conditional selection, and immediate extension.

#include "scalar/integer_helpers.h"

// Check integer results and SCC against the supplied scalar reference.
TEST(s_mov_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *) {
        return rj_arith_s_mov_b32(f, m, d, a);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint64_t result = a;
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_cmov_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t cc, uint32_t *) {
        return rj_arith_s_cmov_b32(f, m, d, ~uint32_t(a), cc);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint64_t result = cc & 1 ? uint32_t(~a) : a;
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_mov_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *) {
        return rj_arith_s_mov_b64(f, m, d, a);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t result = a;
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_cmov_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t, uint32_t cc, uint32_t *) {
        return rj_arith_s_cmov_b64(f, m, d, ~uint64_t(a), cc);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t result = cc & 1 ? uint64_t(~a) : a;
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_getpc_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *) {
        return rj_arith_s_getpc_b64(f, m, d, a);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint64_t result = raw_a + 4;
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_movk_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_movk_i32(f, m, d, uint16_t(b));
      },
      [](uint64_t, uint64_t raw_b, uint32_t cc) {
        uint32_t b = uint32_t(raw_b);
        uint64_t result = uint32_t(rocjitsu::arith::test::signed_word(uint16_t(b), 16));
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_cmovk_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t, uint64_t b, uint32_t cc, uint32_t *) {
        return rj_arith_s_cmovk_i32(f, m, d, uint16_t(b), cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result =
            cc & 1 ? uint32_t(rocjitsu::arith::test::signed_word(uint16_t(b), 16)) : a;
        return std::make_pair(result, cc);
      },
      false);
}
