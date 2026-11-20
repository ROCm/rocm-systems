// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar moves, conditional selection, and immediate extension.

#include "rj_arith/detail/rj_arith_scalar.h"
#include "scalar/integer_helpers.h"

#include <array>

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
        const uint64_t next = (raw_a + 4) & 0x0000ffffffffffffULL;
        uint64_t result = (next & (1ULL << 47)) ? next | 0xffff000000000000ULL : next;
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

// PC export wraps at 48 bits; only gfx11 sign-extends the exported address.
TEST(s_getpc_b64_gfx12, Arithmetic) {
  for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
    for (const auto &test : {std::array<uint64_t, 3>{0x0000800012345000ULL, 0xffff800012345004ULL,
                                                     0x0000800012345004ULL},
                             std::array<uint64_t, 3>{0x0000fffffffffffcULL, 0, 0}}) {
      uint64_t result;
      ASSERT_EQ(rj_arith_s_getpc_b64(semantics, 0, &result, test[0]), RJ_ARITH_SUCCESS);
      EXPECT_EQ(result, test[1]);
      ASSERT_EQ(rj_arith_s_getpc_b64_gfx12(semantics, 0, &result, test[0]), RJ_ARITH_SUCCESS);
      EXPECT_EQ(result, test[2]);
    }
  }
}
