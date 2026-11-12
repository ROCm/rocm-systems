// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar bitfield extraction, insertion, and mask generation.

#include "scalar/integer_helpers.h"

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bitset0_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_bitset0_b32(f, m, d, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = a & ~(1ULL << (b % 32));
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bitset1_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_bitset1_b32(f, m, d, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = a | (1ULL << (b % 32));
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bfe_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_bfe_u32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = rocjitsu::arith::test::field_word(a, uint32_t(b), 32, false);
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bfe_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_bfe_i32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = rocjitsu::arith::test::field_word(a, uint32_t(b), 32, true);
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bfm_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_bfm_b32(f, m, d, a, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        for (int bit = 0; bit < int(a % 32); ++bit) {
          if (bit + int(b % 32) < 32) {
            result |= 1ULL << (bit + int(b % 32));
          }
        }
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bitset0_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_bitset0_b64(f, m, d, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = a & ~(1ULL << (b % 64));
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bitset1_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_bitset1_b64(f, m, d, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = a | (1ULL << (b % 64));
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bfe_u64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_bfe_u64(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = rocjitsu::arith::test::field_word(a, uint32_t(b), 64, false);
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bfe_i64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_bfe_i64(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = rocjitsu::arith::test::field_word(a, uint32_t(b), 64, true);
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bfm_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_bfm_b64(f, m, d, a, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = 0;
        for (int bit = 0; bit < int(a % 64); ++bit) {
          if (bit + int(b % 64) < 64) {
            result |= 1ULL << (bit + int(b % 64));
          }
        }
        return std::make_pair(result, cc);
      },
      false);
}
