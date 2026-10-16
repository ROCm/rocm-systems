// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar shifts and shift/boolean combinations.

#include "scalar/integer_helpers.h"

// Check integer results and SCC against the supplied scalar reference.
TEST(s_lshl_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_lshl_b32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int shift = int(b % 32);
        for (int bit = 0; bit < 32; ++bit) {
          int source = bit - shift;
          bool set = source < 0 ? false : source >= 32 ? false : bool((a >> source) & 1);
          result |= uint64_t(set) << bit;
        }
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_lshr_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_lshr_b32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int shift = int(b % 32);
        for (int bit = 0; bit < 32; ++bit) {
          int source = bit + shift;
          bool set = source < 0 ? false : source >= 32 ? false : bool((a >> source) & 1);
          result |= uint64_t(set) << bit;
        }
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_ashr_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_ashr_i32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int shift = int(b % 32);
        for (int bit = 0; bit < 32; ++bit) {
          int source = bit + shift;
          bool set = source < 0     ? false
                     : source >= 32 ? bool(a >> (32 - 1))
                                    : bool((a >> source) & 1);
          result |= uint64_t(set) << bit;
        }
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_lshl_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_lshl_b64(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = 0;
        int shift = int(b % 64);
        for (int bit = 0; bit < 64; ++bit) {
          int source = bit - shift;
          bool set = source < 0 ? false : source >= 64 ? false : bool((a >> source) & 1);
          result |= uint64_t(set) << bit;
        }
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_lshr_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_lshr_b64(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = 0;
        int shift = int(b % 64);
        for (int bit = 0; bit < 64; ++bit) {
          int source = bit + shift;
          bool set = source < 0 ? false : source >= 64 ? false : bool((a >> source) & 1);
          result |= uint64_t(set) << bit;
        }
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_ashr_i64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_ashr_i64(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = 0;
        int shift = int(b % 64);
        for (int bit = 0; bit < 64; ++bit) {
          int source = bit + shift;
          bool set = source < 0     ? false
                     : source >= 64 ? bool(a >> (64 - 1))
                                    : bool((a >> source) & 1);
          result |= uint64_t(set) << bit;
        }
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_lshl1_add_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_lshl1_add_u32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = (uint64_t(a) << 1) + b;
        cc = result > UINT32_MAX;
        result = uint32_t(result);
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_lshl2_add_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_lshl2_add_u32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = (uint64_t(a) << 2) + b;
        cc = result > UINT32_MAX;
        result = uint32_t(result);
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_lshl3_add_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_lshl3_add_u32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = (uint64_t(a) << 3) + b;
        cc = result > UINT32_MAX;
        result = uint32_t(result);
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_lshl4_add_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_lshl4_add_u32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = (uint64_t(a) << 4) + b;
        cc = result > UINT32_MAX;
        result = uint32_t(result);
        return std::make_pair(result, cc);
      },
      true);
}
