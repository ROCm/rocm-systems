// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar integer arithmetic and SCC results.

#include "scalar/integer_helpers.h"

// Check integer results and SCC against the supplied scalar reference.
TEST(s_add_co_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_add_co_u32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = int64_t(a), y = int64_t(b);
        int64_t value = x + y;
        result = uint32_t(value);
        cc = uint64_t(value) >> 32;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_sub_co_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_sub_co_u32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = int64_t(a), y = int64_t(b);
        int64_t value = x - y;
        result = uint32_t(value);
        cc = value < 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_add_co_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_add_co_i32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = rocjitsu::arith::test::signed_word(a, 32),
                y = rocjitsu::arith::test::signed_word(b, 32);
        int64_t value = x + y;
        result = uint32_t(value);
        cc = value < -2147483648LL || value > 2147483647LL;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_sub_co_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_sub_co_i32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = rocjitsu::arith::test::signed_word(a, 32),
                y = rocjitsu::arith::test::signed_word(b, 32);
        int64_t value = x - y;
        result = uint32_t(value);
        cc = value < -2147483648LL || value > 2147483647LL;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_add_co_ci_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t cc,
         uint32_t *out_cc) { return rj_arith_s_add_co_ci_u32(f, m, d, a, b, out_cc, cc); },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = int64_t(a), y = int64_t(b);
        int64_t value = x + y + (cc & 1);
        result = uint32_t(value);
        cc = uint64_t(value) >> 32;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_sub_co_ci_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t cc,
         uint32_t *out_cc) { return rj_arith_s_sub_co_ci_u32(f, m, d, a, b, out_cc, cc); },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = int64_t(a), y = int64_t(b);
        int64_t value = x - y - (cc & 1);
        result = uint32_t(value);
        cc = value < 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_add_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_add_u32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = int64_t(a), y = int64_t(b);
        int64_t value = x + y;
        result = uint32_t(value);
        cc = uint64_t(value) >> 32;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_sub_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_sub_u32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = int64_t(a), y = int64_t(b);
        int64_t value = x - y;
        result = uint32_t(value);
        cc = value < 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_add_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_add_i32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = rocjitsu::arith::test::signed_word(a, 32),
                y = rocjitsu::arith::test::signed_word(b, 32);
        int64_t value = x + y;
        result = uint32_t(value);
        cc = value < -2147483648LL || value > 2147483647LL;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_sub_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_sub_i32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = rocjitsu::arith::test::signed_word(a, 32),
                y = rocjitsu::arith::test::signed_word(b, 32);
        int64_t value = x - y;
        result = uint32_t(value);
        cc = value < -2147483648LL || value > 2147483647LL;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_addc_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t cc,
         uint32_t *out_cc) { return rj_arith_s_addc_u32(f, m, d, a, b, cc, out_cc); },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = int64_t(a), y = int64_t(b);
        int64_t value = x + y + (cc & 1);
        result = uint32_t(value);
        cc = uint64_t(value) >> 32;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_subb_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t cc,
         uint32_t *out_cc) { return rj_arith_s_subb_u32(f, m, d, a, b, cc, out_cc); },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = int64_t(a), y = int64_t(b);
        int64_t value = x - y - (cc & 1);
        result = uint32_t(value);
        cc = value < 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_addk_co_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_addk_co_i32(f, m, d, uint16_t(b), out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = rocjitsu::arith::test::signed_word(a, 32),
                y = rocjitsu::arith::test::signed_word(uint16_t(b), 16);
        int64_t value = x + y;
        result = uint32_t(value);
        cc = value < -2147483648LL || value > 2147483647LL;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_addk_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_addk_i32(f, m, d, uint16_t(b), out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = rocjitsu::arith::test::signed_word(a, 32),
                y = rocjitsu::arith::test::signed_word(uint16_t(b), 16);
        int64_t value = x + y;
        result = uint32_t(value);
        cc = value < -2147483648LL || value > 2147483647LL;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_mulk_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_mulk_i32(f, m, d, uint16_t(b));
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t x = rocjitsu::arith::test::signed_word(a, 32),
                y = rocjitsu::arith::test::signed_word(uint16_t(b), 16);
        int64_t value = x * y;
        result = uint32_t(value);
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_min_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_min_u32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        cc = a < b;
        result = cc ? a : b;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_min_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_min_i32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        cc = rocjitsu::arith::test::signed_word(a, 32) < rocjitsu::arith::test::signed_word(b, 32);
        result = cc ? a : b;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_max_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_max_u32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        cc = a > b;
        result = cc ? a : b;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_max_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_max_i32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        cc = rocjitsu::arith::test::signed_word(a, 32) > rocjitsu::arith::test::signed_word(b, 32);
        result = cc ? a : b;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_abs_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_abs_i32(f, m, d, a, out_cc);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint64_t result = 0;
        int64_t value = rocjitsu::arith::test::signed_word(uint32_t(a), 32);
        result = uint32_t(value < 0 ? -value : value);
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_absdiff_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_absdiff_i32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = 0;
        int64_t value = rocjitsu::arith::test::signed_word(uint32_t(a - b), 32);
        result = uint32_t(value < 0 ? -value : value);
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_sext_i32_i8, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *) {
        return rj_arith_s_sext_i32_i8(f, m, d, a);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint64_t result = uint32_t(rocjitsu::arith::test::signed_word(a, 8));
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_sext_i32_i16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *) {
        return rj_arith_s_sext_i32_i16(f, m, d, a);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint64_t result = uint32_t(rocjitsu::arith::test::signed_word(a, 16));
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_mul_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_mul_i32(f, m, d, a, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = uint32_t(uint64_t(a) * b);
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_mul_hi_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_mul_hi_u32(f, m, d, a, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = (uint64_t(a) * b) >> 32;
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_mul_hi_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_mul_hi_i32(f, m, d, a, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = uint64_t(rocjitsu::arith::test::signed_word(a, 32) *
                                   rocjitsu::arith::test::signed_word(b, 32)) >>
                          32;
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_add_nc_u64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_add_nc_u64(f, m, d, a, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = a + b;
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_sub_nc_u64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_sub_nc_u64(f, m, d, a, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = a - b;
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_mul_u64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_mul_u64(f, m, d, a, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = a * b;
        return std::make_pair(result, cc);
      },
      false);
}
