// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar integer comparisons and bit tests.

#include "scalar/compare_helpers.h"

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmpk_eq_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmpk_eq_i32(f, m, d, a, uint16_t(b));
      },
      [](auto a, auto b) {
        return rocjitsu::arith::test::signed_word(a, 32) ==
               rocjitsu::arith::test::signed_word(uint16_t(b), 16);
      });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmpk_lg_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmpk_lg_i32(f, m, d, a, uint16_t(b));
      },
      [](auto a, auto b) {
        return rocjitsu::arith::test::signed_word(a, 32) !=
               rocjitsu::arith::test::signed_word(uint16_t(b), 16);
      });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmpk_gt_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmpk_gt_i32(f, m, d, a, uint16_t(b));
      },
      [](auto a, auto b) {
        return rocjitsu::arith::test::signed_word(a, 32) >
               rocjitsu::arith::test::signed_word(uint16_t(b), 16);
      });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmpk_ge_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmpk_ge_i32(f, m, d, a, uint16_t(b));
      },
      [](auto a, auto b) {
        return rocjitsu::arith::test::signed_word(a, 32) >=
               rocjitsu::arith::test::signed_word(uint16_t(b), 16);
      });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmpk_lt_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmpk_lt_i32(f, m, d, a, uint16_t(b));
      },
      [](auto a, auto b) {
        return rocjitsu::arith::test::signed_word(a, 32) <
               rocjitsu::arith::test::signed_word(uint16_t(b), 16);
      });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmpk_le_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmpk_le_i32(f, m, d, a, uint16_t(b));
      },
      [](auto a, auto b) {
        return rocjitsu::arith::test::signed_word(a, 32) <=
               rocjitsu::arith::test::signed_word(uint16_t(b), 16);
      });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmpk_eq_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmpk_eq_u32(f, m, d, a, uint16_t(b));
      },
      [](auto a, auto b) { return uint32_t(a) == uint16_t(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmpk_lg_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmpk_lg_u32(f, m, d, a, uint16_t(b));
      },
      [](auto a, auto b) { return uint32_t(a) != uint16_t(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmpk_gt_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmpk_gt_u32(f, m, d, a, uint16_t(b));
      },
      [](auto a, auto b) { return uint32_t(a) > uint16_t(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmpk_ge_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmpk_ge_u32(f, m, d, a, uint16_t(b));
      },
      [](auto a, auto b) { return uint32_t(a) >= uint16_t(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmpk_lt_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmpk_lt_u32(f, m, d, a, uint16_t(b));
      },
      [](auto a, auto b) { return uint32_t(a) < uint16_t(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmpk_le_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmpk_le_u32(f, m, d, a, uint16_t(b));
      },
      [](auto a, auto b) { return uint32_t(a) <= uint16_t(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_eq_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_eq_i32(f, m, d, a, b);
      },
      [](auto a, auto b) {
        return rocjitsu::arith::test::signed_word(a, 32) ==
               rocjitsu::arith::test::signed_word(b, 32);
      });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_lg_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_lg_i32(f, m, d, a, b);
      },
      [](auto a, auto b) {
        return rocjitsu::arith::test::signed_word(a, 32) !=
               rocjitsu::arith::test::signed_word(b, 32);
      });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_gt_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_gt_i32(f, m, d, a, b);
      },
      [](auto a, auto b) {
        return rocjitsu::arith::test::signed_word(a, 32) >
               rocjitsu::arith::test::signed_word(b, 32);
      });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_ge_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_ge_i32(f, m, d, a, b);
      },
      [](auto a, auto b) {
        return rocjitsu::arith::test::signed_word(a, 32) >=
               rocjitsu::arith::test::signed_word(b, 32);
      });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_lt_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_lt_i32(f, m, d, a, b);
      },
      [](auto a, auto b) {
        return rocjitsu::arith::test::signed_word(a, 32) <
               rocjitsu::arith::test::signed_word(b, 32);
      });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_le_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_le_i32(f, m, d, a, b);
      },
      [](auto a, auto b) {
        return rocjitsu::arith::test::signed_word(a, 32) <=
               rocjitsu::arith::test::signed_word(b, 32);
      });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_eq_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_eq_u32(f, m, d, a, b);
      },
      [](auto a, auto b) { return uint32_t(a) == uint32_t(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_lg_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_lg_u32(f, m, d, a, b);
      },
      [](auto a, auto b) { return uint32_t(a) != uint32_t(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_gt_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_gt_u32(f, m, d, a, b);
      },
      [](auto a, auto b) { return uint32_t(a) > uint32_t(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_ge_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_ge_u32(f, m, d, a, b);
      },
      [](auto a, auto b) { return uint32_t(a) >= uint32_t(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_lt_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_lt_u32(f, m, d, a, b);
      },
      [](auto a, auto b) { return uint32_t(a) < uint32_t(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_le_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_le_u32(f, m, d, a, b);
      },
      [](auto a, auto b) { return uint32_t(a) <= uint32_t(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_bitcmp0_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_bitcmp0_b32(f, m, d, a, b);
      },
      [](auto a, auto b) { return ((a >> (b % 32)) & 1) == 0; });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_bitcmp1_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_bitcmp1_b32(f, m, d, a, b);
      },
      [](auto a, auto b) { return ((a >> (b % 32)) & 1) == 1; });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_bitcmp0_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<64, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_bitcmp0_b64(f, m, d, a, b);
      },
      [](auto a, auto b) { return ((a >> (b % 64)) & 1) == 0; });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_bitcmp1_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<64, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_bitcmp1_b64(f, m, d, a, b);
      },
      [](auto a, auto b) { return ((a >> (b % 64)) & 1) == 1; });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_eq_u64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<64, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_eq_u64(f, m, d, a, b);
      },
      [](auto a, auto b) { return uint64_t(a) == uint64_t(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_lg_u64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<64, false>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_lg_u64(f, m, d, a, b);
      },
      [](auto a, auto b) { return uint64_t(a) != uint64_t(b); });
}
