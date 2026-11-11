// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar FP comparisons and FP-status results.

#include "scalar/compare_helpers.h"

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_lt_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_lt_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return a < b; });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_lt_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<16, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_lt_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return a < b; });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_eq_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_eq_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return a == b; });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_eq_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<16, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_eq_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return a == b; });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_le_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_le_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return a <= b; });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_le_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<16, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_le_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return a <= b; });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_gt_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_gt_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return a > b; });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_gt_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<16, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_gt_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return a > b; });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_lg_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_lg_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return a != b && !std::isnan(a) && !std::isnan(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_lg_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<16, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_lg_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return a != b && !std::isnan(a) && !std::isnan(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_ge_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_ge_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return a >= b; });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_ge_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<16, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_ge_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return a >= b; });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_o_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_o_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_o_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<16, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_o_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_u_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_u_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return std::isnan(a) || std::isnan(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_u_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<16, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_u_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return std::isnan(a) || std::isnan(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_nge_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_nge_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return !(a >= b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_nge_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<16, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_nge_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return !(a >= b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_nlg_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_nlg_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return !(a != b) || std::isnan(a) || std::isnan(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_nlg_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<16, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_nlg_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return !(a != b) || std::isnan(a) || std::isnan(b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_ngt_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_ngt_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return !(a > b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_ngt_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<16, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_ngt_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return !(a > b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_nle_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_nle_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return !(a <= b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_nle_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<16, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_nle_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return !(a <= b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_neq_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_neq_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return !(a == b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_neq_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<16, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_neq_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return !(a == b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_nlt_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<32, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_nlt_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return !(a < b); });
}

// Check scalar predicates, SCC aliases, and FP-status expectations.
TEST(s_cmp_nlt_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_compare<16, true>(
      [](auto f, auto m, auto d, auto a, auto b, auto status) {
        (void)status;
        return rj_arith_s_cmp_nlt_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b) { return !(a < b); });
}
