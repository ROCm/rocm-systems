// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar FP comparisons and FP-status results.

#include "rj_arith/detail/rj_arith_scalar.h"
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

// gfx1250 inherits ordered/unordered CPU comparisons, but no empirical FP-status model.
TEST(ScalarCompareGfx1250, PredicatesAndPolicy) {
  using Call = int (*)(uint64_t, uint64_t, uint32_t *, uint32_t, uint32_t, uint32_t *);
  struct Case {
    Call call;
    int width;
    int predicate;
  };
  const Case cases[] = {
      {rj_arith_s_cmp_lt_f32_gfx1250, 32, 1},   {rj_arith_s_cmp_lt_f16_gfx1250, 16, 1},
      {rj_arith_s_cmp_eq_f32_gfx1250, 32, 2},   {rj_arith_s_cmp_eq_f16_gfx1250, 16, 2},
      {rj_arith_s_cmp_le_f32_gfx1250, 32, 3},   {rj_arith_s_cmp_le_f16_gfx1250, 16, 3},
      {rj_arith_s_cmp_gt_f32_gfx1250, 32, 4},   {rj_arith_s_cmp_gt_f16_gfx1250, 16, 4},
      {rj_arith_s_cmp_lg_f32_gfx1250, 32, 5},   {rj_arith_s_cmp_lg_f16_gfx1250, 16, 5},
      {rj_arith_s_cmp_ge_f32_gfx1250, 32, 6},   {rj_arith_s_cmp_ge_f16_gfx1250, 16, 6},
      {rj_arith_s_cmp_o_f32_gfx1250, 32, 7},    {rj_arith_s_cmp_o_f16_gfx1250, 16, 7},
      {rj_arith_s_cmp_u_f32_gfx1250, 32, 8},    {rj_arith_s_cmp_u_f16_gfx1250, 16, 8},
      {rj_arith_s_cmp_nge_f32_gfx1250, 32, 9},  {rj_arith_s_cmp_nge_f16_gfx1250, 16, 9},
      {rj_arith_s_cmp_nlg_f32_gfx1250, 32, 10}, {rj_arith_s_cmp_nlg_f16_gfx1250, 16, 10},
      {rj_arith_s_cmp_ngt_f32_gfx1250, 32, 11}, {rj_arith_s_cmp_ngt_f16_gfx1250, 16, 11},
      {rj_arith_s_cmp_nle_f32_gfx1250, 32, 12}, {rj_arith_s_cmp_nle_f16_gfx1250, 16, 12},
      {rj_arith_s_cmp_neq_f32_gfx1250, 32, 13}, {rj_arith_s_cmp_neq_f16_gfx1250, 16, 13},
      {rj_arith_s_cmp_nlt_f32_gfx1250, 32, 14}, {rj_arith_s_cmp_nlt_f16_gfx1250, 16, 14}};
  for (const auto &test : cases) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (bool flush : {false, true}) {
        for (int i = 0; i < 100; ++i) {
          const uint32_t a = test.width == 16
                                 ? rocjitsu::arith::test::compare_class_value<16>(i % 10)
                                 : rocjitsu::arith::test::compare_class_value<32>(i % 10);
          const uint32_t b = test.width == 16
                                 ? rocjitsu::arith::test::compare_class_value<16>(i / 10)
                                 : rocjitsu::arith::test::compare_class_value<32>(i / 10);
          const bool effective_flush = flush && semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY;
          const float x =
              test.width == 16
                  ? rocjitsu::arith::test::compare_float_value<16>(a, 0, 0, effective_flush)
                  : rocjitsu::arith::test::compare_float_value<32>(a, 0, 0, effective_flush);
          const float y =
              test.width == 16
                  ? rocjitsu::arith::test::compare_float_value<16>(b, 0, 0, effective_flush)
                  : rocjitsu::arith::test::compare_float_value<32>(b, 0, 0, effective_flush);
          const bool ordered = !std::isnan(x) && !std::isnan(y);
          const bool expected[] = {
              false,     x<y, x == y, x <= y, x> y, ordered && x != y, x >= y,    ordered, !ordered,
              !(x >= y), !ordered || x == y,        !(x > y),          !(x <= y), x != y,  !(x < y),
              true};
          uint32_t out = a, status = 0x80000000U;
          const int error = test.call(semantics | (flush ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0),
                                      0, &out, out, b, &status);
          const bool strict = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                              semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
          ASSERT_EQ(error, strict ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS : RJ_ARITH_SUCCESS);
          EXPECT_EQ(out, strict ? a : uint32_t(expected[test.predicate]));
          EXPECT_EQ(status, 0x80000000U);
        }
      }
    }
    uint32_t out = 123, status = 456;
    rocjitsu::arith::test::expect_invalid_flags(
        [&](auto flags, auto mode) {
          int error = test.call(flags, mode, &out, 0, 0, &status);
          EXPECT_EQ(out, 123U);
          EXPECT_EQ(status, 456U);
          return error;
        },
        0);
  }
}
