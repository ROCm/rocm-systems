// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar shifts and shift/boolean combinations.

#include "integer/shift_reference.h"
#include "scalar/integer_helpers.h"

namespace {

// Reuse the bit-by-bit vector shift reference; scalar shifts additionally set SCC.
template <typename Word, typename Call>
void check_shift(Call call, rocjitsu::arith::test::Shift operation) {
  rocjitsu::arith::test::check_scalar_integer<Word>(
      [=](uint64_t flags, uint64_t mode, Word *d, uint64_t a, uint64_t b, uint32_t, uint32_t *scc) {
        return call(flags, mode, d, a, b, scc);
      },
      [=](uint64_t a, uint64_t b, uint32_t) {
        uint64_t result =
            rocjitsu::arith::test::shift_reference(8 * sizeof(Word), operation, uint32_t(b), a);
        return std::make_pair(result, uint32_t(result != 0));
      },
      true);
}

} // namespace

// Check integer results and SCC against the supplied scalar reference.
TEST(s_lshl_b32, Arithmetic) {
  check_shift<uint32_t>(rj_arith_s_lshl_b32, rocjitsu::arith::test::Shift::Left);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_lshr_b32, Arithmetic) {
  check_shift<uint32_t>(rj_arith_s_lshr_b32, rocjitsu::arith::test::Shift::LogicalRight);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_ashr_i32, Arithmetic) {
  check_shift<uint32_t>(rj_arith_s_ashr_i32, rocjitsu::arith::test::Shift::ArithmeticRight);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_lshl_b64, Arithmetic) {
  check_shift<uint64_t>(rj_arith_s_lshl_b64, rocjitsu::arith::test::Shift::Left);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_lshr_b64, Arithmetic) {
  check_shift<uint64_t>(rj_arith_s_lshr_b64, rocjitsu::arith::test::Shift::LogicalRight);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_ashr_i64, Arithmetic) {
  check_shift<uint64_t>(rj_arith_s_ashr_i64, rocjitsu::arith::test::Shift::ArithmeticRight);
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
