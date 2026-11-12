// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar boolean operations and SCC results.

#include "scalar/integer_helpers.h"

// Check integer results and SCC against the supplied scalar reference.
TEST(s_and_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_and_b32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = (a & b) & 0xffffffffULL;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_or_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_or_b32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = (a | b) & 0xffffffffULL;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_xor_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_xor_b32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = (a ^ b) & 0xffffffffULL;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_nand_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_nand_b32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = (~(a & b)) & 0xffffffffULL;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_nor_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_nor_b32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = (~(a | b)) & 0xffffffffULL;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_xnor_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_xnor_b32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = (~(a ^ b)) & 0xffffffffULL;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_and_not1_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_and_not1_b32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = (a & ~b) & 0xffffffffULL;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_or_not1_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_or_not1_b32(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = (a | ~b) & 0xffffffffULL;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_not_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_not_b32(f, m, d, a, out_cc);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint64_t result = (~a) & 0xffffffffULL;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_brev_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *) {
        return rj_arith_s_brev_b32(f, m, d, a);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint64_t result = rocjitsu::arith::test::reverse_word(a, 32);
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_and_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_and_b64(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = (a & b) & UINT64_MAX;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_or_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_or_b64(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = (a | b) & UINT64_MAX;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_xor_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_xor_b64(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = (a ^ b) & UINT64_MAX;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_nand_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_nand_b64(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = (~(a & b)) & UINT64_MAX;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_nor_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_nor_b64(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = (~(a | b)) & UINT64_MAX;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_xnor_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_xnor_b64(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = (~(a ^ b)) & UINT64_MAX;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_and_not1_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_and_not1_b64(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = (a & ~b) & UINT64_MAX;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_or_not1_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_or_not1_b64(f, m, d, a, b, out_cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = (a | ~b) & UINT64_MAX;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_not_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_not_b64(f, m, d, a, out_cc);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t result = (~a) & UINT64_MAX;
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_brev_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *) {
        return rj_arith_s_brev_b64(f, m, d, a);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t result = rocjitsu::arith::test::reverse_word(a, 64);
        return std::make_pair(result, cc);
      },
      false);
}
