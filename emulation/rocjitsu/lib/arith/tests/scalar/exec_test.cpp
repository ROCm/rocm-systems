// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for EXEC boolean operations, saved masks, and SCC results.

#include "scalar/integer_helpers.h"

namespace {
// Check EXEC transformations, saved masks, and SCC results for the supplied entry point.
template <class Word, class Call> void check_exec(Call call, int truth, bool save) {
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      std::minstd_rand random;
      for (int sample = 0; sample < 128; ++sample) {
        Word a = Word(rocjitsu::arith::test::scalar_input(random, sample)),
             old = Word(rocjitsu::arith::test::scalar_input(random, (sample * 3 + 1) % 128)),
             expected = 0;
        // truth encodes the two-input boolean function, independently of the instruction.
        for (int bit = 0; bit < int(sizeof(Word) * 8); ++bit) {
          int index = int((a >> bit) & 1) | int(((old >> bit) & 1) << 1);
          expected |= Word((truth >> index) & 1) << bit;
        }
        for (bool alias : {false, true}) {
          Word d = 123, exec_mask = old;
          uint32_t scc = 456;
          ASSERT_EQ(call(cpu | semantics, 0, &d, a, old, alias ? &d : &exec_mask, &scc),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(d, alias || !save ? expected : old);
          EXPECT_EQ(exec_mask, alias ? old : expected);
          EXPECT_EQ(scc, uint32_t(expected != 0));
          ASSERT_EQ(call(cpu | semantics, 0, &d, a, old, nullptr, &scc), RJ_ARITH_SUCCESS);
          EXPECT_EQ(d, save ? old : expected);
        }
      }
    }
  }
  Word d = 123, e = 789;
  uint32_t cc = 456;
  rocjitsu::arith::test::expect_invalid_flags(
      [&](auto f, auto m) {
        int error = call(f, m, &d, 0, 0, &e, &cc);
        EXPECT_EQ(d, Word(123));
        EXPECT_EQ(e, Word(789));
        EXPECT_EQ(cc, 456U);
        return error;
      },
      0);
}
} // namespace

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_and_saveexec_b32, Arithmetic) { check_exec<uint32_t>(rj_arith_s_and_saveexec_b32, 8, true); }

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_or_saveexec_b32, Arithmetic) { check_exec<uint32_t>(rj_arith_s_or_saveexec_b32, 14, true); }

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_xor_saveexec_b32, Arithmetic) { check_exec<uint32_t>(rj_arith_s_xor_saveexec_b32, 6, true); }

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_nand_saveexec_b32, Arithmetic) {
  check_exec<uint32_t>(rj_arith_s_nand_saveexec_b32, 7, true);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_nor_saveexec_b32, Arithmetic) { check_exec<uint32_t>(rj_arith_s_nor_saveexec_b32, 1, true); }

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_xnor_saveexec_b32, Arithmetic) {
  check_exec<uint32_t>(rj_arith_s_xnor_saveexec_b32, 9, true);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_and_not0_saveexec_b32, Arithmetic) {
  check_exec<uint32_t>(rj_arith_s_and_not0_saveexec_b32, 4, true);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_or_not0_saveexec_b32, Arithmetic) {
  check_exec<uint32_t>(rj_arith_s_or_not0_saveexec_b32, 13, true);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_and_not1_saveexec_b32, Arithmetic) {
  check_exec<uint32_t>(rj_arith_s_and_not1_saveexec_b32, 2, true);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_or_not1_saveexec_b32, Arithmetic) {
  check_exec<uint32_t>(rj_arith_s_or_not1_saveexec_b32, 11, true);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_and_not0_wrexec_b32, Arithmetic) {
  check_exec<uint32_t>(rj_arith_s_and_not0_wrexec_b32, 4, false);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_and_not1_wrexec_b32, Arithmetic) {
  check_exec<uint32_t>(rj_arith_s_and_not1_wrexec_b32, 2, false);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_and_saveexec_b64, Arithmetic) { check_exec<uint64_t>(rj_arith_s_and_saveexec_b64, 8, true); }

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_or_saveexec_b64, Arithmetic) { check_exec<uint64_t>(rj_arith_s_or_saveexec_b64, 14, true); }

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_xor_saveexec_b64, Arithmetic) { check_exec<uint64_t>(rj_arith_s_xor_saveexec_b64, 6, true); }

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_nand_saveexec_b64, Arithmetic) {
  check_exec<uint64_t>(rj_arith_s_nand_saveexec_b64, 7, true);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_nor_saveexec_b64, Arithmetic) { check_exec<uint64_t>(rj_arith_s_nor_saveexec_b64, 1, true); }

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_xnor_saveexec_b64, Arithmetic) {
  check_exec<uint64_t>(rj_arith_s_xnor_saveexec_b64, 9, true);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_and_not0_saveexec_b64, Arithmetic) {
  check_exec<uint64_t>(rj_arith_s_and_not0_saveexec_b64, 4, true);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_or_not0_saveexec_b64, Arithmetic) {
  check_exec<uint64_t>(rj_arith_s_or_not0_saveexec_b64, 13, true);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_and_not1_saveexec_b64, Arithmetic) {
  check_exec<uint64_t>(rj_arith_s_and_not1_saveexec_b64, 2, true);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_or_not1_saveexec_b64, Arithmetic) {
  check_exec<uint64_t>(rj_arith_s_or_not1_saveexec_b64, 11, true);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_and_not0_wrexec_b64, Arithmetic) {
  check_exec<uint64_t>(rj_arith_s_and_not0_wrexec_b64, 4, false);
}

// Check EXEC transformations, saved masks, and SCC results.
TEST(s_and_not1_wrexec_b64, Arithmetic) {
  check_exec<uint64_t>(rj_arith_s_and_not1_wrexec_b64, 2, false);
}
