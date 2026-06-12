// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cts_common.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>

// Per-ISA generated test data headers.
#include "cts_data/cdna1/vector_bitop3.h"
#include "cts_data/cdna2/vector_bitop3.h"
#include "cts_data/cdna3/vector_bitop3.h"
#include "cts_data/cdna4/vector_bitop3.h"
#include "cts_data/gfx1250/vector_bitop3.h"
#include "cts_data/rdna1/vector_bitop3.h"
#include "cts_data/rdna2/vector_bitop3.h"
#include "cts_data/rdna3/vector_bitop3.h"
#include "cts_data/rdna3_5/vector_bitop3.h"
#include "cts_data/rdna4/vector_bitop3.h"

namespace {

using namespace rocjitsu;
using namespace rocjitsu::cts;

template <typename TestCaseT>
void run_vector_bitop3_tests(const IsaConfig &isa_cfg, const TestCaseT *tests, size_t count,
                             const char *label) {
  CtsFixture fixture(isa_cfg);
  uint32_t passed = 0;
  uint32_t failed = 0;
  constexpr uint32_t TEST_LANE = 0;

  for (size_t i = 0; i < count; ++i) {
    const auto &tc = tests[i];
    fixture.reset();

    fixture.write_vgpr(0, TEST_LANE, tc.inputs[0]);
    fixture.write_vgpr(2, TEST_LANE, tc.inputs[1]);
    fixture.write_vgpr(4, TEST_LANE, tc.inputs[2]);

    uint32_t words[4] = {tc.encoding[0], tc.encoding[1], 0u, 0u};
    bool ok = fixture.decode_execute(words);
    if (!ok) {
      ++failed;
      continue;
    }

    uint32_t actual = fixture.read_vgpr(6, TEST_LANE);
    if (actual != tc.expected_output) {
      ++failed;
      fprintf(stderr, "[%s/%s] case %zu (tt=0x%02X): output 0x%08X != expected 0x%08X\n",
              std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i,
              tc.truth_table, actual, tc.expected_output);
      continue;
    }

    ++passed;
  }

  EXPECT_EQ(failed, 0u) << isa_cfg.name << ": " << failed << " of " << count << " " << label
                        << " tests failed";
  fprintf(stderr, "[%s] %s: %u/%zu passed\n", std::string(isa_cfg.name).c_str(), label, passed,
          count);
}

#define CTS_VECTOR_BITOP3_TEST(ISA_NAME)                                                           \
  TEST(CtsVectorBitop3, ISA_NAME) {                                                                \
    const auto &cfg = *std::find_if(std::begin(kIsaConfigs), std::end(kIsaConfigs),                \
                                    [](const IsaConfig &c) { return c.name == #ISA_NAME; });       \
    if constexpr (ISA_NAME::NUM_VECTOR_BITOP3_TESTS > 0) {                                         \
      run_vector_bitop3_tests(cfg, ISA_NAME::VECTOR_BITOP3_TESTS,                                  \
                              ISA_NAME::NUM_VECTOR_BITOP3_TESTS, "vector_bitop3");                 \
    } else {                                                                                       \
      fprintf(stderr, "[%s] vector_bitop3: no tests (instruction not present)\n", #ISA_NAME);      \
    }                                                                                              \
  }

CTS_VECTOR_BITOP3_TEST(cdna1)
CTS_VECTOR_BITOP3_TEST(cdna2)
CTS_VECTOR_BITOP3_TEST(cdna3)
CTS_VECTOR_BITOP3_TEST(cdna4)
CTS_VECTOR_BITOP3_TEST(rdna1)
CTS_VECTOR_BITOP3_TEST(rdna2)
CTS_VECTOR_BITOP3_TEST(rdna3)
CTS_VECTOR_BITOP3_TEST(rdna3_5)
CTS_VECTOR_BITOP3_TEST(rdna4)
CTS_VECTOR_BITOP3_TEST(gfx1250)

} // namespace
