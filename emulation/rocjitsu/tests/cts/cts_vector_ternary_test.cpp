// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cts_common.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>

// Per-ISA generated test data headers.
#include "cts_data/cdna1/vector_alu_ternary.h"
#include "cts_data/cdna2/vector_alu_ternary.h"
#include "cts_data/cdna3/vector_alu_ternary.h"
#include "cts_data/cdna4/vector_alu_ternary.h"
#include "cts_data/gfx1250/vector_alu_ternary.h"
#include "cts_data/rdna1/vector_alu_ternary.h"
#include "cts_data/rdna2/vector_alu_ternary.h"
#include "cts_data/rdna3/vector_alu_ternary.h"
#include "cts_data/rdna3_5/vector_alu_ternary.h"
#include "cts_data/rdna4/vector_alu_ternary.h"

namespace {

using namespace rocjitsu;
using namespace rocjitsu::cts;

template <typename TestCaseT>
void run_vector_ternary_tests(const IsaConfig &isa_cfg, const TestCaseT *tests, size_t count,
                              const char *label) {
  CtsFixture fixture(isa_cfg);
  uint32_t passed = 0;
  uint32_t failed = 0;
  constexpr uint32_t TEST_LANE = 0;

  for (size_t i = 0; i < count; ++i) {
    const auto &tc = tests[i];
    fixture.reset();

    // VOP3 ternary: v0=src0, v2=src1, v4=src2, v6=dst.
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
      fprintf(stderr,
              "[%s/%s] case %zu: output 0x%08X != expected 0x%08X "
              "(inputs: 0x%08X, 0x%08X, 0x%08X)\n",
              std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i, actual,
              tc.expected_output, tc.inputs[0], tc.inputs[1], tc.inputs[2]);
      continue;
    }

    ++passed;
  }

  EXPECT_EQ(failed, 0u) << isa_cfg.name << ": " << failed << " of " << count << " " << label
                        << " tests failed";
  fprintf(stderr, "[%s] %s: %u/%zu passed\n", std::string(isa_cfg.name).c_str(), label, passed,
          count);
}

#define CTS_VECTOR_TERNARY_TEST(ISA_NAME)                                                          \
  TEST(CtsVectorTernary, ISA_NAME) {                                                               \
    const auto &cfg = *std::find_if(std::begin(kIsaConfigs), std::end(kIsaConfigs),                \
                                    [](const IsaConfig &c) { return c.name == #ISA_NAME; });       \
    run_vector_ternary_tests(cfg, ISA_NAME::VECTOR_ALU_TERNARY_TESTS,                              \
                             ISA_NAME::NUM_VECTOR_ALU_TERNARY_TESTS, "vector_alu_ternary");        \
  }

CTS_VECTOR_TERNARY_TEST(cdna1)
CTS_VECTOR_TERNARY_TEST(cdna2)
CTS_VECTOR_TERNARY_TEST(cdna3)
CTS_VECTOR_TERNARY_TEST(cdna4)
CTS_VECTOR_TERNARY_TEST(rdna1)
CTS_VECTOR_TERNARY_TEST(rdna2)
CTS_VECTOR_TERNARY_TEST(rdna3)
CTS_VECTOR_TERNARY_TEST(rdna3_5)
CTS_VECTOR_TERNARY_TEST(rdna4)
CTS_VECTOR_TERNARY_TEST(gfx1250)

} // namespace
