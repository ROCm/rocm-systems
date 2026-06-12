// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cts_common.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>

// Per-ISA generated test data headers.
#include "cts_data/cdna1/mad_64.h"
#include "cts_data/cdna2/mad_64.h"
#include "cts_data/cdna3/mad_64.h"
#include "cts_data/cdna4/mad_64.h"
#include "cts_data/gfx1250/mad_64.h"
#include "cts_data/rdna1/mad_64.h"
#include "cts_data/rdna2/mad_64.h"
#include "cts_data/rdna3/mad_64.h"
#include "cts_data/rdna3_5/mad_64.h"
#include "cts_data/rdna4/mad_64.h"

namespace {

using namespace rocjitsu;
using namespace rocjitsu::cts;

template <typename TestCaseT>
void run_mad_64_tests(const IsaConfig &isa_cfg, const TestCaseT *tests, size_t count,
                      const char *label) {
  CtsFixture fixture(isa_cfg);
  uint32_t passed = 0;
  uint32_t failed = 0;
  constexpr uint32_t TEST_LANE = 0;

  for (size_t i = 0; i < count; ++i) {
    const auto &tc = tests[i];
    fixture.reset();

    // v0=src0, v2=src1, v4:v5=src2 (64-bit), v8:v9=dst (64-bit).
    fixture.write_vgpr(0, TEST_LANE, tc.inputs[0]);
    fixture.write_vgpr(2, TEST_LANE, tc.inputs[1]);
    fixture.write_vgpr(4, TEST_LANE, tc.inputs[2]);
    fixture.write_vgpr(5, TEST_LANE, tc.inputs[3]);

    uint32_t words[4] = {tc.encoding[0], tc.encoding[1], 0u, 0u};
    bool ok = fixture.decode_execute(words);
    if (!ok) {
      ++failed;
      continue;
    }

    uint32_t actual_lo = fixture.read_vgpr(8, TEST_LANE);
    uint32_t actual_hi = fixture.read_vgpr(9, TEST_LANE);
    if (actual_lo != tc.expected_lo || actual_hi != tc.expected_hi) {
      ++failed;
      fprintf(stderr,
              "[%s/%s] case %zu: output 0x%08X_%08X != expected 0x%08X_%08X "
              "(inputs: 0x%08X 0x%08X 0x%08X_%08X)\n",
              std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i, actual_hi,
              actual_lo, tc.expected_hi, tc.expected_lo, tc.inputs[0], tc.inputs[1], tc.inputs[3],
              tc.inputs[2]);
      continue;
    }

    ++passed;
  }

  EXPECT_EQ(failed, 0u) << isa_cfg.name << ": " << failed << " of " << count << " " << label
                        << " tests failed";
  fprintf(stderr, "[%s] %s: %u/%zu passed\n", std::string(isa_cfg.name).c_str(), label, passed,
          count);
}

#define CTS_MAD_64_TEST(ISA_NAME)                                                                  \
  TEST(CtsMad64, ISA_NAME) {                                                                       \
    const auto &cfg = *std::find_if(std::begin(kIsaConfigs), std::end(kIsaConfigs),                \
                                    [](const IsaConfig &c) { return c.name == #ISA_NAME; });       \
    if constexpr (ISA_NAME::NUM_MAD_64_TESTS > 0) {                                                \
      run_mad_64_tests(cfg, ISA_NAME::MAD_64_TESTS, ISA_NAME::NUM_MAD_64_TESTS, "mad_64");         \
    } else {                                                                                       \
      fprintf(stderr, "[%s] mad_64: no tests (instruction not present)\n", #ISA_NAME);             \
    }                                                                                              \
  }

CTS_MAD_64_TEST(cdna1)
CTS_MAD_64_TEST(cdna2)
CTS_MAD_64_TEST(cdna3)
CTS_MAD_64_TEST(cdna4)
CTS_MAD_64_TEST(rdna1)
CTS_MAD_64_TEST(rdna2)
CTS_MAD_64_TEST(rdna3)
CTS_MAD_64_TEST(rdna3_5)
CTS_MAD_64_TEST(rdna4)
CTS_MAD_64_TEST(gfx1250)

} // namespace
