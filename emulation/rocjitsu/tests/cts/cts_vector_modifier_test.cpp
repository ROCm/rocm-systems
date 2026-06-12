// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cts_common.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

// Per-ISA generated test data headers.
#include "cts_data/cdna1/vector_modifier.h"
#include "cts_data/cdna2/vector_modifier.h"
#include "cts_data/cdna3/vector_modifier.h"
#include "cts_data/cdna4/vector_modifier.h"
#include "cts_data/gfx1250/vector_modifier.h"
#include "cts_data/rdna1/vector_modifier.h"
#include "cts_data/rdna2/vector_modifier.h"
#include "cts_data/rdna3/vector_modifier.h"
#include "cts_data/rdna3_5/vector_modifier.h"
#include "cts_data/rdna4/vector_modifier.h"

namespace {

using namespace rocjitsu;
using namespace rocjitsu::cts;

static bool is_nan_bits(uint32_t bits) {
  uint32_t exp = (bits >> 23) & 0xFF;
  uint32_t mant = bits & 0x007FFFFF;
  return exp == 0xFF && mant != 0;
}

static bool fp_bits_match(uint32_t actual, uint32_t expected) {
  if (is_nan_bits(actual) && is_nan_bits(expected))
    return true;
  return actual == expected;
}

template <typename TestCaseT>
void run_vector_modifier_tests(const IsaConfig &isa_cfg, const TestCaseT *tests, size_t count,
                               const char *label) {
  CtsFixture fixture(isa_cfg);
  uint32_t passed = 0;
  uint32_t failed = 0;
  constexpr uint32_t TEST_LANE = 0;

  for (size_t i = 0; i < count; ++i) {
    const auto &tc = tests[i];
    fixture.reset();

    // VOP3 unary: v0=src0, v4=dst.
    // VOP3 binary: v0=src0, v2=src1, v4=dst.
    fixture.write_vgpr(0, TEST_LANE, tc.inputs[0]);
    if (tc.num_inputs >= 2)
      fixture.write_vgpr(2, TEST_LANE, tc.inputs[1]);

    uint32_t words[4] = {tc.encoding[0], tc.encoding[1], 0u, 0u};
    bool ok = fixture.decode_execute(words);
    if (!ok) {
      ++failed;
      continue;
    }

    uint32_t actual = fixture.read_vgpr(4, TEST_LANE);
    if (!fp_bits_match(actual, tc.expected_output)) {
      ++failed;
      float f_actual, f_expected;
      std::memcpy(&f_actual, &actual, 4);
      std::memcpy(&f_expected, &tc.expected_output, 4);
      fprintf(stderr,
              "[%s/%s] case %zu: output 0x%08X (%g) != expected 0x%08X (%g) "
              "(inputs: 0x%08X",
              std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i, actual,
              f_actual, tc.expected_output, f_expected, tc.inputs[0]);
      if (tc.num_inputs >= 2)
        fprintf(stderr, ", 0x%08X", tc.inputs[1]);
      fprintf(stderr, ")\n");
      continue;
    }

    ++passed;
  }

  EXPECT_EQ(failed, 0u) << isa_cfg.name << ": " << failed << " of " << count << " " << label
                        << " tests failed";
  fprintf(stderr, "[%s] %s: %u/%zu passed\n", std::string(isa_cfg.name).c_str(), label, passed,
          count);
}

#define CTS_VECTOR_MODIFIER_TEST(ISA_NAME)                                                         \
  TEST(CtsVectorModifier, ISA_NAME) {                                                              \
    const auto &cfg = *std::find_if(std::begin(kIsaConfigs), std::end(kIsaConfigs),                \
                                    [](const IsaConfig &c) { return c.name == #ISA_NAME; });       \
    run_vector_modifier_tests(cfg, ISA_NAME::VECTOR_MODIFIER_TESTS,                                \
                              ISA_NAME::NUM_VECTOR_MODIFIER_TESTS, "vector_modifier");             \
  }

CTS_VECTOR_MODIFIER_TEST(cdna1)
CTS_VECTOR_MODIFIER_TEST(cdna2)
CTS_VECTOR_MODIFIER_TEST(cdna3)
CTS_VECTOR_MODIFIER_TEST(cdna4)
CTS_VECTOR_MODIFIER_TEST(rdna1)
CTS_VECTOR_MODIFIER_TEST(rdna2)
CTS_VECTOR_MODIFIER_TEST(rdna3)
CTS_VECTOR_MODIFIER_TEST(rdna3_5)
CTS_VECTOR_MODIFIER_TEST(rdna4)
CTS_VECTOR_MODIFIER_TEST(gfx1250)

} // namespace
