// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cts_common.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>

// Per-ISA generated test data headers.
#include "cts_data/cdna1/scalar_cmp.h"
#include "cts_data/cdna2/scalar_cmp.h"
#include "cts_data/cdna3/scalar_cmp.h"
#include "cts_data/cdna4/scalar_cmp.h"
#include "cts_data/gfx1250/scalar_cmp.h"
#include "cts_data/rdna1/scalar_cmp.h"
#include "cts_data/rdna2/scalar_cmp.h"
#include "cts_data/rdna3/scalar_cmp.h"
#include "cts_data/rdna3_5/scalar_cmp.h"
#include "cts_data/rdna4/scalar_cmp.h"

namespace {

using namespace rocjitsu;
using namespace rocjitsu::cts;

template <typename TestCaseT>
void run_scalar_cmp_tests(const IsaConfig &isa_cfg, const TestCaseT *tests, size_t count) {
  CtsFixture fixture(isa_cfg);
  uint32_t passed = 0;
  uint32_t failed = 0;

  for (size_t i = 0; i < count; ++i) {
    const auto &tc = tests[i];
    fixture.reset();

    // SOPC: s0 = src0, s2 = src1.
    fixture.write_sgpr(0, tc.inputs[0]);
    fixture.write_sgpr(2, tc.inputs[1]);

    uint32_t words[4] = {tc.encoding[0], tc.encoding[1], 0u, 0u};
    bool ok = fixture.decode_execute(words);
    if (!ok) {
      ++failed;
      continue;
    }

    bool actual_scc = fixture.read_scc();
    if (actual_scc != tc.expected_scc) {
      ++failed;
      fprintf(stderr,
              "[%s/%s] case %zu: SCC %d != expected %d "
              "(inputs: 0x%08X, 0x%08X)\n",
              std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i,
              static_cast<int>(actual_scc), static_cast<int>(tc.expected_scc), tc.inputs[0],
              tc.inputs[1]);
      continue;
    }

    ++passed;
  }

  EXPECT_EQ(failed, 0u) << isa_cfg.name << ": " << failed << " of " << count
                        << " scalar CMP tests failed";
  fprintf(stderr, "[%s] scalar CMP: %u/%zu passed\n", std::string(isa_cfg.name).c_str(), passed,
          count);
}

#define CTS_SCALAR_CMP_TEST(ISA_NAME)                                                              \
  TEST(CtsScalarCmp, ISA_NAME) {                                                                   \
    const auto &cfg = *std::find_if(std::begin(kIsaConfigs), std::end(kIsaConfigs),                \
                                    [](const IsaConfig &c) { return c.name == #ISA_NAME; });       \
    run_scalar_cmp_tests(cfg, ISA_NAME::SCALAR_CMP_TESTS, ISA_NAME::NUM_SCALAR_CMP_TESTS);         \
  }

CTS_SCALAR_CMP_TEST(cdna1)
CTS_SCALAR_CMP_TEST(cdna2)
CTS_SCALAR_CMP_TEST(cdna3)
CTS_SCALAR_CMP_TEST(cdna4)
CTS_SCALAR_CMP_TEST(rdna1)
CTS_SCALAR_CMP_TEST(rdna2)
CTS_SCALAR_CMP_TEST(rdna3)
CTS_SCALAR_CMP_TEST(rdna3_5)
CTS_SCALAR_CMP_TEST(rdna4)
CTS_SCALAR_CMP_TEST(gfx1250)

} // namespace
