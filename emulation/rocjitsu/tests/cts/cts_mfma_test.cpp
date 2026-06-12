// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cts_common.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>

// Per-ISA generated test data headers.
#include "cts_data/cdna1/mfma.h"
#include "cts_data/cdna2/mfma.h"
#include "cts_data/cdna3/mfma.h"
#include "cts_data/cdna4/mfma.h"
#include "cts_data/gfx1250/mfma.h"
#include "cts_data/rdna1/mfma.h"
#include "cts_data/rdna2/mfma.h"
#include "cts_data/rdna3/mfma.h"
#include "cts_data/rdna3_5/mfma.h"
#include "cts_data/rdna4/mfma.h"

namespace {

using namespace rocjitsu;
using namespace rocjitsu::cts;

template <typename TestCaseT>
void run_mfma_tests(const IsaConfig &isa_cfg, const TestCaseT *tests, size_t count,
                    const char *label) {
  CtsFixture fixture(isa_cfg);
  uint32_t passed = 0;
  uint32_t failed = 0;

  for (size_t i = 0; i < count; ++i) {
    const auto &tc = tests[i];
    fixture.reset();

    uint64_t exec_val = (isa_cfg.wf_size == 64) ? ~0ULL : ((1ULL << isa_cfg.wf_size) - 1);
    fixture.set_exec(exec_val);

    for (uint32_t v = 0; v < tc.num_src0_vgprs; ++v)
      for (uint32_t lane = 0; lane < tc.wf_size; ++lane)
        fixture.write_vgpr(tc.src0_base + v, lane, tc.src0_data[v * tc.wf_size + lane]);

    for (uint32_t v = 0; v < tc.num_src1_vgprs; ++v)
      for (uint32_t lane = 0; lane < tc.wf_size; ++lane)
        fixture.write_vgpr(tc.src1_base + v, lane, tc.src1_data[v * tc.wf_size + lane]);

    if (tc.has_acc && tc.acc_data)
      for (uint32_t v = 0; v < tc.num_dst_vgprs; ++v)
        for (uint32_t lane = 0; lane < tc.wf_size; ++lane)
          fixture.write_vgpr(tc.acc_base + v, lane, tc.acc_data[v * tc.wf_size + lane]);

    uint32_t words[4] = {tc.encoding[0], tc.encoding[1], 0u, 0u};
    bool ok = fixture.decode_execute(words);
    if (!ok) {
      ++failed;
      fprintf(stderr, "[%s/%s] case %zu: decode_execute failed\n",
              std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i);
      continue;
    }

    uint32_t mismatches = 0;
    for (uint32_t v = 0; v < tc.num_dst_vgprs; ++v) {
      for (uint32_t lane = 0; lane < tc.wf_size; ++lane) {
        uint32_t actual = fixture.read_vgpr(tc.dst_base + v, lane);
        uint32_t exp = tc.expected[v * tc.wf_size + lane];
        if (actual != exp) {
          if (mismatches < 5)
            fprintf(stderr, "[%s/%s] case %zu: v%u{%u} = 0x%08X, expected 0x%08X\n",
                    std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i,
                    tc.dst_base + v, lane, actual, exp);
          ++mismatches;
        }
      }
    }

    if (mismatches > 0) {
      ++failed;
      if (mismatches > 5)
        fprintf(stderr, "[%s/%s] case %zu: ... and %u more mismatches\n",
                std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i,
                mismatches - 5);
    } else {
      ++passed;
    }
  }

  EXPECT_EQ(failed, 0u) << isa_cfg.name << ": " << failed << " of " << count << " " << label
                        << " tests failed";
  fprintf(stderr, "[%s] %s: %u/%zu passed\n", std::string(isa_cfg.name).c_str(), label, passed,
          count);
}

#define CTS_MFMA_TEST(ISA_NAME)                                                                    \
  TEST(CtsMfma, ISA_NAME) {                                                                        \
    const auto &cfg = *std::find_if(std::begin(kIsaConfigs), std::end(kIsaConfigs),                \
                                    [](const IsaConfig &c) { return c.name == #ISA_NAME; });       \
    if constexpr (ISA_NAME::NUM_MFMA_TESTS > 0) {                                                  \
      run_mfma_tests(cfg, ISA_NAME::MFMA_TESTS, ISA_NAME::NUM_MFMA_TESTS, "mfma");                 \
    } else {                                                                                       \
      fprintf(stderr, "[%s] mfma: no tests (instruction not present)\n", #ISA_NAME);               \
    }                                                                                              \
  }

CTS_MFMA_TEST(cdna1)
CTS_MFMA_TEST(cdna2)
CTS_MFMA_TEST(cdna3)
CTS_MFMA_TEST(cdna4)
CTS_MFMA_TEST(rdna1)
CTS_MFMA_TEST(rdna2)
CTS_MFMA_TEST(rdna3)
CTS_MFMA_TEST(rdna3_5)
CTS_MFMA_TEST(rdna4)
CTS_MFMA_TEST(gfx1250)

} // namespace
