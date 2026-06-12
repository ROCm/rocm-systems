// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cts_common.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>

#include "cts_data/cdna1/ds.h"
#include "cts_data/cdna1/ds_atomic.h"
#include "cts_data/cdna2/ds.h"
#include "cts_data/cdna2/ds_atomic.h"
#include "cts_data/cdna3/ds.h"
#include "cts_data/cdna3/ds_atomic.h"
#include "cts_data/cdna4/ds.h"
#include "cts_data/cdna4/ds_atomic.h"
#include "cts_data/gfx1250/ds.h"
#include "cts_data/gfx1250/ds_atomic.h"
#include "cts_data/rdna1/ds.h"
#include "cts_data/rdna1/ds_atomic.h"
#include "cts_data/rdna2/ds.h"
#include "cts_data/rdna2/ds_atomic.h"
#include "cts_data/rdna3/ds.h"
#include "cts_data/rdna3/ds_atomic.h"
#include "cts_data/rdna3_5/ds.h"
#include "cts_data/rdna3_5/ds_atomic.h"
#include "cts_data/rdna4/ds.h"
#include "cts_data/rdna4/ds_atomic.h"

namespace {

using namespace rocjitsu;
using namespace rocjitsu::cts;

template <typename TestCaseT>
void run_ds_read_tests(const IsaConfig &isa_cfg, const TestCaseT *tests, size_t count,
                       const char *label) {
  CtsFixture fixture(isa_cfg);
  uint32_t passed = 0;
  uint32_t failed = 0;
  constexpr uint32_t TEST_LANE = 0;

  for (size_t i = 0; i < count; ++i) {
    const auto &tc = tests[i];
    fixture.reset();
    fixture.set_exec(1ULL << TEST_LANE);
    fixture.clear_lds();

    if (tc.is_64bit) {
      fixture.write_lds64(tc.lds_addr,
                          static_cast<uint64_t>(tc.lds_data_hi) << 32 | tc.lds_data_lo);
    } else {
      fixture.write_lds32(tc.lds_addr, tc.lds_data_lo);
    }

    fixture.write_vgpr(0, TEST_LANE, tc.addr_value);

    uint32_t words[4] = {tc.encoding[0], tc.encoding[1], 0u, 0u};
    bool ok = fixture.decode_execute(words);
    if (!ok) {
      ++failed;
      continue;
    }

    uint32_t actual_lo = fixture.read_vgpr(6, TEST_LANE);
    bool match = (actual_lo == tc.expected_lo);

    if (tc.is_64bit) {
      uint32_t actual_hi = fixture.read_vgpr(7, TEST_LANE);
      match = match && (actual_hi == tc.expected_hi);
      if (!match) {
        ++failed;
        fprintf(stderr, "[%s/%s] case %zu: got 0x%08X_%08X != expected 0x%08X_%08X (addr=0x%X)\n",
                std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i, actual_hi,
                actual_lo, tc.expected_hi, tc.expected_lo, tc.lds_addr);
        continue;
      }
    } else if (!match) {
      ++failed;
      fprintf(stderr, "[%s/%s] case %zu: got 0x%08X != expected 0x%08X (addr=0x%X)\n",
              std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i, actual_lo,
              tc.expected_lo, tc.lds_addr);
      continue;
    }

    ++passed;
  }

  EXPECT_EQ(failed, 0u) << isa_cfg.name << ": " << failed << " of " << count << " " << label
                        << " tests failed";
  fprintf(stderr, "[%s] %s: %u/%zu passed\n", std::string(isa_cfg.name).c_str(), label, passed,
          count);
}

template <typename TestCaseT>
void run_ds_write_tests(const IsaConfig &isa_cfg, const TestCaseT *tests, size_t count,
                        const char *label) {
  CtsFixture fixture(isa_cfg);
  uint32_t passed = 0;
  uint32_t failed = 0;
  constexpr uint32_t TEST_LANE = 0;

  for (size_t i = 0; i < count; ++i) {
    const auto &tc = tests[i];
    fixture.reset();
    fixture.set_exec(1ULL << TEST_LANE);
    fixture.clear_lds();

    fixture.write_vgpr(0, TEST_LANE, tc.addr_value);
    fixture.write_vgpr(2, TEST_LANE, tc.data_lo);
    if (tc.is_64bit) {
      fixture.write_vgpr(3, TEST_LANE, tc.data_hi);
    }

    uint32_t words[4] = {tc.encoding[0], tc.encoding[1], 0u, 0u};
    bool ok = fixture.decode_execute(words);
    if (!ok) {
      ++failed;
      continue;
    }

    bool match;
    if (tc.is_64bit) {
      uint64_t actual = fixture.read_lds64(tc.expected_lds_addr);
      uint64_t expected = static_cast<uint64_t>(tc.expected_hi) << 32 | tc.expected_lo;
      match = (actual == expected);
      if (!match) {
        ++failed;
        fprintf(stderr, "[%s/%s] case %zu: LDS[0x%X] = 0x%016lX != expected 0x%016lX\n",
                std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i,
                tc.expected_lds_addr, actual, expected);
        continue;
      }
    } else {
      uint32_t actual = fixture.read_lds32(tc.expected_lds_addr);
      match = (actual == tc.expected_lo);
      if (!match) {
        ++failed;
        fprintf(stderr, "[%s/%s] case %zu: LDS[0x%X] = 0x%08X != expected 0x%08X\n",
                std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i,
                tc.expected_lds_addr, actual, tc.expected_lo);
        continue;
      }
    }

    ++passed;
  }

  EXPECT_EQ(failed, 0u) << isa_cfg.name << ": " << failed << " of " << count << " " << label
                        << " tests failed";
  fprintf(stderr, "[%s] %s: %u/%zu passed\n", std::string(isa_cfg.name).c_str(), label, passed,
          count);
}

template <typename TestCaseT>
void run_ds_read2_tests(const IsaConfig &isa_cfg, const TestCaseT *tests, size_t count,
                        const char *label) {
  CtsFixture fixture(isa_cfg);
  uint32_t passed = 0;
  uint32_t failed = 0;
  constexpr uint32_t TEST_LANE = 0;

  for (size_t i = 0; i < count; ++i) {
    const auto &tc = tests[i];
    fixture.reset();
    fixture.set_exec(1ULL << TEST_LANE);
    fixture.clear_lds();

    fixture.write_lds32(tc.lds_addr0, tc.lds_data0);
    fixture.write_lds32(tc.lds_addr1, tc.lds_data1);
    fixture.write_vgpr(0, TEST_LANE, tc.addr_value);

    uint32_t words[4] = {tc.encoding[0], tc.encoding[1], 0u, 0u};
    bool ok = fixture.decode_execute(words);
    if (!ok) {
      ++failed;
      continue;
    }

    uint32_t actual0 = fixture.read_vgpr(6, TEST_LANE);
    uint32_t actual1 = fixture.read_vgpr(7, TEST_LANE);

    if (actual0 != tc.expected_vdst0 || actual1 != tc.expected_vdst1) {
      ++failed;
      fprintf(stderr, "[%s/%s] case %zu: v6=0x%08X(exp 0x%08X) v7=0x%08X(exp 0x%08X)\n",
              std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i, actual0,
              tc.expected_vdst0, actual1, tc.expected_vdst1);
      continue;
    }

    ++passed;
  }

  EXPECT_EQ(failed, 0u) << isa_cfg.name << ": " << failed << " of " << count << " " << label
                        << " tests failed";
  fprintf(stderr, "[%s] %s: %u/%zu passed\n", std::string(isa_cfg.name).c_str(), label, passed,
          count);
}

template <typename TestCaseT>
void run_ds_write2_tests(const IsaConfig &isa_cfg, const TestCaseT *tests, size_t count,
                         const char *label) {
  CtsFixture fixture(isa_cfg);
  uint32_t passed = 0;
  uint32_t failed = 0;
  constexpr uint32_t TEST_LANE = 0;

  for (size_t i = 0; i < count; ++i) {
    const auto &tc = tests[i];
    fixture.reset();
    fixture.set_exec(1ULL << TEST_LANE);
    fixture.clear_lds();

    fixture.write_vgpr(0, TEST_LANE, tc.addr_value);
    fixture.write_vgpr(2, TEST_LANE, tc.data0);
    fixture.write_vgpr(4, TEST_LANE, tc.data1);

    uint32_t words[4] = {tc.encoding[0], tc.encoding[1], 0u, 0u};
    bool ok = fixture.decode_execute(words);
    if (!ok) {
      ++failed;
      continue;
    }

    uint32_t actual0 = fixture.read_lds32(tc.expected_lds_addr0);
    uint32_t actual1 = fixture.read_lds32(tc.expected_lds_addr1);

    if (actual0 != tc.expected_lds0 || actual1 != tc.expected_lds1) {
      ++failed;
      fprintf(stderr,
              "[%s/%s] case %zu: LDS[0x%X]=0x%08X(exp 0x%08X) LDS[0x%X]=0x%08X(exp 0x%08X)\n",
              std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i,
              tc.expected_lds_addr0, actual0, tc.expected_lds0, tc.expected_lds_addr1, actual1,
              tc.expected_lds1);
      continue;
    }

    ++passed;
  }

  EXPECT_EQ(failed, 0u) << isa_cfg.name << ": " << failed << " of " << count << " " << label
                        << " tests failed";
  fprintf(stderr, "[%s] %s: %u/%zu passed\n", std::string(isa_cfg.name).c_str(), label, passed,
          count);
}

template <typename TestCaseT>
void run_ds_atomic_rtn_tests(const IsaConfig &isa_cfg, const TestCaseT *tests, size_t count,
                             const char *label) {
  CtsFixture fixture(isa_cfg);
  uint32_t passed = 0;
  uint32_t failed = 0;
  constexpr uint32_t TEST_LANE = 0;

  for (size_t i = 0; i < count; ++i) {
    const auto &tc = tests[i];
    fixture.reset();
    fixture.set_exec(1ULL << TEST_LANE);
    fixture.clear_lds();

    fixture.write_lds32(tc.lds_addr, tc.initial_lds);
    fixture.write_vgpr(0, TEST_LANE, tc.addr_value);
    fixture.write_vgpr(2, TEST_LANE, tc.data_operand);

    uint32_t words[4] = {tc.encoding[0], tc.encoding[1], 0u, 0u};
    bool ok = fixture.decode_execute(words);
    if (!ok) {
      ++failed;
      continue;
    }

    uint32_t actual_vdst = fixture.read_vgpr(6, TEST_LANE);
    uint32_t actual_lds = fixture.read_lds32(tc.lds_addr);

    if (actual_vdst != tc.expected_vdst || actual_lds != tc.expected_lds) {
      ++failed;
      fprintf(stderr,
              "[%s/%s] case %zu: vdst=0x%08X(exp 0x%08X) LDS=0x%08X(exp 0x%08X) "
              "(initial=0x%08X operand=0x%08X)\n",
              std::string(isa_cfg.name).c_str(), std::string(tc.mnemonic).c_str(), i, actual_vdst,
              tc.expected_vdst, actual_lds, tc.expected_lds, tc.initial_lds, tc.data_operand);
      continue;
    }

    ++passed;
  }

  EXPECT_EQ(failed, 0u) << isa_cfg.name << ": " << failed << " of " << count << " " << label
                        << " tests failed";
  fprintf(stderr, "[%s] %s: %u/%zu passed\n", std::string(isa_cfg.name).c_str(), label, passed,
          count);
}

#define CTS_DS_TEST(ISA_NAME)                                                                      \
  TEST(CtsDs, ISA_NAME) {                                                                          \
    const auto &cfg = *std::find_if(std::begin(kIsaConfigs), std::end(kIsaConfigs),                \
                                    [](const IsaConfig &c) { return c.name == #ISA_NAME; });       \
    if constexpr (ISA_NAME::NUM_DS_READ_TESTS > 0) {                                               \
      run_ds_read_tests(cfg, ISA_NAME::DS_READ_TESTS, ISA_NAME::NUM_DS_READ_TESTS, "ds_read");     \
    }                                                                                              \
    if constexpr (ISA_NAME::NUM_DS_WRITE_TESTS > 0) {                                              \
      run_ds_write_tests(cfg, ISA_NAME::DS_WRITE_TESTS, ISA_NAME::NUM_DS_WRITE_TESTS, "ds_write"); \
    }                                                                                              \
    if constexpr (ISA_NAME::NUM_DS_READ2_TESTS > 0) {                                              \
      run_ds_read2_tests(cfg, ISA_NAME::DS_READ2_TESTS, ISA_NAME::NUM_DS_READ2_TESTS, "ds_read2"); \
    }                                                                                              \
    if constexpr (ISA_NAME::NUM_DS_WRITE2_TESTS > 0) {                                             \
      run_ds_write2_tests(cfg, ISA_NAME::DS_WRITE2_TESTS, ISA_NAME::NUM_DS_WRITE2_TESTS,           \
                          "ds_write2");                                                            \
    }                                                                                              \
    if constexpr (ISA_NAME::NUM_DS_ATOMIC_RTN_TESTS > 0) {                                         \
      run_ds_atomic_rtn_tests(cfg, ISA_NAME::DS_ATOMIC_RTN_TESTS,                                  \
                              ISA_NAME::NUM_DS_ATOMIC_RTN_TESTS, "ds_atomic_rtn");                 \
    }                                                                                              \
  }

CTS_DS_TEST(cdna1)
CTS_DS_TEST(cdna2)
CTS_DS_TEST(cdna3)
CTS_DS_TEST(cdna4)
CTS_DS_TEST(rdna1)
CTS_DS_TEST(rdna2)
CTS_DS_TEST(rdna3)
CTS_DS_TEST(rdna3_5)
CTS_DS_TEST(rdna4)
CTS_DS_TEST(gfx1250)

} // namespace
