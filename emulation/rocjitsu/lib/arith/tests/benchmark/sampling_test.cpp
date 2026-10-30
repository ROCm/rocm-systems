// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for interleaved fixed-repetition sampling, medians, and terminal progress.

#include "benchmark/csv.h"
#include "benchmark/ordering.h"
#include "benchmark/sampling.h"
#include "benchmark/workloads.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#if defined(__linux__)
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#endif

// Verify repetition-major execution and the exact number of samples per workload.
TEST(BenchmarkSampling, FixedInterleavedRepetitions) {
  std::vector<rocjitsu::arith::bench::Benchmark> cases(3);
  std::vector<int> order;
  for (int i = 0; i < 2; ++i) {
    cases[i].sample = [&, i, n = 0](double min_ms) mutable {
      EXPECT_DOUBLE_EQ(min_ms, 1);
      order.push_back(i);
      return 100.0 + i + n++;
    };
  }
  rocjitsu::arith::bench::Options options;
  EXPECT_EQ(options.repetitions, 5);
  rocjitsu::arith::bench::run(cases, options);
  EXPECT_EQ(order, (std::vector<int>{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}));
  EXPECT_EQ(cases[0].samples.size(), 5U);
  EXPECT_EQ(cases[1].samples.size(), 5U);
  EXPECT_DOUBLE_EQ(cases[0].median, 102);
  EXPECT_DOUBLE_EQ(cases[1].median, 103);
  EXPECT_TRUE(cases[2].samples.empty());

  // Rerunning replaces samples; even counts average the middle pair.
  options.repetitions = 2;
  rocjitsu::arith::bench::run(cases, options);
  EXPECT_EQ(cases[0].samples.size(), 2U);
  EXPECT_DOUBLE_EQ(cases[0].median, 105.5);
  options.repetitions = 1;
  rocjitsu::arith::bench::run(cases, options);
  EXPECT_EQ(cases[0].samples.size(), 1U);
  EXPECT_DOUBLE_EQ(cases[0].median, 107);
}

// Verify that computing a median sorts initially unordered samples.
TEST(BenchmarkSampling, MedianSortsSamples) {
  std::vector<rocjitsu::arith::bench::Benchmark> cases(1);
  cases[0].sample = [n = 0](double) mutable {
    const double samples[] = {105, 101, 104, 102, 103};
    return samples[n++];
  };
  rocjitsu::arith::bench::run(cases, {});
  EXPECT_DOUBLE_EQ(cases[0].median, 103);
}

// Check that a zero latency is rejected instead of entering the sample set.
TEST(BenchmarkSampling, InvalidMeasurementFails) {
  std::vector<rocjitsu::arith::bench::Benchmark> cases(1);
  cases[0].sample = [](double) { return 0.0; };
  EXPECT_THROW(rocjitsu::arith::bench::run(cases, {}), std::runtime_error);
}

// Check natural ordering of numeric runs within instruction names.
TEST(BenchmarkSampling, AlphanumericInstructionOrder) {
  std::vector<std::string> names = {"v_dot16",         "v_dot02",    "v_dot8",
                                    "v_dot2",          "v_dot2_w64", "v_dot2_w32_gfx12",
                                    "v_dot2_w32_gfx9", "v_dot2_w32"};
  std::sort(names.begin(), names.end(), rocjitsu::arith::bench::alphanumeric_less);
  const std::vector<std::string> expected = {"v_dot2",           "v_dot2_w32", "v_dot2_w32_gfx9",
                                             "v_dot2_w32_gfx12", "v_dot2_w64", "v_dot02",
                                             "v_dot8",           "v_dot16"};
  EXPECT_EQ(names, expected);
  EXPECT_FALSE(rocjitsu::arith::bench::alphanumeric_less("v_dot2", "v_dot2"));
  EXPECT_TRUE(rocjitsu::arith::bench::alphanumeric_less("v_dot99999999999999999999",
                                                        "v_dot100000000000000000000"));
}

// Check minimal progress output for a non-terminal stream.
TEST(BenchmarkSampling, ProgressReportsRepetitions) {
  std::vector<rocjitsu::arith::bench::Benchmark> cases(2);
  cases[0].sample = [](double) { return 100.0; };
  rocjitsu::arith::bench::Options options;
  options.repetitions = 3;
  FILE *progress = std::tmpfile();
  ASSERT_NE(progress, nullptr);
  rocjitsu::arith::bench::run(cases, options, progress);
  std::rewind(progress);
  char buffer[1024];
  size_t size = std::fread(buffer, 1, sizeof(buffer), progress);
  std::fclose(progress);
  EXPECT_EQ(std::string(buffer, size), "1 benchmarks; 3 repetitions each\n"
                                       "  Repetition 1/3\n  Repetition 2/3\n  Repetition 3/3\n");
}

// Exercise actual terminal detection and bytes, without a timing-sensitive test.
#if defined(__linux__)
// Check carriage-return progress and the final newline on a terminal.
TEST(BenchmarkSampling, TerminalProgressOverwritesAndFinishes) {
  int master = posix_openpt(O_RDWR | O_NOCTTY | O_NONBLOCK);
  ASSERT_GE(master, 0);
  ASSERT_EQ(grantpt(master), 0);
  ASSERT_EQ(unlockpt(master), 0);
  FILE *progress = std::fopen(ptsname(master), "w");
  ASSERT_NE(progress, nullptr);
  std::vector<rocjitsu::arith::bench::Benchmark> cases(1);
  cases[0].sample = [](double) { return 100.0; };
  rocjitsu::arith::bench::Options options;
  options.repetitions = 1;
  rocjitsu::arith::bench::run(cases, options, progress);
  char buffer[1024];
  // A PTY read may end between writes, including just before the final newline.
  // Wait for the complete line rather than assuming one read drains the stream.
  std::string output;
  while (output.find('\n') == std::string::npos) {
    pollfd ready{master, POLLIN, 0};
    if (poll(&ready, 1, 1000) <= 0) {
      break;
    }
    auto size = read(master, buffer, sizeof(buffer));
    if (size <= 0) {
      break;
    }
    output.append(buffer, size);
  }
  std::fclose(progress);
  close(master);
  ASSERT_FALSE(output.empty());
  EXPECT_EQ(output.front(), '\r');
  EXPECT_EQ(output.back(), '\n');
  EXPECT_NE(output.find("0.0% | elapsed "), std::string::npos);
  EXPECT_NE(output.find("remaining --"), std::string::npos);
  EXPECT_NE(output.find("\r100.0% | elapsed "), std::string::npos);
  EXPECT_NE(output.find("remaining 0.0s"), std::string::npos);
  EXPECT_EQ(output.find('\x1b'), std::string::npos);
  EXPECT_EQ(std::count(output.begin(), output.end(), '\n'), 1);
}
#endif

// Check joined variant labels and CSV quoting/escaping.
TEST(BenchmarkCsv, VariantField) {
  EXPECT_EQ(rocjitsu::arith::bench::variant_field({}), "");
  EXPECT_EQ(rocjitsu::arith::bench::variant_field({"accumulator reset"}), "accumulator reset");
  EXPECT_EQ(rocjitsu::arith::bench::variant_field({"accumulator reset", "other"}),
            "\"accumulator reset, other\"");
  EXPECT_EQ(rocjitsu::arith::bench::variant_field({"a \"quote\""}), "\"a \"\"quote\"\"\"");
}

// Check category-first, then instruction-name ordering.
TEST(BenchmarkSampling, CategoryThenInstructionOrder) {
  std::vector<rocjitsu::arith::bench::Benchmark> cases(5);
  cases[0].category = "wmma";
  cases[0].instruction = "v_a";
  cases[1].category = "fp10";
  cases[1].instruction = "v_a";
  cases[2].category = "fp2";
  cases[2].instruction = "v_op10";
  cases[3].category = "fp2";
  cases[3].instruction = "v_op2";
  cases[3].cpu = "first";
  cases[4] = cases[3];
  cases[4].cpu = "second";
  std::stable_sort(cases.begin(), cases.end(), rocjitsu::arith::bench::benchmark_less);
  EXPECT_EQ(cases[0].cpu, "first");
  EXPECT_EQ(cases[1].cpu, "second");
  EXPECT_EQ(cases[2].instruction, "v_op10");
  EXPECT_EQ(cases[3].category, "fp10");
  EXPECT_EQ(cases[4].category, "wmma");
}

// Verify that the NaN workload changes exactly one floating-point input element.
TEST(BenchmarkInputs, OneQuietNanElement) {
  struct Case {
    const char *instruction;
    int operand;
    uint32_t low, mask, high;
  };

  const Case cases[] = {
      {"v_add_f16_w32", 0, 0x7e00U, 0xffffU, 0},
      {"v_add_f32_w64", 0, 0x7fc00000U, UINT32_MAX, 0},
      {"v_add_f64_w32", 0, 0, UINT32_MAX, 0x7ff80000U},
      {"v_fma_mixlo_f16_w32", 0, 0x7fc00000U, UINT32_MAX, 0},
      {"v_frexp_exp_i32_f64_w32", 0, 0, UINT32_MAX, 0x7ff80000U},
      {"v_wmma_f32_16x16x16_f16_w32", 2, 0x7fc00000U, UINT32_MAX, 0},
      {"v_wmma_bf16_16x16x16_bf16_w64", 2, 0x7fc0U, 0xffffU, 0},
      {"v_wmma_f32_16x16x16_bf16_w64_gfx12", 0, 0x7fc0U, 0xffffU, 0},
      {"v_swmmac_f32_16x16x32_f16_w32", 0, 0x7e00U, 0xffffU, 0},
      {"v_dot4_f32_fp8_bf8_w32", 0, 0x7fU, 0xffU, 0},
      {"v_dot4_f32_bf8_fp8_w64", 0, 0x7eU, 0xffU, 0},
      {"v_fmac_f32_w32", 0, 0x7fc00000U, UINT32_MAX, 0},
  };
  for (auto test : cases) {
    SCOPED_TRACE(test.instruction);
    rocjitsu::arith::bench::Operands finite(test.instruction), nan(test.instruction, true);
    ASSERT_TRUE(nan.floating_point);
    finite.data[test.operand][0][5] = (finite.data[test.operand][0][5] & ~test.mask) | test.low;
    if (test.high) {
      finite.data[test.operand][1][5] = test.high;
    }
    // Exactly the selected element changed, including both words for FP64.
    EXPECT_EQ(std::memcmp(finite.data, nan.data, sizeof(nan.data)), 0);
    nan.reset();
    nan.reset_accumulator();
    EXPECT_EQ(std::memcmp(finite.data, nan.data, sizeof(nan.data)), 0);
  }
  rocjitsu::arith::bench::Operands integer("v_wmma_i32_16x16x16_iu8_w32", true);
  EXPECT_FALSE(integer.floating_point);
}
