// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "sanitizer_diagnostic_test_util.h"

#include <gtest/gtest-spi.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>

namespace {

void expect_diagnosed_exit(const std::string &output, bool abort_process) {
  const auto fatal_exit = [](int status) {
    return ::testing::ExitedWithCode(EXIT_FAILURE)(status) ||
           ::testing::KilledBySignal(SIGABRT)(status);
  };
  EXPECT_EXIT(
      {
        std::fputs(output.c_str(), stderr);
        std::fflush(stderr);
        if (abort_process)
          std::abort();
        std::_Exit(EXIT_FAILURE);
      },
      fatal_exit, rocjitsu::test::SanitizerFreeDiagnostic("rocjitsu: daemon connection lost"));
}

TEST(SanitizerDiagnosticDeathTest, AcceptsExpectedDiagnostic) {
  expect_diagnosed_exit("rocjitsu: daemon connection lost\n", false);
}

TEST(SanitizerDiagnosticDeathTest, RejectsReportsAfterExpectedDiagnostic) {
  for (const char *report :
       {"ERROR: AddressSanitizer: heap-use-after-free", "WARNING: ThreadSanitizer: data race",
        "MemorySanitizer: use-of-uninitialized-value",
        "ERROR: LeakSanitizer: detected memory leaks", "UndefinedBehaviorSanitizer:DEADLYSIGNAL",
        "test.cpp:42: runtime error: signed integer overflow"}) {
    for (bool abort_process : {false, true}) {
      SCOPED_TRACE(report);
      SCOPED_TRACE(abort_process);
      const std::string output = std::string("rocjitsu: daemon connection lost\n") + report;
      EXPECT_NONFATAL_FAILURE(expect_diagnosed_exit(output, abort_process),
                              "without sanitizer diagnostics");
    }
  }
}

} // namespace
