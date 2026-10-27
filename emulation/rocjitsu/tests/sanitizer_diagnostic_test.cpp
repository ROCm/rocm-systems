// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "sanitizer_diagnostic_test_util.h"

#include <gtest/gtest-spi.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

constexpr char kExpectedDiagnostic[] = "rocjitsu: daemon connection lost";

TEST(SanitizerFreeDiagnosticTest, RequiresDiagnosticWithoutSanitizerReports) {
  const ::testing::Matcher<const std::string &> matcher =
      rocjitsu::test::SanitizerFreeDiagnostic(kExpectedDiagnostic);
  const std::string expected = std::string(kExpectedDiagnostic) + "\n";
  EXPECT_TRUE(matcher.Matches(expected));
  EXPECT_FALSE(matcher.Matches("rocjitsu: unrelated failure\n"));
  for (const char *report :
       {"ERROR: AddressSanitizer: heap-use-after-free", "WARNING: ThreadSanitizer: data race",
        "MemorySanitizer: use-of-uninitialized-value",
        "ERROR: LeakSanitizer: detected memory leaks", "UndefinedBehaviorSanitizer:DEADLYSIGNAL",
        "test.cpp:42: runtime error: signed integer overflow"}) {
    SCOPED_TRACE(report);
    EXPECT_FALSE(matcher.Matches(expected + report));
  }
}

[[noreturn]] void exit_after_sanitizer_report() {
  std::fputs("rocjitsu: daemon connection lost\n"
             "ERROR: AddressSanitizer: heap-use-after-free\n",
             stderr);
  std::fflush(stderr);
  std::_Exit(EXIT_FAILURE);
}

// A sanitizer can exit with the same status as a fatal GPU failure. The report
// in the child's captured output must still fail the death test.
TEST(SanitizerFreeDiagnosticDeathTest, RejectsReportAfterAcceptedExit) {
  EXPECT_NONFATAL_FAILURE(EXPECT_EXIT(exit_after_sanitizer_report(),
                                      ::testing::ExitedWithCode(EXIT_FAILURE),
                                      rocjitsu::test::SanitizerFreeDiagnostic(kExpectedDiagnostic)),
                          "without sanitizer diagnostics");
}

} // namespace
