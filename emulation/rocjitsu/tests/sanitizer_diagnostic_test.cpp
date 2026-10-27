// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "instruction_failure_test_util.h"
#include "sanitizer_diagnostic_test_util.h"

#include <gtest/gtest-spi.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>

namespace {

void expect_diagnosed_exit(const std::string &output, bool abort_process,
                           const std::string &pattern = "rocjitsu: daemon connection lost") {
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
      fatal_exit, rocjitsu::test::ExpectedFailureDiagnostic(pattern));
}

TEST(NativeFailureDiagnosticTest, RejectsReportsAfterExpectedDiagnostic) {
  const ::testing::Matcher<const std::string &> diagnostic =
      rocjitsu::test::ExpectedFailureDiagnostic("rocjitsu: daemon connection lost");
  EXPECT_TRUE(diagnostic.Matches("rocjitsu: daemon connection lost\n"));
  EXPECT_FALSE(diagnostic.Matches("rocjitsu: unrelated failure\n"));
  for (const char *report :
       {"ERROR: AddressSanitizer: heap-use-after-free", "WARNING: ThreadSanitizer: data race",
        "MemorySanitizer: use-of-uninitialized-value",
        "ERROR: LeakSanitizer: detected memory leaks", "UndefinedBehaviorSanitizer:DEADLYSIGNAL",
        "test.cpp:42: runtime error: signed integer overflow",
        "test: test.cpp:42: void fail(): Assertion `false' failed.",
        "Assertion failed: (false), function fail, file test.cpp, line 42.",
        "test.cpp:42: assertion failed: false", "assertion failure: false"}) {
    SCOPED_TRACE(report);
    const std::string output = std::string("rocjitsu: daemon connection lost\n") + report;
    EXPECT_FALSE(diagnostic.Matches(output));
  }
}

TEST(NativeFailureDiagnosticTest, DoesNotCombineSeparateLinesIntoAnAssertionReport) {
  const ::testing::Matcher<const std::string &> diagnostic =
      rocjitsu::test::ExpectedFailureDiagnostic("rocjitsu: daemon connection lost");
  const std::string output = "rocjitsu: daemon connection lost\n"
                             "Assertion reporting enabled\n"
                             "the GPU instruction failed\n";
  EXPECT_TRUE(diagnostic.Matches(output));
}

TEST(SanitizerDiagnosticDeathTest, RejectsAssertionAbortAfterExpectedDiagnostic) {
  EXPECT_NONFATAL_FAILURE(
      expect_diagnosed_exit(
          std::format("rocjitsu: daemon connection lost\n"
                      "CU cu0: wf0 could not execute {} at pc=0x1040 "
                      "(pid=0 qid=1 dispatch=1 wg=0): unimplemented instruction\n"
                      "test: test.cpp:42: void fail(): Assertion `false' failed.\n",
                      rocjitsu::test::kUnimplementedInstructionMnemonic),
          true, rocjitsu::test::kUnimplementedInstructionDiagnostic),
      "without sanitizer or assertion diagnostics");
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
  EXPECT_NONFATAL_FAILURE(
      EXPECT_EXIT(exit_after_sanitizer_report(), ::testing::ExitedWithCode(EXIT_FAILURE),
                  rocjitsu::test::ExpectedFailureDiagnostic("rocjitsu: daemon connection lost")),
      "without sanitizer or assertion diagnostics");
}

} // namespace
