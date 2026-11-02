// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "instruction_failure_test_util.h"
#include "sanitizer_diagnostic_test_util.h"

#include <gtest/gtest-spi.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <string>
#include <thread>

namespace {

constexpr char kDisconnectDiagnostic[] = "rocjitsu: daemon connection lost";

enum class ChildExit { Failure, Abort, Terminate };

std::string unimplemented_instruction_report() {
  return std::format("CU cu0: wf0 could not execute {} at pc=0x1040 "
                     "(pid=0 qid=1 dispatch=1 wg=0): unimplemented instruction\n",
                     rocjitsu::test::kUnimplementedInstructionMnemonic);
}

std::string rejected_report_failure() {
  return std::string("without ") + rocjitsu::test::kUnexpectedFailureDiagnostics;
}

void expect_diagnosed_exit(const std::string &output, ChildExit child_exit,
                           const std::string &pattern) {
  const auto fatal_exit = [](int status) {
    return ::testing::ExitedWithCode(EXIT_FAILURE)(status) ||
           ::testing::KilledBySignal(SIGABRT)(status);
  };
  EXPECT_EXIT(
      {
        std::fputs(output.c_str(), stderr);
        std::fflush(stderr);
        if (child_exit == ChildExit::Abort)
          std::abort();
        if (child_exit == ChildExit::Terminate) {
          // Destroying a joinable thread calls std::terminate(), which reports
          // through the C++ runtime's own handler before aborting.
          std::thread joinable([] {});
        }
        std::_Exit(EXIT_FAILURE);
      },
      fatal_exit, rocjitsu::test::ExpectedFailureDiagnostic(pattern));
}

TEST(NativeFailureDiagnosticTest, RejectsReportsAfterExpectedDiagnostic) {
  const ::testing::Matcher<const std::string &> diagnostic =
      rocjitsu::test::ExpectedFailureDiagnostic(kDisconnectDiagnostic);
  EXPECT_TRUE(diagnostic.Matches("rocjitsu: daemon connection lost\n"));
  EXPECT_FALSE(diagnostic.Matches("rocjitsu: unrelated failure\n"));
  for (const char *report :
       {"ERROR: AddressSanitizer: heap-use-after-free", "WARNING: ThreadSanitizer: data race",
        "MemorySanitizer: use-of-uninitialized-value",
        "ERROR: LeakSanitizer: detected memory leaks", "UndefinedBehaviorSanitizer:DEADLYSIGNAL",
        "test.cpp:42: runtime error: signed integer overflow",
        "test: test.cpp:42: void fail(): Assertion `false' failed.",
        "Assertion failed: (false), function fail, file test.cpp, line 42.",
        "test.cpp:42: assertion failed: false", "assertion failure: false",
        "terminate called after throwing an instance of 'std::runtime_error'",
        "terminate called without an active exception",
        "libc++abi: terminating due to uncaught exception of type std::runtime_error: failed"}) {
    SCOPED_TRACE(report);
    const std::string output = std::string("rocjitsu: daemon connection lost\n") + report;
    EXPECT_FALSE(diagnostic.Matches(output));
  }
}

TEST(NativeFailureDiagnosticTest, DoesNotCombineSeparateLinesIntoAnAssertionReport) {
  const ::testing::Matcher<const std::string &> diagnostic =
      rocjitsu::test::ExpectedFailureDiagnostic(kDisconnectDiagnostic);
  const std::string output = "rocjitsu: daemon connection lost\n"
                             "Assertion reporting enabled\n"
                             "the GPU instruction failed\n";
  EXPECT_TRUE(diagnostic.Matches(output));
}

TEST(NativeFailureDiagnosticDeathTest, RejectsAssertionAbortAfterExpectedDiagnostic) {
  EXPECT_NONFATAL_FAILURE(
      expect_diagnosed_exit(unimplemented_instruction_report() +
                                "test: test.cpp:42: void fail(): Assertion `false' failed.\n",
                            ChildExit::Abort, rocjitsu::test::kUnimplementedInstructionDiagnostic),
      rejected_report_failure());
}

TEST(NativeFailureDiagnosticDeathTest, RejectsTerminateAfterExpectedDiagnostic) {
  EXPECT_NONFATAL_FAILURE(
      expect_diagnosed_exit(unimplemented_instruction_report(), ChildExit::Terminate,
                            rocjitsu::test::kUnimplementedInstructionDiagnostic),
      rejected_report_failure());
}

// A sanitizer can exit with the same status as a fatal GPU failure. The report
// in the child's captured output must still fail the death test.
TEST(NativeFailureDiagnosticDeathTest, RejectsSanitizerReportAfterAcceptedExit) {
  EXPECT_NONFATAL_FAILURE(expect_diagnosed_exit("rocjitsu: daemon connection lost\n"
                                                "ERROR: AddressSanitizer: heap-use-after-free\n",
                                                ChildExit::Failure, kDisconnectDiagnostic),
                          rejected_report_failure());
}

} // namespace
