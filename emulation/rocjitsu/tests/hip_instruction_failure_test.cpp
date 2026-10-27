// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_instruction_failure_test.cpp
/// @brief Fatal simulated instructions terminate the native HIP process.
///
/// Requires the simulator interposer via LD_PRELOAD=librocjitsu.so.

#include "sanitizer_diagnostic_test_util.h"

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <thread>
#include <unistd.h>

#include <gtest/gtest.h>

namespace {
constexpr int kInitializationFailure = 10;
constexpr int kLaunchFailure = 11;
constexpr int kSynchronizationReturned = 12;
enum class FailureKind { Unsupported, Decode };

bool native_instruction_failure(int status) {
  // HIP may abort from its queue-error callback before the engine's fatal exit.
  // Reject successful exits, setup/launch/synchronization failures, and every
  // other signal; the instruction-specific diagnostic is checked separately.
  return ::testing::ExitedWithCode(EXIT_FAILURE)(status) ||
         ::testing::KilledBySignal(SIGABRT)(status);
}

class DaemonFailureDiagnostic {
public:
  using is_gtest_matcher = void;
  DaemonFailureDiagnostic(const int &status, FailureKind kind) : status_(status), kind_(kind) {}

  bool MatchAndExplain(const std::string &output, ::testing::MatchResultListener *listener) const {
    // Match the diagnostic to the actual exit, so an unrelated abort cannot
    // pass merely because the disconnect monitor also printed its message.
    const char *pattern = "rocjitsu: daemon connection lost";
    if (::testing::KilledBySignal(SIGABRT)(status_))
      pattern = kind_ == FailureKind::Decode ? "Queue error: HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION:"
                                             : "Queue error: HSA_STATUS_ERROR_EXCEPTION:";
    const ::testing::Matcher<const std::string &> diagnostic =
        rocjitsu::test::SanitizerFreeDiagnostic(pattern);
    return diagnostic.MatchAndExplain(output, listener);
  }

  void DescribeTo(std::ostream *stream) const {
    *stream << "contains the daemon disconnect or HIP queue-error diagnostic matching its exit "
               "without sanitizer diagnostics";
  }

  void DescribeNegationTo(std::ostream *stream) const {
    *stream << "contains sanitizer diagnostics or does not contain the daemon diagnostic "
               "matching its exit";
  }

private:
  const int &status_;
  FailureKind kind_;
};
} // namespace

__global__ void unsupported_instruction() {
  // s_rfe_restore_b64 is decoded but has no execution implementation on CDNA3/4.
  asm volatile(".long 0x95800000");
}

__global__ void undecodable_instruction() { asm volatile(".long 0xbfff0000"); }

__global__ void valid_instruction(int *output) { *output = 17; }

[[noreturn]] void wait_for_instruction_failure(FailureKind kind) {
  if (hipInit(0) != hipSuccess)
    std::_Exit(kInitializationFailure);
  if (kind == FailureKind::Unsupported)
    unsupported_instruction<<<4, 64>>>();
  else
    undecodable_instruction<<<4, 64>>>();
  if (hipGetLastError() != hipSuccess)
    std::_Exit(kLaunchFailure);
  // Queue-owner exception delivery can return a HIP error before the engine
  // thread finishes shutdown. Successful synchronization is always a failure.
  if (hipDeviceSynchronize() == hipSuccess)
    std::_Exit(kSynchronizationReturned);
  std::this_thread::sleep_for(std::chrono::seconds(2));
  std::_Exit(kSynchronizationReturned);
}

TEST(HipInstructionFailureTest, UnsupportedInstructionExitsWithFailure) {
  // Re-exec so the child owns its HIP runtime and interposer engine threads.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      wait_for_instruction_failure(FailureKind::Unsupported), native_instruction_failure,
      rocjitsu::test::SanitizerFreeDiagnostic(
          "CU .*could not execute s_rfe_restore_b64.*pc=.*pid=.*qid=.*dispatch=.*unimplemented "
          "instruction"));
}

TEST(HipInstructionFailureTest, DecodeFailureExitsWithFailure) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(wait_for_instruction_failure(FailureKind::Decode), native_instruction_failure,
              rocjitsu::test::SanitizerFreeDiagnostic(
                  "CU .*could not decode.*pc=.*pid=.*qid=.*dispatch=.*0xbfff0000"));
}

namespace {
void expect_daemon_instruction_failure(FailureKind kind) {
  // The outer fixture checks the daemon's own failure exit. Its instruction
  // diagnostic stays there; this client can abort from HIP before disconnect.
  int status = 0;
  const auto fatal_exit = [&](int actual_status) {
    status = actual_status;
    return native_instruction_failure(status);
  };
  const ::testing::Matcher<const std::string &> diagnostic = DaemonFailureDiagnostic{status, kind};
  EXPECT_EXIT(wait_for_instruction_failure(kind), fatal_exit, diagnostic);
}
} // namespace

TEST(HipDaemonInstructionFailureTest, UnsupportedInstructionExitsWithFailure) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  expect_daemon_instruction_failure(FailureKind::Unsupported);
}

TEST(HipDaemonInstructionFailureTest, DecodeFailureExitsWithFailure) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  expect_daemon_instruction_failure(FailureKind::Decode);
}

TEST(HipDaemonInstructionFailureTest, ValidClientExitsWhenAnotherClientFails) {
  const char *ready_path = std::getenv("ROCJITSU_TEST_CLIENT_READY");
  if (!ready_path)
    GTEST_SKIP() << "requires a second client controlled by the daemon fixture";
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto run_valid_client = [&] {
    int *device_output = nullptr;
    if (hipInit(0) != hipSuccess || hipMalloc(&device_output, sizeof(int)) != hipSuccess)
      std::_Exit(kInitializationFailure);
    valid_instruction<<<1, 1>>>(device_output);
    if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess)
      std::_Exit(kLaunchFailure);
    int output = 0;
    if (hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost) != hipSuccess ||
        output != 17)
      std::_Exit(kSynchronizationReturned);
    // Only release the faulting client after this one completed valid work.
    const int ready = ::open(ready_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (ready < 0 || ::close(ready) != 0)
      std::_Exit(kInitializationFailure);
    for (;;) {
      valid_instruction<<<1, 1>>>(device_output);
      if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::_Exit(kSynchronizationReturned);
      }
    }
  };
  EXPECT_EXIT(run_valid_client(), ::testing::ExitedWithCode(EXIT_FAILURE),
              rocjitsu::test::SanitizerFreeDiagnostic("rocjitsu: daemon connection lost"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
