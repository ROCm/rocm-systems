// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_instruction_failure_test.cpp
/// @brief Fatal simulated instructions terminate the native HIP process.
///
/// Requires the simulator interposer via LD_PRELOAD=librocjitsu.so.

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <thread>

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
} // namespace

__global__ void unsupported_instruction() {
  // s_cbranch_i_fork is decoded but has no execution implementation on CDNA3/4.
  asm volatile(".long 0xb8000000");
}

__global__ void undecodable_instruction() { asm volatile(".long 0xbfff0000"); }

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
  EXPECT_EXIT(wait_for_instruction_failure(FailureKind::Unsupported), native_instruction_failure,
              "CU .*could not execute s_cbranch_i_fork.*pc=.*pid=.*qid=.*dispatch=.*unimplemented "
              "instruction");
}

TEST(HipInstructionFailureTest, DecodeFailureExitsWithFailure) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(wait_for_instruction_failure(FailureKind::Decode), native_instruction_failure,
              "CU .*could not decode.*pc=.*pid=.*qid=.*dispatch=.*0xbfff0000");
}

// The outer DaemonTest fixture owns the daemon. Its diagnostics stay in that
// process; the client reports loss of the connection after the failed run.
TEST(HipDaemonInstructionFailureTest, UnsupportedInstructionExitsWithFailure) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(wait_for_instruction_failure(FailureKind::Unsupported),
              ::testing::ExitedWithCode(EXIT_FAILURE), "rocjitsu: daemon connection lost");
}

TEST(HipDaemonInstructionFailureTest, DecodeFailureExitsWithFailure) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(wait_for_instruction_failure(FailureKind::Decode),
              ::testing::ExitedWithCode(EXIT_FAILURE), "rocjitsu: daemon connection lost");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
