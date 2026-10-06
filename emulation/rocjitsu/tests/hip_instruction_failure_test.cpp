// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

__global__ void unsupported_instruction() {
  // s_cbranch_i_fork is decoded but has no execution implementation on CDNA3/4.
  asm volatile(".long 0xb8000000");
}

[[noreturn]] void wait_for_unsupported_instruction() {
  if (hipInit(0) != hipSuccess)
    std::_Exit(10);
  unsupported_instruction<<<4, 64>>>();
  if (hipGetLastError() != hipSuccess)
    std::_Exit(11);
  (void)hipDeviceSynchronize();
  std::_Exit(12);
}

TEST(HipInstructionFailureTest, UnsupportedInstructionExitsWithFailure) {
  // Re-exec so the child owns its HIP runtime and interposer engine threads.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(wait_for_unsupported_instruction(), ::testing::ExitedWithCode(EXIT_FAILURE),
              "rocjitsu: local VM failed: .*s_cbranch_i_fork.*pid=.*qid=.*dispatch=.*unimplemented "
              "instruction");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
