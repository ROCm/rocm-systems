/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

// Entry point for the GPU-free InterceptQueue logic unit tests. These tests exercise
// pure decision logic (see core/inc/intercept_queue_logic.h) and do not require the HSA
// runtime or a GPU, so they run as an ordinary host executable under CTest.

#include "gtest/gtest.h"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
