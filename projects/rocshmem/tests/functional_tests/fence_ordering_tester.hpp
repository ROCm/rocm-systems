/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#ifndef TESTS_FENCE_ORDERING_TESTER_HPP
#define TESTS_FENCE_ORDERING_TESTER_HPP

#include "tester.hpp"

class FenceOrderingTester : public Tester {
 public:
  explicit FenceOrderingTester(TesterArguments args);
  ~FenceOrderingTester() override;

  void resetBuffers(size_t size) override;
  void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                    size_t size) override;
  void verifyResults(size_t size) override;

 private:
  char *s_buf{nullptr};
  char *r_buf{nullptr};
  uint64_t *signal{nullptr};
  int *error_count{nullptr};
  int last_loop{0};  // actual loop count passed to the last launchKernel
};

#endif  // TESTS_FENCE_ORDERING_TESTER_HPP
