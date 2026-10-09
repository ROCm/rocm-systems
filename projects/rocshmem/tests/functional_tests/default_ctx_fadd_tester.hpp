/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef _DEFAULT_CTX_FADD_TESTER_HPP_
#define _DEFAULT_CTX_FADD_TESTER_HPP_

#include "tester.hpp"

// Every thread of PE 0 does fetch_add(+1) on one counter at PE 1 over the default context; the returns must be a permutation.
class DefaultCTXFAddTester : public Tester {
 public:
  explicit DefaultCTXFAddTester(TesterArguments args);
  virtual ~DefaultCTXFAddTester();

 protected:
  virtual void resetBuffers(uint64_t size) override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop, uint64_t size) override;

  virtual void verifyResults(uint64_t size) override;

  uint64_t *counter = nullptr;
  uint64_t *fetched = nullptr;
  size_t max_fetches = 0;
  size_t num_fetches = 0;
};

#endif
