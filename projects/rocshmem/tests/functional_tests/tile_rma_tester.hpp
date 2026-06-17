/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#ifndef _TILE_RMA_TESTER_HPP_
#define _TILE_RMA_TESTER_HPP_

#include "tester.hpp"

// Forward declaration
template <typename T>
class SymmetricTensorBuffer;

/******************************************************************************
 * HOST TESTER CLASS
 *****************************************************************************/
class TileRMATester : public Tester {
 public:
  explicit TileRMATester(TesterArguments args);
  virtual ~TileRMATester();

 protected:
  virtual void resetBuffers(uint64_t size) override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            uint64_t size) override;

  virtual void verifyResults(uint64_t size) override;

  float *source = nullptr;
  float *dest = nullptr;

  // Symmetric heap allocations
  SymmetricTensorBuffer<float> *local_alloc = nullptr;
  SymmetricTensorBuffer<float> *remote_alloc = nullptr;
};

#endif
