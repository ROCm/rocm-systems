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

  // Default tile dimensions (rows × cols) when no -s is specified.
  static constexpr int DEFAULT_TILE_ROWS = 64;
  static constexpr int DEFAULT_TILE_COLS = 64;
  // Maximum col count for the arbitrary-stride layout (row_stride=257, col_stride=3):
  // tile_extent_1 * col_stride must fit within row_stride → floor(257/3) = 85.
  static constexpr int ARBITRARY_MAX_COLS = 85;

  // Maximum tile dimensions — set from max_msg_size at construction time.
  // tile_extent_0 (rows) is fixed; tile_extent_1 (cols) scales with message size.
  // These bound the allocated buffer size.
  int tile_extent_0 = DEFAULT_TILE_ROWS;
  int tile_extent_1 = DEFAULT_TILE_COLS;


  // Symmetric heap allocations
  SymmetricTensorBuffer<float> *local_alloc = nullptr;
  SymmetricTensorBuffer<float> *remote_alloc = nullptr;
};

#endif
