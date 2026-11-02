/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#ifndef TILE_ND_RMA_TESTER_HPP
#define TILE_ND_RMA_TESTER_HPP

#include "tester.hpp"

using namespace rocshmem;

class TileNDRMATester : public Tester {
 public:
  // Fixed tile dimensions — no message-size sweep.
  static constexpr int D0_3D = 8;
  static constexpr int D1_3D = 16;
  static constexpr int D2_3D = 4;
  static constexpr int TILE_ELEMS_3D = D0_3D * D1_3D * D2_3D;  // 512

  static constexpr int D0_4D = 2;
  static constexpr int D1_4D = 8;
  static constexpr int D2_4D = 8;
  static constexpr int D3_4D = 4;
  static constexpr int TILE_ELEMS_4D = D0_4D * D1_4D * D2_4D * D3_4D;  // 512

  // Both 3D and 4D tiles have the same total element count.
  static constexpr int TILE_ELEMS = 512;

  explicit TileNDRMATester(TesterArguments args);
  ~TileNDRMATester() override;

  void resetBuffers(uint64_t size) override;
  void launchKernel(dim3 gridSize, dim3 blockSize, int loop, uint64_t size) override;
  void verifyResults(uint64_t size) override;

 private:
  float* source_buf{nullptr};  // owns both symmetric allocations
  float* dest_buf{nullptr};
  float* source{nullptr};      // points into source_buf or dest_buf
  float* dest{nullptr};        // points into source_buf or dest_buf
};

#endif  // TILE_ND_RMA_TESTER_HPP
