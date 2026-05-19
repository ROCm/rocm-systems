/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef TILE_BROADCAST_TESTER_HPP
#define TILE_BROADCAST_TESTER_HPP

#include "tester.hpp"

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNELS
 *****************************************************************************/
// Thread-level broadcast test - single WG, single wave
__global__ void TileBroadcastThreadTest(rocshmem_team_t team,
                                        float *source, float *dest,
                                        int tile_extent_0, int tile_extent_1,
                                        int my_world_pe, int pe_root,
                                        ShmemContextType ctx_type,
                                        int *error_flag);

// Wave-level broadcast test - single WG, single wave
__global__ void TileBroadcastWaveTest(rocshmem_team_t team,
                                      float *source, float *dest,
                                      int tile_extent_0, int tile_extent_1,
                                      int my_world_pe, int pe_root,
                                      ShmemContextType ctx_type,
                                      int wf_size,
                                      int *error_flag);

// Workgroup-level broadcast test - multiple WGs with different teams
__global__ void TileBroadcastTest(rocshmem_team_t *teams, int num_teams,
                                  float *source, float *dest,
                                  int tile_extent_0, int tile_extent_1,
                                  int my_world_pe, int pe_root,
                                  ShmemContextType ctx_type,
                                  int *error_flag);

/******************************************************************************
 * HOST TESTER CLASS
 *****************************************************************************/
class TileBroadcastTester : public Tester {
 public:
  explicit TileBroadcastTester(TesterArguments args);

  virtual ~TileBroadcastTester();

  virtual void resetBuffers(size_t size) override;

  virtual void preLaunchKernel() override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            size_t size) override;

  virtual void postLaunchKernel() override;

  virtual void verifyResults(size_t size) override;

 protected:
  rocshmem_team_t *teams;  // Array of teams
  int num_teams;           // Number of teams to create

  float *source;           // Source tile data
  float *dest;             // Destination tile data
  int tile_extent_0;       // Tile dimension 0
  int tile_extent_1;       // Tile dimension 1

  int *error_flag;         // Device error flag for verification
};

#endif  // TILE_BROADCAST_TESTER_HPP
