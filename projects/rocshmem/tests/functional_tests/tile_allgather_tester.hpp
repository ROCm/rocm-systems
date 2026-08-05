/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef TILE_ALLGATHER_TESTER_HPP
#define TILE_ALLGATHER_TESTER_HPP

#include "tester.hpp"

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNELS
 *****************************************************************************/
// Thread-level allgather test - single WG, single wave
__global__ void TileAllgatherThreadTest(rocshmem_team_t team,
                                        float *source, float *dest,
                                        int tile_extent_0, int tile_extent_1,
                                        int my_world_pe, int n_pes,
                                        ShmemContextType ctx_type,
                                        int *error_flag);

// Wave-level allgather test - single WG, single wave
__global__ void TileAllgatherWaveTest(rocshmem_team_t team,
                                      float *source, float *dest,
                                      int tile_extent_0, int tile_extent_1,
                                      int my_world_pe, int n_pes,
                                      ShmemContextType ctx_type,
                                      int wf_size,
                                      int *error_flag);

// Workgroup-level allgather test - multiple WGs with different teams
__global__ void TileAllgatherTest(rocshmem_team_t *teams, int num_teams,
                                   float *source, float *dest,
                                   int tile_extent_0, int tile_extent_1,
                                   int my_world_pe, int n_pes,
                                   ShmemContextType ctx_type,
                                   int *error_flag);

/******************************************************************************
 * HOST TESTER CLASS
 *****************************************************************************/
class TileAllgatherTester : public Tester {
 public:
  explicit TileAllgatherTester(TesterArguments args);

  virtual ~TileAllgatherTester();

  virtual void resetBuffers(size_t size) override;

  virtual void preLaunchKernel() override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            size_t size) override;

  virtual void postLaunchKernel() override;

  virtual void verifyResults(size_t size) override;

 protected:
  rocshmem_team_t *teams;  // Array of teams
  int num_teams;           // Number of teams to create

  float *source;           // Source tile data (one tile per PE)
  float *dest;             // Destination tile data (n_pes tiles per PE)
  int tile_extent_0;       // Tile dimension 0
  int tile_extent_1;       // Tile dimension 1

  int *error_flag;         // Device error flag for verification
};

#endif  // TILE_ALLGATHER_TESTER_HPP
