/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef TILE_REDUCE_TESTER_HPP
#define TILE_REDUCE_TESTER_HPP

#include "tester.hpp"

using namespace rocshmem;

__global__ void TileReduceThreadTest(rocshmem_team_t team, float *source,
                                     float *sum_dest, float *max_dest,
                                     float *min_dest, short *short_source,
                                     short *short_sum_dest,
                                     short *short_max_dest,
                                     short *short_min_dest, int *int_source,
                                     int *int_sum_dest, int *int_max_dest,
                                     int *int_min_dest, long *long_source,
                                     long *long_sum_dest, long *long_max_dest,
                                     long *long_min_dest, int tile_extent_0,
                                     int tile_extent_1, int my_world_pe,
                                     int n_pes, int root,
                                     ShmemContextType ctx_type,
                                     int *error_flag);

__global__ void TileReduceWaveTest(rocshmem_team_t team, float *source,
                                   float *sum_dest, float *max_dest,
                                   float *min_dest, short *short_source,
                                   short *short_sum_dest,
                                   short *short_max_dest,
                                   short *short_min_dest, int *int_source,
                                   int *int_sum_dest, int *int_max_dest,
                                   int *int_min_dest, long *long_source,
                                   long *long_sum_dest, long *long_max_dest,
                                   long *long_min_dest, int tile_extent_0,
                                   int tile_extent_1, int my_world_pe,
                                   int n_pes, int root,
                                   ShmemContextType ctx_type, int wf_size,
                                   int *error_flag);

__global__ void TileReduceTest(rocshmem_team_t *teams, int num_teams,
                               float *source, float *sum_dest, float *max_dest,
                               float *min_dest, short *short_source,
                               short *short_sum_dest, short *short_max_dest,
                               short *short_min_dest, int *int_source,
                               int *int_sum_dest, int *int_max_dest,
                               int *int_min_dest, long *long_source,
                               long *long_sum_dest, long *long_max_dest,
                               long *long_min_dest, int tile_extent_0,
                               int tile_extent_1, int my_world_pe, int n_pes,
                               int root, ShmemContextType ctx_type,
                               int *error_flag);

class TileReduceTester : public Tester {
 public:
  explicit TileReduceTester(TesterArguments args);

  virtual ~TileReduceTester();

  virtual void resetBuffers(size_t size) override;

  virtual void preLaunchKernel() override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            size_t size) override;

  virtual void postLaunchKernel() override;

  virtual void verifyResults(size_t size) override;

 protected:
  rocshmem_team_t *teams;
  int num_teams;

  float *source;
  float *sum_dest;
  float *max_dest;
  float *min_dest;
  short *short_source;
  short *short_sum_dest;
  short *short_max_dest;
  short *short_min_dest;
  int *int_source;
  int *int_sum_dest;
  int *int_max_dest;
  int *int_min_dest;
  long *long_source;
  long *long_sum_dest;
  long *long_max_dest;
  long *long_min_dest;
  int tile_extent_0;
  int tile_extent_1;

  int *error_flag;
};

#endif  // TILE_REDUCE_TESTER_HPP
