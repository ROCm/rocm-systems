/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <hip/cooperative_groups/hip_reduce.h>
#include "hip_test_params.hh"
#include <cmd_options.hh>

namespace {
constexpr int kMaxGPUs = 8;
}  // namespace

constexpr int MaxGPUs = 8;

inline bool operator==(const dim3& l, const dim3& r) {
  return l.x == r.x && l.y == r.y && l.z == r.z;
}

inline bool operator!=(const dim3& l, const dim3& r) { return !(l == r); }

__device__ inline unsigned int thread_rank_in_grid() {
  const auto block_size = blockDim.x * blockDim.y * blockDim.z;
  const auto block_rank_in_grid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto thread_rank_in_block =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  return block_rank_in_grid * block_size + thread_rank_in_block;
}

template <class T> bool CheckDimensions(unsigned int device, T kernel, dim3 blocks, dim3 threads) {
  hipDeviceProp_t props;
  int max_blocks_per_sm = 0;
  int num_sm = 0;
  HIP_CHECK(hipSetDevice(device));
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel,
                                                         threads.x * threads.y * threads.z, 0));

  HIP_CHECK(hipGetDeviceProperties(&props, device));
  num_sm = props.multiProcessorCount;

  if ((blocks.x * blocks.y * blocks.z) > max_blocks_per_sm * num_sm ||
       blocks.x <= 0 || blocks.y <= 0 || blocks.z <= 0 ||
       threads.x <= 0 || threads.y <= 0 || threads.z <= 0) {
    return false;
  }

  return true;
}

inline double GetTestReductionFactor() { return cmd_options.cg_reduction_factor * 0.01; }

/**
 * Prints one line per Catch GENERATE combination when HIP_COOP_LOG_DIMS is set (any non-empty
 * value except "0"). Example: HIP_COOP_LOG_DIMS=1 ./coopGrpTest ...
 */
inline void CoopLogGridDimsIfEnabled(const dim3& grid_blocks, const dim3& thread_block,
                                     const char* combo_tag) {
  const char* env = std::getenv("HIP_COOP_LOG_DIMS");
  if (env == nullptr || env[0] == '\0' || (env[0] == '0' && env[1] == '\0')) {
    return;
  }
  int grid_mult_n = 0;
  int thread_mult_n = 0;
  if (std::strcmp(combo_tag, "shuffle") == 0) {
    grid_mult_n = CooperativeBlockGridMultiplierListSizeShuffle();
    thread_mult_n = CooperativeThreadMultiplierListSizeShuffle();
  } else if (std::strcmp(combo_tag, "thread_only") == 0) {
    grid_mult_n = 1;
    thread_mult_n = CooperativeThreadMultiplierListSizeFull();
  } else if (std::strncmp(combo_tag, "strix", 5) == 0) {
    grid_mult_n = -1;
    thread_mult_n = -1;
  } else {
    grid_mult_n = CooperativeBlockGridMultiplierListSize();
    thread_mult_n = CooperativeThreadMultiplierListSizeFull();
  }
  if (grid_mult_n < 0) {
    std::fprintf(stderr,
                 "[HIP_COOP_LOG_DIMS] %s level=%d grid=%u,%u,%u block=%u,%u,%u | "
                 "grid_mult_n=n/a thread_mult_n=n/a | total_est=n/a (strix hardcoded GENERATE)\n",
                 combo_tag, CurrentTestLevelNumber(), grid_blocks.x, grid_blocks.y, grid_blocks.z,
                 thread_block.x, thread_block.y, thread_block.z);
  } else {
    int Mg = grid_mult_n;
    int Mt = thread_mult_n;
    int n_block = CooperativeBlockGridGeneratorUnionCount(Mg);
    int n_thread = CooperativeThreadGeneratorUnionCount(Mt);
    int total_est = -1;
    if (std::strcmp(combo_tag, "thread_only") == 0) {
      n_block = 1;
      total_est = n_thread;
    } else {
      total_est = n_block * n_thread;
    }
    std::fprintf(stderr,
                 "[HIP_COOP_LOG_DIMS] %s level=%d grid=%u,%u,%u block=%u,%u,%u | "
                 "grid_mult_n=%d thread_mult_n=%d | n_block_gen=%d n_thread_gen=%d total_est=%d "
                 "(Catch2: union per GENERATE; two GENERATEs multiply)\n",
                 combo_tag, CurrentTestLevelNumber(), grid_blocks.x, grid_blocks.y, grid_blocks.z,
                 thread_block.x, thread_block.y, thread_block.z, grid_mult_n, thread_mult_n, n_block,
                 n_thread, total_est);
  }
  std::fflush(stderr);
}

inline void CoopLogMultiGridCaseIfEnabled(unsigned int test_case) {
  const char* env = std::getenv("HIP_COOP_LOG_DIMS");
  if (env == nullptr || env[0] == '\0' || (env[0] == '0' && env[1] == '\0')) {
    return;
  }
  const int max_cases = CooperativeMultiGridTestCaseCount();
  std::fprintf(stderr,
               "[HIP_COOP_LOG_DIMS] multi_grid test_case=%u (1-based %u/%d) level=%d | "
               "total_test_case_combinations=%d\n",
               test_case, test_case + 1, max_cases, CurrentTestLevelNumber(), max_cases);
  std::fflush(stderr);
}
