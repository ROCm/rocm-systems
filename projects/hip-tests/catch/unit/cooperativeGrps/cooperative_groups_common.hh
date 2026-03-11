/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <hip/cooperative_groups/hip_reduce.h>
#include <cmd_options.hh>
#include "../math/math_common.hh"
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
