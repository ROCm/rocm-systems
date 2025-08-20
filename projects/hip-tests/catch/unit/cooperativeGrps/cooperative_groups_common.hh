/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

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
