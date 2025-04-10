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

#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <hip_test_common.hh>

template <bool use_mask> __global__ void warp_reduce(const int* in, int* out, int warp_size) {
  extern __shared__ int shared_scratch[];

  size_t thread_id = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  shared_scratch[thread_id] = in[idx];

  size_t warp_id = thread_id / warp_size;
  size_t lane_id = thread_id % warp_size;
  for (unsigned int s = 1; s < warp_size; s *= 2) {
    if (lane_id % (2 * s) == 0) {
      shared_scratch[thread_id] += shared_scratch[thread_id + s];
    }
    if constexpr (use_mask) {
      uint64_t mask = 0;
      for (int i = 0; i < warp_size; i += s) {
        mask |= (1 << i);
      }
      __syncwarp(mask);
    } else {
      __syncwarp();
    }
  }

  if (lane_id == 0) {
    out[warp_id + (blockDim.x * blockIdx.x) / warp_size] = shared_scratch[thread_id];
  }
}

void run_warp_test(int warp_size, int block_size, bool use_mask) {
  constexpr int n_blocks = 2;
  size_t size = block_size * n_blocks;
  size_t n_warps = size / warp_size;

  // construct
  int* in_d = nullptr;
  int* out_d = nullptr;
  HIP_CHECK(hipMalloc(&in_d, size * sizeof(int)));
  HIP_CHECK(hipMalloc(&out_d, n_warps * sizeof(int)));

  // Set values
  std::vector<int> in_h(size);
  {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_int_distribution<int> dist{0, 84};
    auto gen = [&]() { return dist(mersenne_engine); };
    std::generate(in_h.begin(), in_h.end(), gen);
  }
  HIP_CHECK(hipMemcpy(in_d, in_h.data(), size * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(out_d, 0, n_warps * sizeof(int)));

  // execute reduction
  if (use_mask) {
    warp_reduce<true><<<n_blocks, block_size, block_size * sizeof(int)>>>(in_d, out_d, warp_size);
  } else {
    warp_reduce<false><<<n_blocks, block_size, block_size * sizeof(int)>>>(in_d, out_d, warp_size);
  }
  HIP_CHECK(hipStreamSynchronize(0));

  // validate
  std::vector<int> out_h(n_warps);
  HIP_CHECK(hipMemcpy(out_h.data(), out_d, n_warps * sizeof(int), hipMemcpyDeviceToHost));
  auto itt = in_h.begin();
  for (const auto val : out_h) {
    int expected = std::accumulate(itt, itt + warp_size, 0);
    REQUIRE(val == expected);
    itt += warp_size;
  }

  // release
  HIP_CHECK(hipFree(in_d));
  HIP_CHECK(hipFree(out_d));
}

/**
 * Test Description
 * ------------------------
 *  - Validates the warp synchronize behaviour. This launches a reduce with divergent threads
 *    within the warp, these threads are then synchronized with __syncwarp. This is tested with
 *    with and without a mask, the mask test only syncs the threads that access shared memory.
 * ------------------------
 *  - unit/warp/syncwarp.cc
 * Test requirements
 */
TEST_CASE("Unit_Warp_Sync", "") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  int warp_size = device_properties.warpSize;
  int block_size = device_properties.maxThreadsPerBlock;

  SECTION("syncwarp without mask") { run_warp_test(warp_size, block_size, false); }

  SECTION("syncwarp with mask") { run_warp_test(warp_size, block_size, true); }
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
