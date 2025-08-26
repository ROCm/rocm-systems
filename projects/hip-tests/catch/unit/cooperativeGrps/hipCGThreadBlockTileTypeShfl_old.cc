/*
Copyright (c) 2020 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

#include "hip_cg_common.hh"

namespace cg = cooperative_groups;

enum class TiledGroupShflTests { shflDown, shflXor, shflUp };

template <unsigned int tileSz>
__device__ int reduction_kernel_shfl_down(cg::thread_block_tile<tileSz> const& g,
                                          volatile int val) {
  int sz = g.size();

  for (int i = sz / 2; i > 0; i >>= 1) {
    val += g.shfl_down(val, i);
  }

  // Choose the 0'th indexed thread that holds the reduction value to return
  if (g.thread_rank() == 0) {
    return val;
  }
  // Rest of the threads return no useful values
  else {
    return -1;
  }
}

template <unsigned int tileSz>
__device__ int reduction_kernel_shfl_xor(cg::thread_block_tile<tileSz> const& g, int val) {
  int sz = g.size();

  for (int i = sz / 2; i > 0; i >>= 1) {
    val += g.shfl_xor(val, i);
  }

  // Choose the 0'th indexed thread that holds the reduction value to return
  if (g.thread_rank() == 0) {
    return val;
  }
  // Rest of the threads return no useful values
  else {
    return -1;
  }
}

template <unsigned int tileSz>
__device__ int prefix_sum_kernel(cg::thread_block_tile<tileSz> const& g, volatile int val) {
  int sz = g.size();
#pragma unroll
  for (int i = 1; i < sz; i <<= 1) {
    int temp = g.shfl_up(val, i);

    if (g.thread_rank() >= i) {
      val += temp;
    }
  }
  return val;
}

template <unsigned int tile_size>
static __global__ void kernel_cg_group_partition_static(int* result,
                                                        TiledGroupShflTests shfl_test) {
  cg::thread_block thread_block_CG_ty = cg::this_thread_block();
  int input, output_sum;

  // Choose a leader thread to print the results
  if (thread_block_CG_ty.thread_rank() == 0) {
    printf(" Creating %d groups, of tile size %d threads:\n\n",
           (int)thread_block_CG_ty.size() / tile_size, tile_size);
  }

  thread_block_CG_ty.sync();

  cg::thread_block_tile<tile_size> tiled_part = cg::tiled_partition<tile_size>(thread_block_CG_ty);

  input = tiled_part.thread_rank();

  switch (shfl_test) {
    case (TiledGroupShflTests::shflDown):
      output_sum = reduction_kernel_shfl_down(tiled_part, input);
      break;
    case (TiledGroupShflTests::shflXor):
      output_sum = reduction_kernel_shfl_xor(tiled_part, input);
      break;
    case (TiledGroupShflTests::shflUp):
      output_sum = prefix_sum_kernel(tiled_part, input);
      result[thread_block_CG_ty.thread_rank()] = output_sum;
  }

  if (tiled_part.thread_rank() == 0 && shfl_test != TiledGroupShflTests::shflUp) {
    printf("   Sum of all ranks 0..%d in this tiled_part group is %d\n", tiled_part.size() - 1,
           output_sum);
    result[thread_block_CG_ty.thread_rank() / (tile_size)] = output_sum;
  }
}

static void expected_result_calc(int* expected_result, int tile_size, int size,
                                 TiledGroupShflTests shfl_test) {
  switch (shfl_test) {
    case (TiledGroupShflTests::shflDown):
    case (TiledGroupShflTests::shflXor): {
      int expected_sum = ((tile_size - 1) * tile_size / 2);
      for (int i = 0; i < size; i++) {
        expected_result[i] = expected_sum;
      }
      break;
    }
    case (TiledGroupShflTests::shflUp): {
      for (int i = 0; i < size / tile_size; i++) {
        int acc = 0;
        for (int j = 0; j < tile_size; j++) {
          acc += j;
          expected_result[i * tile_size + j] = acc;
        }
      }
      break;
    }
  }
}

template <unsigned int tile_size> static void test_group_partition(TiledGroupShflTests shfl_test) {
  int block_size = 1;
  int threads_per_blk = 64;

  int num_elem = (block_size * threads_per_blk) / tile_size;
  if (shfl_test == TiledGroupShflTests::shflUp) {
    num_elem = block_size * threads_per_blk;
  }

  int* expected_result = new int[num_elem];

  int* result_dev = NULL;
  int* result_host = NULL;

  HIP_CHECK(hipHostMalloc(&result_host, num_elem * sizeof(int), hipHostMallocDefault));
  memset(result_host, 0, num_elem * sizeof(int));

  HIP_CHECK(hipMalloc(&result_dev, num_elem * sizeof(int)));

  // Launch Kernel
  hipLaunchKernelGGL(kernel_cg_group_partition_static<tile_size>, block_size, threads_per_blk,
                     threads_per_blk * sizeof(int), 0, result_dev, shfl_test);
  HIP_CHECK(hipDeviceSynchronize());


  HIP_CHECK(hipMemcpy(result_host, result_dev, sizeof(int) * num_elem, hipMemcpyDeviceToHost));

  expected_result_calc(expected_result, tile_size, num_elem, shfl_test);
  compareResults(expected_result, result_host, num_elem * sizeof(int));

  // Free all allocated memory on host and device
  HIP_CHECK(hipFree(result_dev));
  HIP_CHECK(hipHostFree(result_host));
  delete[] expected_result;
}

TEST_CASE("Unit_hipCGThreadBlockTileType_Shfl") {
  // Use default device for validating the test
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  TiledGroupShflTests shfl_test = GENERATE(
      TiledGroupShflTests::shflDown, TiledGroupShflTests::shflXor, TiledGroupShflTests::shflUp);
  test_group_partition<2>(shfl_test);
  test_group_partition<4>(shfl_test);
  test_group_partition<8>(shfl_test);
  test_group_partition<16>(shfl_test);
  test_group_partition<32>(shfl_test);
}
