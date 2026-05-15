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

enum class ThreadBlockTypeTests { basicApi, baseType, publicApi };

static __global__ void kernel_cg_thread_block_type(int* size_dev, int* thd_rank_dev, int* sync_dev,
                                                   dim3* group_index_dev, dim3* thd_index_dev,
                                                   dim3* group_dim_dev) {
  cg::thread_block tb = cg::this_thread_block();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
  // Test size
  size_dev[gIdx] = tb.size();

  // Test thread_rank
  thd_rank_dev[gIdx] = tb.thread_rank();

  // Test sync
  __shared__ int sm[2];
  if (threadIdx.x == 0)
    sm[0] = 10;
  else if (threadIdx.x == 1)
    sm[1] = 20;
  tb.sync();
  sync_dev[gIdx] = sm[1] * sm[0];

  // Test group_index
  group_index_dev[gIdx] = tb.group_index();

  // Test thread_index
  thd_index_dev[gIdx] = tb.thread_index();

  // Test group_dim aka number of threads in a block
  group_dim_dev[gIdx] = tb.group_dim();
}

static __global__ void kernel_cg_thread_block_type_via_base_type(int* size_dev, int* thd_rank_dev,
                                                                 int* sync_dev) {
  cg::thread_group tg = cg::this_thread_block();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test size
  size_dev[gIdx] = tg.size();

  // Test thread_rank
  thd_rank_dev[gIdx] = tg.thread_rank();

  // Test sync
  __shared__ int sm[2];
  if (threadIdx.x == 0)
    sm[0] = 10;
  else if (threadIdx.x == 1)
    sm[1] = 20;
  tg.sync();
  sync_dev[gIdx] = sm[1] * sm[0];
}

static __global__ void kernel_cg_thread_block_type_via_public_api(int* size_dev, int* thd_rank_dev,
                                                                  int* sync_dev) {
  cg::thread_block tb = cg::this_thread_block();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test group_size api
  size_dev[gIdx] = cg::group_size(tb);

  // Test thread_rank api
  thd_rank_dev[gIdx] = cg::thread_rank(tb);

  // Test sync api
  __shared__ int sm[2];
  if (threadIdx.x == 0)
    sm[0] = 10;
  else if (threadIdx.x == 1)
    sm[1] = 20;
  cg::sync(tb);
  sync_dev[gIdx] = sm[1] * sm[0];
}

static void test_cg_thread_block_type(ThreadBlockTypeTests test_type, int block_size) {
  int num_bytes = sizeof(int) * 2 * block_size;
  int num_dim3_bytes = sizeof(dim3) * 2 * block_size;
  int *size_dev, *size_host;
  int *thd_rank_dev, *thd_rank_host;
  int *sync_dev, *sync_host;
  dim3 *group_index_dev, *group_index_host;
  dim3 *thd_index_dev, *thd_index_host;
  dim3 *group_dim_dev, *group_dim_host;

  // Allocate device memory
  HIP_CHECK(hipMalloc(&size_dev, num_bytes));
  HIP_CHECK(hipMalloc(&thd_rank_dev, num_bytes));
  HIP_CHECK(hipMalloc(&sync_dev, num_bytes));

  // Allocate host memory
  HIP_CHECK(hipHostMalloc(&size_host, num_bytes));
  HIP_CHECK(hipHostMalloc(&thd_rank_host, num_bytes));
  HIP_CHECK(hipHostMalloc(&sync_host, num_bytes));

  switch (test_type) {
    case (ThreadBlockTypeTests::basicApi):
      HIP_CHECK(hipMalloc(&group_index_dev, num_dim3_bytes));
      HIP_CHECK(hipMalloc(&thd_index_dev, num_dim3_bytes));
      HIP_CHECK(hipMalloc(&group_dim_dev, num_dim3_bytes));
      HIP_CHECK(hipHostMalloc(&group_index_host, num_dim3_bytes));
      HIP_CHECK(hipHostMalloc(&thd_index_host, num_dim3_bytes));
      HIP_CHECK(hipHostMalloc(&group_dim_host, num_dim3_bytes));

      hipLaunchKernelGGL(kernel_cg_thread_block_type, 2, block_size, 0, 0, size_dev, thd_rank_dev,
                         sync_dev, group_index_dev, thd_index_dev, group_dim_dev);
      break;
    case (ThreadBlockTypeTests::baseType):
      hipLaunchKernelGGL(kernel_cg_thread_block_type_via_base_type, 2, block_size, 0, 0, size_dev,
                         thd_rank_dev, sync_dev);
      break;
    case (ThreadBlockTypeTests::publicApi):
      hipLaunchKernelGGL(kernel_cg_thread_block_type_via_public_api, 2, block_size, 0, 0, size_dev,
                         thd_rank_dev, sync_dev);
  }

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(size_host, size_dev, num_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(thd_rank_host, thd_rank_dev, num_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sync_host, sync_dev, num_bytes, hipMemcpyDeviceToHost));
  if (test_type == ThreadBlockTypeTests::basicApi) {
    HIP_CHECK(hipMemcpy(group_index_host, group_index_dev, num_dim3_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(thd_index_host, thd_index_dev, num_dim3_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(group_dim_host, group_dim_dev, num_dim3_bytes, hipMemcpyDeviceToHost));
  }

  // Validate results for both blocks together
  for (int i = 0; i < 2 * block_size; ++i) {
    ASSERT_EQUAL(size_host[i], block_size);
    ASSERT_EQUAL(thd_rank_host[i], i % block_size);
    ASSERT_EQUAL(sync_host[i], 200);
    if (test_type == ThreadBlockTypeTests::basicApi) {
      ASSERT_EQUAL(group_index_host[i].x, (uint)i / block_size);
      ASSERT_EQUAL(group_index_host[i].y, 0);
      ASSERT_EQUAL(group_index_host[i].z, 0);
      ASSERT_EQUAL(thd_index_host[i].x, (uint)i % block_size);
      ASSERT_EQUAL(thd_index_host[i].y, 0);
      ASSERT_EQUAL(thd_index_host[i].z, 0);
      ASSERT_EQUAL(group_dim_host[i].x, block_size);
      ASSERT_EQUAL(group_dim_host[i].y, 1);
      ASSERT_EQUAL(group_dim_host[i].z, 1);
    }
  }

  // Free device memory
  HIP_CHECK(hipFree(size_dev));
  HIP_CHECK(hipFree(thd_rank_dev));
  HIP_CHECK(hipFree(sync_dev));

  // Free host memory
  HIP_CHECK(hipHostFree(size_host));
  HIP_CHECK(hipHostFree(thd_rank_host));
  HIP_CHECK(hipHostFree(sync_host));

  if (test_type == ThreadBlockTypeTests::basicApi) {
    HIP_CHECK(hipFree(group_index_dev));
    HIP_CHECK(hipFree(thd_index_dev));
    HIP_CHECK(hipFree(group_dim_dev));
    HIP_CHECK(hipHostFree(group_index_host));
    HIP_CHECK(hipHostFree(thd_index_host));
    HIP_CHECK(hipHostFree(group_dim_host));
  }
}


TEST_CASE("Unit_hipCGThreadBlockType") {
  // Use default device for validating the test
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  ThreadBlockTypeTests test_type = ThreadBlockTypeTests::basicApi;

  SECTION("Default thread block API test") { test_type = ThreadBlockTypeTests::basicApi; }

  SECTION("Base type thread block API test") { test_type = ThreadBlockTypeTests::baseType; }

  SECTION("Public API thread block test") { test_type = ThreadBlockTypeTests::publicApi; }

  // Test for blockSizes in powers of 2
  int max_threads_per_blk = device_properties.maxThreadsPerBlock;
  for (int block_size = 2; block_size <= max_threads_per_blk; block_size = block_size * 2) {
    test_cg_thread_block_type(test_type, block_size);
  }

  // Test for random block_size, but the sequence is the same every execution
  srand(0);
  for (int i = 0; i < 10; i++) {
    // Test fails for only 1 thread per block
    test_cg_thread_block_type(test_type, max(2, rand() % max_threads_per_blk));
  }
}
