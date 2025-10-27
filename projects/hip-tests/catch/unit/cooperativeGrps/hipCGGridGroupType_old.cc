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

enum class GridTypeTests { gridGroupType, baseType, publicApi };

static __device__ int gm[2];

static __global__ void kernel_cg_grid_group_type(int* size_dev, int* thd_rank_dev,
                                                 int* is_valid_dev, int* sync_dev,
                                                 dim3* group_dim_dev) {
  cg::grid_group gg = cg::this_grid();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test size
  size_dev[gIdx] = gg.size();

  // Test thread_rank
  thd_rank_dev[gIdx] = gg.thread_rank();

  // Test is_valid
  is_valid_dev[gIdx] = gg.is_valid();

  // Test sync
  if (blockIdx.x == 0 && threadIdx.x == 0)
    gm[0] = 10;
  else if (blockIdx.x == 1 && threadIdx.x == 0)
    gm[1] = 20;
  gg.sync();
  sync_dev[gIdx] = gm[1] * gm[0];

   // Test group_dim aka number of thread blocks in a grid
  group_dim_dev[gIdx] = gg.group_dim();
}

static __global__ void kernel_cg_grid_group_type_via_base_type(int* size_dev, int* thd_rank_dev,
                                                               int* is_valid_dev, int* sync_dev) {
  cg::thread_group tg = cg::this_grid();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test size
  size_dev[gIdx] = tg.size();

  // Test thread_rank
  thd_rank_dev[gIdx] = tg.thread_rank();

  // Test is_valid
#ifdef __HIP_PLATFORM_AMD__
  is_valid_dev[gIdx] = tg.is_valid();
#else
  // Cuda has no thread_group.is_valid()
  is_valid_dev[gIdx] = true;
#endif

  // Test sync
  if (blockIdx.x == 0 && threadIdx.x == 0)
    gm[0] = 10;
  else if (blockIdx.x == 1 && threadIdx.x == 0)
    gm[1] = 20;
  tg.sync();
  sync_dev[gIdx] = gm[1] * gm[0];
}

static __global__ void kernel_cg_grid_group_type_via_public_api(int* size_dev, int* thd_rank_dev,
                                                                int* is_valid_dev, int* sync_dev,
                                                                dim3* group_dim_dev) {
  cg::grid_group gg = cg::this_grid();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test group_size api
  size_dev[gIdx] = cg::group_size(gg);

  // Test thread_rank api
  thd_rank_dev[gIdx] = cg::thread_rank(gg);

  // Test is_valid api
  is_valid_dev[gIdx] = gg.is_valid();

  // Test sync
  if (blockIdx.x == 0 && threadIdx.x == 0)
    gm[0] = 10;
  else if (blockIdx.x == 1 && threadIdx.x == 0)
    gm[1] = 20;
  cg::sync(gg);
  sync_dev[gIdx] = gm[1] * gm[0];

  // Test group_dim aka number of thread blocks in a grid
  group_dim_dev[gIdx] = gg.group_dim();
}

static __global__ void coop_kernel(unsigned int* first_array, unsigned int* second_array,
                                   unsigned int loops, unsigned int array_len) {
  cg::grid_group grid = cg::this_grid();
  unsigned int rank = grid.thread_rank();
  unsigned int grid_size = grid.size();

  for (int i = 0; i < loops; i++) {
    // The goal of this loop is to directly add in values from
    // array one into array two, on a per-wave basis.
    for (int offset = rank; offset < array_len; offset += grid_size) {
      second_array[offset] += first_array[offset];
    }

    grid.sync();

    // The goal of this loop is to pull data the "mirror" lane in
    // array two and add it back into array one. This causes inter-
    // thread swizzling.
    for (int offset = rank; offset < array_len; offset += grid_size) {
      unsigned int swizzle_offset = array_len - offset - 1;
      first_array[offset] += second_array[swizzle_offset];
    }

    grid.sync();
  }
}

static __global__ void test_kernel(unsigned int* atomic_val, unsigned int* array,
                                   unsigned int loops) {
  cg::grid_group grid = cg::this_grid();
  unsigned rank = grid.thread_rank();

  int offset = blockIdx.x;
  for (int i = 0; i < loops; i++) {
    // Make the last thread run way behind everyone else.
    // If the barrier below fails, then the other threads may hit the
    // atomicInc instruction many times before the last thread ever gets to it.
    // As such, without the barrier, the last array entry will eventually
    // contain a very large value, defined by however many times the other
    // wavefronts make it through this loop.
    // If the barrier works, then it will likely contain some number
    // near "total number of blocks". It will be the last wavefront to
    // reach the atomicInc, but everyone will have only hit the atomic once.
    if (rank == (grid.size() - 1)) {
      long long time_diff = 0;
      long long last_clock = clock64();
      do {
        long long cur_clock = clock64();
        if (cur_clock > last_clock) {
          time_diff += (cur_clock - last_clock);
        }
        // If it rolls over, we don't know how much to add to catch up.
        // So just ignore those slipped cycles.
        last_clock = cur_clock;
      } while (time_diff < 1000000);
    }

    if (threadIdx.x == 0) {
      array[offset] = atomicInc(&atomic_val[0], UINT_MAX);
    }
    grid.sync();
    offset += gridDim.x;
  }
}

__global__ void test_kernel_gfx11(unsigned int* atomic_val, unsigned int* array,
                                  unsigned int loops) {
#if HT_AMD
  cg::grid_group grid = cg::this_grid();
  unsigned rank = grid.thread_rank();

  int offset = blockIdx.x;
  for (int i = 0; i < loops; i++) {
    // Make the last thread run way behind everyone else.
    // If the barrier below fails, then the other threads may hit the
    // atomicInc instruction many times before the last thread ever gets
    // to it.
    // As such, without the barrier, the last array entry will eventually
    // contain a very large value, defined by however many times the other
    // wavefronts make it through this loop.
    // If the barrier works, then it will likely contain some number
    // near "total number of blocks". It will be the last wavefront to
    // reach the atomicInc, but everyone will have only hit the atomic once.
    if (rank == (grid.size() - 1)) {
      long long time_diff = 0;
      long long last_clock = clock_function();
      do {
        long long cur_clock = clock_function();
        if (cur_clock > last_clock) {
          time_diff += (cur_clock - last_clock);
        }
        // If it rolls over, we don't know how much to add to catch up.
        // So just ignore those slipped cycles.
        last_clock = cur_clock;
      } while (time_diff < 1000000);
    }

    if (threadIdx.x == 0) {
      array[offset] = atomicInc(&atomic_val[0], UINT_MAX);
    }
    grid.sync();
    offset += gridDim.x;
  }
#endif
}

static void verify_coop_buffers(unsigned int* host_input, unsigned int* first_array,
                                unsigned int* second_array, unsigned int loops,
                                unsigned int array_len) {
  unsigned int* expected_first_array = host_input;
  unsigned int* expected_second_array =
      reinterpret_cast<unsigned int*>(malloc(sizeof(unsigned int) * array_len));
  memset(expected_second_array, 0, sizeof(unsigned int) * array_len);

  for (int i = 0; i < loops; i++) {
    for (int offset = 0; offset < array_len; offset++) {
      expected_second_array[offset] += expected_first_array[offset];
    }

    for (int offset = 0; offset < array_len; offset++) {
      unsigned int swizzle_offset = array_len - offset - 1;
      expected_first_array[offset] += expected_second_array[swizzle_offset];
    }
  }

  for (int i = 0; i < array_len; i++) {
    REQUIRE(first_array[i] == expected_first_array[i]);
    REQUIRE(second_array[i] == expected_second_array[i]);
  }

  free(expected_second_array);
}

static void verify_barrier_buffer(unsigned int loops, unsigned int warps,
                                  unsigned int* host_buffer) {
  unsigned int max_in_this_loop = 0;
  for (unsigned int i = 0; i < loops; i++) {
    max_in_this_loop += warps;
    for (unsigned int j = 0; j < warps; j++) {
      REQUIRE(host_buffer[i * warps + j] <= max_in_this_loop);
    }
  }
}

template <typename F> static void test_cg_grid_group_type(F kernel_func, int block_size, GridTypeTests kernel_type) {
  int num_bytes = sizeof(int) * 2 * block_size;
  int num_dim3_bytes = sizeof(dim3) * 2 * block_size;
  int *size_dev, *size_host;
  int *thd_rank_dev, *thd_rank_host;
  int *is_valid_dev, *is_valid_host;
  int *sync_dev, *sync_host;
  dim3 *group_dim_dev, *group_dim_host;

  // Allocate device memory
  HIP_CHECK(hipMalloc(&size_dev, num_bytes));
  HIP_CHECK(hipMalloc(&thd_rank_dev, num_bytes));
  HIP_CHECK(hipMalloc(&is_valid_dev, num_bytes));
  HIP_CHECK(hipMalloc(&sync_dev, num_bytes));
  HIP_CHECK(hipMalloc(&group_dim_dev, num_dim3_bytes));

  // Allocate host memory
  HIP_CHECK(hipHostMalloc(&size_host, num_bytes));
  HIP_CHECK(hipHostMalloc(&thd_rank_host, num_bytes));
  HIP_CHECK(hipHostMalloc(&is_valid_host, num_bytes));
  HIP_CHECK(hipHostMalloc(&sync_host, num_bytes));
  HIP_CHECK(hipHostMalloc(&group_dim_host, num_dim3_bytes));

  // Launch Kernel
  void* params[5];
  params[0] = &size_dev;
  params[1] = &thd_rank_dev;
  params[2] = &is_valid_dev;
  params[3] = &sync_dev;
  params[4] = &group_dim_dev;
  HIP_CHECK(hipLaunchCooperativeKernel(kernel_func, 2, block_size, params, 0, 0));

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(size_host, size_dev, num_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(thd_rank_host, thd_rank_dev, num_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(is_valid_host, is_valid_dev, num_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sync_host, sync_dev, num_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(group_dim_host, group_dim_dev, num_dim3_bytes, hipMemcpyDeviceToHost));

  // Validate results for both blocks together
  for (int i = 0; i < 2 * block_size; ++i) {
    ASSERT_EQUAL(size_host[i], 2 * block_size);
    ASSERT_EQUAL(thd_rank_host[i], i);
    ASSERT_EQUAL(is_valid_host[i], 1);
    ASSERT_EQUAL(sync_host[i], 200);
    if(kernel_type != GridTypeTests::baseType){
      ASSERT_EQUAL(group_dim_host[i].x, 2);           
      ASSERT_EQUAL(group_dim_host[i].y, 1);          
      ASSERT_EQUAL(group_dim_host[i].z, 1); 
    }
  }

  // Free device memory
  HIP_CHECK(hipFree(size_dev));
  HIP_CHECK(hipFree(thd_rank_dev));
  HIP_CHECK(hipFree(is_valid_dev));
  HIP_CHECK(hipFree(sync_dev));
  HIP_CHECK(hipFree(group_dim_dev));

  // Free host memory
  HIP_CHECK(hipHostFree(size_host));
  HIP_CHECK(hipHostFree(thd_rank_host));
  HIP_CHECK(hipHostFree(is_valid_host));
  HIP_CHECK(hipHostFree(sync_host));
  HIP_CHECK(hipHostFree(group_dim_host));
}

TEST_CASE("Unit_hipCGGridGroupType_Basic") {
  // Use default device for validating the test
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  void* kernel_func;
  GridTypeTests kernel_type = GridTypeTests::gridGroupType;

  SECTION("Default grid group API test") {
    kernel_func = reinterpret_cast<void*>(kernel_cg_grid_group_type);
    kernel_type = GridTypeTests::gridGroupType;
  }
#if HT_AMD
  SECTION("Base type grid group API test") {
    kernel_func = reinterpret_cast<void*>(kernel_cg_grid_group_type_via_base_type);
    kernel_type = GridTypeTests::baseType;
  }
#endif

  SECTION("Public API grid group test") {
    kernel_func = reinterpret_cast<void*>(kernel_cg_grid_group_type_via_public_api);
    kernel_type = GridTypeTests::publicApi;
  }

  // Test for block_size in powers of 2
  int max_threads_per_blk = device_properties.maxThreadsPerBlock;
  for (int block_size = 2; block_size <= max_threads_per_blk; block_size = block_size * 2) {
    test_cg_grid_group_type(kernel_func, block_size, kernel_type);
  }

  // Test for random blockSizes, but the sequence is the same every execution
  srand(0);
  for (int i = 0; i < 10; i++) {
    // Test fails for only 1 thread per block
    test_cg_grid_group_type(kernel_func, max(2, rand() % max_threads_per_blk), kernel_type);
  }
}

TEST_CASE("Unit_hipCGGridGroupType_DataSharing") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));

  hipDeviceProp_t device_properties;

  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  int loops = GENERATE(1, 2, 3, 4);
  int width = GENERATE(512, 1024, 2048, 4096);

  // Launch enough waves to fill up all of the GPU
  int warp_size = device_properties.warpSize;
  int num_sms = device_properties.multiProcessorCount;

  // Calculate the device occupancy to know how many blocks can be run.
  int max_blocks_per_sm;
  HIP_CHECK(
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, coop_kernel, warp_size, 0));

  int num_blocks = max_blocks_per_sm * num_sms;

  // Create Streams
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocate and initialize data

  // Alocate the host input buffer, and two device buffers
  unsigned int* input_buffer =
      reinterpret_cast<unsigned int*>(malloc(sizeof(unsigned int) * width));
  for (int i = 0; i < width; i++) {
    input_buffer[i] = i;
  }

  unsigned int *dev_mem_1, *host_mem_1;
  host_mem_1 = reinterpret_cast<unsigned int*>(malloc(sizeof(unsigned int) * width));
  HIP_CHECK(hipMalloc(&dev_mem_1, sizeof(unsigned int) * width));
  HIP_CHECK(hipMemcpyAsync(dev_mem_1, input_buffer, sizeof(unsigned int) * width,
                           hipMemcpyHostToDevice, stream));

  unsigned int *dev_mem_2, *host_mem_2;
  host_mem_2 = reinterpret_cast<unsigned int*>(malloc(sizeof(unsigned int) * width));
  HIP_CHECK(hipMalloc(&dev_mem_2, sizeof(unsigned int) * width));
  HIP_CHECK(hipMemsetAsync(dev_mem_2, 0, width * sizeof(unsigned int), stream));

  // Launch the kernels
  INFO("Launching a cooperative kernel with " << num_blocks << " blocks, each with " << warp_size
                                              << " threads");

  void* coop_params[4];
  coop_params[0] = reinterpret_cast<void*>(&dev_mem_1);
  coop_params[1] = reinterpret_cast<void*>(&dev_mem_2);
  coop_params[2] = reinterpret_cast<void*>(&loops);
  coop_params[3] = reinterpret_cast<void*>(&width);
  HIP_CHECK(hipLaunchCooperativeKernel(coop_kernel, num_blocks, warp_size, coop_params, 0, stream));

  // Read back the buffers and print out their data
  HIP_CHECK(hipMemcpyAsync(host_mem_1, dev_mem_1, sizeof(unsigned int) * width,
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipMemcpyAsync(host_mem_2, dev_mem_2, sizeof(unsigned int) * width,
                           hipMemcpyDeviceToHost, stream));

  HIP_CHECK(hipStreamSynchronize(stream));

  verify_coop_buffers(input_buffer, host_mem_1, host_mem_2, loops, width);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(dev_mem_1));
  HIP_CHECK(hipFree(dev_mem_2));
  free(input_buffer);
  free(host_mem_1);
  free(host_mem_2);
}

TEST_CASE("Unit_hipCGGridGroupType_Barrier") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));

  hipDeviceProp_t device_properties;

  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  uint32_t loops = GENERATE(1, 2, 3, 4);
  uint32_t warps = GENERATE(4, 8, 16, 32);
  uint32_t block_size = 1;

  // Test whether the requested size will fit on the GPU
  int max_blocks_per_sm;
  int warp_size = device_properties.warpSize;
  int num_sms = device_properties.multiProcessorCount;

  int num_threads_in_block = block_size * warp_size;

  auto test_kernel_used = IsGfx11() ? test_kernel_gfx11 : test_kernel;
  // Calculate the device occupancy to know how many blocks can be run.
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, test_kernel_used,
                                                         num_threads_in_block, 0));

  int requested_blocks = warps / block_size;
  if (requested_blocks > max_blocks_per_sm * num_sms) {
    INFO("Too many blocks requested!");
    REQUIRE(false);
  }

  // Each block will output a single value per loop.
  uint32_t total_buffer_len = requested_blocks * loops;

  // Alocate the buffer that will hold the kernel's output, and which will
  // also be used to globally synchronize during GWS initialization
  unsigned int* host_buffer =
      reinterpret_cast<unsigned int*>(calloc(total_buffer_len, sizeof(unsigned int)));

  unsigned int* kernel_buffer;
  HIP_CHECK(hipMalloc(&kernel_buffer, sizeof(unsigned int) * total_buffer_len));
  HIP_CHECK(hipMemcpy(kernel_buffer, host_buffer, sizeof(unsigned int) * total_buffer_len,
                      hipMemcpyHostToDevice));

  unsigned int* kernel_atomic;
  HIP_CHECK(hipMalloc(&kernel_atomic, sizeof(unsigned int)));
  HIP_CHECK(hipMemset(kernel_atomic, 0, sizeof(unsigned int)));

  // Launch the kernel
  INFO("Launching a cooperative kernel with " << warps << " warps in " << requested_blocks
                                              << " thread blocks");

  void* params[3];
  params[0] = reinterpret_cast<void*>(&kernel_atomic);
  params[1] = reinterpret_cast<void*>(&kernel_buffer);
  params[2] = reinterpret_cast<void*>(&loops);
  HIP_CHECK(hipLaunchCooperativeKernel(test_kernel_used, requested_blocks, num_threads_in_block,
                                       params, 0, 0));

  // Read back the buffer to host
  HIP_CHECK(hipMemcpy(host_buffer, kernel_buffer, sizeof(unsigned int) * total_buffer_len,
                      hipMemcpyDeviceToHost));

  verify_barrier_buffer(loops, requested_blocks, host_buffer);

  HIP_CHECK(hipFree(kernel_buffer));
  HIP_CHECK(hipFree(kernel_atomic));
  free(host_buffer);
}
