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

static __global__ void kernel_cg_multi_grid_group_type(int* grid_rank_dev, int* size_dev,
                                                       int* thd_rank_dev, int* is_valid_dev,
                                                       int* sync_dev, int* sync_result,
                                                       int* num_grids_dev) {
  cg::multi_grid_group mg = cg::this_multi_grid();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test num_grids
  num_grids_dev[gIdx] = mg.num_grids();

  // Test grid_rank
  grid_rank_dev[gIdx] = mg.grid_rank();

  // Test size
  size_dev[gIdx] = mg.size();

  // Test thread_rank
  thd_rank_dev[gIdx] = mg.thread_rank();

  // Test is_valid
  is_valid_dev[gIdx] = mg.is_valid();

  // Test sync
  //
  // Eech thread assign 1 to their respective location
  sync_dev[gIdx] = 1;
  // Grid level sync
  cg::this_grid().sync();
  // Thread 0 from work-group 0 of current grid (gpu) does grid level reduction
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (uint i = 1; i < gridDim.x * blockDim.x; ++i) {
      sync_dev[0] += sync_dev[i];
    }
    sync_result[mg.grid_rank() + 1] = sync_dev[0];
  }
  // multi-grid level sync
  mg.sync();
  // grid (gpu) 0 does final reduction across all grids (gpus)
  if (mg.grid_rank() == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
    sync_result[0] = 0;
    for (uint i = 1; i <= mg.num_grids(); ++i) {
      sync_result[0] += sync_result[i];
    }
  }
}

static __global__ void kernel_cg_multi_grid_group_type_via_base_type(
    int* grid_rank_dev, int* size_dev, int* thd_rank_dev, int* is_valid_dev, int* sync_dev,
    int* sync_result) {
  cg::thread_group tg =
      cg::this_multi_grid();  // This can work if _CG_ABI_EXPERIMENTAL defined on Cuda

  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test size
  size_dev[gIdx] = tg.size();

  // Test thread_rank
  grid_rank_dev[gIdx] = cg::this_multi_grid().grid_rank();
  thd_rank_dev[gIdx] = tg.thread_rank();

  // Test is_valid
#ifdef __HIP_PLATFORM_AMD__
  is_valid_dev[gIdx] = tg.is_valid();
#else
  // Cuda has no thread_group.is_valid()
  is_valid_dev[gIdx] = true;
#endif
  // Test sync
  //
  // Eech thread assign 1 to their respective location
  sync_dev[gIdx] = 1;
  // Grid level sync
  cg::this_grid().sync();
  // Thread 0 from work-group 0 of current grid (gpu) does grid level reduction
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (uint i = 1; i < gridDim.x * blockDim.x; ++i) {
      sync_dev[0] += sync_dev[i];
    }
    sync_result[cg::this_multi_grid().grid_rank() + 1] = sync_dev[0];
  }
  // multi-grid level sync
  tg.sync();
  // grid (gpu) 0 does final reduction across all grids (gpus)
  if (cg::this_multi_grid().grid_rank() == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
    sync_result[0] = 0;
    for (uint i = 1; i <= cg::this_multi_grid().num_grids(); ++i) {
      sync_result[0] += sync_result[i];
    }
  }
}

static __global__ void kernel_cg_multi_grid_group_type_via_public_api(
    int* grid_rank_dev, int* size_dev, int* thd_rank_dev, int* is_valid_dev, int* sync_dev,
    int* sync_result) {
  cg::multi_grid_group mg = cg::this_multi_grid();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test group_size api
  size_dev[gIdx] = cg::group_size(mg);

  // Test thread_rank api
  grid_rank_dev[gIdx] = cg::this_multi_grid().grid_rank();
  thd_rank_dev[gIdx] = cg::thread_rank(mg);

  // Test is_valid api
  is_valid_dev[gIdx] = mg.is_valid();

  // Test sync api
  //
  // Eech thread assign 1 to their respective location
  sync_dev[gIdx] = 1;
  // Grid level sync
  cg::sync(cg::this_grid());
  // Thread 0 from work-group 0 of current grid (gpu) does grid level reduction
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (uint i = 1; i < gridDim.x * blockDim.x; ++i) {
      sync_dev[0] += sync_dev[i];
    }
    sync_result[cg::this_multi_grid().grid_rank() + 1] = sync_dev[0];
  }
  // multi-grid level sync via public api
  cg::sync(mg);
  // grid (gpu) 0 does final reduction across all grids (gpus)
  if (cg::this_multi_grid().grid_rank() == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
    sync_result[0] = 0;
    for (uint i = 1; i <= cg::this_multi_grid().num_grids(); ++i) {
      sync_result[0] += sync_result[i];
    }
  }
}

__global__ void test_kernel(unsigned int* atomic_val, unsigned int* array, uint32_t loops,
                            unsigned int* per_loop_atomic, unsigned int* grid_counters,
                            unsigned int* recorded_values, unsigned int* grid_values) {
  cg::grid_group grid = cg::this_grid();
  cg::multi_grid_group mgrid = cg::this_multi_grid();
  unsigned rank = grid.thread_rank();
  int grid_id = mgrid.grid_rank();
  int num_grids = mgrid.num_grids();
  int blocks_seen = 0;
  int grid_blocks = gridDim.x * gridDim.y * gridDim.z;
  int offset = blockIdx.x;
  for (int i = 0; i < loops; i++) {
    // Make the last thread run way behind everyone else.
    // If the grid sync below fails, then the other threads may hit the
    // atomicInc instruction many times before the last thread ever gets to it.
    // If the sync works, then it will likely contain "total number of blocks"*iter
    if (rank == (grid.size() - 1)) {
      // The last wavefront should spin on this loop's atomic value
      // until all of the other wavefronts have incremented the
      // per-loop atomic and hit the grid.sync()
#if HT_AMD
      while (__hip_atomic_load(&per_loop_atomic[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) <
             (grid_blocks - 1)) {
        __builtin_amdgcn_s_sleep(127);
      }

      // Give the other waves time to maybe go around the loop again
      // if the barrier has failed
      __builtin_amdgcn_s_sleep(127);
#else  // CUDA does not seem to need an ordered atomic load
      while (per_loop_atomic[i] < (grid_blocks - 1)) {
      }
#endif
    }

    if (threadIdx.x == 0) {
      atomicInc(&per_loop_atomic[i], UINT_MAX);
      array[offset + blocks_seen] = atomicInc(atomic_val, UINT_MAX);
    }
    grid.sync();
    blocks_seen += grid_blocks;

    // Each grid updates its own counter
    if (rank == (grid.size() - 1)) {
      grid_values[grid_id] = atomicAdd(&grid_counters[grid_id], grid_id + 1);
    }
    mgrid.sync();

    // After mgrid sync, read the next grid's counter
    recorded_values[(grid_id * loops + i)] = grid_values[(grid_id + 1) % num_grids];
    mgrid.sync();
  }
}

static void verify_barrier_buffer(unsigned int loops, unsigned int warps, unsigned int* host_buffer,
                                  unsigned int num_devs) {
  unsigned int max_in_this_loop = 0;
  for (unsigned int i = 0; i < loops; i++) {
    max_in_this_loop += (warps * num_devs);
    for (unsigned int j = 0; j < warps; j++) {
      REQUIRE(host_buffer[i * warps + j] <= max_in_this_loop);
    }
  }
}

// Function to verify recorded readings
static void verify_recorded_values(unsigned int* recorded_values, uint32_t loops, int num_devices) {
  for (uint32_t i = 0; i < loops; i++) {
    for (int grid_id = 0; grid_id < num_devices; grid_id++) {
      // Determine the expected value from the next grid's counter
      int next_grid_id = (grid_id + 1) % num_devices;
      // The expected value should be the sum of the increments from previous loops
      unsigned int expected_value = i * (next_grid_id + 1);
      // Check the recorded value
      unsigned int recorded_value = recorded_values[grid_id * loops + i];
      REQUIRE(recorded_value == expected_value);
      INFO("Mismatch at loop " << i << " for grid " << grid_id << ": expected " << expected_value
                               << ", got " << recorded_value);
    }
  }
}

template <typename F> static void test_cg_multi_grid_group_type(F kernel_func, int num_devices,
                                                                int block_size,
                                                                bool specific_api_test) {
  // Create a stream each device
  hipStream_t stream[MaxGPUs];
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());  // Make sure work is done on this device
    HIP_CHECK(hipStreamCreate(&stream[i]));
  }

  // Allocate host and device memory
  int num_bytes = sizeof(int) * 2 * block_size;
  int *num_grids_dev[MaxGPUs], *num_grids_host[MaxGPUs];
  int *grid_rank_dev[MaxGPUs], *grid_rank_host[MaxGPUs];
  int *size_dev[MaxGPUs], *size_host[MaxGPUs];
  int *thd_rank_dev[MaxGPUs], *thd_rank_host[MaxGPUs];
  int *is_valid_dev[MaxGPUs], *is_valid_host[MaxGPUs];
  int *sync_dev[MaxGPUs], *sync_result;
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));

    if (specific_api_test) {
      HIP_CHECK(hipMalloc(&num_grids_dev[i], num_bytes));
      HIP_CHECK(hipHostMalloc(&num_grids_host[i], num_bytes));
    }

    HIP_CHECK(hipMalloc(&grid_rank_dev[i], num_bytes));
    HIP_CHECK(hipMalloc(&size_dev[i], num_bytes));
    HIP_CHECK(hipMalloc(&thd_rank_dev[i], num_bytes));
    HIP_CHECK(hipMalloc(&is_valid_dev[i], num_bytes));
    HIP_CHECK(hipMalloc(&sync_dev[i], num_bytes));

    HIP_CHECK(hipHostMalloc(&grid_rank_host[i], num_bytes));
    HIP_CHECK(hipHostMalloc(&size_host[i], num_bytes));
    HIP_CHECK(hipHostMalloc(&thd_rank_host[i], num_bytes));
    HIP_CHECK(hipHostMalloc(&is_valid_host[i], num_bytes));

    if (i == 0) {
      HIP_CHECK(
          hipHostMalloc(&sync_result, sizeof(int) * (num_devices + 1), hipHostMallocCoherent));
    }
  }

  // Launch Kernel
  int NumKernelArgs = 6;
  if (specific_api_test) {
    NumKernelArgs = 7;
  }
  hipLaunchParams* launchParamsList = new hipLaunchParams[num_devices];
  std::vector<void*> args(MaxGPUs * NumKernelArgs);
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));

    args[i * NumKernelArgs] = &grid_rank_dev[i];
    args[i * NumKernelArgs + 1] = &size_dev[i];
    args[i * NumKernelArgs + 2] = &thd_rank_dev[i];
    args[i * NumKernelArgs + 3] = &is_valid_dev[i];
    args[i * NumKernelArgs + 4] = &sync_dev[i];
    args[i * NumKernelArgs + 5] = &sync_result;
    if (specific_api_test) {
      args[i * NumKernelArgs + 6] = &num_grids_dev[i];
    }

    launchParamsList[i].func = reinterpret_cast<void*>(kernel_func);
    launchParamsList[i].gridDim = 2;
    launchParamsList[i].blockDim = block_size;
    launchParamsList[i].sharedMem = 0;
    launchParamsList[i].stream = stream[i];
    launchParamsList[i].args = &args[i * NumKernelArgs];
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList, num_devices, 0));

  // Copy result from device to host
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    if (specific_api_test) {
      HIP_CHECK(hipMemcpy(num_grids_host[i], num_grids_dev[i], num_bytes, hipMemcpyDeviceToHost));
    }
    HIP_CHECK(hipMemcpy(grid_rank_host[i], grid_rank_dev[i], num_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(size_host[i], size_dev[i], num_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(thd_rank_host[i], thd_rank_dev[i], num_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(is_valid_host[i], is_valid_dev[i], num_bytes, hipMemcpyDeviceToHost));
  }

  // Validate results
  int grids_seen[MaxGPUs];
  for (int i = 0; i < num_devices; ++i) {
    for (int j = 0; j < 2 * block_size; ++j) {
      if (specific_api_test) {
        ASSERT_EQUAL(num_grids_host[i][j], num_devices);
      }
      ASSERT_GE(grid_rank_host[i][j], 0);
      ASSERT_LE(grid_rank_host[i][j], num_devices - 1);
      ASSERT_EQUAL(grid_rank_host[i][j], grid_rank_host[i][0]);
      ASSERT_EQUAL(size_host[i][j], num_devices * 2 * block_size);
      int gridRank = grid_rank_host[i][j];
      ASSERT_EQUAL(thd_rank_host[i][j], (gridRank * 2 * block_size) + j);
      ASSERT_EQUAL(is_valid_host[i][j], 1);
    }
    ASSERT_EQUAL(sync_result[i + 1], 2 * block_size);

    // Validate uniqueness property of grid rank
    grids_seen[i] = grid_rank_host[i][0];
    for (int k = 0; k < i; ++k) {
      INFO("Grid rank in multi-gpu setup should be unique");
      REQUIRE(grids_seen[k] != grids_seen[i]);
    }
  }
  ASSERT_EQUAL(sync_result[0], num_devices * 2 * block_size);

  // Free host and device memory
  delete[] launchParamsList;
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));

    if (specific_api_test) {
      HIP_CHECK(hipFree(num_grids_dev[i]));
      HIP_CHECK(hipHostFree(num_grids_host[i]));
    }

    HIP_CHECK(hipFree(grid_rank_dev[i]));
    HIP_CHECK(hipFree(size_dev[i]));
    HIP_CHECK(hipFree(thd_rank_dev[i]));
    HIP_CHECK(hipFree(is_valid_dev[i]));
    HIP_CHECK(hipFree(sync_dev[i]));

    if (i == 0) {
      HIP_CHECK(hipHostFree(sync_result));
    }
    HIP_CHECK(hipHostFree(grid_rank_host[i]));
    HIP_CHECK(hipHostFree(size_host[i]));
    HIP_CHECK(hipHostFree(thd_rank_host[i]));
    HIP_CHECK(hipHostFree(is_valid_host[i]));
    HIP_CHECK(hipStreamDestroy(stream[i]));
  }
}

TEST_CASE("Unit_hipCGMultiGridGroupType_Basic") {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  num_devices = min(num_devices, MaxGPUs);

  // Set `max_threads_per_blk` by taking minimum among all available devices
  int max_threads_per_blk = INT_MAX;
  hipDeviceProp_t device_properties;
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipGetDeviceProperties(&device_properties, i));
    if (!device_properties.cooperativeMultiDeviceLaunch) {
      HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
      return;
    }
    max_threads_per_blk = min(max_threads_per_blk, device_properties.maxThreadsPerBlock);
  }

  void* kernel_func;
  bool specific_api_test = false;

  SECTION("Default multi grid group API test") {
    kernel_func = reinterpret_cast<void*>(kernel_cg_multi_grid_group_type);
    specific_api_test = true;
  }

  SECTION("Base type multi grid group API test") {
    kernel_func = reinterpret_cast<void*>(kernel_cg_multi_grid_group_type_via_base_type);
  }

  SECTION("Public API multi grid group test") {
    kernel_func = reinterpret_cast<void*>(kernel_cg_multi_grid_group_type_via_public_api);
  }

  // Test for blockSizes in powers of 2
  for (int block_size = 2; block_size <= max_threads_per_blk; block_size = block_size * 2) {
    test_cg_multi_grid_group_type(kernel_func, num_devices, block_size, specific_api_test);
  }

  // Test for random blockSizes, but the sequence is the same every execution
  srand(0);
  for (int i = 0; i < 10; i++) {
    // Test fails for 0 thread per block
    test_cg_multi_grid_group_type(kernel_func, num_devices, max(2, rand() % max_threads_per_blk),
                                  specific_api_test);
  }
}

TEST_CASE("Unit_hipCGMultiGridGroupType_Barrier") {
  int num_devices = 0;
  uint32_t loops = GENERATE(1, 2, 3, 4);
  uint32_t warps = GENERATE(4, 8, 16, 32);
  uint32_t block_size = 1;

  HIP_CHECK(hipGetDeviceCount(&num_devices));
  if (num_devices < 2) {
    HipTest::HIP_SKIP_TEST("Device number is < 2");
    return;
  }

  std::vector<hipDeviceProp_t> device_properties(num_devices);
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipGetDeviceProperties(&device_properties[i], i));
    if (!device_properties[i].cooperativeMultiDeviceLaunch) {
      HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
      return;
    }
  }

  // Test whether the requested size will fit on the GPU
  std::vector<int> warp_sizes(num_devices);
  std::vector<int> num_sms(num_devices);
  int warp_size = INT_MAX;
  int num_sm = INT_MAX;
  for (int i = 0; i < num_devices; i++) {
    warp_sizes[i] = device_properties[i].warpSize;
    if (warp_sizes[i] < warp_size) {
      warp_size = warp_sizes[i];
    }
    num_sms[i] = device_properties[i].multiProcessorCount;
    if (num_sms[i] < num_sm) {
      num_sm = num_sms[i];
    }
  }

  int num_threads_in_block = block_size * warp_size;

  // Calculate the device occupancy to know how many blocks can be run.
  std::vector<int> max_blocks_per_sm_arr(num_devices);
  int max_blocks_per_sm = INT_MAX;
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm_arr[i], test_kernel,
                                                           num_threads_in_block, 0));
    if (max_blocks_per_sm_arr[i] < max_blocks_per_sm) {
      max_blocks_per_sm = max_blocks_per_sm_arr[i];
    }
  }

  int requested_blocks = warps / block_size;

  // Each block will output a single value per loop.
  uint32_t total_buffer_len = requested_blocks * loops;

  // Alocate the buffer that will hold the kernel's output, and which will
  // also be used to globally synchronize during GWS initialization
  std::vector<unsigned int*> host_buffer(num_devices);
  std::vector<unsigned int*> kernel_buffer(num_devices);
  std::vector<unsigned int*> kernel_atomic(num_devices);
  std::vector<hipStream_t> streams(num_devices);
  std::vector<unsigned int*> per_loop_atomic(num_devices);
  // Allocate and initialize grid-specific counters and values using hipHostMalloc
  unsigned int *grid_counters, *recorded_values, *grid_values;
  HIP_CHECK(hipHostMalloc(&grid_counters, sizeof(unsigned int) * num_devices));
  HIP_CHECK(hipHostMalloc(&recorded_values, sizeof(unsigned int) * num_devices * loops));
  HIP_CHECK(hipHostMalloc(&grid_values, sizeof(unsigned int) * num_devices));
  HIP_CHECK(hipMemset(grid_counters, 0, num_devices * sizeof(unsigned int)));
  HIP_CHECK(hipMemset(recorded_values, 0, num_devices * loops * sizeof(unsigned int)));
  HIP_CHECK(hipMemset(grid_values, 0, num_devices * sizeof(unsigned int)));

  for (int i = 0; i < num_devices; i++) {
    host_buffer[i] =
        reinterpret_cast<unsigned int*>(calloc(total_buffer_len, sizeof(unsigned int)));
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipMalloc(&kernel_buffer[i], sizeof(unsigned int) * total_buffer_len));
    HIP_CHECK(hipMemcpy(kernel_buffer[i], host_buffer[i], sizeof(unsigned int) * total_buffer_len,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&kernel_atomic[i], sizeof(unsigned int)));
    HIP_CHECK(hipMemset(kernel_atomic[i], 0, sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&per_loop_atomic[i], loops * sizeof(unsigned int)));
    HIP_CHECK(hipMemset(per_loop_atomic[i], 0, loops * sizeof(unsigned int)));
    HIP_CHECK(hipStreamCreate(&streams[i]));
  }

  // Launch the kernels
  INFO("Launching a cooperative kernel with " << warps << " warps in " << requested_blocks
                                              << " thread blocks");

  std::vector<std::vector<void*>> dev_params(num_devices, std::vector<void*>(7, nullptr));
  std::vector<hipLaunchParams> md_params(num_devices);
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    dev_params[i][0] = reinterpret_cast<void*>(&kernel_atomic[i]);
    dev_params[i][1] = reinterpret_cast<void*>(&kernel_buffer[i]);
    dev_params[i][2] = reinterpret_cast<void*>(&loops);
    dev_params[i][3] = reinterpret_cast<void*>(&per_loop_atomic[i]);
    dev_params[i][4] = reinterpret_cast<void*>(&grid_counters);
    dev_params[i][5] = reinterpret_cast<void*>(&recorded_values);
    dev_params[i][6] = reinterpret_cast<void*>(&grid_values);
    md_params[i].func = reinterpret_cast<void*>(test_kernel);
    md_params[i].gridDim = requested_blocks;
    md_params[i].blockDim = num_threads_in_block;
    md_params[i].sharedMem = 0;
    md_params[i].stream = streams[i];
    md_params[i].args = dev_params[i].data();
  }

  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(md_params.data(), num_devices, 0));
  HIP_CHECK(hipDeviceSynchronize());

  // Read back the buffer to host
  for (int dev = 0; dev < num_devices; dev++) {
    HIP_CHECK(hipMemcpy(host_buffer[dev], kernel_buffer[dev],
                        sizeof(unsigned int) * total_buffer_len, hipMemcpyDeviceToHost));
  }

  for (unsigned int dev = 0; dev < num_devices; dev++) {
    verify_barrier_buffer(loops, requested_blocks, host_buffer[dev], num_devices);
  }

  // Verify the recorded values
  verify_recorded_values(recorded_values, loops, num_devices);

  HIP_CHECK(hipHostFree(grid_counters));
  HIP_CHECK(hipHostFree(recorded_values));
  HIP_CHECK(hipHostFree(grid_values));
  for (int k = 0; k < num_devices; ++k) {
    HIP_CHECK(hipFree(kernel_buffer[k]));
    HIP_CHECK(hipFree(kernel_atomic[k]));
    HIP_CHECK(hipFree(per_loop_atomic[k]));
    HIP_CHECK(hipStreamDestroy(streams[k]));
    free(host_buffer[k]);
  }
}
