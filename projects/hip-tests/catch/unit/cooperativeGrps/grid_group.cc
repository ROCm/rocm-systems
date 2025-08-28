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
#include "cooperative_groups_common.hh"

#include <cpu_grid.h>
#include <resource_guards.hh>
#include <utils.hh>

/**
 * @addtogroup grid_group grid_group
 * @{
 * @ingroup DeviceLanguageTest
 * Contains unit tests for all grid_group APIs
 */

namespace cg = cooperative_groups;

static __global__ void grid_group_size_getter(unsigned int* sizes) {
  sizes[thread_rank_in_grid()] = cg::this_grid().size();
}

static __global__ void grid_group_thread_rank_getter(unsigned int* thread_ranks) {
  thread_ranks[thread_rank_in_grid()] = cg::this_grid().thread_rank();
}

static __global__ void grid_group_is_valid_getter(unsigned int* is_valid_flags) {
  is_valid_flags[thread_rank_in_grid()] = cg::this_grid().is_valid();
}

static __global__ void grid_group_non_member_size_getter(unsigned int* sizes) {
  sizes[thread_rank_in_grid()] = cg::group_size(cg::this_grid());
}

static __global__ void grid_group_non_member_thread_rank_getter(unsigned int* thread_ranks) {
  thread_ranks[thread_rank_in_grid()] = cg::thread_rank(cg::this_grid());
}

static __global__ void sync_kernel(unsigned int* atomic_val, unsigned int* per_loop_atomic,
                                   unsigned int* array, unsigned int loops) {
  cg::grid_group grid = cg::this_grid();
  unsigned rank = grid.thread_rank();

  int blocks_seen = 0;
  int grid_blocks = gridDim.x * gridDim.y * gridDim.z;

  for (int iter = 0; iter < loops; iter++) {
    // Make the last thread run way behind everyone else.
    // If the sync below fails, then the other threads may hit the
    // atomicInc instruction many times before the last thread ever gets to it.
    // If the sync works, then it will likely contain "total number of blocks"*iter
    if (rank == (grid.size() - 1)) {
      // The last wavefront should spin on this loop's atomic value
      // until all of the other wavefronts have incremented the
      // per-loop atomic and hit the grid.sync()
#if HT_AMD
      while (__hip_atomic_load(&per_loop_atomic[iter], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) <
             (grid_blocks - 1)) {
        __builtin_amdgcn_s_sleep(127);
      }

      // Give the other waves time to maybe go around the loop again
      // if the barrier has failed
      __builtin_amdgcn_s_sleep(127);
#else  // CUDA does not seem to need an ordered atomic load
      while (per_loop_atomic[iter] < (grid_blocks - 1)) {
      }
#endif
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 &&
        threadIdx.z == blockDim.z - 1) {
      atomicInc(&per_loop_atomic[iter], UINT_MAX);
      array[((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) + blocks_seen] =
          atomicInc(&atomic_val[0], UINT_MAX);
    }
    grid.sync();
    blocks_seen += grid_blocks;
  }
}

/**
 * Test Description
 * ------------------------
 *  - Launches kernels that write the return values of size, thread_rank and is_valid member
 * functions to an output array that is validated on the host side. The kernels are run
 * sequentially, reusing the output array, to avoid running out of device memory for large kernel
 * launches.
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/grid_group.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Device supports cooperative launch
 */
TEST_CASE("Unit_Grid_Group_Getters_Positive_Basic") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  const auto blocks = GenerateBlockDimensions();
  const auto threads = GenerateThreadDimensions();
  if (!CheckDimensions(device, grid_group_size_getter, blocks, threads)) return;
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);

  const CPUGrid grid(blocks, threads);

  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          grid.thread_count_ * sizeof(unsigned int));

  // Launch Kernel
  unsigned int* uint_arr_dev_ptr = uint_arr_dev.ptr();
  void* params[1];
  params[0] = &uint_arr_dev_ptr;

  HIP_CHECK(hipLaunchCooperativeKernel(grid_group_size_getter, blocks, threads, params, 0, 0));

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(
      hipLaunchCooperativeKernel(grid_group_thread_rank_getter, blocks, threads, params, 0, 0));

  // Verify grid_group.size() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
             [size = grid.thread_count_](uint32_t) { return size; });

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipLaunchCooperativeKernel(grid_group_is_valid_getter, blocks, threads, params, 0, 0));

  // Verify grid_group.thread_rank() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_, [](uint32_t i) { return i; });

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify grid_group.is_valid() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_, [](uint32_t) { return 1; });
}

/**
 * Test Description
 * ------------------------
 *    - Launches kernels that write the return values of size and thread_rank non-member functions
 * to an output array that is validated on the host side. The kernels are run sequentially, reusing
 * the output array, to avoid running out of device memory for large kernel launches.
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/grid_group.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Device supports cooperative launch
 */
TEST_CASE("Unit_Grid_Group_Getters_Via_Non_Member_Functions_Positive_Basic") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  const auto blocks = GenerateBlockDimensions();
  const auto threads = GenerateThreadDimensions();
  if (!CheckDimensions(device, grid_group_non_member_size_getter, blocks, threads)) return;
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);

  const CPUGrid grid(blocks, threads);

  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          grid.thread_count_ * sizeof(unsigned int));

  // Launch Kernel
  unsigned int* uint_arr_dev_ptr = uint_arr_dev.ptr();
  void* params[1];
  params[0] = &uint_arr_dev_ptr;

  HIP_CHECK(
      hipLaunchCooperativeKernel(grid_group_non_member_size_getter, blocks, threads, params, 0, 0));

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipLaunchCooperativeKernel(grid_group_non_member_thread_rank_getter, blocks, threads,
                                       params, 0, 0));

  // Verify grid_group.size() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
             [size = grid.thread_count_](uint32_t) { return size; });

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify grid_group.thread_rank() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_, [](uint32_t i) { return i; });
}

/**
 * Test Description
 * ------------------------
 *    - Launches a kernel where the last thread in a block atomically increments a global variable
 * within a work loop. The value returned from this atomic increment entirely depends on the order
 * the threads arrive at the atomic instruction. Each thread then stores the result in the global
 * array based on its block id. A wait loop is inserted into the last thread so that it runs behind
 * all other threads. If the sync doesn't work, the other threads will increment the atomic variable
 * many times before the last thread gets to it and it will read a very large value. If the sync
 * works, each thread will increment the variable once per loop iteration and the last thread will
 * contain total number of blocks * loop iteration.
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/grid_group.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Device supports cooperative launch
 */
TEST_CASE("Unit_Grid_Group_Sync_Positive_Basic") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  auto loops = GENERATE(2, 4, 8, 16);
  // Launch params for this test are hardcoded as a workaround for an issue reported
  // SWDEV-429791. When fixed, please enable calls to GenerateBlock/ThreadDimensions()
  const auto blocks =
      GENERATE_COPY(dim3(5, 5, 5), dim3(330, 1, 1), dim3(1, 330, 1), dim3(1, 1, 330));
  const auto threads =
      GENERATE_COPY(dim3(16, 8, 8), dim3(32, 32, 1), dim3(64, 8, 2), dim3(16, 16, 3));
  if (!CheckDimensions(device, sync_kernel, blocks, threads)) return;
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);

  const CPUGrid grid(blocks, threads);
  unsigned int array_len = grid.block_count_ * loops;

  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              array_len * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          array_len * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> atomic_val(LinearAllocs::hipMalloc, sizeof(unsigned int));
  LinearAllocGuard<unsigned int> per_loop_atomic_val(LinearAllocs::hipMalloc,
                                                     loops * sizeof(unsigned int));
  HIP_CHECK(hipMemset(atomic_val.ptr(), 0, sizeof(unsigned int)));
  HIP_CHECK(hipMemset(per_loop_atomic_val.ptr(), 0, loops * sizeof(unsigned int)));

  // Launch Kernel
  unsigned int* uint_arr_dev_ptr = uint_arr_dev.ptr();
  unsigned int* atomic_val_ptr = atomic_val.ptr();
  unsigned int* per_loop_atomic_val_ptr = per_loop_atomic_val.ptr();
  void* params[4];
  params[0] = reinterpret_cast<void*>(&atomic_val_ptr);
  params[1] = reinterpret_cast<void*>(&per_loop_atomic_val_ptr);
  params[2] = reinterpret_cast<void*>(&uint_arr_dev_ptr);
  params[3] = reinterpret_cast<void*>(&loops);

  HIP_CHECK(hipLaunchCooperativeKernel(sync_kernel, blocks, threads, params, 0, 0));

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), array_len * sizeof(*uint_arr.ptr()),
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipDeviceSynchronize());

  // Verify host buffer values
  unsigned int max_in_this_loop = 0;
  for (unsigned int i = 0; i < loops; i++) {
    max_in_this_loop += grid.block_count_;
    unsigned int j = 0;
    for (j = 0; j < grid.block_count_ - 1; j++) {
      REQUIRE(uint_arr.ptr()[i * grid.block_count_ + j] < max_in_this_loop);
    }
    REQUIRE(uint_arr.ptr()[i * grid.block_count_ + j] == max_in_this_loop - 1);
  }
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
