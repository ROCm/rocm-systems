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
#include "cg_common_kernels.hh"

#include <cpu_grid.h>
#include <resource_guards.hh>
#include <utils.hh>

/**
 * @addtogroup multi_grid_group multi_grid_group
 * @{
 * @ingroup DeviceLanguageTest
 * Contains unit tests for all multi_grid_group APIs
 */

namespace cg = cooperative_groups;

template <typename BaseType = cg::multi_grid_group>
static __global__ void multi_grid_group_size_getter(unsigned int* sizes) {
  const BaseType group = cg::this_multi_grid();
  sizes[thread_rank_in_grid()] = group.size();
}

template <typename BaseType = cg::multi_grid_group>
static __global__ void multi_grid_group_thread_rank_getter(unsigned int* thread_ranks) {
  const BaseType group = cg::this_multi_grid();
  thread_ranks[thread_rank_in_grid()] = group.thread_rank();
}

template <typename BaseType = cg::multi_grid_group>
static __global__ void multi_grid_group_is_valid_getter(unsigned int* is_valid_flags) {
  const BaseType group = cg::this_multi_grid();
  is_valid_flags[thread_rank_in_grid()] = static_cast<unsigned int>(group.is_valid());
}

static __global__ void multi_grid_group_num_grids_getter(unsigned int* num_grids) {
  num_grids[thread_rank_in_grid()] = cg::this_multi_grid().num_grids();
}

static __global__ void multi_grid_group_grid_rank_getter(unsigned int* grid_ranks) {
  grid_ranks[thread_rank_in_grid()] = cg::this_multi_grid().grid_rank();
}

static __global__ void multi_grid_group_non_member_size_getter(unsigned int* sizes) {
  sizes[thread_rank_in_grid()] = cg::group_size(cg::this_multi_grid());
}

static __global__ void multi_grid_group_non_member_thread_rank_getter(unsigned int* thread_ranks) {
  thread_ranks[thread_rank_in_grid()] = cg::thread_rank(cg::this_multi_grid());
}

static __global__ void sync_kernel(unsigned int* atomic_val, unsigned int* global_array,
                                   unsigned int* array, uint32_t loops) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  cooperative_groups::multi_grid_group mgrid = cooperative_groups::this_multi_grid();
  unsigned rank = grid.thread_rank();
  unsigned global_rank = mgrid.thread_rank();

  int offset = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  for (int i = 0; i < loops; i++) {
    // Make the last thread run way behind everyone else.
    // If the sync below fails, then the other threads may hit the
    // atomicInc instruction many times before the last thread ever gets to it.
    // If the sync works, then it will likely contain "total number of blocks"*i
    if (rank == (grid.size() - 1)) {
      busy_wait(100000);
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 &&
        threadIdx.z == blockDim.z - 1) {
      array[offset] = atomicInc(atomic_val, UINT_MAX);
    }
    grid.sync();

    // Make the last thread in the entire multi-grid run way behind
    // everyone else.
    if (global_rank == (mgrid.size() - 1)) {
      busy_wait(100000);
    }
    // During even iterations, add into your own array entry
    // During odd iterations, add into next array entry
    unsigned grid_rank = mgrid.grid_rank();
    unsigned inter_gpu_offset = (grid_rank + 1) % mgrid.num_grids();
    if (rank == (grid.size() - 1)) {
      if (i % 2 == 0) {
        global_array[grid_rank] += 2;
      } else {
        global_array[inter_gpu_offset] *= 2;
      }
    }
    mgrid.sync();
    offset += gridDim.x * gridDim.y * gridDim.z;
  }
}

static void get_multi_grid_dims(dim3& grid_dim, dim3& block_dim, unsigned int device,
                                unsigned int test_case) {
  hipDeviceProp_t props;
  HIP_CHECK(hipSetDevice(device))
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  int sm = props.multiProcessorCount;
  auto warp_size = getWarpSize();
  std::vector<dim3> block_dim_values = {dim3(1, 1, 1),
                                        dim3(props.maxThreadsDim[0], 1, 1),
                                        dim3(1, props.maxThreadsDim[1], 1),
                                        dim3(1, 1, props.maxThreadsDim[2]),
                                        dim3(16, 8, 8),
                                        dim3(32, 32, 1),
                                        dim3(64, 8, 2),
                                        dim3(16, 16, 3),
                                        dim3(warp_size - 1, 3, 3),
                                        dim3(warp_size + 1, 3, 3)};
  std::vector<dim3> grid_dim_values = {dim3(1, 1, 1),
                                       dim3(static_cast<int>(0.5 * sm), 1, 3),
                                       dim3(4, static_cast<int>(0.5 * sm), 1),
                                       dim3(1, 1, static_cast<int>(0.5 * sm)),
                                       dim3(sm, 2, 1),
                                       dim3(2, sm, 1),
                                       dim3(1, sm, 2),
                                       dim3(3, 3, 3)};

  grid_dim = grid_dim_values[test_case % grid_dim_values.size()];
  block_dim = block_dim_values[test_case % block_dim_values.size()];
}

/**
 * Test Description
 * ------------------------
 *  - Launches kernels that write the return values of size, thread_rank, grid_rank, num_grids and
 * is_valid member functions to an output array that is validated on the host side. The kernels are
 * run sequentially, reusing the output array, to avoid running out of device memory for large
 * kernel launches.
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/multi_grid_group.c
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Devices support cooperative multi device launch
 */
TEST_CASE("Unit_Multi_Grid_Group_Getters_Positive_Basic") {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  num_devices = min(num_devices, kMaxGPUs);

  std::vector<hipDeviceProp_t> device_properties(num_devices);
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipGetDeviceProperties(&device_properties[i], i));
    if (!device_properties[i].cooperativeMultiDeviceLaunch) {
      HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
      return;
    }
  }
  const auto test_case = GENERATE(range(0, 20));
  std::vector<dim3> grid_dims(num_devices);
  std::vector<dim3> block_dims(num_devices);
  for (int i = 0; i < num_devices; i++) {
    get_multi_grid_dims(grid_dims[i], block_dims[i], i, test_case);
    if (!CheckDimensions(i, multi_grid_group_size_getter<cg::multi_grid_group>, grid_dims[i],
                         block_dims[i]))
      return;
    INFO("Grid dimensions dev " << i << " : x " << grid_dims[i].x << ", y " << grid_dims[i].y
                                << ", z " << grid_dims[i].z);
    INFO("Block dimensions dev " << i << " : x " << block_dims[i].x << ", y " << block_dims[i].y
                                 << ", z " << block_dims[i].z);
  }

  CPUMultiGrid multi_grid(num_devices, grid_dims.data(), block_dims.data());

  std::vector<StreamGuard> streams;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr_dev;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr;
  std::vector<unsigned int*> uint_arr_dev_ptr(num_devices);
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
    streams.emplace_back(Streams::created);

    uint_arr_dev.emplace_back(LinearAllocs::hipMalloc,
                              multi_grid.grids_[i].thread_count_ * sizeof(unsigned int));
    uint_arr_dev_ptr[i] = uint_arr_dev[i].ptr();
    uint_arr.emplace_back(LinearAllocs::hipHostMalloc,
                          multi_grid.grids_[i].thread_count_ * sizeof(unsigned int));
  }

  // Launch Kernel
  std::vector<hipLaunchParams> launchParamsList(num_devices);
  std::vector<void*> args(num_devices);
  for (int i = 0; i < num_devices; i++) {
    args[i] = &uint_arr_dev_ptr[i];

    launchParamsList[i].func =
        reinterpret_cast<void*>(multi_grid_group_size_getter<cg::multi_grid_group>);
    launchParamsList[i].gridDim = grid_dims[i];
    launchParamsList[i].blockDim = block_dims[i];
    launchParamsList[i].sharedMem = 0;
    launchParamsList[i].stream = streams[i].stream();
    launchParamsList[i].args = &args[i];
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList.data(), num_devices, 0));
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        multi_grid.grids_[i].thread_count_ * sizeof(*uint_arr[i].ptr()),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    launchParamsList[i].func =
        reinterpret_cast<void*>(multi_grid_group_thread_rank_getter<cg::multi_grid_group>);
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList.data(), num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    // Verify multi_grid_group.size() values
    ArrayAllOf(uint_arr[i].ptr(), multi_grid.grids_[i].thread_count_,
               [size = multi_grid.thread_count_](uint32_t) { return size; });
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        multi_grid.grids_[i].thread_count_ * sizeof(*uint_arr[i].ptr()),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    launchParamsList[i].func = reinterpret_cast<void*>(multi_grid_group_grid_rank_getter);
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList.data(), num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    // Verify multi_grid_group.thread_rank() values
    const auto multi_grid_thread0_rank = multi_grid.thread0_rank_in_multi_grid(i);
    ArrayInRange(uint_arr[i].ptr(), multi_grid.grids_[i].thread_count_, multi_grid_thread0_rank,
                 multi_grid_thread0_rank + multi_grid.grids_[i].thread_count_);
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        multi_grid.grids_[i].thread_count_ * sizeof(*uint_arr[i].ptr()),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    launchParamsList[i].func = reinterpret_cast<void*>(multi_grid_group_num_grids_getter);
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList.data(), num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    // Verify multi_grid_group.grid_rank() values
    ArrayFindIfNot(uint_arr[i].ptr(), static_cast<unsigned int>(i),
                   multi_grid.grids_[i].thread_count_);

    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        multi_grid.grids_[i].thread_count_ * sizeof(*uint_arr[i].ptr()),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    launchParamsList[i].func =
        reinterpret_cast<void*>(multi_grid_group_is_valid_getter<cg::multi_grid_group>);
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList.data(), num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    // Verify multi_grid_group.num_grids() values
    ArrayFindIfNot(uint_arr[i].ptr(), static_cast<unsigned int>(num_devices),
                   multi_grid.grids_[i].thread_count_);

    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        multi_grid.grids_[i].thread_count_ * sizeof(*uint_arr[i].ptr()),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Verify multi_grid_group.is_valid() values
    ArrayFindIfNot(uint_arr[i].ptr(), 1U, multi_grid.grids_[i].thread_count_);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Launches kernels that write the return values of size, thread_rank and is_valid member
 * functions to an output array that is validated on the host side, while treating the
 * multi_grid_group as a thread group. The kernels are run sequentially, reusing the output array,
 * to avoid running out of device memory for large kernel launches.
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/multi_grid_group.c
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Devices support cooperative multi device launch
 */
TEST_CASE("Unit_Multi_Grid_Group_Getters_Positive_Base_Type") {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  num_devices = min(num_devices, kMaxGPUs);
  std::vector<hipDeviceProp_t> device_properties(num_devices);
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipGetDeviceProperties(&device_properties[i], i));
    if (!device_properties[i].cooperativeMultiDeviceLaunch) {
      HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
      return;
    }
  }

  const auto test_case = GENERATE(range(0, 20));
  std::vector<dim3> grid_dims(num_devices);
  std::vector<dim3> block_dims(num_devices);
  for (int i = 0; i < num_devices; i++) {
    get_multi_grid_dims(grid_dims[i], block_dims[i], i, test_case);
    if (!CheckDimensions(i, multi_grid_group_size_getter<cg::multi_grid_group>, grid_dims[i],
                         block_dims[i]))
      return;
    INFO("Grid dimensions dev " << i << " : x " << grid_dims[i].x << ", y " << grid_dims[i].y
                                << ", z " << grid_dims[i].z);
    INFO("Block dimensions dev " << i << " : x " << block_dims[i].x << ", y " << block_dims[i].y
                                 << ", z " << block_dims[i].z);
  }

  CPUMultiGrid multi_grid(num_devices, grid_dims.data(), block_dims.data());

  std::vector<StreamGuard> streams;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr_dev;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr;
  std::vector<unsigned int*> uint_arr_dev_ptr(num_devices);
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
    streams.emplace_back(Streams::created);

    uint_arr_dev.emplace_back(LinearAllocs::hipMalloc,
                              multi_grid.grids_[i].thread_count_ * sizeof(unsigned int));
    uint_arr_dev_ptr[i] = uint_arr_dev[i].ptr();
    uint_arr.emplace_back(LinearAllocs::hipHostMalloc,
                          multi_grid.grids_[i].thread_count_ * sizeof(unsigned int));
  }

  // Launch Kernel
  std::vector<hipLaunchParams> launchParamsList(num_devices);
  std::vector<void*> args(num_devices);
  for (int i = 0; i < num_devices; i++) {
    args[i] = &uint_arr_dev_ptr[i];

    launchParamsList[i].func =
        reinterpret_cast<void*>(multi_grid_group_size_getter<cg::thread_group>);
    launchParamsList[i].gridDim = grid_dims[i];
    launchParamsList[i].blockDim = block_dims[i];
    launchParamsList[i].sharedMem = 0;
    launchParamsList[i].stream = streams[i].stream();
    launchParamsList[i].args = &args[i];
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList.data(), num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        multi_grid.grids_[i].thread_count_ * sizeof(*uint_arr[i].ptr()),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    launchParamsList[i].func =
        reinterpret_cast<void*>(multi_grid_group_thread_rank_getter<cg::thread_group>);
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList.data(), num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    // Verify multi_grid_group.size() values
    ArrayFindIfNot(uint_arr[i].ptr(), multi_grid.thread_count_, multi_grid.grids_[i].thread_count_);
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        multi_grid.grids_[i].thread_count_ * sizeof(*uint_arr[i].ptr()),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
#if HT_AMD
    launchParamsList[i].func =
        reinterpret_cast<void*>(multi_grid_group_is_valid_getter<cg::thread_group>);
#else
    launchParamsList[i].func =
        reinterpret_cast<void*>(multi_grid_group_is_valid_getter<cg::multi_grid_group>);
#endif
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList.data(), num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    // Verify multi_grid_group.thread_rank() values
    const auto multi_grid_thread0_rank = multi_grid.thread0_rank_in_multi_grid(i);
    ArrayInRange(uint_arr[i].ptr(), multi_grid.grids_[i].thread_count_, multi_grid_thread0_rank,
                 multi_grid_thread0_rank + multi_grid.grids_[i].thread_count_);
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        multi_grid.grids_[i].thread_count_ * sizeof(*uint_arr[i].ptr()),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Verify multi_grid_group.is_valid() values
    ArrayFindIfNot(uint_arr[i].ptr(), 1U, multi_grid.grids_[i].thread_count_);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Launches kernels that write the return values of size and thread_rank non-member functions
 * to an output array that is validated on the host side. The kernels are run sequentially, reusing
 * the output array, to avoid running out of device memory for large kernel launches.
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/multi_grid_group.c
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Devices support cooperative multi device launch
 */
TEST_CASE("Unit_Multi_Grid_Group_Getters_Positive_Non_Member_Functions") {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  num_devices = min(num_devices, kMaxGPUs);

  std::vector<hipDeviceProp_t> device_properties(num_devices);
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipGetDeviceProperties(&device_properties[i], i));
    if (!device_properties[i].cooperativeMultiDeviceLaunch) {
      HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
      return;
    }
  }
  const auto test_case = GENERATE(range(0, 20));
  std::vector<dim3> grid_dims(num_devices);
  std::vector<dim3> block_dims(num_devices);
  for (int i = 0; i < num_devices; i++) {
    get_multi_grid_dims(grid_dims[i], block_dims[i], i, test_case);
    if (!CheckDimensions(i, multi_grid_group_size_getter<cg::multi_grid_group>, grid_dims[i],
                         block_dims[i]))
      return;
    INFO("Grid dimensions dev " << i << " : x " << grid_dims[i].x << ", y " << grid_dims[i].y
                                << ", z " << grid_dims[i].z);
    INFO("Block dimensions dev " << i << " : x " << block_dims[i].x << ", y " << block_dims[i].y
                                 << ", z " << block_dims[i].z);
  }

  CPUMultiGrid multi_grid(num_devices, grid_dims.data(), block_dims.data());

  std::vector<StreamGuard> streams;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr_dev;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr;
  std::vector<unsigned int*> uint_arr_dev_ptr(num_devices);
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
    streams.emplace_back(Streams::created);

    uint_arr_dev.emplace_back(LinearAllocs::hipMalloc,
                              multi_grid.grids_[i].thread_count_ * sizeof(unsigned int));
    uint_arr_dev_ptr[i] = uint_arr_dev[i].ptr();
    uint_arr.emplace_back(LinearAllocs::hipHostMalloc,
                          multi_grid.grids_[i].thread_count_ * sizeof(unsigned int));
  }

  // Launch Kernel
  std::vector<hipLaunchParams> launchParamsList(num_devices);
  std::vector<void*> args(num_devices);
  for (int i = 0; i < num_devices; i++) {
    args[i] = &uint_arr_dev_ptr[i];

    launchParamsList[i].func = reinterpret_cast<void*>(multi_grid_group_non_member_size_getter);
    launchParamsList[i].gridDim = grid_dims[i];
    launchParamsList[i].blockDim = block_dims[i];
    launchParamsList[i].sharedMem = 0;
    launchParamsList[i].stream = streams[i].stream();
    launchParamsList[i].args = &args[i];
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList.data(), num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        multi_grid.grids_[i].thread_count_ * sizeof(*uint_arr[i].ptr()),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    launchParamsList[i].func =
        reinterpret_cast<void*>(multi_grid_group_non_member_thread_rank_getter);
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList.data(), num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    // Verify multi_grid_group.size() values
    ArrayFindIfNot(uint_arr[i].ptr(), multi_grid.thread_count_, multi_grid.grids_[i].thread_count_);
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        multi_grid.grids_[i].thread_count_ * sizeof(*uint_arr[i].ptr()),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    // Verify multi_grid_group.thread_rank() values
    const auto multi_grid_thread0_rank = multi_grid.thread0_rank_in_multi_grid(i);
    ArrayInRange(uint_arr[i].ptr(), multi_grid.grids_[i].thread_count_, multi_grid_thread0_rank,
                 multi_grid_thread0_rank + multi_grid.grids_[i].thread_count_);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Launches a kernel to multiple gpus which tests sync of separate grids and sync of the entire
 * multi grid. The last thread in a block in a grid atomically increments a global variable within a
 * work loop. The value returned from this atomic increment entirely depends on the order the
 * threads arrive at the atomic instruction. Each thread then stores the result in the global array
 * based on its block id. A wait loop is inserted into the last thread so that it runs behind all
 * other threads. If the grid sync doesn't work, the other threads will increment the atomic
 * variable many times before the last thread gets to it and it will read a very large value. If the
 * grid sync works, each thread will increment the variable once per loop iteration and the last
 * thread will contain total number of blocks * loop iteration. In the end of the work loop, a value
 * is added into grid's own global array entry during even iterations and during odd iterations, a
 * value of the next grid is multiplied. A wait loop is inserted into the last thread in the entire
 * multi-grid so that it runs behind all the other threads. If the multi grid sync doesn't work the
 * two global array entries will end up being out of sync, because the intermingling of adds and
 * multiplies will not be aligned between the devices.
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/multi_grid_group.c
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Devices support cooperative multi device launch
 */
TEST_CASE("Unit_Multi_Grid_Group_Positive_Sync") {
  CHECK_IMAGE_SUPPORT
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  num_devices = min(num_devices, kMaxGPUs);

  std::vector<hipDeviceProp_t> device_properties(num_devices);
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipGetDeviceProperties(&device_properties[i], i));
    if (!device_properties[i].cooperativeMultiDeviceLaunch) {
      HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
      return;
    }
  }
  auto loops = GENERATE(2, 4, 8, 16);
  const auto test_case = GENERATE(range(0, 20));
  std::vector<dim3> grid_dims(num_devices);
  std::vector<dim3> block_dims(num_devices);
  for (int i = 0; i < num_devices; i++) {
    get_multi_grid_dims(grid_dims[i], block_dims[i], i, test_case);
    if (!CheckDimensions(i, sync_kernel, grid_dims[i], block_dims[i])) return;
    INFO("Grid dimensions dev " << i << " : x " << grid_dims[i].x << ", y " << grid_dims[i].y
                                << ", z " << grid_dims[i].z);
    INFO("Block dimensions dev " << i << " : x " << block_dims[i].x << ", y " << block_dims[i].y
                                 << ", z " << block_dims[i].z);
  }

  CPUMultiGrid multi_grid(num_devices, grid_dims.data(), block_dims.data());

  std::vector<StreamGuard> streams;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr_dev;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr;
  std::vector<LinearAllocGuard<unsigned int>> atomic_val;
  std::vector<unsigned int*> uint_arr_dev_ptr(num_devices);
  std::vector<unsigned int*> atomic_val_ptr(num_devices);
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
    streams.emplace_back(Streams::created);

    // Allocate grid sync arrays
    unsigned int array_len = multi_grid.grids_[i].block_count_ * loops;
    uint_arr_dev.emplace_back(LinearAllocs::hipMalloc, array_len * sizeof(unsigned int));
    uint_arr_dev_ptr[i] = uint_arr_dev[i].ptr();
    uint_arr.emplace_back(LinearAllocs::hipHostMalloc, array_len * sizeof(unsigned int));

    atomic_val.emplace_back(LinearAllocs::hipMalloc, sizeof(unsigned int));
    HIP_CHECK(hipMemset(atomic_val[i].ptr(), 0, sizeof(unsigned int)));
    atomic_val_ptr[i] = atomic_val[i].ptr();
  }
  // Allocate multi_grid sync array
  LinearAllocGuard<unsigned int> global_arr(LinearAllocs::hipHostMalloc,
                                            num_devices * sizeof(unsigned int));
  HIP_CHECK(hipMemset(global_arr.ptr(), 0, num_devices * sizeof(unsigned int)));
  unsigned int* global_arr_ptr = global_arr.ptr();

  std::vector<std::vector<void*>> dev_params(num_devices, std::vector<void*>(4, nullptr));
  std::vector<hipLaunchParams> md_params(num_devices);
  for (int i = 0; i < num_devices; i++) {
    dev_params[i][0] = reinterpret_cast<void*>(&atomic_val_ptr[i]);
    dev_params[i][1] = reinterpret_cast<void*>(&global_arr_ptr);
    dev_params[i][2] = reinterpret_cast<void*>(&uint_arr_dev_ptr[i]);
    dev_params[i][3] = reinterpret_cast<void*>(&loops);

    md_params[i].func = reinterpret_cast<void*>(sync_kernel);
    md_params[i].gridDim = grid_dims[i];
    md_params[i].blockDim = block_dims[i];
    md_params[i].sharedMem = 0;
    md_params[i].stream = streams[i].stream();
    md_params[i].args = dev_params[i].data();
  }

  // Launch Kernel
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(md_params.data(), num_devices, 0));
  HIP_CHECK(hipDeviceSynchronize());

  // Read back the grid sync buffer to host
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    unsigned int array_len = multi_grid.grids_[i].block_count_ * loops;
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(), array_len * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
  }

  HIP_CHECK(hipDeviceSynchronize());

  // Verify grid sync host array values
  for (int i = 0; i < num_devices; i++) {
    unsigned int max_in_this_loop = 0;
    for (unsigned int j = 0; j < loops; j++) {
      max_in_this_loop += multi_grid.grids_[i].block_count_;
      unsigned int k = 0;
      for (k = 0; k < multi_grid.grids_[i].block_count_ - 1; k++) {
        REQUIRE(uint_arr[i].ptr()[j * multi_grid.grids_[i].block_count_ + k] < max_in_this_loop);
      }
      REQUIRE(uint_arr[i].ptr()[j * multi_grid.grids_[i].block_count_ + k] == max_in_this_loop - 1);
    }
  }

  // Verify multi_grid sync array values
  const auto f = [loops](unsigned int) -> unsigned int {
    unsigned int desired_val = 0;
    for (int j = 0; j < loops; j++) {
      if (j % 2 == 0) {
        desired_val += 2;
      } else {
        desired_val *= 2;
      }
    }
    return desired_val;
  };
  ArrayAllOf(global_arr.ptr(), num_devices, f);
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
