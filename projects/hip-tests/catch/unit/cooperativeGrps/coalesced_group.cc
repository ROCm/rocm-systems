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

#include <cmd_options.hh>
#include <cpu_grid.h>
#include <resource_guards.hh>

/**
 * @addtogroup coalesced_group coalesced_group
 * @{
 * @ingroup DeviceLanguageTest
 * Contains unit tests for all coalesced_group basic APIs
 */

namespace cg = cooperative_groups;

template <typename BaseType = cg::coalesced_group>
static __global__ void coalesced_group_size_getter(unsigned int* sizes, uint64_t active_mask) {
#if (__GFX8__ || __GFX9__)
  constexpr unsigned int ksize = 64;
#else
  constexpr unsigned int ksize = 32;
#endif
  const cg::thread_block_tile<ksize> tile = cg::tiled_partition<ksize>(cg::this_thread_block());
  if (active_mask & (static_cast<uint64_t>(1) << tile.thread_rank())) {
    BaseType active = cg::coalesced_threads();
    sizes[thread_rank_in_grid()] = active.size();
  }
}

template <typename BaseType = cg::coalesced_group>
static __global__ void coalesced_group_thread_rank_getter(unsigned int* thread_ranks,
                                                          uint64_t active_mask) {
#if (__GFX8__ || __GFX9__)
  constexpr unsigned int ksize = 64;
#else
  constexpr unsigned int ksize = 32;
#endif
  const cg::thread_block_tile<ksize> tile = cg::tiled_partition<ksize>(cg::this_thread_block());
  if (active_mask & (static_cast<uint64_t>(1) << tile.thread_rank())) {
    BaseType active = cg::coalesced_threads();
    thread_ranks[thread_rank_in_grid()] = active.thread_rank();
  }
}

static __global__ void coalesced_group_non_member_size_getter(unsigned int* sizes,
                                                              uint64_t active_mask) {
#if (__GFX8__ || __GFX9__)
  constexpr unsigned int ksize = 64;
#else
  constexpr unsigned int ksize = 32;
#endif
  const cg::thread_block_tile<ksize> tile = cg::tiled_partition<ksize>(cg::this_thread_block());
  if (active_mask & (static_cast<uint64_t>(1) << tile.thread_rank())) {
    cg::coalesced_group active = cg::coalesced_threads();
    sizes[thread_rank_in_grid()] = cg::group_size(active);
  }
}

static __global__ void coalesced_group_non_member_thread_rank_getter(unsigned int* thread_ranks,
                                                                     uint64_t active_mask) {
#if (__GFX8__ || _GFX9__)
  constexpr unsigned int ksize = 64;
#else
  constexpr unsigned int ksize = 32;
#endif
  const cg::thread_block_tile<ksize> tile = cg::tiled_partition<ksize>(cg::this_thread_block());
  if (active_mask & (static_cast<uint64_t>(1) << tile.thread_rank())) {
    cg::coalesced_group active = cg::coalesced_threads();
    thread_ranks[thread_rank_in_grid()] = cg::thread_rank(active);
  }
}

static unsigned int get_active_thread_count(uint64_t active_mask, unsigned int partition_size) {
  unsigned int active_thread_count = 0;
  for (int i = 0; i < partition_size; i++) {
    if (active_mask & (static_cast<uint64_t>(1) << i)) active_thread_count++;
  }
  return active_thread_count;
}

static uint64_t get_active_mask(unsigned int test_case, size_t warp_size) {
  uint64_t active_mask = 0;
  switch (test_case) {
    case 0:  // 1st thread
      active_mask = 1;
      break;
    case 1:  // last thread
      active_mask = static_cast<uint64_t>(1) << (warp_size - 1);
      break;
    case 2:  // all threads
      active_mask = 0xFFFFFFFFFFFFFFFF;
      break;
    case 3:  // every second thread
      active_mask = 0xAAAAAAAAAAAAAAAA;
      break;
    default:  // random
      static std::mt19937_64 mt(test_case);
      std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
      active_mask = dist(mt);
  }
  return active_mask;
}

/**
 * Test Description
 * ------------------------
 *  - Launches kernels that write the return values of size and thread_rank member
 * functions of coalesced groups, created according to the generated mask, to an output array that
 * is validated on the host side. The kernels are run sequentially, reusing the output array, to
 * avoid running out of device memory for large kernel launches
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/coalesced_group.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Coalesced_Group_Getters_Positive_Basic") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  size_t warp_size = static_cast<size_t>(device_properties.warpSize);

  const auto blocks = GenerateBlockDimensionsForShuffle();
  const auto threads = GenerateThreadDimensionsForShuffle();
  auto test_case = GENERATE(range(0, 4));
  uint64_t active_mask = get_active_mask(test_case, warp_size);
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  INFO("Coalesced group mask: " << active_mask);
  const CPUGrid grid(blocks, threads);

  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          grid.thread_count_ * sizeof(unsigned int));
  HIP_CHECK(hipMemset(uint_arr_dev.ptr(), 0, grid.thread_count_ * sizeof(unsigned int)));

  // Launch Kernel
  coalesced_group_size_getter<<<blocks, threads>>>(uint_arr_dev.ptr(), active_mask);

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemset(uint_arr_dev.ptr(), 0, grid.thread_count_ * sizeof(unsigned int)));
  HIP_CHECK(hipDeviceSynchronize());
  coalesced_group_thread_rank_getter<<<blocks, threads>>>(uint_arr_dev.ptr(), active_mask);

  // Verify coalesced_group.size() values
  unsigned int coalesced_size = 0;
  const auto partitions_in_block = (grid.threads_in_block_count_ + warp_size - 1) / warp_size;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % warp_size;
    // If the number of threads in a block is not a multiple of warp size, the
    // last warp will have inactive threads and coalesced group size must be recalculated
    if (rank_in_block == (partitions_in_block - 1) * warp_size) {
      unsigned int partition_size =
          grid.threads_in_block_count_ - (partitions_in_block - 1) * warp_size;
      coalesced_size = get_active_thread_count(active_mask, partition_size);
    } else if (rank_in_block == 0) {
      coalesced_size = get_active_thread_count(active_mask, warp_size);
    }
    if (active_mask & (static_cast<uint64_t>(1) << rank_in_partition)) {
      if (uint_arr.ptr()[i] != coalesced_size) {
        REQUIRE(uint_arr.ptr()[i] == coalesced_size);
      }
    }
  }

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify coalesced_group.thread_rank() values
  unsigned int coalesced_rank = 0;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % warp_size;
    if (rank_in_partition == 0) coalesced_rank = 0;

    if (active_mask & (static_cast<uint64_t>(1) << rank_in_partition)) {
      if (uint_arr.ptr()[i] != coalesced_rank) {
        REQUIRE(uint_arr.ptr()[i] == coalesced_rank);
      }
      coalesced_rank++;
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Launches kernels that write the return values of size and thread_rank member functions to an
 * output array that is validated on the host side, while treating the coalesced group, created
 * according to the generated mask, as a thread group. The kernels are run sequentially, reusing the
 * output array, to avoid running out of device memory for large kernel launches
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/coalesced_group.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Coalesced_Group_Getters_Via_Base_Type_Positive_Basic") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));


  size_t warp_size = static_cast<size_t>(device_properties.warpSize);

  const auto blocks = GenerateBlockDimensionsForShuffle();
  const auto threads = GenerateThreadDimensionsForShuffle();
  auto test_case = GENERATE(range(0, 4));
  uint64_t active_mask = get_active_mask(test_case, warp_size);
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  INFO("Coalesced group mask: " << active_mask);

  const CPUGrid grid(blocks, threads);

  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          grid.thread_count_ * sizeof(unsigned int));
  HIP_CHECK(hipMemset(uint_arr_dev.ptr(), 0, grid.thread_count_ * sizeof(unsigned int)));

  // Launch Kernel
  coalesced_group_size_getter<<<blocks, threads>>>(uint_arr_dev.ptr(), active_mask);

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemset(uint_arr_dev.ptr(), 0, grid.thread_count_ * sizeof(unsigned int)));
  HIP_CHECK(hipDeviceSynchronize());
  coalesced_group_thread_rank_getter<<<blocks, threads>>>(uint_arr_dev.ptr(), active_mask);

  // Verify coalesced_group.size() values
  unsigned int coalesced_size = 0;
  const auto partitions_in_block = (grid.threads_in_block_count_ + warp_size - 1) / warp_size;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % warp_size;
    // If the number of threads in a block is not a multiple of warp size, the
    // last warp will have inactive threads and coalesced group size must be recalculated
    if (rank_in_block == (partitions_in_block - 1) * warp_size) {
      unsigned int partition_size =
          grid.threads_in_block_count_ - (partitions_in_block - 1) * warp_size;
      coalesced_size = get_active_thread_count(active_mask, partition_size);
    } else if (rank_in_block == 0) {
      coalesced_size = get_active_thread_count(active_mask, warp_size);
    }
    if (active_mask & (static_cast<uint64_t>(1) << rank_in_partition)) {
      if (uint_arr.ptr()[i] != coalesced_size) {
        REQUIRE(uint_arr.ptr()[i] == coalesced_size);
      }
    }
  }

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify coalesced_group.thread_rank() values
  unsigned int coalesced_rank = 0;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % warp_size;
    if (rank_in_partition == 0) coalesced_rank = 0;

    if (active_mask & (static_cast<uint64_t>(1) << rank_in_partition)) {
      if (uint_arr.ptr()[i] != coalesced_rank) {
        REQUIRE(uint_arr.ptr()[i] == coalesced_rank);
      }
      coalesced_rank++;
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Launches kernels that write the return values of size and thread_rank non-member functions
 * of coalesced groups, created according to the generated mask, to an output array that is
 * validated on the host side. The kernels are run sequentially, reusing the output array, to avoid
 * running out of device memory for large kernel launches.
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/coalesced_group.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Coalesced_Group_Getters_Via_Non_Member_Functions_Positive_Basic") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  size_t warp_size = static_cast<size_t>(device_properties.warpSize);

  const auto blocks = GenerateBlockDimensionsForShuffle();
  const auto threads = GenerateThreadDimensionsForShuffle();
  auto test_case = GENERATE(range(0, 4));
  uint64_t active_mask = get_active_mask(test_case, warp_size);
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  INFO("Coalesced group mask: " << active_mask);

  const CPUGrid grid(blocks, threads);

  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          grid.thread_count_ * sizeof(unsigned int));
  HIP_CHECK(hipMemset(uint_arr_dev.ptr(), 0, grid.thread_count_ * sizeof(unsigned int)));

  // Launch Kernel
  coalesced_group_non_member_size_getter<<<blocks, threads>>>(uint_arr_dev.ptr(), active_mask);

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemset(uint_arr_dev.ptr(), 0, grid.thread_count_ * sizeof(unsigned int)));
  HIP_CHECK(hipDeviceSynchronize());
  coalesced_group_non_member_thread_rank_getter<<<blocks, threads>>>(uint_arr_dev.ptr(),
                                                                     active_mask);

  // Verify coalesced_group.size() values
  unsigned int coalesced_size = 0;
  const auto partitions_in_block = (grid.threads_in_block_count_ + warp_size - 1) / warp_size;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % warp_size;
    // If the number of threads in a block is not a multiple of warp size, the
    // last warp will have inactive threads and coalesced group size must be recalculated
    if (rank_in_block == (partitions_in_block - 1) * warp_size) {
      unsigned int partition_size =
          grid.threads_in_block_count_ - (partitions_in_block - 1) * warp_size;
      coalesced_size = get_active_thread_count(active_mask, partition_size);
    } else if (rank_in_block == 0) {
      coalesced_size = get_active_thread_count(active_mask, warp_size);
    }
    if (active_mask & (static_cast<uint64_t>(1) << rank_in_partition)) {
      if (uint_arr.ptr()[i] != coalesced_size) {
        REQUIRE(uint_arr.ptr()[i] == coalesced_size);
      }
    }
  }

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify coalesced_group.thread_rank() values
  unsigned int coalesced_rank = 0;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % warp_size;
    if (rank_in_partition == 0) coalesced_rank = 0;

    if (active_mask & (static_cast<uint64_t>(1) << rank_in_partition)) {
      if (uint_arr.ptr()[i] != coalesced_rank) {
        REQUIRE(uint_arr.ptr()[i] == coalesced_rank);
      }
      coalesced_rank++;
    }
  }
}

template <typename T> __global__ void coalesced_group_shfl_up(T* const out,
                                                              const unsigned int delta,
                                                              const uint64_t active_mask) {
#if (__GFX8__ || __GFX9__)
  constexpr unsigned int ksize = 64;
#else
  constexpr unsigned int ksize = 32;
#endif
  const cg::thread_block_tile<ksize> tile = cg::tiled_partition<ksize>(cg::this_thread_block());
  if (active_mask & (static_cast<uint64_t>(1) << tile.thread_rank())) {
    cg::coalesced_group active = cg::coalesced_threads();
    T var = static_cast<T>(active.thread_rank());
    out[thread_rank_in_grid()] = active.shfl_up(var, delta);
  }
}

template <typename T> void CoalescedGroupShflUpTestImpl() {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  size_t warp_size = static_cast<size_t>(device_properties.warpSize);

  const auto blocks = GenerateBlockDimensionsForShuffle();
  const auto threads = GenerateThreadDimensionsForShuffle();
  auto test_case = GENERATE(range(0, 4));
  uint64_t active_mask = get_active_mask(test_case, warp_size);
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  INFO("Coalesced group mask: " << active_mask);
  unsigned int active_thread_count = get_active_thread_count(active_mask, warp_size);

  auto delta = GENERATE(range(static_cast<size_t>(0), static_cast<size_t>(getWarpSize())));
  delta = delta % active_thread_count;
  INFO("Delta: " << delta);
  CPUGrid grid(blocks, threads);

  const auto alloc_size = grid.thread_count_ * sizeof(T);
  LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

  coalesced_group_shfl_up<T><<<blocks, threads>>>(arr_dev.ptr(), delta, active_mask);
  HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  unsigned int coalesced_rank = 0;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % warp_size;
    if (rank_in_partition == 0) coalesced_rank = 0;
    if (active_mask & (static_cast<uint64_t>(1) << rank_in_partition)) {
      int target = coalesced_rank - delta;
      target = target < 0 ? coalesced_rank : target;
      if (arr.ptr()[i] != target) {
        REQUIRE(arr.ptr()[i] == target);
      }
      coalesced_rank++;
    }
  }
}


/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle up behavior of coalesced group, created according to the generated
 * mask, for various delta values
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/coalesced_group.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Coalesced_Group_Shfl_Up_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  CoalescedGroupShflUpTestImpl<TestType>();
}

template <typename T> __global__ void coalesced_group_shfl_down(T* const out,
                                                                const unsigned int delta,
                                                                const uint64_t active_mask) {
#if (__GFX8__ || __GFX9__)
  constexpr unsigned int ksize = 64;
#else
  constexpr unsigned int ksize = 32;
#endif
  const cg::thread_block_tile<ksize> tile = cg::tiled_partition<ksize>(cg::this_thread_block());
  if (active_mask & (static_cast<uint64_t>(1) << tile.thread_rank())) {
    cg::coalesced_group active = cg::coalesced_threads();
    T var = static_cast<T>(active.thread_rank());
    out[thread_rank_in_grid()] = active.shfl_down(var, delta);
  }
}

template <typename T> void CoalescedGroupShflDownTest() {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  size_t warp_size = static_cast<size_t>(device_properties.warpSize);

  const auto blocks = GenerateBlockDimensionsForShuffle();
  const auto threads = GenerateThreadDimensionsForShuffle();
  auto test_case = GENERATE(range(0, 4));
  uint64_t active_mask = get_active_mask(test_case, warp_size);
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  INFO("Coalesced group mask: " << active_mask);
  unsigned int active_thread_count = get_active_thread_count(active_mask, warp_size);

  auto delta = GENERATE(range(static_cast<size_t>(0), static_cast<size_t>(getWarpSize())));
  delta = delta % active_thread_count;
  INFO("Delta: " << delta);
  CPUGrid grid(blocks, threads);

  const auto alloc_size = grid.thread_count_ * sizeof(T);
  LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

  coalesced_group_shfl_down<T><<<blocks, threads>>>(arr_dev.ptr(), delta, active_mask);
  HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  unsigned int coalesced_rank = 0;
  unsigned int coalesced_size = 0;
  const auto partitions_in_block = (grid.threads_in_block_count_ + warp_size - 1) / warp_size;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % warp_size;
    if (rank_in_partition == 0) coalesced_rank = 0;
    // If the number of threads in a block is not a multiple of warp size, the
    // last warp will have inactive threads and coalesced group size must be recalculated
    if (rank_in_block == (partitions_in_block - 1) * warp_size) {
      unsigned int partition_size =
          grid.threads_in_block_count_ - (partitions_in_block - 1) * warp_size;
      coalesced_size = get_active_thread_count(active_mask, partition_size);
    } else if (rank_in_block == 0) {
      coalesced_size = get_active_thread_count(active_mask, warp_size);
    }
    if (active_mask & (static_cast<uint64_t>(1) << rank_in_partition)) {
      int target = coalesced_rank + delta;
      target = target < coalesced_size ? target : coalesced_rank;
      if (arr.ptr()[i] != target) {
        REQUIRE(arr.ptr()[i] == target);
      }
      coalesced_rank++;
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle down behavior of coalesced group, created according to the generated
 * mask, for various delta values
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/coalesced_group.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Coalesced_Group_Shfl_Down_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  CoalescedGroupShflDownTest<TestType>();
}

template <typename T> __global__ void coalesced_group_shfl(T* const out, uint8_t* target_lanes,
                                                           const uint64_t active_mask) {
#if (__GFX8__ || __GFX9__)
  constexpr unsigned int ksize = 64;
#else
  constexpr unsigned int ksize = 32;
#endif
  const cg::thread_block_tile<ksize> tile = cg::tiled_partition<ksize>(cg::this_thread_block());
  if (active_mask & (static_cast<uint64_t>(1) << tile.thread_rank())) {
    cg::coalesced_group active = cg::coalesced_threads();
    T var = static_cast<T>(active.thread_rank());
    out[thread_rank_in_grid()] = active.shfl(var, target_lanes[active.thread_rank()]);
    ;
  }
}

template <typename T> void CoalescedGroupShflTest() {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  size_t warp_size = static_cast<size_t>(device_properties.warpSize);

  const auto blocks = GenerateBlockDimensionsForShuffle();
  const auto threads = GenerateThreadDimensionsForShuffle();
  auto test_case = GENERATE(range(0, 4));
  uint64_t active_mask = get_active_mask(test_case, warp_size);
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  INFO("Coalesced group mask: " << active_mask);
  unsigned int active_thread_count = get_active_thread_count(active_mask, warp_size);
  CPUGrid grid(blocks, threads);

  const auto alloc_size = grid.thread_count_ * sizeof(T);
  LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

  LinearAllocGuard<uint8_t> target_lanes_dev(LinearAllocs::hipMalloc,
                                             active_thread_count * sizeof(uint8_t));
  LinearAllocGuard<uint8_t> target_lanes(LinearAllocs::hipHostMalloc,
                                         active_thread_count * sizeof(uint8_t));
  // Generate a couple different combinations for target lanes
  for (auto i = 0u; i < active_thread_count; ++i) {
    target_lanes.ptr()[i] = active_thread_count - 1 - i;
  }

  HIP_CHECK(hipMemcpy(target_lanes_dev.ptr(), target_lanes.ptr(),
                      active_thread_count * sizeof(uint8_t), hipMemcpyHostToDevice));
  coalesced_group_shfl<T><<<blocks, threads>>>(arr_dev.ptr(), target_lanes_dev.ptr(), active_mask);
  HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());


  unsigned int coalesced_rank = 0;
  unsigned int coalesced_size = 0;
  const auto partitions_in_block = (grid.threads_in_block_count_ + warp_size - 1) / warp_size;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % warp_size;
    if (rank_in_partition == 0) coalesced_rank = 0;
    // If the number of threads in a block is not a multiple of warp size, the
    // last warp will have inactive threads and coalesced group size must be recalculated
    if (rank_in_block == (partitions_in_block - 1) * warp_size) {
      unsigned int partition_size =
          grid.threads_in_block_count_ - (partitions_in_block - 1) * warp_size;
      coalesced_size = get_active_thread_count(active_mask, partition_size);
    } else if (rank_in_block == 0) {
      coalesced_size = get_active_thread_count(active_mask, warp_size);
    }
    if (active_mask & (static_cast<uint64_t>(1) << rank_in_partition)) {
      auto target = target_lanes.ptr()[coalesced_rank];
      if (target >= coalesced_size) {
#if HT_NVIDIA
        target = 0;
#else
        target %= coalesced_size;
#endif
      }
      if (arr.ptr()[i] != target) {
        REQUIRE(arr.ptr()[i] == target);
      }
      coalesced_rank++;
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle behavior of of coalesced group, created according to the generated
 * mask, for generated shuffle target lanes
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/coalesced_group.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Coalesced_Group_Shfl_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  CoalescedGroupShflTest<TestType>();
}

static inline std::mt19937& GetRandomGenerator() {
  static std::mt19937 mt(11);
  return mt;
}

template <typename T> static inline T GenerateRandomInteger(const T min, const T max) {
  std::uniform_int_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

template <bool use_global, typename T>
__global__ void coalesced_group_sync_check(T* global_data, unsigned int* wait_modifiers,
                                           const uint64_t active_mask) {
#if (__GFX8__ || __GFX9__)
  constexpr unsigned int ksize = 64;
#else
  constexpr unsigned int ksize = 32;
#endif

  extern __shared__ uint8_t shared_data[];
  T* const data = use_global ? global_data : reinterpret_cast<T*>(shared_data);
  const auto tid = cg::this_grid().thread_rank();
  const auto block = cg::this_thread_block();
  const cg::thread_block_tile<ksize> partition = cg::tiled_partition<ksize>(block);

  const auto data_idx = [&block](unsigned int i) { return use_global ? i : (i % block.size()); };

  const auto partition_rank = block.thread_rank() / partition.size();

  const auto block_base_idx = tid / block.size() * block.size();
  const auto tile_base_idx = block_base_idx + partition_rank * partition.size();
  const auto wait_modifier = wait_modifiers[tid];

  if (active_mask & (static_cast<uint64_t>(1) << partition.thread_rank())) {
    cg::coalesced_group active = cg::coalesced_threads();
    busy_wait(wait_modifier);
    data[data_idx(tid)] = active.thread_rank();
    active.sync();
    bool valid = true;
    for (auto i = 0; i < active.size(); ++i) {
      const auto expected = (active.thread_rank() + i) % active.size();
      unsigned int active_count = 0;
      int offset = -1;
      while (active_count <= expected) {
        offset++;
        if (active_mask & (static_cast<uint64_t>(1) << offset)) active_count++;
      }

      if (!(valid &= (data[data_idx(tile_base_idx + offset)] == expected))) {
        break;
      }
    }
    active.sync();
    data[data_idx(tid)] = valid;

    if constexpr (!use_global) {
      global_data[tid] = data[data_idx(tid)];
    }
  }
}

template <bool global_memory, typename T> void CoalescedGroupSyncTest() {
  int device;
  hipDeviceProp_t device_properties;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  size_t warp_size = static_cast<size_t>(device_properties.warpSize);

  const auto randomized_run_count = GENERATE(range(0, cmd_options.cg_iterations));
  const auto blocks = GenerateBlockDimensionsForShuffle();
  const auto threads = GenerateThreadDimensionsForShuffle();
  auto test_case = GENERATE(range(0, 4));
  uint64_t active_mask = get_active_mask(test_case, warp_size);
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  INFO("Coalesced group mask: " << active_mask);
  CPUGrid grid(blocks, threads);

  const auto alloc_size = grid.thread_count_ * sizeof(T);
  const auto alloc_size_per_block = alloc_size / grid.block_count_;

  int max_shared_mem_per_block = 0;
  HIP_CHECK(hipDeviceGetAttribute(&max_shared_mem_per_block,
                                  hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
  if (!global_memory && (max_shared_mem_per_block < alloc_size_per_block)) {
    return;
  }

  LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);
  LinearAllocGuard<unsigned int> wait_modifiers_dev(LinearAllocs::hipMalloc,
                                                    grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> wait_modifiers(LinearAllocs::hipHostMalloc,
                                                grid.thread_count_ * sizeof(unsigned int));
  if (randomized_run_count != 0) {
    std::generate(wait_modifiers.ptr(), wait_modifiers.ptr() + grid.thread_count_,
                  [] { return GenerateRandomInteger(0u, 1500u); });
  } else {
    std::fill_n(wait_modifiers.ptr(), grid.thread_count_, 0u);
  }

  const auto shared_memory_size = global_memory ? 0u : alloc_size_per_block;
  HIP_CHECK(hipMemcpy(wait_modifiers_dev.ptr(), wait_modifiers.ptr(),
                      grid.thread_count_ * sizeof(unsigned int), hipMemcpyHostToDevice));

  coalesced_group_sync_check<global_memory><<<blocks, threads, shared_memory_size>>>(
      arr_dev.ptr(), wait_modifiers_dev.ptr(), active_mask);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % warp_size;
    if (active_mask & (static_cast<uint64_t>(1) << rank_in_partition)) {
      if (arr.ptr()[i] != 1) {
        REQUIRE(arr.ptr()[i] == 1);
      }
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Launches a kernel where blocks are devided into coalesced groups and every thread writes its
 * intra-tile rank into an array slot determined by its grid-wide linear index. The array is either
 * in global or dynamic shared memory based on a compile time switch, and the test is run for arrays
 * of 1, 2, and 4 byte elements. Before the write each thread executes a busy wait loop for a random
 * amount of clock cycles, the amount being read from an input array. After the write a sync for
 * active threads is performed and each thread validates that it can read the expected values that
 * other active threads within the same coalesced group have written to their respective array
 * slots. Each thread begins the validation from a given offset from its own index. For the first
 * run of the test, all the offsets are zero, so memory reads should be coalesced as adjacent
 * threads read from adjacent memory locations. On subsequent runs the offsets are randomized for
 * each thread, leading to non-coalesced reads and cache thrashing.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/coalesced_group.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Coalesced_Group_Sync_Positive_Basic", "", uint8_t, uint16_t, uint32_t) {
  SECTION("Global memory") { CoalescedGroupSyncTest<true, TestType>(); }
  SECTION("Shared memory") { CoalescedGroupSyncTest<false, TestType>(); }
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
