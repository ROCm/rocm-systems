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
#include <optional>
#include <resource_guards.hh>
#include <utils.hh>

#include <cmd_options.hh>

/**
 * @addtogroup thread_block thread_block
 * @{
 * @ingroup DeviceLanguageTest
 * Contains unit tests for all thread_block APIs
 */

namespace cg = cooperative_groups;

template <typename BaseType = cg::thread_block>
static __global__ void thread_block_size_getter(unsigned int* sizes) {
  const BaseType group = cg::this_thread_block();
  sizes[thread_rank_in_grid()] = group.size();
}

template <typename BaseType = cg::thread_block>
static __global__ void thread_block_thread_rank_getter(unsigned int* thread_ranks) {
  const BaseType group = cg::this_thread_block();
  thread_ranks[thread_rank_in_grid()] = group.thread_rank();
}

static __global__ void thread_block_group_indices_getter(dim3* group_indices) {
  group_indices[thread_rank_in_grid()] = cg::this_thread_block().group_index();
}

static __global__ void thread_block_thread_indices_getter(dim3* thread_indices) {
  thread_indices[thread_rank_in_grid()] = cg::this_thread_block().thread_index();
}

static __global__ void thread_block_non_member_size_getter(unsigned int* sizes) {
  sizes[thread_rank_in_grid()] = cg::group_size(cg::this_thread_block());
}

static __global__ void thread_block_non_member_thread_rank_getter(unsigned int* thread_ranks) {
  thread_ranks[thread_rank_in_grid()] = cg::thread_rank(cg::this_thread_block());
}

/**
 * Test Description
 * ------------------------
 *    - Launches kernels that write the return values of size, thread_rank, group_index, and
 * thread_index member functions to an output array that is validated on the host side. The kernels
 * are run sequentially, reusing the output array, to avoid running out of device memory for large
 * kernel launches.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Thread_Block_Getters_Positive_Basic") {
  const auto blocks = GenerateBlockDimensions();
  const auto threads = GenerateThreadDimensions();
  if (blocks.x <= 0 || blocks.y <= 0 || blocks.z <= 0 || threads.x <= 0 || threads.y <= 0 ||
      threads.z <= 0) {
    return;
  }
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  const CPUGrid grid(blocks, threads);

  {
    LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                                grid.thread_count_ * sizeof(unsigned int));
    LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                            grid.thread_count_ * sizeof(unsigned int));

    thread_block_size_getter<<<blocks, threads>>>(uint_arr_dev.ptr());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                        grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    thread_block_thread_rank_getter<<<blocks, threads>>>(uint_arr_dev.ptr());
    HIP_CHECK(hipGetLastError());

    // Validate thread_block.size() values
    ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
               [size = grid.threads_in_block_count_](uint32_t) { return size; });

    HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                        grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Validate thread_block.thread_rank() values
    ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
               [&grid](uint32_t i) { return grid.thread_rank_in_block(i).value(); });
  }

  {
    LinearAllocGuard<dim3> dim3_arr_dev(LinearAllocs::hipMalloc, grid.thread_count_ * sizeof(dim3));
    LinearAllocGuard<dim3> dim3_arr(LinearAllocs::hipHostMalloc, grid.thread_count_ * sizeof(dim3));

    thread_block_group_indices_getter<<<blocks, threads>>>(dim3_arr_dev.ptr());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(dim3_arr.ptr(), dim3_arr_dev.ptr(),
                        grid.thread_count_ * sizeof(*dim3_arr.ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    thread_block_thread_indices_getter<<<blocks, threads>>>(dim3_arr_dev.ptr());
    HIP_CHECK(hipGetLastError());

    // Validate thread_block.group_index() values
    ArrayAllOf(dim3_arr.ptr(), grid.thread_count_,
               [&grid](uint32_t i) { return grid.block_idx(i).value(); });

    HIP_CHECK(hipMemcpy(dim3_arr.ptr(), dim3_arr_dev.ptr(),
                        grid.thread_count_ * sizeof(*dim3_arr.ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Validate thread_block.thread_index() values
    ArrayAllOf(dim3_arr.ptr(), grid.thread_count_,
               [&grid](uint32_t i) { return grid.thread_idx(i).value(); });
  }
}

/**
 * Test Description
 * ------------------------
 *    - Launches kernels that write the return values of size and thread_rank member functions to an
 * output array that is validated on the host side, while treating the thread block as a thread
 * group. The kernels are run sequentially, reusing the output array, to avoid running out of device
 * memory for large kernel launches.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Thread_Block_Getters_Via_Base_Type_Positive_Basic") {
  const auto blocks = GenerateBlockDimensions();
  const auto threads = GenerateThreadDimensions();
  if (blocks.x <= 0 || blocks.y <= 0 || blocks.z <= 0 || threads.x <= 0 || threads.y <= 0 ||
      threads.z <= 0) {
    return;
  }
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);

  const CPUGrid grid(blocks, threads);

  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          grid.thread_count_ * sizeof(unsigned int));

  thread_block_size_getter<cg::thread_group><<<blocks, threads>>>(uint_arr_dev.ptr());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  thread_block_thread_rank_getter<cg::thread_group><<<blocks, threads>>>(uint_arr_dev.ptr());
  HIP_CHECK(hipGetLastError());

  // Validate thread_block.size() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
             [size = grid.threads_in_block_count_](uint32_t) { return size; });

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Validate thread_block.thread_rank() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
             [&grid](uint32_t i) { return grid.thread_rank_in_block(i).value(); });
}

/**
 * Test Description
 * ------------------------
 *    - Launches kernels that write the return values of size and thread_rank non-member functions
 * to an output array that is validated on the host side. The kernels are run sequentially, reusing
 * the output array, to avoid running out of device memory for large kernel launches.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Thread_Block_Getters_Via_Non_Member_Functions_Positive_Basic") {
  const auto blocks = GenerateBlockDimensions();
  const auto threads = GenerateThreadDimensions();
  if (blocks.x <= 0 || blocks.y <= 0 || blocks.z <= 0 || threads.x <= 0 || threads.y <= 0 ||
      threads.z <= 0) {
    return;
  }
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);

  const CPUGrid grid(blocks, threads);

  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          grid.thread_count_ * sizeof(unsigned int));

  thread_block_non_member_size_getter<<<blocks, threads>>>(uint_arr_dev.ptr());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  thread_block_non_member_thread_rank_getter<<<blocks, threads>>>(uint_arr_dev.ptr());
  HIP_CHECK(hipGetLastError());

  // Validate thread_block.size() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
             [size = grid.threads_in_block_count_](uint32_t) { return size; });

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Validate thread_block.thread_rank() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
             [&grid](uint32_t i) { return grid.thread_rank_in_block(i).value(); });
}


template <bool use_global, typename T>
__global__ void thread_block_sync_check(T* global_data, unsigned int* wait_modifiers,
                                        unsigned int* read_offsets) {
  extern __shared__ uint8_t shared_data[];
  T* const data = use_global ? global_data : reinterpret_cast<T*>(shared_data);
  const auto block = cg::this_thread_block();
  constexpr T divisor = 255;
  const auto tid = block.thread_rank();
  const auto wait_modifier = wait_modifiers[tid];
  const auto read_offset = read_offsets[tid];
  busy_wait(wait_modifier);
  data[tid] = tid % divisor;
  block.sync();
  bool valid = true;
  for (auto i = 0; i < block.size(); ++i) {
    const auto offset = block.size() + read_offset;
    const auto expected = (tid + offset + i) % block.size();
    if (!(valid &= (data[expected] == expected % divisor))) {
      break;
    }
  }
  block.sync();
  data[tid] = valid;
  if constexpr (!use_global) {
    global_data[tid] = data[tid];
  }
}

static inline std::mt19937& GetRandomGenerator() {
  // With a static seed the tests will remain consistent between runs, yet it relieves the problem
  // of predetermining a set of modifiers by hand. The sets of modifiers could actually be
  // determined at compile time if std::random objects could operate in a constexpr context.
  static std::mt19937 mt(17);
  return mt;
}

template <typename T> static inline T GenerateRandomInteger(const T min, const T max) {
  std::uniform_int_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

template <bool global_memory, typename T> void ThreadBlockSyncTest() {
  const auto randomized_run_count = GENERATE(range(0, cmd_options.cg_iterations));
  INFO("Run number: " << randomized_run_count + 1);
  const auto blocks = dim3(1, 1, 1);
  const auto threads = GenerateThreadDimensions();
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  CPUGrid grid(blocks, threads);

  const auto alloc_size = grid.thread_count_ * sizeof(T);
  int max_shared_mem_per_block = 0;
  HIP_CHECK(hipDeviceGetAttribute(&max_shared_mem_per_block,
                                  hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
  if (!global_memory && max_shared_mem_per_block < alloc_size) {
    return;
  }
  LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

  LinearAllocGuard<unsigned int> wait_modifiers_dev(LinearAllocs::hipMalloc,
                                                    grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> wait_modifiers(LinearAllocs::hipHostMalloc,
                                                grid.thread_count_ * sizeof(unsigned int));
  std::generate(wait_modifiers.ptr(), wait_modifiers.ptr() + grid.thread_count_,
                [&] { return GenerateRandomInteger(0u, 1500u); });

  LinearAllocGuard<unsigned int> read_offsets_dev(LinearAllocs::hipMalloc,
                                                  grid.thread_count_ * sizeof(unsigned int));
  std::vector<unsigned int> read_offsets(grid.thread_count_, 0u);
  if (randomized_run_count != 0) {
    std::generate(read_offsets.begin(), read_offsets.end(),
                  [&] { return GenerateRandomInteger(0u, grid.thread_count_); });
  }

  const auto shared_memory_size = global_memory ? 0u : alloc_size;
  HIP_CHECK(hipMemcpy(wait_modifiers_dev.ptr(), wait_modifiers.ptr(),
                      grid.thread_count_ * sizeof(unsigned int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(read_offsets_dev.ptr(), read_offsets.data(),
                      grid.thread_count_ * sizeof(unsigned int), hipMemcpyHostToDevice));

  thread_block_sync_check<global_memory><<<blocks, threads, shared_memory_size>>>(
      arr_dev.ptr(), wait_modifiers_dev.ptr(), read_offsets_dev.ptr());
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(std::all_of(arr.ptr(), arr.ptr() + grid.thread_count_, [](unsigned int e) { return e; }));
}

/**
 * Test Description
 * ------------------------
 *    - Launches a kernel wherein every thread writes its grid-wide linear index into an array. The
 * array is either in global or dynamic shared memory based on a compile time switch, and the test
 * is run for arrays of 1, 2, and 4 byte elements. Before the write each thread executes a busy wait
 * loop for a random amount of clock cycles, the amount being read from an input array. After the
 * write a block-wide sync is performed and each thread validates that it can read the expected
 * values that other threads have written to their respective array slots. Each thread begins the
 * validation from a given offset from its own index. For the first run of the test, all the offsets
 * are zero, so memory reads should be coalesced as adjacent threads read from adjacent memory
 * locations. On subsequent runs the offsets are randomized for each thread, leading to
 * non-coalesced reads and cache thrashing.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Thread_Block_Sync_Positive_Basic", "", uint8_t, uint16_t, uint32_t) {
  SECTION("Global memory") { ThreadBlockSyncTest<true, TestType>(); }
  SECTION("Shared memory") { ThreadBlockSyncTest<false, TestType>(); }
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
