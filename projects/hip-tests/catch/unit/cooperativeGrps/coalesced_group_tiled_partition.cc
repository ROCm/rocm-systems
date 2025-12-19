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

#include <bitset>
#include <optional>
#include <resource_guards.hh>
#include <utils.hh>

#include <cmd_options.hh>
#include <cpu_grid.h>
#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

/**
 * @addtogroup coalesced_group_tile coalesced_group_tile
 * @{
 * @ingroup DeviceLanguageTest
 * Contains unit tests for partitioning of coalesced groups into tiled partitions
 */

namespace cg = cooperative_groups;

namespace {
#if HT_AMD
constexpr auto kMaskMin = std::numeric_limits<uint64_t>().min();
constexpr auto kMaskLimit = std::numeric_limits<uint64_t>().max();
#else
constexpr auto kMaskMin = std::numeric_limits<uint32_t>().min();
constexpr auto kMaskLimit = std::numeric_limits<uint32_t>().max();
#endif
}  // namespace

static unsigned int GenerateTileSizes() {
#if HT_AMD
  return GENERATE(2u, 4u, 8u, 16u, 32u, 64u);
#else
  return GENERATE(2u, 4u, 8u, 16u, 32u);
#endif
}

static inline std::mt19937& GetRandomGenerator() {
  static std::mt19937 mt(11);
  return mt;
}

template <typename T> static inline T GenerateRandomInteger(const T min, const T max) {
  std::uniform_int_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

static auto coalesce_threads(const uint64_t mask, unsigned int warp_size) {
  std::tuple<std::vector<unsigned int>, unsigned int> res;
  auto& [threads, count] = res;

  for (auto i = 0u; i < warp_size; ++i) {
    if (mask & (1u << i)) {
      threads.push_back(i);
    }
  }

  return res;
}

__device__ bool deactivate_thread(uint64_t* active_masks, unsigned int warp_size) {
  const auto warp = cg::tiled_partition(cg::this_thread_block(), warp_size);
  const auto block = cg::this_thread_block();
  const auto warps_per_block = (block.size() + warp_size - 1) / warp_size;
  const auto block_rank = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto idx = block_rank * warps_per_block + block.thread_rank() / warp_size;
  return !(active_masks[idx] & (1u << warp.thread_rank()));
}

__global__ void coalesced_group_tiled_partition_size_getter(uint64_t* active_masks,
                                                            unsigned int tile_size,
                                                            unsigned int* sizes,
                                                            unsigned int warp_size) {
  if (deactivate_thread(active_masks, warp_size)) {
    return;
  }
  sizes[thread_rank_in_grid()] = cg::tiled_partition(cg::coalesced_threads(), tile_size).size();
}

__global__ void coalesced_group_tiled_partition_thread_rank_getter(uint64_t* active_masks,
                                                                   unsigned int tile_size,
                                                                   unsigned int* sizes,
                                                                   unsigned int warp_size) {
  if (deactivate_thread(active_masks, warp_size)) {
    return;
  }

  sizes[thread_rank_in_grid()] =
      cg::tiled_partition(cg::coalesced_threads(), tile_size).thread_rank();
}

/**
 * Test Description
 * ------------------------
 *    - Deactivates threads based on passed in mask and creates tiled partitions over coalesced
 * threads for each of the valid sizes{2, 4, 8, 16, 32, 64(if AMD)} and writes the return values of
 * size and thread_rank member functions to an output array that is validated on the host side.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/coalesced_group_tiled_partition.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Coalesced_Group_Tiled_Partition_Getters_Positive_Basic") {
  const auto tile_size = GenerateTileSizes();
  INFO("Tile size: " << tile_size);
  auto blocks = GenerateBlockDimensions();
  auto threads = GenerateThreadDimensions();
  int warp_size = getWarpSize();
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  CPUGrid grid(blocks, threads);

  const auto alloc_size = grid.thread_count_ * sizeof(unsigned int);
  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc, alloc_size);

  const auto warps_in_block = (grid.threads_in_block_count_ + warp_size - 1) / warp_size;
  const auto warps_in_grid = warps_in_block * grid.block_count_;
  LinearAllocGuard<uint64_t> active_masks_dev(LinearAllocs::hipMalloc,
                                              warps_in_grid * sizeof(uint64_t));
  LinearAllocGuard<uint64_t> active_masks(LinearAllocs::hipHostMalloc,
                                          warps_in_grid * sizeof(uint64_t));

  std::generate(active_masks.ptr(), active_masks.ptr() + warps_in_grid, [] {
    return GenerateRandomInteger((uint64_t)0, std::numeric_limits<uint64_t>().max());
  });
  HIP_CHECK(hipMemcpy(active_masks_dev.ptr(), active_masks.ptr(), warps_in_grid * sizeof(uint64_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemsetAsync(uint_arr_dev.ptr(), 0, alloc_size));
  coalesced_group_tiled_partition_size_getter<<<blocks, threads>>>(
      active_masks_dev.ptr(), tile_size, uint_arr_dev.ptr(), warp_size);
  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemsetAsync(uint_arr_dev.ptr(), 0, alloc_size));
  coalesced_group_tiled_partition_thread_rank_getter<<<blocks, threads>>>(
      active_masks_dev.ptr(), tile_size, uint_arr_dev.ptr(), warp_size);

  const auto tail = warps_in_block * warp_size - grid.threads_in_block_count_;

  // validate size
  for (auto i = 0u; i < warps_in_grid; ++i) {
    uint64_t current_warp_mask = active_masks.ptr()[i];
    const auto shift_amount =
        (tail + 32 * TestContext::get().isNvidia()) * !((i + 1) % warps_in_block);
    current_warp_mask = (current_warp_mask << shift_amount) >> shift_amount;

    const auto [active_threads, active_thread_count] =
        coalesce_threads(current_warp_mask, warp_size);

    const auto tails = tail * (i / warps_in_block) * (i >= warps_in_block);
    const auto num_tiles = (active_thread_count + tile_size - 1) / tile_size;
    const auto tile_tail = num_tiles * tile_size - active_thread_count;
    // Step tile-sized window over active threads
    for (auto t = 0u; t < active_thread_count; t += tile_size) {
      const auto window_start = t;
      const auto window_end = t + tile_size;
      // Iterate through window
      for (auto k = window_start; k < window_end && k < active_thread_count; ++k) {
        const auto global_thread_idx = i * warp_size + active_threads[k] - tails;
        const auto expected_val = tile_size - tile_tail * (t + tile_size >= active_thread_count);
        const auto actual_val = uint_arr.ptr()[global_thread_idx];
        INFO("global index: " << global_thread_idx);
        if (actual_val != expected_val) {
          REQUIRE(actual_val == expected_val);
        }
      }
    }
  }

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // validate rank
  for (auto i = 0u; i < warps_in_grid; ++i) {
    uint64_t current_warp_mask = active_masks.ptr()[i];
    const auto shift_amount =
        (tail + 32 * TestContext::get().isNvidia()) * !((i + 1) % warps_in_block);
    current_warp_mask = (current_warp_mask << shift_amount) >> shift_amount;

    const auto [active_threads, active_thread_count] =
        coalesce_threads(current_warp_mask, warp_size);

    const auto tails = tail * (i / warps_in_block) * (i >= warps_in_block);
    // Step tile-sized window over active threads
    for (auto t = 0u; t < active_thread_count; t += tile_size) {
      const auto window_start = t;
      const auto window_end = t + tile_size;
      // Iterate through window
      for (auto k = window_start; k < window_end && k < active_thread_count; ++k) {
        const auto global_thread_idx = i * warp_size + active_threads[k] - tails;
        const auto expected_val = k % tile_size;
        const auto actual_val = uint_arr.ptr()[global_thread_idx];
        INFO("global index: " << global_thread_idx);
        if (actual_val != expected_val) {
          REQUIRE(actual_val == expected_val);
        }
      }
    }
  }
}


template <typename T>
__global__ void coalesced_group_tiled_partition_shfl_up(uint64_t* active_masks, T* const out,
                                                        const unsigned int tile_size,
                                                        const unsigned int delta,
                                                        unsigned int warp_size) {
  if (deactivate_thread(active_masks, warp_size)) {
    return;
  }
  const auto warp = cg::tiled_partition(cg::this_thread_block(), warp_size);
  T var = static_cast<T>(warp.thread_rank());

  const auto tile = cg::tiled_partition(cg::coalesced_threads(), tile_size);
  out[thread_rank_in_grid()] = tile.shfl_up(var, delta);
}


template <typename T> static void CoalescedGroupTiledPartitonShflUpTestImpl() {
  const auto tile_size = GenerateTileSizes();
  INFO("Tile size: " << tile_size);
  auto blocks = GenerateBlockDimensionsForShuffle();
  auto threads = GenerateThreadDimensionsForShuffle();
  auto warp_size = getWarpSize();
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  const auto delta = GENERATE_COPY(range(0u, tile_size));
  INFO("Delta: " << delta);
  CPUGrid grid(blocks, threads);

  const auto alloc_size = grid.thread_count_ * sizeof(T);
  LinearAllocGuard<T> uint_arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<T> uint_arr(LinearAllocs::hipHostMalloc, alloc_size);

  const auto warps_in_block = (grid.threads_in_block_count_ + warp_size - 1) / warp_size;
  const auto warps_in_grid = warps_in_block * grid.block_count_;
  LinearAllocGuard<uint64_t> active_masks_dev(LinearAllocs::hipMalloc,
                                              warps_in_grid * sizeof(uint64_t));
  LinearAllocGuard<uint64_t> active_masks(LinearAllocs::hipHostMalloc,
                                          warps_in_grid * sizeof(uint64_t));

  std::generate(active_masks.ptr(), active_masks.ptr() + warps_in_grid,
                [] { return GenerateRandomInteger(kMaskMin, kMaskLimit); });
  HIP_CHECK(hipMemcpy(active_masks_dev.ptr(), active_masks.ptr(), warps_in_grid * sizeof(uint64_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemsetAsync(uint_arr_dev.ptr(), 0, alloc_size));
  coalesced_group_tiled_partition_shfl_up<T><<<blocks, threads>>>(
      active_masks_dev.ptr(), uint_arr_dev.ptr(), tile_size, delta, warp_size);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  const auto tail = warps_in_block * warp_size - grid.threads_in_block_count_;

  for (auto i = 0u; i < warps_in_grid; ++i) {
    auto current_warp_mask = active_masks.ptr()[i];
    const auto shift_amount =
        (tail + 32 * TestContext::get().isNvidia()) * !((i + 1) % warps_in_block);
    current_warp_mask = (current_warp_mask << shift_amount) >> shift_amount;

    const auto [active_threads, active_thread_count] =
        coalesce_threads(current_warp_mask, warp_size);

    const auto tails = tail * (i / warps_in_block) * (i >= warps_in_block);
    // Step tile-sized window over active threads
    for (auto t = 0u; t < active_thread_count; t += tile_size) {
      const auto window_start = t + delta;
      const auto window_end = t + tile_size;
      // Iterate through window
      for (auto k = window_start; k < window_end && k < active_thread_count; ++k) {
        const auto global_thread_idx = i * warp_size + active_threads[k] - tails;
        const auto expected_val = active_threads[k - delta];
        const auto actual_val = uint_arr.ptr()[global_thread_idx];
        INFO("global index: " << global_thread_idx);
        if (actual_val != expected_val) {
          REQUIRE(actual_val == expected_val);
        }
      }
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle up behavior of tiled partitions of all valid sizes{2, 4, 8, 16, 32,
 * 64(if AMD)} for delta values of [0, tile size). The partitions are created over a coalesced
 * group, with memberships of threads in the coalesced group being controlled via a passed in active
 * mask. The test is run for all overloads of shfl_up.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/coalesced_group_tiled_partition.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Coalesced_Group_Tiled_Partition_Shfl_Up_Positive_Basic", "", int,
                   unsigned int, long, unsigned long, long long, unsigned long long, float,
                   double) {
  CoalescedGroupTiledPartitonShflUpTestImpl<TestType>();
}


template <typename T>
__global__ void coalesced_group_tiled_partition_shfl_down(uint64_t* active_masks, T* const out,
                                                          const unsigned int tile_size,
                                                          const unsigned int delta,
                                                          unsigned int warp_size) {
  if (deactivate_thread(active_masks, warp_size)) {
    return;
  }
  const auto warp = cg::tiled_partition(cg::this_thread_block(), warp_size);
  T var = static_cast<T>(warp.thread_rank());

  const auto tile = cg::tiled_partition(cg::coalesced_threads(), tile_size);
  out[thread_rank_in_grid()] = tile.shfl_down(var, delta);
}


template <typename T> static void CoalescedGroupTiledPartitonShflDownTestImpl() {
  const auto tile_size = GenerateTileSizes();
  INFO("Tile size: " << tile_size);
  auto blocks = GenerateBlockDimensionsForShuffle();
  auto threads = GenerateThreadDimensionsForShuffle();
  auto warp_size = getWarpSize();
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  const auto delta = GENERATE_COPY(range(0u, tile_size));
  INFO("Delta: " << delta);
  CPUGrid grid(blocks, threads);

  const auto alloc_size = grid.thread_count_ * sizeof(T);
  LinearAllocGuard<T> uint_arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<T> uint_arr(LinearAllocs::hipHostMalloc, alloc_size);

  const auto warps_in_block = (grid.threads_in_block_count_ + warp_size - 1) / warp_size;
  const auto warps_in_grid = warps_in_block * grid.block_count_;
  LinearAllocGuard<uint64_t> active_masks_dev(LinearAllocs::hipMalloc,
                                              warps_in_grid * sizeof(uint64_t));
  LinearAllocGuard<uint64_t> active_masks(LinearAllocs::hipHostMalloc,
                                          warps_in_grid * sizeof(uint64_t));

  std::generate(active_masks.ptr(), active_masks.ptr() + warps_in_grid,
                [] { return GenerateRandomInteger(kMaskMin, kMaskLimit); });
  HIP_CHECK(hipMemcpy(active_masks_dev.ptr(), active_masks.ptr(), warps_in_grid * sizeof(uint64_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemsetAsync(uint_arr_dev.ptr(), 0, alloc_size));
  coalesced_group_tiled_partition_shfl_down<T><<<blocks, threads>>>(
      active_masks_dev.ptr(), uint_arr_dev.ptr(), tile_size, delta, warp_size);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  const auto tail = warps_in_block * warp_size - grid.threads_in_block_count_;

  for (auto i = 0u; i < warps_in_grid; ++i) {
    auto current_warp_mask = active_masks.ptr()[i];
    const auto shift_amount =
        (tail + 32 * TestContext::get().isNvidia()) * !((i + 1) % warps_in_block);
    current_warp_mask = (current_warp_mask << shift_amount) >> shift_amount;

    const auto [active_threads, active_thread_count] =
        coalesce_threads(current_warp_mask, warp_size);

    if (delta >= active_thread_count) {
      continue;
    }

    const auto tails = tail * (i / warps_in_block) * (i >= warps_in_block);
    // Step tile-sized window over active threads
    for (auto t = 0u; t < active_thread_count; t += tile_size) {
      const auto window_start = t;
      const auto window_end = t + tile_size - delta;
      // Iterate through window
      for (auto k = window_start; k < window_end && k < active_thread_count - delta; ++k) {
        const auto global_thread_idx = i * warp_size + active_threads[k] - tails;
        const auto expected_val = active_threads[k + delta];
        const auto actual_val = uint_arr.ptr()[global_thread_idx];
        INFO("global index: " << global_thread_idx);
        if (actual_val != expected_val) {
          REQUIRE(actual_val == expected_val);
        }
      }
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle down behavior of tiled partitions of all valid sizes{2, 4, 8, 16, 32,
 * 64(if AMD)} for delta values of [0, tile size). The partitions are created over a coalesced
 * group, with memberships of threads in the coalesced group being controlled via a passed in active
 * mask. The test is run for all overloads of shfl_down.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/coalesced_group_tiled_partition.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Coalesced_Group_Tiled_Partition_Shfl_Down_Positive_Basic", "", int,
                   unsigned int, long, unsigned long, long long, unsigned long long, float,
                   double) {
  CoalescedGroupTiledPartitonShflDownTestImpl<TestType>();
}


template <typename T>
__global__ void coalesced_group_tiled_partition_shfl(uint64_t* active_masks, uint8_t* target_lanes,
                                                     T* const out, const unsigned int tile_size,
                                                     unsigned int warp_size) {
  if (deactivate_thread(active_masks, warp_size)) {
    return;
  }
  const auto warp = cg::tiled_partition(cg::this_thread_block(), warp_size);
  T var = static_cast<T>(warp.thread_rank());

  const auto tile = cg::tiled_partition(cg::coalesced_threads(), tile_size);
  out[thread_rank_in_grid()] = tile.shfl(var, target_lanes[tile.thread_rank()]);
}

template <typename T> static void CoalescedGroupTiledPartitonShflTestImpl() {
  const auto tile_size = GenerateTileSizes();
  INFO("Tile size: " << tile_size);
  auto blocks = GenerateBlockDimensionsForShuffle();
  auto threads = GenerateThreadDimensionsForShuffle();
  auto warp_size = getWarpSize();
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  CPUGrid grid(blocks, threads);

  const auto alloc_size = grid.thread_count_ * sizeof(T);
  LinearAllocGuard<T> uint_arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<T> uint_arr(LinearAllocs::hipHostMalloc, alloc_size);

  const auto warps_in_block = (grid.threads_in_block_count_ + warp_size - 1) / warp_size;
  const auto warps_in_grid = warps_in_block * grid.block_count_;
  LinearAllocGuard<uint64_t> active_masks_dev(LinearAllocs::hipMalloc,
                                              warps_in_grid * sizeof(uint64_t));
  LinearAllocGuard<uint64_t> active_masks(LinearAllocs::hipHostMalloc,
                                          warps_in_grid * sizeof(uint64_t));
  LinearAllocGuard<uint8_t> target_lanes_dev(LinearAllocs::hipMalloc, tile_size * sizeof(uint8_t));
  LinearAllocGuard<uint8_t> target_lanes(LinearAllocs::hipHostMalloc, tile_size * sizeof(uint8_t));

  std::generate(target_lanes.ptr(), target_lanes.ptr() + tile_size,
                [tile_size] { return GenerateRandomInteger(0, static_cast<int>(2 * tile_size)); });
  std::generate(active_masks.ptr(), active_masks.ptr() + warps_in_grid,
                [] { return GenerateRandomInteger(kMaskMin, kMaskLimit); });
  HIP_CHECK(hipMemcpy(active_masks_dev.ptr(), active_masks.ptr(), warps_in_grid * sizeof(uint64_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(target_lanes_dev.ptr(), target_lanes.ptr(), tile_size * sizeof(uint8_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemsetAsync(uint_arr_dev.ptr(), 0, alloc_size));
  coalesced_group_tiled_partition_shfl<T><<<blocks, threads>>>(
      active_masks_dev.ptr(), target_lanes_dev.ptr(), uint_arr_dev.ptr(), tile_size, warp_size);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  const auto tail = warps_in_block * warp_size - grid.threads_in_block_count_;

  for (auto i = 0u; i < warps_in_grid; ++i) {
    auto current_warp_mask = active_masks.ptr()[i];
    const auto shift_amount =
        (tail + 32 * TestContext::get().isNvidia()) * !((i + 1) % warps_in_block);
    current_warp_mask = (current_warp_mask << shift_amount) >> shift_amount;

    const auto [active_threads, active_thread_count] =
        coalesce_threads(current_warp_mask, warp_size);

    const auto tails = tail * (i / warps_in_block) * (i >= warps_in_block);
    // Step tile-sized window over active threads
    for (auto t = 0u; t < active_thread_count; t += tile_size) {
      const auto window_start = t;
      const auto window_end = t + tile_size;
      // Iterate through window
      for (auto k = window_start; k < window_end && k < active_thread_count; ++k) {
        const auto global_thread_idx = i * warp_size + active_threads[k] - tails;
        const auto target_lane = target_lanes.ptr()[k % tile_size];
        if (target_lane >= tile_size || target_lane >= active_thread_count - t) {
          continue;
        }
        const auto expected_val = active_threads[t + target_lane];
        const auto actual_val = uint_arr.ptr()[global_thread_idx];
        INFO("global index: " << global_thread_idx);
        if (actual_val != expected_val) {
          REQUIRE(actual_val == expected_val);
        }
      }
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle behavior of tiled partitions of all valid sizes{2, 4, 8, 16, 32,
 * 64(if AMD)} for delta values of [0, tile size). The partitions are created over a coalesced
 * group, with memberships of threads in the coalesced group being controlled via a passed in active
 * mask. The test is run for all overloads of shfl.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/coalesced_group_tiled_partition.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Coalesced_Group_Tiled_Partition_Shfl_Positive_Basic", "", int,
                   unsigned int, long, unsigned long, long long, unsigned long long, float,
                   double) {
  CoalescedGroupTiledPartitonShflTestImpl<TestType>();
}


template <bool use_global, typename T>
__global__ void coalesced_group_tiled_partition_sync_check(uint64_t* active_masks, T* global_data,
                                                           unsigned int* wait_modifiers,
                                                           size_t tile_size,
                                                           unsigned int warp_size) {
  if (deactivate_thread(active_masks, warp_size)) {
    return;
  }

  extern __shared__ uint8_t shared_data[];
  T* const data = use_global ? global_data : reinterpret_cast<T*>(shared_data);
  const auto tid = cg::this_grid().thread_rank();
  const auto block = cg::this_thread_block();
  const auto coalesced = cg::coalesced_threads();
  const auto partition = cg::tiled_partition(coalesced, tile_size);
  const auto data_idx = [&block](unsigned int i) { return use_global ? i : (i % block.size()); };

  const auto wait_modifier = wait_modifiers[tid];

  const auto block_rank = tid / block.size();
  const auto warp_rank = block.thread_rank() / warp_size;
  const auto warp_base = block_rank * block.size() + warp_rank * warp_size;
  const auto global_idx = warp_base + coalesced.thread_rank();

  busy_wait(wait_modifier);
  data[data_idx(global_idx)] = partition.thread_rank();
  partition.sync();

  bool valid = true;
  const auto tile_rank = coalesced.thread_rank() / tile_size;
  for (auto i = 0u; i < tile_size; ++i) {
    const auto target_rank_in_tile = (coalesced.thread_rank() + i) % tile_size;
    const auto target_rank_in_warp = tile_rank * tile_size + target_rank_in_tile;
    if (target_rank_in_warp >= coalesced.size()) {
      continue;
    }
    if (!(valid &= (data[data_idx(warp_base + target_rank_in_warp)] == target_rank_in_tile))) {
      break;
    }
  }
  // Validate
  partition.sync();
  data[data_idx(global_idx)] = valid;
  if constexpr (!use_global) {
    global_data[global_idx] = data[data_idx(global_idx)];
  }
}

template <bool global_memory, typename T> void CoalescedGroupTiledPartitionSyncTest() {
  const auto randomized_run_count = GENERATE(range(0, cmd_options.cg_iterations));
  INFO("Run number: " << randomized_run_count + 1);
  const auto tile_size = GenerateTileSizes();
  INFO("Tile size: " << tile_size);
  auto blocks = GenerateBlockDimensionsForShuffle();
  auto threads = GenerateThreadDimensionsForShuffle();
  auto warp_size = getWarpSize();
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
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
  const auto warps_in_block = (grid.threads_in_block_count_ + warp_size - 1) / warp_size;
  const auto warps_in_grid = warps_in_block * grid.block_count_;
  LinearAllocGuard<uint64_t> active_masks_dev(LinearAllocs::hipMalloc,
                                              warps_in_grid * sizeof(uint64_t));
  LinearAllocGuard<uint64_t> active_masks(LinearAllocs::hipHostMalloc,
                                          warps_in_grid * sizeof(uint64_t));
  if (randomized_run_count != 0) {
    std::generate(wait_modifiers.ptr(), wait_modifiers.ptr() + grid.thread_count_,
                  [] { return GenerateRandomInteger(0u, 1500u); });
  } else {
    std::fill_n(wait_modifiers.ptr(), grid.thread_count_, 0u);
  }
  std::generate(active_masks.ptr(), active_masks.ptr() + warps_in_grid,
                [] { return GenerateRandomInteger(kMaskMin, kMaskLimit); });

  HIP_CHECK(hipMemcpy(active_masks_dev.ptr(), active_masks.ptr(), warps_in_grid * sizeof(uint64_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(wait_modifiers_dev.ptr(), wait_modifiers.ptr(),
                      grid.thread_count_ * sizeof(unsigned int), hipMemcpyHostToDevice));

  const auto shared_memory_size = global_memory ? 0u : alloc_size_per_block;
  coalesced_group_tiled_partition_sync_check<global_memory>
      <<<blocks, threads, shared_memory_size>>>(active_masks_dev.ptr(), arr_dev.ptr(),
                                                wait_modifiers_dev.ptr(), tile_size, warp_size);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  const auto tail = warps_in_block * warp_size - grid.threads_in_block_count_;
  for (int i = 0u; i < grid.block_count_; ++i) {
    for (int j = 0u; j < warps_in_block; ++j) {
      const auto warp_idx = i * warps_in_block + j;
      auto mask = active_masks.ptr()[warp_idx];
      const auto shift_amount =
          (tail + 32 * TestContext::get().isNvidia()) * !((warp_idx + 1) % warps_in_block);
      mask = (mask << shift_amount) >> shift_amount;
      const auto active_count = std::bitset<sizeof(mask) * 8>(mask).count();
      const auto start_offset = i * grid.threads_in_block_count_ + j * warp_size;
      const auto end_offset = start_offset + active_count;
      const auto valid =
          std::all_of(arr.ptr() + start_offset, arr.ptr() + end_offset, [](T e) { return e; });
      if (!valid) {
        REQUIRE(valid);
      }
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Launches a kernel wherein threads in each warp are deactivated based on a passed bitmask.
 * Coalesced groups are formed and divided into tiled partitions(size of 2, 4, 8, 16, 32, 64 if AMD)
 * and every thread writes its intra-tile rank into an array slot determined by its global warp rank
 * and coalesced group rank. The array is either in global or dynamic shared memory based on a
 * compile time switch, and the test is run for arrays of 1, 2, and 4 byte elements. Before the
 * write each thread executes a busy wait loop for a random amount of clock cycles, the amount being
 * read from an input array. After the write a tile-wide sync is performed and each thread validates
 * that it can read the expected values that other threads within the same tile have written to
 * their respective array slots.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/coalesced_group_tiled_partition.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
uint64_t counter = 0;
TEMPLATE_TEST_CASE("Unit_Coalesced_Group_Tiled_Partition_Sync_Positive_Basic", "", uint8_t,
                   uint16_t, uint32_t) {
  SECTION("Global memory") { CoalescedGroupTiledPartitionSyncTest<true, TestType>(); }
  SECTION("Shared memory") { CoalescedGroupTiledPartitionSyncTest<false, TestType>(); }
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
