/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#include "warp_common.hh"
#include "cooperative_groups_common.hh"
#include "cg_common_kernels.hh"
#include <array>
#include <random>

#include <cmd_options.hh>
#include <cpu_grid.h>
#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <resource_guards.hh>
#include <utils.hh>
#include <utility>

/**
 * @addtogroup thread_block_tile thread_block_tile
 * @{
 * @ingroup DeviceLanguageTest
 * Contains unit tests for all thread_block_tile APIs and dynamic block partitioning
 */

namespace cg = cooperative_groups;

template <bool dynamic, unsigned int tile_size>
__global__ void thread_block_partition_size_getter(unsigned int* sizes) {
  const auto group = cg::this_thread_block();
  if constexpr (dynamic) {
    sizes[thread_rank_in_grid()] = cg::tiled_partition(group, tile_size).size();
  } else {
    cg::thread_block_tile<tile_size> tiled_partition = cg::tiled_partition<tile_size>(group);
    sizes[thread_rank_in_grid()] = tiled_partition.size();
  }
}

template <bool dynamic, unsigned int tile_size>
__global__ void thread_block_partition_thread_rank_getter(unsigned int* thread_ranks) {
  const auto group = cg::this_thread_block();
  if constexpr (dynamic) {
    thread_ranks[thread_rank_in_grid()] = cg::tiled_partition(group, tile_size).thread_rank();
  } else {
    cg::thread_block_tile<tile_size> tiled_partition = cg::tiled_partition<tile_size>(group);
    thread_ranks[thread_rank_in_grid()] = tiled_partition.thread_rank();
  }
}

template <bool dynamic, size_t tile_size> void BlockPartitionGettersBasicTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    auto blocks = GenerateBlockDimensions();
    auto threads = GenerateThreadDimensions();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(unsigned int);
    LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc, alloc_size);

    thread_block_partition_size_getter<dynamic, tile_size><<<blocks, threads>>>(uint_arr_dev.ptr());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    thread_block_partition_thread_rank_getter<dynamic, tile_size>
        <<<blocks, threads>>>(uint_arr_dev.ptr());
    HIP_CHECK(hipGetLastError());

    ArrayAllOf(uint_arr.ptr(), grid.thread_count_, [&grid](unsigned int i) {
      if constexpr (!dynamic) {
        return tile_size;
      }

      const auto partitions_in_block = (grid.threads_in_block_count_ + tile_size - 1) / tile_size;
      const auto rank_in_block = grid.thread_rank_in_block(i).value();

      const auto tail = partitions_in_block * tile_size - grid.threads_in_block_count_;
      return tile_size - tail * (rank_in_block >= (partitions_in_block - 1) * tile_size);
    });

    HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    ArrayAllOf(uint_arr.ptr(), grid.thread_count_, [&grid](unsigned int i) {
      return grid.thread_rank_in_block(i).value() % tile_size;
    });
  }
}

template <bool dynamic, size_t... tile_sizes> void BlockPartitionGettersBasicTest() {
  static_cast<void>((BlockPartitionGettersBasicTestImpl<dynamic, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Creates tiled partitions for each of the valid sizes{2, 4, 8, 16, 32, 64(if AMD)} and writes
 * the return values of size and thread_rank member functions to an output array that is validated
 * on the host side.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block_tile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Thread_Block_Tile_Getters_Positive_Basic) {
  BlockPartitionGettersBasicTest<false, 2, 4, 8, 16, 32>();
}

/**
 * Test Description
 * ------------------------
 *    - Creates tiled partitions for each of the valid sizes{2, 4, 8, 16, 32, 64(if AMD)} via the
 * dynamic tiled partition api and writes the return values of size and thread_rank member functions
 * to an output array that is validated on host.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block_tile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Thread_Block_Tile_Dynamic_Getters_Positive_Basic) {
  BlockPartitionGettersBasicTest<true, 2, 4, 8, 16, 32>();
}


template <typename T, size_t tile_size>
__global__ void block_tile_shfl_up(T* const out, const unsigned int delta) {
  const cg::thread_block_tile<tile_size> partition =
      cg::tiled_partition<tile_size>(cg::this_thread_block());
  T var = static_cast<T>(partition.thread_rank());
  out[thread_rank_in_grid()] = partition.shfl_up(var, delta);
}

template <typename T, size_t tile_size> void BlockTileShflUpTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    const auto inv_reduction_factor = 1.0 / GetTestReductionFactor();

    auto blocks = GenerateBlockDimensionsForShuffle();
    auto threads = GenerateThreadDimensionsForShuffle();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);

    std::vector<size_t> deltas;
    for (double i = 0; i < tile_size - 1; i += inv_reduction_factor) {
      deltas.emplace_back(static_cast<size_t>(std::floor(i)));
    }
    deltas.emplace_back(tile_size - 1);

    const auto delta = GENERATE_COPY(from_range(deltas.begin(), deltas.end()));
    INFO("Delta: " << delta);

    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    block_tile_shfl_up<T, tile_size><<<blocks, threads>>>(arr_dev.ptr(), delta);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    ArrayAllOf(arr.ptr(), grid.thread_count_, [delta, &grid](unsigned int i) -> std::optional<T> {
      const int rank_in_partition = grid.thread_rank_in_block(i).value() % tile_size;
      const int target = rank_in_partition - delta;
      return target < 0 ? rank_in_partition : target;
    });
  }
}

template <typename T, size_t... tile_sizes> void BlockTileShflUpTest() {
  static_cast<void>((BlockTileShflUpTestImpl<T, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle up behavior of thread block tiles of all valid sizes{2, 4, 8, 16, 32,
 * 64(if AMD)} for delta values of [0, tile size). The test is run for all overloads of shfl_up.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block_tile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEMPLATE_TEST_CASE(Unit_Thread_Block_Tile_Shfl_Up_Positive_Basic, int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  BlockTileShflUpTest<TestType, 2, 16, 32>();
}


template <typename T, size_t tile_size>
__global__ void block_tile_shfl_down(T* const out, const unsigned int delta) {
  const cg::thread_block_tile<tile_size> partition =
      cg::tiled_partition<tile_size>(cg::this_thread_block());
  T var = static_cast<T>(partition.thread_rank());
  out[thread_rank_in_grid()] = partition.shfl_down(var, delta);
}

template <typename T, size_t tile_size> void BlockTileShflDownTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    const auto inv_reduction_factor = 1.0 / GetTestReductionFactor();

    auto blocks = GenerateBlockDimensionsForShuffle();
    auto threads = GenerateThreadDimensionsForShuffle();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);

    std::vector<size_t> deltas;
    for (double i = 0; i < tile_size - 1; i += inv_reduction_factor) {
      deltas.emplace_back(static_cast<size_t>(std::floor(i)));
    }
    deltas.emplace_back(tile_size - 1);

    const auto delta = GENERATE_COPY(from_range(deltas.begin(), deltas.end()));
    INFO("Delta: " << delta);

    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    block_tile_shfl_down<T, tile_size><<<blocks, threads>>>(arr_dev.ptr(), delta);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    ArrayAllOf(arr.ptr(), grid.thread_count_, [delta, &grid](unsigned int i) -> std::optional<T> {
      const auto partitions_in_block = (grid.threads_in_block_count_ + tile_size - 1) / tile_size;
      const auto rank_in_block = grid.thread_rank_in_block(i).value();
      const auto rank_in_group = rank_in_block % tile_size;
      const auto target = rank_in_group + delta;
      if (rank_in_block < (partitions_in_block - 1) * tile_size) {
        return target < tile_size ? target : rank_in_group;
      } else {
        // If the number of threads in a block is not an integer multiple of tile_size, the
        // final(tail end) tile will contain inactive threads.
        // Shuffling from an inactive thread returns an undefined value, accordingly threads that
        // shuffle from one must be skipped
        const auto tail = partitions_in_block * tile_size - grid.threads_in_block_count_;
        return target < tile_size - tail ? std::optional(target) : std::nullopt;
      }
    });
  }
}

template <typename T, size_t... tile_sizes> void BlockTileShflDownTest() {
  static_cast<void>((BlockTileShflDownTestImpl<T, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle down behavior of thread block tiles of all valid sizes{2, 16,
 * 32, 64(if AMD)} for delta values of [0, tile size). The test is run for all overloads of
 * shfl_down.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block_tile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEMPLATE_TEST_CASE(Unit_Thread_Block_Tile_Shfl_Down_Positive_Basic, int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  BlockTileShflDownTest<TestType, 2, 16, 32>();
}


template <typename T, size_t tile_size>
__global__ void block_tile_shfl_xor(T* const out, const unsigned mask) {
  const cg::thread_block_tile<tile_size> partition =
      cg::tiled_partition<tile_size>(cg::this_thread_block());
  T var = static_cast<T>(partition.thread_rank());
  out[thread_rank_in_grid()] = partition.shfl_xor(var, mask);
}

template <typename T, size_t tile_size> void BlockTileShflXORTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    const auto inv_reduction_factor = 1.0 / GetTestReductionFactor();

    auto blocks = GenerateBlockDimensionsForShuffle();
    auto threads = GenerateThreadDimensionsForShuffle();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);

    std::vector<size_t> masks;
    for (double i = 0; i < tile_size - 1; i += inv_reduction_factor) {
        masks.emplace_back(static_cast<size_t>(std::floor(i)));
    }
    masks.emplace_back(tile_size - 1);

    const auto mask = GENERATE_COPY(from_range(masks.begin(), masks.end()));
    INFO("Mask: 0x" << std::hex << mask);

    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    block_tile_shfl_xor<T, tile_size><<<blocks, threads>>>(arr_dev.ptr(), mask);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    const auto f = [mask, &grid](unsigned int i) -> std::optional<T> {
      const auto partitions_in_block = (grid.threads_in_block_count_ + tile_size - 1) / tile_size;
      const auto rank_in_block = grid.thread_rank_in_block(i).value();
      const int rank_in_partition = rank_in_block % tile_size;
      const auto target = rank_in_partition ^ mask;
      if (rank_in_block < (partitions_in_block - 1) * tile_size) {
        return target;
      }
      const auto tail = partitions_in_block * tile_size - grid.threads_in_block_count_;
      return target < tile_size - tail ? std::optional(target) : std::nullopt;
    };
    ArrayAllOf(arr.ptr(), grid.thread_count_, f);
  }
}

template <typename T, size_t... tile_sizes> void BlockTileShflXORTest() {
  static_cast<void>((BlockTileShflXORTestImpl<T, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle xor behavior of thread block tiles of all valid sizes{2, 16, 32,
 * 64(if AMD)} for mask values of [0, tile size). The test is run for all overloads of shfl_xor.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block_tile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEMPLATE_TEST_CASE(Unit_Thread_Block_Tile_Shfl_XOR_Positive_Basic, int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  BlockTileShflXORTest<TestType, 2, 16, 32>();
}

template <typename T, size_t tile_size>
__global__ void block_tile_shfl(T* const out, uint8_t* target_lanes) {
  const cg::thread_block_tile<tile_size> partition =
      cg::tiled_partition<tile_size>(cg::this_thread_block());
  T var = static_cast<T>(partition.thread_rank());
  out[thread_rank_in_grid()] = partition.shfl(var, target_lanes[partition.thread_rank()]);
}

static inline std::mt19937& GetRandomGenerator() {
  static std::mt19937 mt(11);
  return mt;
}

template <typename T> static inline T GenerateRandomInteger(const T min, const T max) {
  std::uniform_int_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

template <typename T, size_t tile_size> void BlockTileShflTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    auto blocks = GenerateBlockDimensionsForShuffle();
    auto threads = GenerateThreadDimensionsForShuffle();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    LinearAllocGuard<uint8_t> target_lanes_dev(LinearAllocs::hipMalloc,
                                               tile_size * sizeof(uint8_t));
    LinearAllocGuard<uint8_t> target_lanes(LinearAllocs::hipHostMalloc,
                                           tile_size * sizeof(uint8_t));
    std::generate(target_lanes.ptr(), target_lanes.ptr() + tile_size,
                  [] { return GenerateRandomInteger(0, static_cast<int>(2 * tile_size)); });

    HIP_CHECK(hipMemcpy(target_lanes_dev.ptr(), target_lanes.ptr(), tile_size * sizeof(uint8_t),
                        hipMemcpyHostToDevice));
    block_tile_shfl<T, tile_size><<<blocks, threads>>>(arr_dev.ptr(), target_lanes_dev.ptr());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    const auto f = [&target_lanes, &grid](unsigned int i) -> std::optional<T> {
      const auto partitions_in_block = (grid.threads_in_block_count_ + tile_size - 1) / tile_size;
      const auto rank_in_block = grid.thread_rank_in_block(i).value();
      const int rank_in_partition = rank_in_block % tile_size;
      const auto target = target_lanes.ptr()[rank_in_partition] % tile_size;
      if (rank_in_block < (partitions_in_block - 1) * tile_size) {
        return target;
      }
      const auto tail = partitions_in_block * tile_size - grid.threads_in_block_count_;
      return target < tile_size - tail ? std::optional(target) : std::nullopt;
    };
    ArrayAllOf(arr.ptr(), grid.thread_count_, f);
  }
}

template <typename T, size_t... tile_sizes> void BlockTileShflTest() {
  static_cast<void>((BlockTileShflTestImpl<T, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle behavior of thread block tiles of all valid sizes{2, 16, 32,
 * 64(if AMD)} for generated shuffle target lanes. The test is run for all overloads of shfl. Test
 * source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block_tile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEMPLATE_TEST_CASE(Unit_Thread_Block_Tile_Shfl_Positive_Basic, int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  BlockTileShflTest<TestType, 2, 16, 32>();
}


template <bool use_global, size_t tile_size, typename T>
__global__ void block_tile_sync_check(T* global_data, unsigned int* wait_modifiers) {
  extern __shared__ uint8_t shared_data[];
  T* const data = use_global ? global_data : reinterpret_cast<T*>(shared_data);
  const auto tid = cg::this_grid().thread_rank();
  const auto block = cg::this_thread_block();
  const cg::thread_block_tile<tile_size> partition =
      cg::tiled_partition<tile_size>(cg::this_thread_block());

  const auto data_idx = [&block](unsigned int i) { return use_global ? i : (i % block.size()); };

  const auto partitions_in_block = (block.size() + partition.size() - 1) / partition.size();
  const auto partition_rank = block.thread_rank() / partition.size();
  const auto tail = partitions_in_block * partition.size() - block.size();
  const auto window_size = partition.size() - tail * (partition_rank == partitions_in_block - 1);

  const auto block_base_idx = tid / block.size() * block.size();
  const auto tile_base_idx = block_base_idx + partition_rank * partition.size();

  const auto wait_modifier = wait_modifiers[tid];
  busy_wait(wait_modifier);
  data[data_idx(tid)] = partition.thread_rank();
  partition.sync();
  bool valid = true;
  for (auto i = 0; i < window_size; ++i) {
    const auto expected = (partition.thread_rank() + i) % window_size;

    if (!(valid &= (data[data_idx(tile_base_idx + expected)] == expected))) {
      break;
    }
  }
  partition.sync();
  data[data_idx(tid)] = valid;
  if constexpr (!use_global) {
    global_data[tid] = data[data_idx(tid)];
  }
}

template <bool global_memory, typename T, size_t tile_size> void BlockTileSyncTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    const auto randomized_run_count = GENERATE(range(0, cmd_options.cg_iterations));
    INFO("Run number: " << randomized_run_count + 1);
    auto blocks = GenerateBlockDimensions();
    auto threads = GenerateThreadDimensions();
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
    if (randomized_run_count != 0) {
      std::generate(wait_modifiers.ptr(), wait_modifiers.ptr() + grid.thread_count_,
                    [] { return GenerateRandomInteger(0u, 1500u); });
    } else {
      std::fill_n(wait_modifiers.ptr(), grid.thread_count_, 0u);
    }

    const auto shared_memory_size = global_memory ? 0u : alloc_size_per_block;
    HIP_CHECK(hipMemcpy(wait_modifiers_dev.ptr(), wait_modifiers.ptr(),
                        grid.thread_count_ * sizeof(unsigned int), hipMemcpyHostToDevice));

    block_tile_sync_check<global_memory, tile_size>
        <<<blocks, threads, shared_memory_size>>>(arr_dev.ptr(), wait_modifiers_dev.ptr());
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(
        std::all_of(arr.ptr(), arr.ptr() + grid.thread_count_, [](unsigned int e) { return e; }));
  }
}

template <bool global_memory, typename T, size_t... tile_sizes> void BlockTileSyncTest() {
  static_cast<void>((BlockTileSyncTestImpl<global_memory, T, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Launches a kernel wherein blocks are divided into tiled partitions(size of 2, 4, 8, 16, 32,
 * 64 if AMD) and every thread writes its intra-tile rank into an array slot determined by its
 * grid-wide linear index. The array is either in global or dynamic shared memory based on a compile
 * time switch, and the test is run for arrays of 1, 2, and 4 byte elements. Before the write each
 * thread executes a busy wait loop for a random amount of clock cycles, the amount being read from
 * an input array. After the write a tile-wide sync is performed and each thread validates that it
 * can read the expected values that other threads within the same tile have written to their
 * respective array slots.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block_tile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEMPLATE_TEST_CASE(Unit_Thread_Block_Tile_Sync_Positive_Basic, uint8_t, uint16_t, uint32_t) {
  SECTION("Global memory") { BlockTileSyncTest<true, TestType, 2, 16, 32>(); }
  SECTION("Shared memory") { BlockTileSyncTest<false, TestType, 2, 16, 32>(); }
}

template <size_t TileSize>
void __global__ simpleSum(int* result)
{
   int sum = 1;
   cg::thread_block mygroup = cg::this_thread_block();
   auto mytile = cg::tiled_partition<TileSize>(mygroup);
   *result = cg::reduce(mytile, sum, cg::plus<int>());
}

template <size_t TileSize>
void __global__ simpleSumSubtiles(int* result)
{
   int sum = 1;
   cg::thread_block mygroup = cg::this_thread_block();
   auto supertile = cg::tiled_partition<TileSize>(mygroup);
   auto subtile = cg::tiled_partition<TileSize / 2>(supertile);
   *result = cg::reduce(subtile, sum, cg::plus<int>());
}

template <size_t TileSize>
void testReduceForTileSize()
{
  LinearAllocGuard<int> h_result(LinearAllocs::malloc, sizeof(int));
  LinearAllocGuard<int> d_result(LinearAllocs::hipMalloc, sizeof(int));
  dim3 gridDim = { 1 };
  dim3 blockDim = { static_cast<unsigned short>(getWarpSize()) };
  void* devicePtr = d_result.ptr();
  void* args[] = { &devicePtr };

  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(simpleSum<TileSize>), gridDim, blockDim, args, 0, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                      h_result.size_bytes(), hipMemcpyDeviceToHost));
  REQUIRE(*h_result.host_ptr() == TileSize);

  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(simpleSumSubtiles<TileSize>), gridDim, blockDim, args, 0, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                      h_result.size_bytes(), hipMemcpyDeviceToHost));
  REQUIRE(*h_result.host_ptr() == TileSize / 2);
}


TEMPLATE_TEST_CASE(Unit_Thread_Block_Tile_Reduce_Basic, int)
{
  unsigned int wavefrontSize = getWarpSize();

  testReduceForTileSize<2>();
  testReduceForTileSize<4>();
  testReduceForTileSize<8>();
  testReduceForTileSize<16>();
  testReduceForTileSize<32>();

  if (wavefrontSize > 32) {
    testReduceForTileSize<64>();
  }
}

template <size_t TileSize>
void __global__ partialSum(int* result)
{
   int sum = 1;
   cg::thread_block mygroup = cg::this_thread_block();
   auto mytile = cg::tiled_partition<TileSize>(mygroup);

  if (threadIdx.x != warpSize - 1) {
     *result = cg::reduce(mytile, sum, cg::plus<int>());
  }
}

TEST_CASE(Unit_Thread_Block_Tile_Reduce_Non_Participating_Threads)
{
  LinearAllocGuard<int> h_result(LinearAllocs::malloc, sizeof(int));
  LinearAllocGuard<int> d_result(LinearAllocs::hipMalloc, sizeof(int));
  dim3 gridDim = { 1 };
  dim3 blockDim = { static_cast<unsigned short>(getWarpSize()) };
  void* devicePtr = d_result.ptr();
  void* args[] = { &devicePtr };
  void* kernelPtr = reinterpret_cast<void*>(getWarpSize() == 32?
                    partialSum<32> :
                    partialSum<64>);

  HIP_CHECK(hipLaunchCooperativeKernel(kernelPtr, gridDim, blockDim, args, 0, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                      h_result.size_bytes(), hipMemcpyDeviceToHost));
  // because a thread did not participate; we get a partial sum; note: this is undefined behaviour
  // on Nvidia
  REQUIRE(*h_result.host_ptr() == getWarpSize() - 1);
}

template <size_t TileSize, class Functor, class T>
void __global__ reduceKernel(T* output, const T* input, unsigned long long* extraMasks)
{
  int tid = threadIdx.x;
  int laneId = tid % warpSize;
  cg::thread_block mygroup = cg::this_thread_block();
  auto mytile = cg::tiled_partition<TileSize>(mygroup);

  for (int i = 0; i < kNumReduces; i++) {
    int idx = warpSize * i + laneId;
    unsigned long long mask = extraMasks[i];
    T& result = output[idx];

    if ((1ull << laneId) & mask) {
      result = cg::reduce(mytile, input[idx], Functor());
    } else {
      result = 0;
    }
  }
}

template <class Functor, class T>
void __global__ reduceKernelCoalesced(T* output, const T* input, unsigned long long* extraMasks)
{
  int tid = threadIdx.x;
  int laneId = tid % warpSize;

  for (int i = 0; i < kNumReduces; i++) {
    int idx = warpSize * i + laneId;
    unsigned long long mask = extraMasks[i];
    T& result = output[idx];

    if ((1ull << laneId) & mask) {
      auto coalesced = cg::coalesced_threads();

      result = cg::reduce(coalesced, input[idx], Functor());
    } else {
      result = 0;
    }
  }
}

// @TileSize the tile size or 0 when testing coalesced groups
template <size_t TileSize, class Op, class T>
void reduceForTypeAndOp()
{
  using distribution = typename DistributionType<T>::type;
  int wavefrontSize = getWarpSize();
  int size_bytes = kNumReduces * sizeof(T) * wavefrontSize;
  LinearAllocGuard<T> h_result(LinearAllocs::malloc, size_bytes);
  LinearAllocGuard<T> d_result(LinearAllocs::hipMalloc, size_bytes);
  LinearAllocGuard<T> h_input(LinearAllocs::malloc, size_bytes);
  LinearAllocGuard<T> d_input(LinearAllocs::hipMalloc, size_bytes);
  LinearAllocGuard<unsigned long long> d_extraMasks(LinearAllocs::hipMalloc,
                                                    kNumReduces * sizeof(unsigned long long));
  LinearAllocGuard<unsigned long long> h_extraMasks(LinearAllocs::malloc,
                                                    kNumReduces * sizeof(unsigned long long));
  std::mt19937_64 gen(Catch::rngSeed());
  dim3 gridDim = { 1 };
  dim3 blockDim = { static_cast<unsigned short>(wavefrontSize) };
  hipError_t status;
  typename distribution::result_type a = std::is_same<T, half>::value? std::numeric_limits<unsigned short>::lowest() :
                                         (std::is_signed<T>::value? -1023 : 0);
  typename distribution::result_type b = std::is_same<T, half>::value? std::numeric_limits<unsigned short>::max() :
                                         1023;
  distribution distInput(a, b);
  int numReduce = 0;
  void* kernelPtr;

  genRandomBuffers(d_input, h_input, distInput, gen, kNumReduces * wavefrontSize);
  genRandomMasks(d_extraMasks,
                 h_extraMasks,
                 gen,
                 kNumReduces);

  std::array<void*, 3> devicePtrs = { d_result.ptr(), d_input.ptr(), d_extraMasks.ptr() };
  void* args[devicePtrs.size()];

  for (int i = 0; i < devicePtrs.size(); i++) {
    args[i] = &devicePtrs[i];
  }

  if constexpr (TileSize == 0) {
    kernelPtr = reinterpret_cast<void*>(reduceKernelCoalesced<Op, T>);
  } else {
    kernelPtr = reinterpret_cast<void*>(reduceKernel<TileSize, Op, T>);
  }

  status = hipLaunchCooperativeKernel(kernelPtr,
                                      gridDim,
                                      blockDim,
                                      args,
                                      0,
                                      nullptr);
  HIP_CHECK(status);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                      h_result.size_bytes(), hipMemcpyDeviceToHost));

  while (numReduce < kNumReduces) {
    for (int laneId = 0; laneId < wavefrontSize; laneId++) {
      unsigned long long mask = ~0ull;
      T result = h_result.host_ptr()[numReduce * wavefrontSize + laneId], expected = 0;
      const T* input = &h_input.host_ptr()[numReduce * wavefrontSize];

      if constexpr (TileSize > 0) {
        mask >>= (64 - TileSize);
        mask <<= ((laneId % wavefrontSize) / TileSize) * TileSize;
      }

      mask &= h_extraMasks.host_ptr()[numReduce];

      if ((1ull << laneId) & mask) {
        Op op {};
        expected = calculateExpected(input, op, mask);
      }

      if constexpr (std::is_integral<T>::value) {
        // for integral types the result should match exactly
        if (result != expected) {
          std::string opName = opToString<T, Op>();
          printMismatch(result, expected, input, mask);
          INFO("Operator: " << opName);
          REQUIRE(result == expected);
        }
      } else {
        compareFloatingPoint(result, expected, mask, h_input.host_ptr());
      }
    }

    numReduce++;
  }
}

template <class Op, class T>
void reduceCoopTiles(const std::index_sequence<>)
{
}

template <class Op, class T, size_t TileSize, size_t... TileSizes>
void reduceCoopTiles(const std::index_sequence<TileSize, TileSizes...>)
{
  const std::index_sequence<TileSizes...> remainingTiles;

  reduceForTypeAndOp<TileSize, Op, T>();
  reduceCoopTiles<Op, T>(remainingTiles);
}

template <bool Coalesced, class Op, class T, int WarpSize>
void runReduceRandomForType()
{
  if constexpr (Coalesced) {
    reduceForTypeAndOp<0, Op, T>();
  } else if constexpr (WarpSize <= 32) {
    std::index_sequence<1, 2, 4, 8, 16, 32> tileSizes;
    reduceCoopTiles<Op, T>(tileSizes);
  } else {
    std::index_sequence<1, 2, 4, 8, 16, 32, 64> tileSizes;
    reduceCoopTiles<Op, T>(tileSizes);
  }
}

template <bool Coalesced, class T, int WarpSize, class Op = void>
void runReduceRandomForOps(const std::tuple<>)
{
}

template <bool Coalesced, class T, int WarpSize, class Op, class... Ops>
void runReduceRandomForOps(const std::tuple<Op, Ops...>)
{
  const std::tuple<Ops...> remainingOps;

  runReduceRandomForType<Coalesced, Op, T, WarpSize>();
  runReduceRandomForOps<Coalesced, T, WarpSize>(remainingOps);
}

// for all the tile sizes and all input types, using random input values, calculates the reduce()
// values. Additionally, randomly make some threads not participate
TEMPLATE_TEST_CASE(Unit_Thread_Block_Tile_Reduce_Random_arithmetic, int, unsigned int, long long,
                   unsigned long long, float, half, double)
{
  std::tuple<cooperative_groups::plus<TestType>,
             cooperative_groups::less<TestType>,
             cooperative_groups::greater<TestType>> types;

  if (getWarpSize() == 32) {
    runReduceRandomForOps<false, TestType, 32>(types);
  } else {
    runReduceRandomForOps<false, TestType, 64>(types);
  }
}

TEMPLATE_TEST_CASE(Unit_Thread_Block_Tile_Reduce_Random_boolean, int, unsigned int, long long,
                   unsigned long long)
{
  std::tuple<cooperative_groups::bit_and<TestType>,
             cooperative_groups::bit_or<TestType>,
             cooperative_groups::bit_xor<TestType>> types;

  if (getWarpSize() == 32) {
    runReduceRandomForOps<false, TestType, 32>(types);
  } else {
    runReduceRandomForOps<false, TestType, 64>(types);
  }
}

// passes a custom operator to cooperative_groups::reduce()
TEST_CASE(Unit_Thread_Block_Tile_Reduce_Custom_Op)
{
  if (getWarpSize() == 32) {
    runReduceRandomForType<false, MaxOfAbsolute<int>, int, 32>();
  } else {
    runReduceRandomForType<false, MaxOfAbsolute<int>, int, 64>();
  }
}

struct Vector {
  int x;
  int y;
};

// given two vector returns the one with the maximum magnitude
struct MaxMagnitude {
  Vector __device__ operator()(Vector lhs, Vector rhs) const
  {
    int lhsMag = lhs.x * lhs.x + lhs.y * lhs.y;
    int rhsMag = rhs.x * rhs.x + rhs.y * rhs.y;

    return lhsMag > rhsMag? lhs : rhs;
  }
};

void __global__ maxMagnitude(Vector* result)
{
  cg::thread_block mygroup = cg::this_thread_block();
  auto mytile = cg::tiled_partition<4>(mygroup);
  MaxMagnitude op;
  Vector input[] = {{ 2, 3 },
                    { 1, 9 },
                    { 0, 7 },
                    { 4, 1} };

  *result = cg::reduce(mytile, input[threadIdx.x], op);
}

// tests that we can pass trivially copyable structs as values to reduce
TEST_CASE(Unit_Thread_Block_Tile_Reduce_Trivially_Copyable_Parameters)
{
  LinearAllocGuard<Vector> h_result(LinearAllocs::malloc, sizeof(Vector));
  LinearAllocGuard<Vector> d_result(LinearAllocs::hipMalloc, sizeof(Vector));
  dim3 gridDim = { 1 };
  dim3 blockDim = { 4 };
  void* devicePtr = d_result.ptr();
  void* args[] = { &devicePtr };
  Vector* result;

  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(maxMagnitude), gridDim, blockDim, args, 0, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                      h_result.size_bytes(), hipMemcpyDeviceToHost));
  result = &h_result.host_ptr()[0];
  REQUIRE((result->x == 1 && result->y == 9));
}

template <size_t NumElems>
using ArrayContainer = std::array<unsigned char, NumElems>;

template <size_t NumElems>
struct Sum {
  ArrayContainer<NumElems> __device__ operator()(const ArrayContainer<NumElems>& lhs,
                                                 const ArrayContainer<NumElems>& rhs) const
  {
    ArrayContainer<NumElems> result;

    for (int i = 0; i < NumElems; i++) {
      result[i] = lhs[i] + rhs[i];
    }

    return result;
  }
};

template <size_t NumElems>
struct Max {
  ArrayContainer<NumElems> __device__ operator()(const ArrayContainer<NumElems>& lhs,
                                                 const ArrayContainer<NumElems>& rhs) const
  {
    ArrayContainer<NumElems> result;

    for (int i = 0; i < NumElems; i++) {
      for (int i = 0; i < NumElems; i++) {
        result[i] = std::max(lhs[i], rhs[i]);
      }
    }

    return result;
  }
};

template <size_t NumElems, class Functor>
__global__ void applyFunctor(ArrayContainer<NumElems>* result)
{
  cg::thread_block mygroup = cg::this_thread_block();
  auto mytile = cg::tiled_partition<32>(mygroup);
  __shared__ ArrayContainer<NumElems> input;
  Functor op;

  if (threadIdx.x < NumElems) {
    input[threadIdx.x] = threadIdx.x;
    __syncwarp();
    *result = cg::reduce(mytile, input, op);
  }
}

template <size_t NumElems, template <size_t> class Functor>
void testReduceSizes()
{
  LinearAllocGuard<ArrayContainer<NumElems>> h_result(LinearAllocs::malloc, sizeof(ArrayContainer<NumElems>));
  LinearAllocGuard<ArrayContainer<NumElems>> d_result(LinearAllocs::hipMalloc, sizeof(ArrayContainer<NumElems>));
  dim3 gridDim = { 1 };
  dim3 blockDim = { 32 };
  void* devicePtr = d_result.ptr();
  void* args[] = { &devicePtr };
  ArrayContainer<NumElems>* result;

  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(applyFunctor<NumElems, Functor<NumElems>>), gridDim, blockDim, args, 0, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                      h_result.size_bytes(), hipMemcpyDeviceToHost));
  result = &h_result.host_ptr()[0];
  INFO("T is of size: " << NumElems);

  for (int i = 0; i < NumElems; i++) {
    INFO("Element: " << i);

    if (std::is_same<Functor<NumElems>, Sum<NumElems>>::value) {
      // the result can be calculated with an arithmetic series formula, modulo 256
      // (we do overflow unsigned char for some indices, but that is defined behaviour)
      REQUIRE((*result)[i] == ((NumElems * (i + i)) / 2) % 256);
    } else {
      REQUIRE((*result)[i] == i);
    }
  }

  if constexpr (NumElems > 1) {
    testReduceSizes<NumElems - 1, Functor>();
  }
}

// we allow any reduction size of T of up to 32 bytes; this tests that reduction works with user-defined
// types in that range; including non-powers of two
TEST_CASE(Unit_Thread_Block_Tile_Reduce_All_Parameter_Sizes)
{
  SECTION("sum") {
    testReduceSizes<32, Sum>();
  }

  SECTION("max") {
    testReduceSizes<32, Max>();
  }
}

struct Point {
    int x;
    int y;

    __device__ Point operator+(const Point& rhs)
    {
      return { x + rhs.x, y + rhs.y };
    }

};

__global__ void sumPoints(Point* result)
{
  cg::thread_block mygroup = cg::this_thread_block();
  auto mytile = cg::tiled_partition<32>(mygroup);
  Point input;
  
  input.x = threadIdx.x;
  input.y = threadIdx.x;

  __syncwarp();
  *result = cg::reduce(mytile, input, cooperative_groups::plus<Point> {});
}

// using a standard functor in the cooperative_groups namespace with a type that is not primitive
TEST_CASE(Unit_Thread_Block_Tile_Reduce_Standard_Op_Custom_Type)
{
  LinearAllocGuard<Point> h_result(LinearAllocs::malloc, sizeof(Point) * 32);
  LinearAllocGuard<Point> d_result(LinearAllocs::hipMalloc, sizeof(Point) * 32);
  dim3 gridDim = { 1 };
  dim3 blockDim = { 32 };
  void* devicePtr = d_result.ptr();
  void* args[] = { &devicePtr };
  int expected = 31 * 16;

  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(sumPoints), gridDim, blockDim, args, 0, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                      h_result.size_bytes(), hipMemcpyDeviceToHost));

  for (int i = 0; i < 32; i++) {
    INFO("Expected x: " << expected << " got: " << *h_result.host_ptr());
    INFO("Expected y: " << expected << " got: " << *h_result.host_ptr());
    REQUIRE((h_result.host_ptr()->x == expected && h_result.host_ptr()->y == expected));
  }
}

TEMPLATE_TEST_CASE(Unit_Thread_Block_Coalesced_Reduce_arithmetic, int, unsigned int, long long,
                   unsigned long long, float, half, double)
{
  std::tuple<cooperative_groups::plus<TestType>,
             cooperative_groups::less<TestType>,
             cooperative_groups::greater<TestType>> ops;

  if (getWarpSize() == 32) {
    runReduceRandomForOps<true, TestType, 32>(ops);
  } else {
    runReduceRandomForOps<true, TestType, 64>(ops);
  }
}


TEMPLATE_TEST_CASE(Unit_Thread_Block_Coalesced_Reduce_boolean, int, unsigned int, long long, unsigned long long)
{
  std::tuple<cooperative_groups::bit_and<TestType>,
             cooperative_groups::bit_or<TestType>,
             cooperative_groups::bit_xor<TestType>> ops;
  if (getWarpSize() == 32) {
    runReduceRandomForOps<true, TestType, 32>(ops);
  } else {
    runReduceRandomForOps<true, TestType, 64>(ops);
  }
} 

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
