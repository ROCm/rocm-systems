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

  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(simpleSum<TileSize>),
                                       gridDim,
                                       blockDim,
                                       args,
                                       0,
                                       nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                      h_result.size_bytes(), hipMemcpyDeviceToHost));
  REQUIRE(*h_result.host_ptr() == TileSize);

  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(simpleSumSubtiles<TileSize>),
                                       gridDim,
                                       blockDim,
                                       args,
                                       0,
                                       nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                      h_result.size_bytes(), hipMemcpyDeviceToHost));
  REQUIRE(*h_result.host_ptr() == TileSize / 2);
}


HIP_TEMPLATE_TEST_CASE(Unit_Thread_Block_Tile_Reduce_Basic, int)
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

// @extraMasks  when testing coalesced_threads, this can be use to simulate
//              divergence
template <size_t TileSize, class Functor, class T>
void __global__ reduceKernel(T* output,
                             const T* input,
                             unsigned long long* extraMasks,
                             AggregationType* aggType)
{
  int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  int laneId = tid % warpSize;
  cg::thread_block mygroup = cg::this_thread_block();
  auto mytile = cg::tiled_partition<TileSize>(mygroup);

  for (int i = 0; i < kNumReduces; i++) {
    int idx = warpSize * i + laneId;
    T& result = output[idx];
    unsigned long long mask = extraMasks[i];

    if ((1ull << laneId) & mask) {
      switch (*aggType) {
      case AggregationType::Reduce:
        result = cg::reduce(mytile, input[idx], Functor());
        break;
      case AggregationType::InclusiveScan:
        result = cg::inclusive_scan(mytile, input[idx], Functor());
        break;
      case AggregationType::ExclusiveScan:
        result = cg::exclusive_scan(mytile, input[idx], Functor());
        break;
      default:
        assert(false && "Unsupported enumeration");
      }

    } else {
      result = 0;
    }
  }
}

// @extraMasks  used to simulate divergence when using coalesced_threads
template <class Functor, class T>
void __global__ reduceKernelCoalesced(T* output,
                                      const T* input,
                                      unsigned long long* extraMasks,
                                      AggregationType* aggType)
{
  int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  int laneId = tid % warpSize;

  for (int i = 0; i < kNumReduces; i++) {
    int idx = warpSize * i + laneId;
    unsigned long long mask = extraMasks[i];
    T& result = output[idx];

    if ((1ull << laneId) & mask) {
      auto coalesced = cg::coalesced_threads();

      switch (*aggType) {
      case AggregationType::Reduce:
        result = cg::reduce(coalesced, input[idx], Functor());
        break;
      case AggregationType::InclusiveScan:
        result = cg::inclusive_scan(coalesced, input[idx], Functor());
        break;
      case AggregationType::ExclusiveScan:
        result = cg::exclusive_scan(coalesced, input[idx], Functor());
        break;
      default:
        assert(false && "Unsupported enumeration");
      }
    } else {
      result = 0;
    }
  }
}

// @TileSize the tile size or 0 when testing coalesced groups
template <size_t TileSize, class Op, class T>
void aggregateForTypeAndOp(AggregationType aggType,
                           dim3 blockDim)
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
  LinearAllocGuard<AggregationType> d_aggType(LinearAllocs::hipMalloc, sizeof(AggregationType));
  std::mt19937_64 gen(Catch::rngSeed());
  dim3 gridDim = { 1 };
  hipError_t status;
  typename distribution::result_type a = std::is_same<T, half>::value? std::numeric_limits<unsigned short>::lowest() :
                                         (std::is_signed<T>::value? -1023 : 0);
  typename distribution::result_type b = std::is_same<T, half>::value? std::numeric_limits<unsigned short>::max() :
                                         1023;
  distribution distInput {a, b};
  int numAggregation = 0;
  void* kernelPtr;
  T expected[64];

  HIP_CHECK(hipMemcpy(d_aggType.ptr(), &aggType, sizeof(aggType), hipMemcpyHostToDevice));
  genRandomBuffers(d_input, h_input, distInput, gen, kNumReduces * wavefrontSize);

  if (TileSize) {
    // tile block case
    std::memset(h_extraMasks.host_ptr(), 0xFF, h_extraMasks.size_bytes());
    HIP_CHECK(hipMemset(d_extraMasks.ptr(), 0xFF, d_extraMasks.size_bytes()));
  } else {
    // coalesced_threads case
    genRandomMasks(d_extraMasks,
                   h_extraMasks,
                   gen,
                   kNumReduces);
  }

  std::array<void*, 4> devicePtrs = { d_result.ptr(), d_input.ptr(), d_extraMasks.ptr(), d_aggType.ptr() };
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

  while (numAggregation < kNumReduces) {
    for (int laneId = 0; laneId < wavefrontSize; laneId++) {
      T result = h_result.host_ptr()[numAggregation * wavefrontSize + laneId];
      int lastLane;
      const T* input = &h_input.host_ptr()[numAggregation * wavefrontSize];
      unsigned long long mask = ~0ull;
      Op op {};
      std::string opName = opToString<T, Op>();
      int resultLane;

      if constexpr (TileSize > 0) {
        mask >>= (64 - TileSize);
        mask <<= ((laneId % wavefrontSize) / TileSize) * TileSize;
      }

      mask &= h_extraMasks.host_ptr()[numAggregation];
      lastLane = 64 - __builtin_clzll(mask) - 1;

      if (laneId > lastLane) {
        continue;
      }

      calculateExpected<T>(expected, input, op, mask, aggType);

      if (aggType == AggregationType::Reduce) {
        resultLane = lastLane;
      } else {
        resultLane = laneId;
      }

      if ((1ull << laneId) & mask) {
        if constexpr (std::is_integral<T>::value) {
          // for integral types the result should match exactly
          // for reduce, the result would be in the last lane whose first bit is on in the mask
          // for scans, the associated result is different in each lane
          if (result != expected[resultLane]) {
            INFO("Aggregation type: " << aggregationTypeToStr(aggType));
            printMismatch(result, expected[resultLane], input, mask, laneId);
            INFO("Operator: " << opName << " mask: 0x" << std::hex << mask);
            REQUIRE(result == expected[resultLane]);
          }
        } else {
          compareFloatingPoint<Op>(result,
                                   expected[resultLane],
                                   mask,
                                   h_input.host_ptr(),
                                   laneId);
        }
      }
    }

    numAggregation++;
  }
}

template <class Op, class T>
void aggregateCoopTiles(AggregationType, dim3, const std::index_sequence<>)
{
}

template <class Op, class T, size_t TileSize, size_t... TileSizes>
void aggregateCoopTiles(AggregationType aggType, dim3 blockDim, const std::index_sequence<TileSize, TileSizes...>)
{
  const std::index_sequence<TileSizes...> remainingTiles;

  aggregateForTypeAndOp<TileSize, Op, T>(aggType, blockDim);
  aggregateCoopTiles<Op, T>(aggType, blockDim, remainingTiles);
}

template <bool Coalesced, class Op, class T, int MaxTileSize>
void runAggregationRandomForType(AggregationType aggType, dim3 blockDim)
{
  if constexpr (Coalesced) {
    aggregateForTypeAndOp<0, Op, T>(aggType, blockDim);
  } else if constexpr (MaxTileSize <= 4) {
    std::index_sequence<1, 2, 4> tileSizes;
    aggregateCoopTiles<Op, T>(aggType, blockDim, tileSizes);
  } else if constexpr (MaxTileSize <= 32) {
    std::index_sequence<1, 2, 4, 8, 16, 32> tileSizes;
    aggregateCoopTiles<Op, T>(aggType, blockDim, tileSizes);
  } else {
    std::index_sequence<1, 2, 4, 8, 16, 32, 64> tileSizes;
    aggregateCoopTiles<Op, T>(aggType, blockDim, tileSizes);
  }
}

template <bool Coalesced, class T, int WarpSize, class Op = void>
void runAggregationRandomForOps(AggregationType aggType, const std::tuple<>)
{
}

template <bool Coalesced, class T, int WarpSize, class Op, class... Ops>
void runAggregationRandomForOps(AggregationType aggType, const std::tuple<Op, Ops...>)
{
  const std::tuple<Ops...> remainingOps;
  int wavefrontSize = getWarpSize();
  dim3 blockDim = {static_cast<unsigned int>(wavefrontSize)};

  runAggregationRandomForType<Coalesced, Op, T, WarpSize>(aggType, blockDim);
  runAggregationRandomForOps<Coalesced, T, WarpSize>(aggType, remainingOps);
}

// for all the tile sizes and all input types, using random input values, calculates the reduce()
// values. Additionally, randomly make some threads not participate for the coalesced_threads case
HIP_TEMPLATE_TEST_CASE(Unit_Thread_Block_Tile_Reduce_Random_arithmetic, int, unsigned int, long long,
                       unsigned long long, float, half, double)
{
  std::tuple<cooperative_groups::plus<TestType>,
             cooperative_groups::less<TestType>,
             cooperative_groups::greater<TestType>> types;
  int wavefrontSize = getWarpSize();
  dim3 blockDim = {static_cast<unsigned int>(wavefrontSize)};

  if (wavefrontSize == 32) {
    runAggregationRandomForOps<false, TestType, 32>(AggregationType::Reduce, types);
  } else {
    runAggregationRandomForOps<false, TestType, 64>(AggregationType::Reduce, types);
  }
}

HIP_TEMPLATE_TEST_CASE(Unit_Thread_Block_Tile_Reduce_Random_boolean, int, unsigned int, long long,
                   unsigned long long)
{
  std::tuple<cooperative_groups::bit_and<TestType>,
             cooperative_groups::bit_or<TestType>,
             cooperative_groups::bit_xor<TestType>> types;

  if (getWarpSize() == 32) {
    runAggregationRandomForOps<false, TestType, 32>(AggregationType::Reduce, types);
  } else {
    runAggregationRandomForOps<false, TestType, 64>(AggregationType::Reduce, types);
  }
}

// passes a custom operator to cooperative_groups::reduce()
HIP_TEST_CASE(Unit_Thread_Block_Tile_Reduce_Custom_Op)
{
  int wavefrontSize = getWarpSize();

  dim3 blockDim = {static_cast<unsigned int>(wavefrontSize)};
  if (wavefrontSize == 32) {
    runAggregationRandomForType<false, MaxOfAbsolute<int>, int, 32>(AggregationType::Reduce, blockDim);
  } else {
    runAggregationRandomForType<false, MaxOfAbsolute<int>, int, 64>(AggregationType::Reduce, blockDim);
  }
}

HIP_TEST_CASE(Unit_Thread_Block_Tile_Scan_Custom_Op)
{
  dim3 blockDim = {static_cast<unsigned int>(getWarpSize())};
  // only using 4 threads to avoid long long overflows
  runAggregationRandomForType<false, NonCommutativeOp<long long>, long long, 4>(
    AggregationType::InclusiveScan, blockDim);
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

void __global__ maxMagnitude(Vector* result, AggregationType* aggregationType)
{
  cg::thread_block mygroup = cg::this_thread_block();
  auto mytile = cg::tiled_partition<4>(mygroup);
  MaxMagnitude op;
  Vector input[] = {{ 2, 3 },
                    { 1, 9 },
                    { 0, 7 },
                    { 4, 1} };

  switch (*aggregationType) {
  case AggregationType::Reduce:
    result[threadIdx.x] = cg::reduce(mytile, input[threadIdx.x], op);
    break;
  case AggregationType::InclusiveScan:
    result[threadIdx.x] = cg::inclusive_scan(mytile, input[threadIdx.x], op);
    break;
  case AggregationType::ExclusiveScan:
    result[threadIdx.x] = cg::exclusive_scan(mytile, input[threadIdx.x], op);
    break;
  default:
    assert(false && "Unexpected aggType");
  }
}

// tests that we can pass trivially copyable structs as values to reduce
HIP_TEST_CASE(Unit_Thread_Block_Tile_Reduce_Trivially_Copyable_Parameters)
{
  dim3 gridDim = { 1 };
  dim3 blockDim = { 4 };
  LinearAllocGuard<Vector> h_result(LinearAllocs::malloc, sizeof(Vector) * blockDim.x);
  LinearAllocGuard<Vector> d_result(LinearAllocs::hipMalloc, sizeof(Vector) * blockDim.x);
  LinearAllocGuard<AggregationType> d_aggType(LinearAllocs::hipMalloc, sizeof(AggregationType));
  std::array<void*, 2> devicePtrs = { d_result.ptr(), d_aggType.ptr() };
  void* args[devicePtrs.size()];
  Vector* result;
  AggregationType aggType = AggregationType::Reduce;

  for (int i = 0; i < devicePtrs.size(); i++) {
    args[i] = &devicePtrs[i];
  }

  HIP_CHECK(hipMemcpy(d_aggType.ptr(), &aggType, sizeof(aggType), hipMemcpyHostToDevice));
  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(maxMagnitude), gridDim, blockDim, args, 0, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                      h_result.size_bytes(), hipMemcpyDeviceToHost));

  for (unsigned int idx = 0; idx < blockDim.x; idx++) {
    result = &h_result.host_ptr()[idx];
    REQUIRE((result->x == 1 && result->y == 9));
  }
}

TEST_CASE(Unit_Thread_Block_Tile_Scan_Trivially_Copyable_Parameters)
{
  dim3 gridDim = { 1 };
  dim3 blockDim = { 4 };
  LinearAllocGuard<Vector> h_result(LinearAllocs::malloc, sizeof(Vector) * blockDim.x);
  LinearAllocGuard<Vector> d_result(LinearAllocs::hipMalloc, sizeof(Vector) * blockDim.x);
  LinearAllocGuard<AggregationType> d_aggType(LinearAllocs::hipMalloc, sizeof(AggregationType));
  std::array<void*, 2> devicePtrs = { d_result.ptr(), d_aggType.ptr() };
  void* args[devicePtrs.size()];
  Vector* result;
  AggregationType aggType;

  for (int i = 0; i < devicePtrs.size(); i++) {
    args[i] = &devicePtrs[i];
  }

  SECTION("inclusive") {
    aggType = AggregationType::InclusiveScan;
    HIP_CHECK(hipMemcpy(d_aggType.ptr(), &aggType, sizeof(aggType), hipMemcpyHostToDevice));
    HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(maxMagnitude), gridDim, blockDim, args, 0, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                        h_result.size_bytes(), hipMemcpyDeviceToHost));
    result = &h_result.host_ptr()[0];
    REQUIRE((result->x == 2 && result->y == 3));
    result = &h_result.host_ptr()[1];
    REQUIRE((result->x == 1 && result->y == 9));
    result = &h_result.host_ptr()[2];
    REQUIRE((result->x == 1 && result->y == 9));
    result = &h_result.host_ptr()[3];
    REQUIRE((result->x == 1 && result->y == 9));
  }

  SECTION("exclusive") {
    aggType = AggregationType::ExclusiveScan;
    HIP_CHECK(hipMemcpy(d_aggType.ptr(), &aggType, sizeof(aggType), hipMemcpyHostToDevice));
    HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(maxMagnitude), gridDim, blockDim, args, 0, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                        h_result.size_bytes(), hipMemcpyDeviceToHost));
    result = &h_result.host_ptr()[0];
    REQUIRE((result->x == 0 && result->y == 0));
    result = &h_result.host_ptr()[1];
    REQUIRE((result->x == 2 && result->y == 3));
    result = &h_result.host_ptr()[2];
    REQUIRE((result->x == 1 && result->y == 9));
    result = &h_result.host_ptr()[3];
    REQUIRE((result->x == 1 && result->y == 9));
  }
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
        result[i] = std::max(lhs[i], rhs[i]);
    }

    return result;
  }
};

template <size_t NumElems, class Functor>
__global__ void applyFunctor(ArrayContainer<NumElems>* result)
{
  cg::thread_block mygroup = cg::this_thread_block();
  auto mytile = cg::tiled_partition<NumElems>(mygroup);
  __shared__ ArrayContainer<NumElems> input;
  Functor op;

  if (threadIdx.x < NumElems) {
    input[threadIdx.x] = threadIdx.x;
  }

  mytile.sync();

  if (threadIdx.x < NumElems) {
    *result = cg::reduce(mytile, input, op);
  }
}

template <size_t NumElems, class Functor>
__global__ void applyScanFunctor(ArrayContainer<NumElems>** result,
                                 AggregationType* aggType)
{
  cg::thread_block mygroup = cg::this_thread_block();
  auto mytile = cg::tiled_partition<NumElems>(mygroup);
  __shared__ ArrayContainer<NumElems> input;
  Functor op;
  unsigned int tid = threadIdx.x +
                     threadIdx.y * blockDim.x +
                     threadIdx.z * blockDim.x * blockDim.y;

  if (threadIdx.x < NumElems) {
    input[threadIdx.x] = threadIdx.x;
  }

  mytile.sync();

  if (threadIdx.x < NumElems) {
    switch (*aggType) {
    case AggregationType::InclusiveScan:
      *(result[tid]) = cg::inclusive_scan(mytile, input, op);
      break;
    case AggregationType::ExclusiveScan:
      *(result[tid]) = cg::exclusive_scan(mytile, input, op);
      break;
    default:
      assert(false && "AggregationType not supported");
    }
  }
}

// tests aggregations of arguments of different sizes (types <= 32 bytes are accepted)
template <size_t NumElems, template <size_t> class Functor>
void testArgsDifferentSizesReduce()
{
  LinearAllocGuard<ArrayContainer<NumElems>> h_result(LinearAllocs::malloc, sizeof(ArrayContainer<NumElems>));
  LinearAllocGuard<ArrayContainer<NumElems>> d_result(LinearAllocs::hipMalloc, sizeof(ArrayContainer<NumElems>));
  dim3 gridDim = { 1 };
  dim3 blockDim = { 32 };
  void* devicePtr = d_result.ptr();
  void* args[] = { &devicePtr };
  ArrayContainer<NumElems>* result;

  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(applyFunctor<NumElems, Functor<NumElems>>),
                                       gridDim, blockDim, args, 0, nullptr));
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
    testArgsDifferentSizesReduce<NumElems / 2, Functor>();
  }
}

// in this case, as opposed to the reduction, were we were only saving one result per reduction,
// we save one result per lane. i.e. we save one ArrayContainer per lane, as each lane will return
// a different ArrayContainer value
template <size_t NumElems, template <size_t> class Functor>
void testArgsDifferentSizesScan(AggregationType aggType)
{
  int wavefrontSize = getWarpSize();
  // one per lane
  LinearAllocGuard<ArrayContainer<NumElems>> h_result[64];
  LinearAllocGuard<ArrayContainer<NumElems>> d_result[64];

  // as we cannot pass d_result directly (because it is an array of
  // LinearAllocGuards), we convert to an array of raw device pointers
  ArrayContainer<NumElems>* h_devicePtrs[64];
  LinearAllocGuard<ArrayContainer<NumElems>*> d_devicePtrs(LinearAllocs::hipMalloc, sizeof(void*) * 64);
  LinearAllocGuard<AggregationType> d_aggType(LinearAllocs::hipMalloc, sizeof(AggregationType));
  dim3 gridDim = { 1 };
  dim3 blockDim = { 32 };
  ArrayContainer<NumElems>* result;

  for (int i = 0; i < wavefrontSize; i++) {
    h_result[i] = LinearAllocGuard<ArrayContainer<NumElems>>(LinearAllocs::malloc,
                                                             sizeof(ArrayContainer<NumElems>));
    d_result[i] = LinearAllocGuard<ArrayContainer<NumElems>>(LinearAllocs::hipMalloc,
                                                             sizeof(ArrayContainer<NumElems>));
  }

  for (int i = 0; i < wavefrontSize; i++) {
    h_devicePtrs[i] = d_result[i].ptr();
  }

  HIP_CHECK(hipMemcpy(d_aggType.ptr(), &aggType, sizeof(aggType), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_devicePtrs.ptr(),
                      h_devicePtrs,
                      d_devicePtrs.size_bytes(),
                      hipMemcpyHostToDevice));

  std::array<void*, 2> devicePtrs = { d_devicePtrs.ptr(), d_aggType.ptr() };
  void* args[devicePtrs.size()];

  for (int i = 0; i < devicePtrs.size(); i++) {
    args[i] = &devicePtrs[i];
  }
  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(applyScanFunctor<NumElems, Functor<NumElems>>),
                                       gridDim, blockDim, args, 0, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());

  for (int i = 0; i < wavefrontSize; i++) {
    HIP_CHECK(hipMemcpy(h_result[i].host_ptr(), d_result[i].ptr(),
                        h_result[i].size_bytes(), hipMemcpyDeviceToHost));
  }

  INFO("T is of size: " << NumElems);

  for (int laneId = 0; laneId < NumElems; laneId++) {
    result = &(h_result[laneId].host_ptr()[0]);

    if (aggType == AggregationType::InclusiveScan) {
      INFO("Lane: " << laneId);

      if (std::is_same<Functor<NumElems>, Sum<NumElems>>::value) {
        // the result can be calculated with an arithmetic series formula, modulo 256
        // (we do overflow unsigned char for some indices, but that is defined behaviour)
        REQUIRE((*result)[laneId] == (((laneId + 1) * (laneId + laneId) / 2) % 256));
      } else {
        REQUIRE((*result)[laneId] == laneId);
      }
    } else if (aggType == AggregationType::ExclusiveScan) {
      if (std::is_same<Functor<NumElems>, Sum<NumElems>>::value) {
        INFO("Lane: " << laneId);

        // the result can be calculated with an arithmetic series formula, modulo 256
        // (we do overflow unsigned char for some indices, but that is defined behaviour)
        if (laneId == 0) {
          REQUIRE((*result)[laneId] == 0);
        } else {
          REQUIRE((*result)[laneId] == (((laneId) * (laneId + laneId) / 2) % 256));
        }
      } else {
        if (laneId == 0) {
          REQUIRE((*result)[laneId] == 0);
        } else {
          REQUIRE((*result)[laneId] == laneId);
        }
      }
    }
  }

  if constexpr (NumElems > 1) {
    testArgsDifferentSizesScan<NumElems / 2, Functor>(aggType);
  }
}

// we allow any reduction size of T of up to 32 bytes; this tests that reduction works with user-defined
// types in that range; including non-powers of two
HIP_TEST_CASE(Unit_Thread_Block_Tile_Reduce_All_Parameter_Sizes)
{
  SECTION("sum") {
    testArgsDifferentSizesReduce<32, Sum>();
  }

  SECTION("max") {
    testArgsDifferentSizesReduce<32, Max>();
  }
}

TEST_CASE(Unit_Thread_Block_Tile_Scan_All_Parameter_Sizes)
{
  SECTION("sum") {
    testArgsDifferentSizesScan<32, Sum>(AggregationType::InclusiveScan);
    testArgsDifferentSizesScan<32, Sum>(AggregationType::ExclusiveScan);
  }

  SECTION("max") {
    testArgsDifferentSizesScan<32, Max>(AggregationType::InclusiveScan);
    testArgsDifferentSizesScan<32, Max>(AggregationType::ExclusiveScan);
  }
}

struct Point {
    int x;
    int y;

    __host__ __device__ Point operator+(const Point& rhs) const
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
HIP_TEST_CASE(Unit_Thread_Block_Tile_Reduce_Standard_Op_Custom_Type)
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

HIP_TEMPLATE_TEST_CASE(Unit_Thread_Block_Coalesced_Reduce_arithmetic, int, unsigned int, long long,
                   unsigned long long, float, half, double)
{
  std::tuple<cooperative_groups::plus<TestType>,
             cooperative_groups::less<TestType>,
             cooperative_groups::greater<TestType>> ops;

  if (getWarpSize() == 32) {
    runAggregationRandomForOps<true, TestType, 32>(AggregationType::Reduce, ops);
  } else {
    runAggregationRandomForOps<true, TestType, 64>(AggregationType::Reduce, ops);
  }
}

HIP_TEMPLATE_TEST_CASE(Unit_Thread_Block_Coalesced_Reduce_boolean, int, unsigned int, long long, unsigned long long)
{
  std::tuple<cooperative_groups::bit_and<TestType>,
             cooperative_groups::bit_or<TestType>,
             cooperative_groups::bit_xor<TestType>> ops;
  if (getWarpSize() == 32) {
    runAggregationRandomForOps<true, TestType, 32>(AggregationType::Reduce, ops);
  } else {
    runAggregationRandomForOps<true, TestType, 64>(AggregationType::Reduce, ops);
  }
}

template <size_t TileSize, AggregationType AggType, class Op, class T>
void __global__ simpleScan(T* result)
{
  T value = threadIdx.x;
  cg::thread_block mygroup = cg::this_thread_block();
  auto mytile = cg::tiled_partition<TileSize>(mygroup);
  Op op;

  if constexpr (AggType == AggregationType::InclusiveScan) {
    result[threadIdx.x] = cg::inclusive_scan(mytile, value, op);
  } else if constexpr (AggType == AggregationType::ExclusiveScan) {
    result[threadIdx.x] = cg::exclusive_scan(mytile, value, op);
  } else if constexpr (AggType == AggregationType::InclusiveScanDefault) {
    result[threadIdx.x] = cg::inclusive_scan(mytile, value);
  } else if constexpr (AggType == AggregationType::ExclusiveScanDefault) {
    result[threadIdx.x] = cg::exclusive_scan(mytile, value);
  } else {
    assert(false && "Unexpected aggType");
  }
}

/// @tparam Op either std::plus or std::less
template <size_t TileSize, AggregationType AggType, class Op, class T>
void testScanForTileSize()
{
  LinearAllocGuard<T> h_result(LinearAllocs::malloc, sizeof(T) * getWarpSize());
  LinearAllocGuard<T> d_result(LinearAllocs::hipMalloc, h_result.size_bytes());
  dim3 gridDim = { 1 };
  dim3 blockDim = { static_cast<unsigned short>(getWarpSize()) };
  T id = 0;
  T accum = 0;
  int pos = 0;
  LinearAllocGuard<AggregationType> d_aggType(LinearAllocs::hipMalloc, sizeof(AggregationType));
  std::array<void*, 2> devicePtrs = { d_result.ptr(), d_aggType.ptr() };
  void* args[devicePtrs.size()];
  AggregationType aggType = AggType;

  if (!isInclusive(aggType)) {
    scanIdentity<T, Op>(id);
  }

  accum = id;

  for (int i = 0; i < devicePtrs.size(); i++) {
    args[i] = &devicePtrs[i];
  }

  HIP_CHECK(hipMemcpy(d_aggType.ptr(), &aggType, sizeof(AggType), hipMemcpyHostToDevice));
  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(simpleScan<TileSize, AggType, Op, T>),
                                       gridDim,
                                       blockDim,
                                       args,
                                       0,
                                       nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                      h_result.size_bytes(), hipMemcpyDeviceToHost));

  while (pos < getWarpSize()) {
    for (int i = 0; i < TileSize; i++) {
      UNSCOPED_INFO("Index: " << pos + i << " tile size: " << TileSize);

      if (isInclusive(aggType)) {
        accum += pos + i;
        INFO("Inclusive scan");
        REQUIRE(h_result.host_ptr()[pos + i] == accum);
      } else {
        using ComparisonType = typename std::conditional<std::is_same<T, half>::value, float, T>::type;
        ComparisonType result, expected;

        INFO("Exclusive scan");

        if constexpr (std::is_same<T, half>::value) {
          // Catch2 cannot print fp16 if there is an error
          result = __half2float(h_result.host_ptr()[pos + i]);
          expected = __half2float(__half2float(accum));
        } else {
          result = h_result.host_ptr()[pos + i];
          expected = accum;
        }

        INFO("Lane: " << (pos + i));
        REQUIRE(result == expected);

        if (i == 0) {
          accum = id;
        }

        if constexpr (std::is_same<Op, cooperative_groups::less<T>>::value) {
          accum = std::min(__half2float(accum), __half2float(pos + i));
        } else {
          accum += pos + i;
        }
      }
    }

    accum = id;
    pos += TileSize;
  }
}

TEST_CASE(Unit_Thread_Block_Tile_Inclusive_Scan_Basic)
{
  using Op = cooperative_groups::plus<int>;
  static constexpr AggregationType AggType = AggregationType::InclusiveScan;

  testScanForTileSize<1, AggType, Op, int>();
  testScanForTileSize<2, AggType, Op, int>();
  testScanForTileSize<4, AggType, Op, int>();
  testScanForTileSize<8, AggType, Op, int>();
  testScanForTileSize<16, AggType, Op, int>();
  testScanForTileSize<32, AggType, Op, int>();
  testScanForTileSize<32, AggregationType::InclusiveScanDefault, Op, int>();

  if (getWarpSize() == 64) {
    testScanForTileSize<64, AggType, Op, int>();
  }
}

TEMPLATE_TEST_CASE(Unit_Thread_Block_Tile_Exclusive_Scan_Basic, int, half)
{
  static constexpr AggregationType AggType = AggregationType::ExclusiveScan;

  SECTION("plus") {
    using Op = cooperative_groups::plus<TestType>;
    testScanForTileSize<1, AggType, Op, TestType>();
    testScanForTileSize<2, AggType, Op, TestType>();
    testScanForTileSize<4, AggType, Op, TestType>();
    testScanForTileSize<8, AggType, Op, TestType>();
    testScanForTileSize<16, AggType, Op, TestType>();
    testScanForTileSize<32, AggType, Op, TestType>();
    testScanForTileSize<32, AggregationType::ExclusiveScanDefault, Op, int>();

    if (getWarpSize() == 64) {
      testScanForTileSize<64, AggType, Op, TestType>();
    }
  }

  SECTION("less") {
    using Op = cooperative_groups::less<TestType>;
    testScanForTileSize<1, AggType, Op, TestType>();
    testScanForTileSize<2, AggType, Op, TestType>();
    testScanForTileSize<4, AggType, Op, TestType>();
    testScanForTileSize<8, AggType, Op, TestType>();
    testScanForTileSize<16, AggType, Op, TestType>();
    testScanForTileSize<32, AggType, Op, TestType>();

    if (getWarpSize() == 64) {
      testScanForTileSize<64, AggType, Op, TestType>();
    }
  }
}

// for all the tile sizes and all input types, using random input values, calculates the scan
// values. Additionally, randomly make some threads not participate for the coalesced_threads case
TEMPLATE_TEST_CASE(Unit_Thread_Block_Tile_Scan_Random_arithmetic,  int, unsigned int, long long,
                   unsigned long long, float, half, double)
{
  std::tuple<cooperative_groups::plus<TestType>,
             cooperative_groups::less<TestType>,
             cooperative_groups::greater<TestType>> types;

  SECTION("inclusive") {
    if (getWarpSize() == 32) {
      runAggregationRandomForOps<false, TestType, 32>(AggregationType::InclusiveScan, types);
    } else {
      runAggregationRandomForOps<false, TestType, 64>(AggregationType::InclusiveScan, types);
    }
  }

  SECTION("exclusive") {
    if (getWarpSize() == 32) {
      runAggregationRandomForOps<false, TestType, 32>(AggregationType::ExclusiveScan, types);
    } else {
      runAggregationRandomForOps<false, TestType, 64>(AggregationType::ExclusiveScan, types);
    }
  }
}

TEMPLATE_TEST_CASE(Unit_Thread_Block_Tile_Scan_Random_boolean, int, unsigned int, long long,
                   unsigned long long)
{
  std::tuple<cooperative_groups::bit_and<TestType>,
             cooperative_groups::bit_or<TestType>,
             cooperative_groups::bit_xor<TestType>> types;

  SECTION("inclusive") {
    if (getWarpSize() == 32) {
      runAggregationRandomForOps<false, TestType, 32>(AggregationType::InclusiveScan, types);
    } else {
      runAggregationRandomForOps<false, TestType, 64>(AggregationType::InclusiveScan, types);
    }
  }

  SECTION("exclusive") {
    if (getWarpSize() == 32) {
      runAggregationRandomForOps<false, TestType, 32>(AggregationType::ExclusiveScan, types);
    } else {
      runAggregationRandomForOps<false, TestType, 64>(AggregationType::ExclusiveScan, types);
    }
  }
}

// make sures that tiled blocks that use the y or z dimension work correctly
TEST_CASE(Unit_Thread_Block_Tile_2D_3D_Blocks)
{
  int wavefrontSize = getWarpSize();

  SECTION("2D") {
    dim3 blockDim {1, 2, static_cast<unsigned int>(wavefrontSize / 2)};

    if (wavefrontSize == 32) {
      runAggregationRandomForType<false, std::plus<int>, int, 32>(AggregationType::Reduce, blockDim);
      runAggregationRandomForType<false, std::plus<int>, int, 32>(AggregationType::InclusiveScan, blockDim);
      runAggregationRandomForType<false, std::plus<int>, int, 32>(AggregationType::ExclusiveScan, blockDim);
    } else {
      runAggregationRandomForType<false, std::plus<int>, int, 64>(AggregationType::Reduce, blockDim);
      runAggregationRandomForType<false, std::plus<int>, int, 64>(AggregationType::InclusiveScan, blockDim);
      runAggregationRandomForType<false, std::plus<int>, int, 64>(AggregationType::ExclusiveScan, blockDim);
    }
  }

  SECTION("3D") {
    dim3 blockDim {static_cast<unsigned int>(wavefrontSize) / 2, 2, 1};

    if (wavefrontSize == 32) {
      runAggregationRandomForType<false, std::plus<int>, int, 32>(AggregationType::Reduce, blockDim);
      runAggregationRandomForType<false, std::plus<int>, int, 32>(AggregationType::InclusiveScan, blockDim);
      runAggregationRandomForType<false, std::plus<int>, int, 32>(AggregationType::ExclusiveScan, blockDim);
    } else {
      runAggregationRandomForType<false, std::plus<int>, int, 64>(AggregationType::Reduce, blockDim);
      runAggregationRandomForType<false, std::plus<int>, int, 64>(AggregationType::InclusiveScan, blockDim);
      runAggregationRandomForType<false, std::plus<int>, int, 64>(AggregationType::ExclusiveScan, blockDim);
    }
  }
}

TEMPLATE_TEST_CASE(Unit_Thread_Block_Coalesced_Scan_arithmetic, int, unsigned int, long long,
                   unsigned long long, float, half, double)
{
  std::tuple<cooperative_groups::plus<TestType>,
             cooperative_groups::less<TestType>,
             cooperative_groups::greater<TestType>> ops;

  SECTION("inclusive") {
    if (getWarpSize() == 32) {
      runAggregationRandomForOps<true, TestType, 32>(AggregationType::InclusiveScan, ops);
    } else {
      runAggregationRandomForOps<true, TestType, 64>(AggregationType::InclusiveScan, ops);
    }
  }

  SECTION("exclusive") {
    if (getWarpSize() == 32) {
      runAggregationRandomForOps<true, TestType, 32>(AggregationType::ExclusiveScan, ops);
    } else {
      runAggregationRandomForOps<true, TestType, 64>(AggregationType::ExclusiveScan, ops);
    }
  }
}

TEMPLATE_TEST_CASE(Unit_Thread_Block_Coalesced_Scan_boolean, int, unsigned int, long long,
                   unsigned long long)
{
  std::tuple<cooperative_groups::bit_and<TestType>,
             cooperative_groups::bit_or<TestType>,
             cooperative_groups::bit_xor<TestType>> ops;

  SECTION("inclusive") {
    if (getWarpSize() == 32) {
      runAggregationRandomForOps<true, TestType, 32>(AggregationType::InclusiveScan, ops);
    } else {
      runAggregationRandomForOps<true, TestType, 64>(AggregationType::InclusiveScan, ops);
    }
  }

  SECTION("exclusive") {
    if (getWarpSize() == 32) {
      runAggregationRandomForOps<true, TestType, 32>(AggregationType::ExclusiveScan, ops);
    } else {
      runAggregationRandomForOps<true, TestType, 64>(AggregationType::ExclusiveScan, ops);
    }
  }
}

void __global__ binaryPartitionCoalesced(int* out, int* ranks)
{
   if (threadIdx.x >= warpSize / 2) {
     // this group will contain the upper part of the threads
     auto coalesced = cg::coalesced_threads();

     // this group is subsequently split in two: on even and odd indexes
     auto partitioned = cg::binary_partition(coalesced, threadIdx.x % 2);

     ranks[threadIdx.x] = partitioned.thread_rank();
     out[threadIdx.x] = cg::inclusive_scan(partitioned, threadIdx.x);
   } else {
     ranks[threadIdx.x] = -1;
     out[threadIdx.x] = -1;
   }
}

template <unsigned int WarpSize>
void __global__ binaryPartitionTiled(int* out, int* ranks)
{
  cg::thread_block mygroup = cg::this_thread_block();
  auto tile = cg::tiled_partition<WarpSize / 2>(mygroup);

  if (tile.meta_group_rank() == 1) {
    auto partitioned = cg::binary_partition(tile, threadIdx.x % 2);
    ranks[threadIdx.x] = partitioned.thread_rank();
    out[threadIdx.x] = cg::inclusive_scan(partitioned, threadIdx.x);
  } else {
    ranks[threadIdx.x] = -1;
    out[threadIdx.x] = -1;
  }
}

TEST_CASE(Unit_Thread_Block_Scan_partition)
{
  int wavefrontSize = getWarpSize();
  LinearAllocGuard<int> h_result(LinearAllocs::malloc, sizeof(int) * wavefrontSize);
  LinearAllocGuard<int> d_result(LinearAllocs::hipMalloc, h_result.size_bytes());
  LinearAllocGuard<int> h_ranks(LinearAllocs::malloc, sizeof(int) * wavefrontSize);
  LinearAllocGuard<int> d_ranks(LinearAllocs::hipMalloc, h_ranks.size_bytes());
  dim3 gridDim = { 1 };
  dim3 blockDim = { static_cast<unsigned short>(getWarpSize()) };
  void* resultsPtr = d_result.ptr();
  void* ranksPtr = d_ranks.ptr();
  void* args[] = { &resultsPtr, &ranksPtr };
  auto checkResults = [&]() {
    int accumEven = 0;
    int accumOdd = 0;

    for (int laneId = 0; laneId < getWarpSize(); laneId++) {
      if (laneId >= wavefrontSize / 2) {
        if (laneId % 2 == 0) {
          accumEven += laneId;
        } else {
          accumOdd += laneId;
        }

        INFO("laneId: " << laneId);
        REQUIRE(h_ranks.host_ptr()[laneId] == (laneId - wavefrontSize / 2) / 2);
        REQUIRE(h_result.host_ptr()[laneId] == (laneId % 2 ? accumOdd : accumEven));
      } else {
        INFO("laneId: " << laneId);
        REQUIRE(h_ranks.host_ptr()[laneId] == -1);
        REQUIRE(h_result.host_ptr()[laneId] == -1);
      }
    }
  };

  // the result of both sections must be the same; in one we use coalesced_threads to sub-divide
  // into higher-order and lower-order threads, in another we just use cg::tiled_partition to do the
  // equivalent
  SECTION("coalesced") {
    HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(binaryPartitionCoalesced),
                                        gridDim,
                                        blockDim,
                                        args,
                                        0,
                                        nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(h_ranks.host_ptr(), d_ranks.ptr(),
                        h_ranks.size_bytes(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                        h_result.size_bytes(), hipMemcpyDeviceToHost));
    checkResults();
  }

  SECTION("tiled") {
    void* kernelPtr = reinterpret_cast<void*>(wavefrontSize == 32?
                                              binaryPartitionTiled<32> : binaryPartitionTiled<64>);
    HIP_CHECK(hipLaunchCooperativeKernel(kernelPtr,
                                         gridDim,
                                         blockDim,
                                         args,
                                         0,
                                         nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(h_ranks.host_ptr(), d_ranks.ptr(),
                        h_ranks.size_bytes(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_result.host_ptr(), d_result.ptr(),
                        h_result.size_bytes(), hipMemcpyDeviceToHost));
    checkResults();
  }
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
