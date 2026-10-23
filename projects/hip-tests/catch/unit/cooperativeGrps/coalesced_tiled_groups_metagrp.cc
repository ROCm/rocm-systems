/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <resource_guards.hh>
#include <utils.hh>


/**
 * @addtogroup coalesced_group thread_block_tile
 * @{
 * @ingroup CooperativeGroupTest
 * meta_group_size()
 * meta_group_rank()
 */

using namespace cooperative_groups;

constexpr auto total_elem = 1 << 16;
constexpr auto block_size = 256;
constexpr auto test_size = 32;

static __global__ void kernel_coalesced_grp(int* mgrpSize, int* mgrpRank) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id % 2 == 0) {
    coalesced_group threadBlockCGTy = coalesced_threads();
    mgrpSize[id] = threadBlockCGTy.meta_group_size();
    mgrpRank[id] = threadBlockCGTy.meta_group_rank();
  } else {
    coalesced_group threadBlockCGTx = coalesced_threads();
    mgrpSize[id] = threadBlockCGTx.meta_group_size();
    mgrpRank[id] = threadBlockCGTx.meta_group_rank();
  }
}

static __global__ void kernel_tiledgrp_threadblk(int* mgrpSize, int* mgrpRank) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  thread_block_tile<test_size> tiledGr = tiled_partition<test_size>(this_thread_block());
  mgrpSize[id] = tiledGr.meta_group_size();
  mgrpRank[id] = tiledGr.meta_group_rank();
}

template <unsigned int tileSize>
static __global__ void kernel_tiledgrp_meta_rank(unsigned int* ranks, bool divergent) {
  const auto block = this_thread_block();
  const auto tile = tiled_partition<tileSize>(block);
  const thread_block_tile<tileSize> erasedTile = tile;
  const auto nestedTile = tiled_partition<tileSize / 2>(tile);
  const auto rank = block.thread_rank();
  if (!divergent || rank % 2 != 0) {
    ranks[3 * rank] = tile.meta_group_rank();
    ranks[3 * rank + 1] = erasedTile.meta_group_rank();
    ranks[3 * rank + 2] = nestedTile.meta_group_rank();
  }
}

template <unsigned int tileSize> static void TiledGroupMetaRankTest(dim3 threads, bool divergent) {
  DYNAMIC_SECTION("Tile size: " << tileSize) {
    constexpr auto rankCount = 3 * block_size;
    constexpr auto allocSize = rankCount * sizeof(unsigned int);
    LinearAllocGuard<unsigned int> ranksDev(LinearAllocs::hipMalloc, allocSize);
    LinearAllocGuard<unsigned int> ranksHost(LinearAllocs::hipHostMalloc, allocSize);
    HIP_CHECK(hipMemset(ranksDev.ptr(), 0xff, allocSize));
    kernel_tiledgrp_meta_rank<tileSize><<<1, threads>>>(ranksDev.ptr(), divergent);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(ranksHost.ptr(), ranksDev.ptr(), allocSize, hipMemcpyDeviceToHost));
    ArrayAllOf(ranksHost.ptr(), rankCount, [divergent](unsigned int index) {
      const auto rank = index / 3;
      if (divergent && rank % 2 == 0) {
        return ~0u;
      }
      return index % 3 == 2 ? (rank % tileSize) / (tileSize / 2) : rank / tileSize;
    });
  }
}

/**
 * Test Description
 * ------------------------
 *    - Verify exact meta-group ranks for typed, type-erased, and nested tiles in 1D, 2D, and
 * 3D blocks, with all lanes active and with only odd lanes active. Cover full-wave and sub-wave
 * tiles on wave32 and wave64 devices.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/coalesced_tiled_groups_metagrp.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_tiled_groups_metagrp_rank) {
  const auto threads = GENERATE(dim3(256), dim3(8, 32), dim3(4, 8, 8));
  const auto divergent = GENERATE(false, true);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  INFO("Odd lanes only: " << divergent);
  TiledGroupMetaRankTest<16>(threads, divergent);
  TiledGroupMetaRankTest<32>(threads, divergent);
#if HT_AMD
  int device = 0;
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&properties, device));
  if (properties.warpSize == 64) {
    TiledGroupMetaRankTest<64>(threads, divergent);
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Verify the values returned by meta_group_size() and
 * meta_group_rank() for thread block tile with thread block
 * group as parent.
 * ------------------------
 *    - catch\unit\cooperativeGrps\coalesced_tiled_groups_metagrp.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_tiled_groups_metagrp_basic) {
  int *mgrpSize_d = nullptr, *mgrpRank_d = nullptr;
  int *mgrpSize_h = nullptr, *mgrpRank_h = nullptr;
  mgrpSize_h = new int[total_elem];
  REQUIRE(mgrpSize_h != nullptr);
  mgrpRank_h = new int[total_elem];
  REQUIRE(mgrpRank_h != nullptr);

  HIP_CHECK(hipMalloc(&mgrpSize_d, total_elem * sizeof(int)));
  HIP_CHECK(hipMalloc(&mgrpRank_d, total_elem * sizeof(int)));
  SECTION("Parent Group = thread block group") {
    hipLaunchKernelGGL(kernel_tiledgrp_threadblk, total_elem / block_size, block_size, 0, 0,
                       mgrpSize_d, mgrpRank_d);
    HIP_CHECK(hipMemcpy(mgrpRank_h, mgrpRank_d, total_elem * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(mgrpSize_h, mgrpSize_d, total_elem * sizeof(int), hipMemcpyDeviceToHost));
    for (int i = 0; i < total_elem; i++) {
      REQUIRE(mgrpRank_h[i] >= 0);
      REQUIRE(mgrpRank_h[i] < (block_size / test_size));
      REQUIRE(mgrpSize_h[i] == (block_size / test_size));
    }
  }
  HIP_CHECK(hipFree(mgrpSize_d));
  HIP_CHECK(hipFree(mgrpRank_d));
  delete[] mgrpSize_h;
  delete[] mgrpRank_h;
}

/**
 * Test Description
 * ------------------------
 *    - Verify the values returned by meta_group_size() and
 * meta_group_rank() for Coalesced Groups.
 * ------------------------
 *    - catch\unit\cooperativeGrps\coalesced_tiled_groups_metagrp.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_coalesced_groups_metagrp_basic) {
  int *mgrpSize_d = nullptr, *mgrpRank_d = nullptr;
  int *mgrpSize_h = nullptr, *mgrpRank_h = nullptr;
  mgrpSize_h = new int[total_elem];
  REQUIRE(mgrpSize_h != nullptr);
  mgrpRank_h = new int[total_elem];
  REQUIRE(mgrpRank_h != nullptr);

  HIP_CHECK(hipMalloc(&mgrpSize_d, total_elem * sizeof(int)));
  HIP_CHECK(hipMalloc(&mgrpRank_d, total_elem * sizeof(int)));

  hipLaunchKernelGGL(kernel_coalesced_grp, total_elem / block_size, block_size, 0, 0, mgrpSize_d,
                     mgrpRank_d);
  HIP_CHECK(hipMemcpy(mgrpRank_h, mgrpRank_d, total_elem * sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(mgrpSize_h, mgrpSize_d, total_elem * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < total_elem; i++) {
    REQUIRE(mgrpRank_h[i] == 0);
    REQUIRE(mgrpSize_h[i] == 1);
  }
  HIP_CHECK(hipFree(mgrpSize_d));
  HIP_CHECK(hipFree(mgrpRank_d));
  delete[] mgrpSize_h;
  delete[] mgrpRank_h;
}

/**
 * End doxygen group CooperativeGroupTest.
 * @}
 */
