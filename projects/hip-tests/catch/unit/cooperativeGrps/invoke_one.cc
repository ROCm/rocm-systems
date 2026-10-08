/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

#include <algorithm>
#include <vector>

namespace cg = cooperative_groups;

// Counts how many threads of the block ran the invocable.
static __global__ void invoke_one_block(unsigned int* count) {
  cg::thread_block block = cg::this_thread_block();
  cg::invoke_one(block, [&]() { atomicAdd(count, 1u); });
}

// Same, but the group is a tile, so the invocable runs once per tile.
template <unsigned int TileSize> static __global__ void invoke_one_tile(unsigned int* count) {
  auto tile = cg::tiled_partition<TileSize>(cg::this_thread_block());
  cg::invoke_one(tile, [&]() { atomicAdd(count, 1u); });
}

// Only every third thread enters the branch, so the group is a subset of the wavefront.
static __global__ void invoke_one_coalesced(unsigned int* count) {
  if (threadIdx.x % 3 == 0) {
    cg::coalesced_group active = cg::coalesced_threads();
    cg::invoke_one(active, [&]() { atomicAdd(count, 1u); });
  }
}

// Records the rank of the selected thread.
static __global__ void invoke_one_selected_rank(unsigned int* rank) {
  cg::thread_block block = cg::this_thread_block();
  cg::invoke_one(block, [&]() { *rank = block.thread_rank(); });
}

// Passes arguments through to the invocable.
static __global__ void invoke_one_args(unsigned int* out) {
  cg::thread_block block = cg::this_thread_block();
  cg::invoke_one(
      block, [](unsigned int* o, unsigned int a, unsigned int b) { *o = a + b; }, out, 20u, 22u);
}

// The largest return value the broadcast supports.
struct Big32 {
  unsigned int words[8];
};

// Only threads 5 and above take the branch, so the selected thread is not lane 0.
static __global__ void invoke_one_broadcast_coalesced(unsigned int* out) {
  if (threadIdx.x >= 5) {
    cg::coalesced_group active = cg::coalesced_threads();
    out[threadIdx.x] =
        cg::invoke_one_broadcast(active, [&]() { return 100u + active.thread_rank(); });
  }
}

// Each tile broadcasts the value produced by its own selected thread.
template <unsigned int TileSize> static __global__ void invoke_one_broadcast_tile(unsigned int* out) {
  auto tile = cg::tiled_partition<TileSize>(cg::this_thread_block());
  out[threadIdx.x] = cg::invoke_one_broadcast(tile, [&]() { return threadIdx.x; });
}

// Broadcasts a 32 byte value and checks every word of it.
static __global__ void invoke_one_broadcast_big(unsigned int* out) {
  auto tile = cg::tiled_partition<16>(cg::this_thread_block());
  Big32 value = cg::invoke_one_broadcast(tile, [&]() {
    Big32 v;
    for (unsigned int i = 0; i < 8; ++i) v.words[i] = threadIdx.x + i;
    return v;
  });

  const unsigned int selected = (threadIdx.x / 16) * 16;
  unsigned int matches = 1;
  for (unsigned int i = 0; i < 8; ++i) matches &= (value.words[i] == selected + i);
  out[threadIdx.x] = matches;
}

// Warp aggregated atomic. one atomic for the group, distinct for each thread.
static __global__ void invoke_one_broadcast_aggregated_atomic(unsigned int* counter,
                                                              unsigned int* out) {
  cg::coalesced_group active = cg::coalesced_threads();
  unsigned int base =
      cg::invoke_one_broadcast(active, [&]() { return atomicAdd(counter, active.size()); });
  out[threadIdx.x] = base + active.thread_rank();
}

/**
 * Test Description
 * ------------------------
 *  - Verifies that cg::invoke_one runs the supplied invocable on exactly one thread of the
 *    calling group, for thread_block, thread_block_tile and coalesced_group, and that the
 *    arguments are passed through to the invocable.
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/invoke_one.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 10.2
 */
HIP_TEST_CASE(Unit_cg_invoke_one) {
  const unsigned int warp_size = static_cast<unsigned int>(getWarpSize());
  const unsigned int block_size = warp_size;
  const unsigned int blocks = 4;

  unsigned int* d_result;
  HIP_CHECK(hipMalloc(&d_result, sizeof(unsigned int)));
  HIP_CHECK(hipMemset(d_result, 0, sizeof(unsigned int)));

  unsigned int result = 0;

  SECTION("thread_block") {
    invoke_one_block<<<blocks, block_size>>>(d_result);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(&result, d_result, sizeof(unsigned int), hipMemcpyDeviceToHost));
    REQUIRE(result == blocks);
  }

  SECTION("thread_block_tile<8>") {
    invoke_one_tile<8><<<blocks, block_size>>>(d_result);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(&result, d_result, sizeof(unsigned int), hipMemcpyDeviceToHost));
    REQUIRE(result == blocks * (block_size / 8));
  }

  SECTION("thread_block_tile<16>") {
    invoke_one_tile<16><<<blocks, block_size>>>(d_result);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(&result, d_result, sizeof(unsigned int), hipMemcpyDeviceToHost));
    REQUIRE(result == blocks * (block_size / 16));
  }

  SECTION("coalesced_group") {
    // A single block of one wavefront forms a single coalesced group in the branch.
    invoke_one_coalesced<<<1, block_size>>>(d_result);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(&result, d_result, sizeof(unsigned int), hipMemcpyDeviceToHost));
    REQUIRE(result == 1);
  }

  SECTION("selected thread belongs to the group") {
    // The selection is not guaranteed to be deterministic, only that it is a group member.
    invoke_one_selected_rank<<<1, block_size>>>(d_result);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(&result, d_result, sizeof(unsigned int), hipMemcpyDeviceToHost));
    REQUIRE(result < block_size);
  }

  SECTION("arguments are passed to the invocable") {
    invoke_one_args<<<1, block_size>>>(d_result);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(&result, d_result, sizeof(unsigned int), hipMemcpyDeviceToHost));
    REQUIRE(result == 42);
  }

  HIP_CHECK(hipFree(d_result));
}

/**
 * Test Description
 * ------------------------
 *  - Verifies that cg::invoke_one_broadcast runs the supplied invocable on exactly one thread of
 *    the calling group and distributes the returned value to all threads of the group, for
 *    coalesced_group and thread_block_tile, including a 32 byte return value.
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/invoke_one.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 10.2
 */
HIP_TEST_CASE(Unit_cg_invoke_one_broadcast) {
  const unsigned int warp_size = static_cast<unsigned int>(getWarpSize());
  const unsigned int block_size = warp_size;
  const size_t bytes = sizeof(unsigned int) * block_size;

  unsigned int* d_out;
  HIP_CHECK(hipMalloc(&d_out, bytes));
  HIP_CHECK(hipMemset(d_out, 0, bytes));

  std::vector<unsigned int> out(block_size, 0);

  SECTION("coalesced_group") {
    invoke_one_broadcast_coalesced<<<1, block_size>>>(d_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    for (unsigned int i = 5; i < block_size; ++i) {
      INFO("thread: " << i);
      REQUIRE(out[i] == 100);
    }
  }

  SECTION("thread_block_tile<16>") {
    invoke_one_broadcast_tile<16><<<1, block_size>>>(d_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    for (unsigned int i = 0; i < block_size; ++i) {
      INFO("thread: " << i);
      REQUIRE(out[i] == (i / 16) * 16);
    }
  }

  SECTION("32 byte return value") {
    invoke_one_broadcast_big<<<1, block_size>>>(d_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    for (unsigned int i = 0; i < block_size; ++i) {
      INFO("thread: " << i);
      REQUIRE(out[i] == 1);
    }
  }

  SECTION("warp aggregated atomic") {
    unsigned int* d_counter;
    HIP_CHECK(hipMalloc(&d_counter, sizeof(unsigned int)));
    HIP_CHECK(hipMemset(d_counter, 0, sizeof(unsigned int)));

    invoke_one_broadcast_aggregated_atomic<<<1, block_size>>>(d_counter, d_out);
    HIP_CHECK(hipGetLastError());

    unsigned int counter = 0;
    HIP_CHECK(hipMemcpy(&counter, d_counter, sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(out.data(), d_out, bytes, hipMemcpyDeviceToHost));

    // A single atomic was issued for the whole group.
    REQUIRE(counter == block_size);

    // Every thread got a distinct slot.
    std::sort(out.begin(), out.end());
    for (unsigned int i = 0; i < block_size; ++i) {
      INFO("slot: " << i);
      REQUIRE(out[i] == i);
    }

    HIP_CHECK(hipFree(d_counter));
  }

  HIP_CHECK(hipFree(d_out));
}
