// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Cross-wave prefix allocation distilled from PyTorch's BF16 top-k gather.
/// The same observable contract uses wave32 RDNA/CDNA5 and wave64 CDNA3/4
/// ballot ranks, LDS atomic reservation, lane broadcast, and native paired
/// LDS-load encodings.

#include "consan_device_test_support.h"

#include <algorithm>
#include <array>
#include <vector>

namespace {

constexpr uint32_t kThreadCount = 1024u;
#if defined(RJ_CONSAN_TOPK_WAVE64)
constexpr uint32_t kWaveSize = 64u;
#else
constexpr uint32_t kWaveSize = 32u;
#endif
constexpr uint32_t kWaveCount = kThreadCount / kWaveSize;
constexpr uint32_t kTileWordsPerLane = 64u;
constexpr uint32_t kTileResultWordsPerLane = 8u;
// Keep the same permuted threshold-selection idiom as PyTorch TopK, but select
// enough lanes that Sampled observes the missing-barrier conflict reliably.
constexpr uint32_t kTopKThreshold = 512u;

struct DeviceState {
  uint32_t synchronize;
  uint32_t final_counter;
  uint32_t builtin_final_counter;
  uint32_t direct_prefix[kWaveCount];
  uint32_t broadcast_prefix[kWaveCount];
  uint32_t observed_prefix[kThreadCount];
  uint32_t observed_rank[kThreadCount];
  uint32_t observed_payload[kThreadCount];
  uint32_t observed_builtin_payload[kThreadCount];
  uint32_t observed_tile[kWaveSize * kTileResultWordsPerLane];
  uint32_t control[kThreadCount];
};

__host__ __device__ constexpr uint32_t tile_value(uint32_t lane, uint32_t word) {
  return 0x5a000000u | (lane << 8u) | word;
}

__host__ __device__ constexpr bool topk_candidate(uint32_t tid) {
#if defined(RJ_CONSAN_TOPK_WAVE64)
  // Retain a sparse, permuted TopK tail while filling the first eight waves.
  // The full prefix makes producer slots land on later wave boundaries, so
  // every deterministic Sampled bank has cross-owner producer/consumer
  // overlap rather than relying on an accidental sparse-rank intersection.
  if (tid < 8u * kWaveSize)
    return true;
#endif
  return ((tid * 73u) % kThreadCount) > kTopKThreshold;
}

#if defined(RJ_CONSAN_TOPK_LEGACY_LDS)
#define WAVE32_LDS_WAIT "s_waitcnt lgkmcnt(0)\n\t"
#else
#define WAVE32_LDS_WAIT "s_wait_dscnt 0\n\t"
#endif

__device__ __forceinline__ uint32_t lds_add_return(volatile uint32_t *address, uint32_t value) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  uint32_t result = 0u;
  asm volatile("ds_add_rtn_u32 %0, %1, %2\n\t" WAVE32_LDS_WAIT
               : "=v"(result)
               : "v"(lds_address), "v"(value)
               : "memory");
  return result;
}

__device__ __forceinline__ uint32_t wave_broadcast(uint32_t source_lane, uint32_t value) {
  uint32_t result = 0u;
  asm volatile("ds_bpermute_b32 %0, %1, %2\n\t" WAVE32_LDS_WAIT
               : "=v"(result)
               : "v"(source_lane << 2u), "v"(value)
               : "memory");
  return result;
}

__device__ __forceinline__ void reserve_warp_space(bool candidate, uint32_t &counter,
                                                   uint32_t &start_index, uint32_t &local_rank,
                                                   uint32_t &candidate_count) {
  const unsigned long long ballot = __ballot(candidate);
  candidate_count = static_cast<uint32_t>(__popcll(ballot));
  const uint32_t lane = threadIdx.x % kWaveSize;
  if (candidate_count > 0u && lane == 0u)
    start_index = atomicAdd(&counter, candidate_count);
  start_index = __shfl(start_index, 0);
  const unsigned long long lower_lanes = (1ull << lane) - 1ull;
  local_rank = static_cast<uint32_t>(__popcll(ballot & lower_lanes));
}

#if defined(RJ_CONSAN_TOPK_WAVE64)
#define DEFINE_LDS_LOAD_PAIR(name, offset0, offset1)                                               \
  __device__ __forceinline__ uint2 name(uint32_t address) {                                        \
    register uint32_t value0 asm("v20") = 0u;                                                      \
    register uint32_t value1 asm("v21") = 0u;                                                      \
    asm volatile("ds_read2_b32 v[20:21], %2 offset0:" #offset0 " offset1:" #offset1                \
                 "\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE WAVE32_LDS_WAIT                                \
                 : "+v"(value0), "+v"(value1)                                                      \
                 : "v"(address)                                                                    \
                 : "memory");                                                                      \
    return make_uint2(value0, value1);                                                             \
  }

DEFINE_LDS_LOAD_PAIR(load_pair_0_8, 0, 8)
DEFINE_LDS_LOAD_PAIR(load_pair_16_24, 16, 24)
DEFINE_LDS_LOAD_PAIR(load_pair_32_40, 32, 40)
DEFINE_LDS_LOAD_PAIR(load_pair_48_56, 48, 56)
#else
#define DEFINE_LDS_LOAD_PAIR(name, offset0, offset1)                                               \
  __device__ __forceinline__ uint2 name(uint32_t address) {                                        \
    register uint32_t value0 asm("v20") = 0u;                                                      \
    register uint32_t value1 asm("v21") = 0u;                                                      \
    asm volatile("ds_load_2addr_b32 v[20:21], %2 offset0:" #offset0 " offset1:" #offset1           \
                 "\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE WAVE32_LDS_WAIT                                \
                 : "+v"(value0), "+v"(value1)                                                      \
                 : "v"(address)                                                                    \
                 : "memory");                                                                      \
    return make_uint2(value0, value1);                                                             \
  }

DEFINE_LDS_LOAD_PAIR(load_pair_0_8, 0, 8)
DEFINE_LDS_LOAD_PAIR(load_pair_16_24, 16, 24)
DEFINE_LDS_LOAD_PAIR(load_pair_32_40, 32, 40)
DEFINE_LDS_LOAD_PAIR(load_pair_48_56, 48, 56)
#endif

#undef DEFINE_LDS_LOAD_PAIR

__global__ __launch_bounds__(kThreadCount) void wave32_topk_prefix_for_consan(DeviceState *state) {
  __shared__ uint32_t counter;
  __shared__ uint32_t builtin_counter;
  __shared__ uint32_t payload[kThreadCount];
  __shared__ uint32_t tile[kWaveSize][kTileWordsPerLane];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % kWaveSize;
  const uint32_t wave = tid / kWaveSize;
  const bool candidate = topk_candidate(tid);
  const uint64_t candidate_mask = __ballot(candidate);
  const uint32_t low_rank = __mbcnt_lo(static_cast<uint32_t>(candidate_mask), 0u);
#if defined(RJ_CONSAN_TOPK_WAVE64)
  const uint32_t local_rank = __mbcnt_hi(static_cast<uint32_t>(candidate_mask >> 32u), low_rank);
#else
  const uint32_t local_rank = low_rank;
#endif
  const uint32_t active_count = static_cast<uint32_t>(__popcll(candidate_mask));
  // PyTorch's HIP gatherTopK reserves space from lane zero even when lane zero
  // is not one of the selected candidates, then broadcasts the returned base.
  constexpr uint32_t source_lane = 0u;

  state->control[tid] = 0x32000000u | tid;
  state->observed_rank[tid] = local_rank;
  if (tid == 0u) {
    counter = 0u;
    builtin_counter = 0u;
  }
  if (tid < kThreadCount)
    payload[tid] = 0u;
  for (uint32_t index = tid; index < kWaveSize * kTileWordsPerLane; index += kThreadCount) {
    const uint32_t tile_lane = index / kTileWordsPerLane;
    const uint32_t tile_word = index % kTileWordsPerLane;
    tile[tile_lane][tile_word] = tile_value(tile_lane, tile_word);
  }
  __syncthreads();

  if (wave == 0u) {
    const uint32_t tile_address =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&tile[lane][0]) & 0xffffu);
    const uint2 pair0 = load_pair_0_8(tile_address);
    const uint2 pair1 = load_pair_16_24(tile_address);
    const uint2 pair2 = load_pair_32_40(tile_address);
    const uint2 pair3 = load_pair_48_56(tile_address);
    const uint2 pairs[] = {pair0, pair1, pair2, pair3};
    for (uint32_t pair = 0; pair < 4u; ++pair) {
      state->observed_tile[lane * kTileResultWordsPerLane + 2u * pair] = pairs[pair].x;
      state->observed_tile[lane * kTileResultWordsPerLane + 2u * pair + 1u] = pairs[pair].y;
    }
  }

  uint32_t prefix = 0xdeadc0deu;
  if (lane == source_lane && active_count > 0u) {
    prefix = lds_add_return(&counter, active_count);
    state->direct_prefix[wave] = prefix;
  }
  const uint32_t broadcast = wave_broadcast(source_lane, prefix);
  state->observed_prefix[tid] = broadcast;
  if (lane == 0u)
    state->broadcast_prefix[wave] = broadcast;
  if (candidate)
    consan_device_test::store_lds_word(&payload[broadcast + local_rank], tid + 1u);

  uint32_t builtin_prefix = 0u;
  uint32_t builtin_rank = 0u;
  uint32_t builtin_count = 0u;
  reserve_warp_space(candidate, builtin_counter, builtin_prefix, builtin_rank, builtin_count);
  if (candidate) {
    const uint32_t write_index = builtin_prefix + builtin_rank;
    state->observed_builtin_payload[write_index] = tid + 1u;
  }

  if (state->synchronize)
    __syncthreads();

  state->observed_payload[tid] = consan_device_test::load_lds_word(&payload[tid]);
  if (tid == 0u) {
    state->final_counter = counter;
    state->builtin_final_counter = builtin_counter;
  }
}

void run_wave32_topk_prefix(bool synchronize) {
  DeviceState state{};
  state.synchronize = synchronize;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  wave32_topk_prefix_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  uint32_t total_candidates = 0u;
  std::array<bool, kThreadCount> claimed_slots{};
  for (uint32_t wave = 0; wave < kWaveCount; ++wave) {
    uint32_t candidate_count = 0u;
    for (uint32_t lane = 0; lane < kWaveSize; ++lane)
      candidate_count += topk_candidate(wave * kWaveSize + lane);
    total_candidates += candidate_count;
    if (candidate_count == 0u)
      continue;
    EXPECT_EQ(state.broadcast_prefix[wave], state.direct_prefix[wave]) << "wave=" << wave;
    ASSERT_LE(state.direct_prefix[wave] + candidate_count, kThreadCount) << "wave=" << wave;
    for (uint32_t rank = 0; rank < candidate_count; ++rank) {
      const uint32_t slot = state.direct_prefix[wave] + rank;
      EXPECT_FALSE(claimed_slots[slot]) << "wave=" << wave << " slot=" << slot;
      claimed_slots[slot] = true;
    }
  }

  for (uint32_t tid = 0; tid < kThreadCount; ++tid) {
    const uint32_t lane = tid % kWaveSize;
    const uint32_t wave = tid / kWaveSize;
    uint32_t wave_candidate_count = 0u;
    for (uint32_t candidate_lane = 0; candidate_lane < kWaveSize; ++candidate_lane)
      wave_candidate_count += topk_candidate(wave * kWaveSize + candidate_lane);
    uint32_t expected_rank = 0u;
    for (uint32_t prior_lane = 0; prior_lane < lane; ++prior_lane)
      expected_rank += topk_candidate(wave * kWaveSize + prior_lane);
    EXPECT_EQ(state.observed_rank[tid], expected_rank)
        << "tid=" << tid << " wave=" << wave << " lane=" << lane;
    if (wave_candidate_count > 0u)
      EXPECT_EQ(state.observed_prefix[tid], state.direct_prefix[wave])
          << "tid=" << tid << " wave=" << wave << " lane=" << lane;
  }

  constexpr uint32_t tile_words[] = {0u, 8u, 16u, 24u, 32u, 40u, 48u, 56u};
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    for (uint32_t result = 0; result < kTileResultWordsPerLane; ++result)
      EXPECT_EQ(state.observed_tile[lane * kTileResultWordsPerLane + result],
                tile_value(lane, tile_words[result]))
          << "lane=" << lane << " result=" << result;

  if (synchronize) {
    EXPECT_EQ(state.final_counter, total_candidates);
    EXPECT_EQ(state.builtin_final_counter, total_candidates);
    std::vector<uint32_t> expected_payload;
    std::vector<uint32_t> observed_payload(state.observed_payload,
                                           state.observed_payload + total_candidates);
    std::vector<uint32_t> observed_builtin_payload(
        state.observed_builtin_payload, state.observed_builtin_payload + total_candidates);
    std::vector<uint32_t> all_observed_builtin_payload;
    for (uint32_t slot = 0; slot < kThreadCount; ++slot) {
      if (state.observed_builtin_payload[slot] != 0u) {
        all_observed_builtin_payload.push_back(state.observed_builtin_payload[slot]);
        EXPECT_LT(slot, total_candidates) << "payload=" << state.observed_builtin_payload[slot];
      }
    }
    for (uint32_t tid = 0; tid < kThreadCount; ++tid) {
      if (topk_candidate(tid))
        expected_payload.push_back(tid + 1u);
    }
    std::sort(expected_payload.begin(), expected_payload.end());
    std::sort(observed_payload.begin(), observed_payload.end());
    std::sort(observed_builtin_payload.begin(), observed_builtin_payload.end());
    std::sort(all_observed_builtin_payload.begin(), all_observed_builtin_payload.end());
    EXPECT_EQ(observed_payload, expected_payload);
    EXPECT_EQ(observed_builtin_payload, expected_payload);
    EXPECT_EQ(all_observed_builtin_payload, expected_payload);
  }
  for (uint32_t tid = 0; tid < kThreadCount; ++tid)
    EXPECT_EQ(state.control[tid], 0x32000000u | tid) << "thread=" << tid;
}

TEST(ConSanDeviceTopKPrefixTest, Correct) { run_wave32_topk_prefix(true); }

TEST(ConSanDeviceTopKPrefixTest, Incorrect) { run_wave32_topk_prefix(false); }

} // namespace

#undef WAVE32_LDS_WAIT
