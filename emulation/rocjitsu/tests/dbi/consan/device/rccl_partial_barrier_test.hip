// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Reusable partial-workgroup barrier distilled from RCCL's device primitives.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kThreadCount = 128u;
constexpr uint32_t kWorkerWaveCount = 2u;
constexpr uint32_t kStageCount = 2u;
constexpr uint32_t kMaximumWaveSize = 64u;
constexpr uint32_t kTailLaneCount = 3u;
constexpr uint32_t kFinalBarrierTarget = 6u;
constexpr uint32_t kObservedCanary = 0xfeedc0deu;
constexpr uint32_t kControlCanary = 0xdeadbeefu;

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_size;
  uint32_t active_lanes;
  uint32_t barrier_counts[3];
  uint4 observed[kStageCount][kMaximumWaveSize];
  uint32_t control[kThreadCount];
};

__host__ __device__ uint4 rank_filled_chunk(uint32_t stage, uint32_t producer_wave, uint32_t lane) {
  const uint32_t base = 0x1000u * (stage + 1u) + 0x100u * (producer_wave + 1u) + 16u * lane;
  return make_uint4(base, base + 3u, base + 7u, base + 11u);
}

__host__ __device__ uint4 reduce_rank_filled_chunks(uint32_t stage, uint32_t lane) {
  const uint4 first = rank_filled_chunk(stage, 0u, lane);
  const uint4 second = rank_filled_chunk(stage, 1u, lane);
  return make_uint4(first.x + second.x, first.y + second.y, first.z + second.z, first.w + second.w);
}

__device__ void rccl_partial_barrier(volatile uint32_t *barriers, uint32_t wave, uint32_t lane,
                                     uint32_t &barrier_next, bool acquire_publication) {
  if (lane != 0u)
    return;

  barrier_next += kWorkerWaveCount;
  __hip_atomic_fetch_add(&barriers[0], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
  // Wait until both releases are complete using relaxed operations. This is
  // only a scheduling rendezvous: it creates no happens-before edge for the
  // payload and therefore cannot substitute for the acquire under test. It
  // does make each RCCL-style polling generation execute exactly one semantic
  // acquire, independent of wave scheduling or emulator timing.
  __hip_atomic_fetch_add(&barriers[2], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  while (__hip_atomic_load(&barriers[2], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) <
         barrier_next) {
    __builtin_amdgcn_s_sleep(1u);
  }
  // Keep the release and first polling load as separate instrumentation
  // boundaries. gfx1250 otherwise clauses the adjacent memory operations;
  // the neutral scalar instruction does not alter RCCL's ordering contract.
  asm volatile("s_nop 0" ::: "memory");

  // RCCL's barrier_generic has every wave leader publish to one generation
  // counter and poll that same counter with acquire. The incorrect member
  // changes only the final consumer acquire address. Both leaders still
  // release to the real counter, so neither wave can hang.
  volatile uint32_t *poll = &barriers[0];
  if (wave == 1u && !acquire_publication)
    poll = &barriers[1];
  if (__hip_atomic_load(poll, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) < barrier_next)
    __builtin_trap();
  asm volatile("s_wakeup" ::: "memory");
}

__global__
__launch_bounds__(kThreadCount) void rccl_partial_barrier_for_consan(DeviceState *state) {
  __shared__ __align__(16) uint4 chunks[kWorkerWaveCount][kMaximumWaveSize];
  __shared__ volatile uint32_t barriers[3];

  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0x7b000000u | tid;

  if (tid == 0u) {
    barriers[0] = 0u;
    // The incorrect member polls this already-ready but unrelated generation
    // counter. Its acquire completes without importing either worker's
    // release sequence, matching a wrong-address partial-barrier failure.
    barriers[1] = kFinalBarrierTarget;
    barriers[2] = 0u;
  }
  __syncthreads();

  // On wave32 targets the other two waves model RCCL's non-worker portion of
  // a block. On wave64 targets the same contract naturally occupies the whole
  // 128-thread block.
  if (wave >= kWorkerWaveCount)
    return;

  uint32_t barrier_next = 0u;
  uint4 first_stage{};

  if (lane < state->active_lanes)
    chunks[wave][lane] = rank_filled_chunk(0u, wave, lane);
  asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  rccl_partial_barrier(barriers, wave, lane, barrier_next, true);

  if (wave == 1u && lane < state->active_lanes) {
    const uint4 first = chunks[0][lane];
    const uint4 second = chunks[1][lane];
    first_stage =
        make_uint4(first.x + second.x, first.y + second.y, first.z + second.z, first.w + second.w);
  }

  // The second generation protects reuse of the same communication scratch.
  rccl_partial_barrier(barriers, wave, lane, barrier_next, true);

  if (lane < state->active_lanes)
    chunks[wave][lane] = rank_filled_chunk(1u, wave, lane);
  asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  rccl_partial_barrier(barriers, wave, lane, barrier_next, state->synchronize || wave == 0u);

  if (wave == 1u && lane < state->active_lanes) {
    const uint4 first = chunks[0][lane];
    const uint4 second = chunks[1][lane];
    const uint4 second_stage =
        make_uint4(first.x + second.x, first.y + second.y, first.z + second.z, first.w + second.w);
    state->observed[0][lane] = first_stage;
    state->observed[1][lane] = second_stage;
  }
  if (wave == 0u && lane == 0u) {
    state->barrier_counts[0] = barriers[0];
    state->barrier_counts[1] = barriers[1];
    state->barrier_counts[2] = barriers[2];
  }
}

void expect_uint4_eq(const uint4 &actual, const uint4 &expected, uint32_t stage, uint32_t lane) {
  EXPECT_EQ(actual.x, expected.x) << "stage=" << stage << " lane=" << lane << " word=0";
  EXPECT_EQ(actual.y, expected.y) << "stage=" << stage << " lane=" << lane << " word=1";
  EXPECT_EQ(actual.z, expected.z) << "stage=" << stage << " lane=" << lane << " word=2";
  EXPECT_EQ(actual.w, expected.w) << "stage=" << stage << " lane=" << lane << " word=3";
}

void run_rccl_partial_barrier(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  ASSERT_TRUE(wave_size == 32u || wave_size == 64u);

  DeviceState state{};
  state.synchronize = synchronize;
  state.wave_size = wave_size;
  state.active_lanes = wave_size - kTailLaneCount;
  state.barrier_counts[0] = kControlCanary;
  state.barrier_counts[1] = kControlCanary;
  state.barrier_counts[2] = kControlCanary;
  for (uint32_t stage = 0u; stage < kStageCount; ++stage)
    for (uint32_t lane = 0u; lane < kMaximumWaveSize; ++lane)
      state.observed[stage][lane] =
          make_uint4(kObservedCanary, kObservedCanary, kObservedCanary, kObservedCanary);
  for (uint32_t &control : state.control)
    control = kControlCanary;

  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  rccl_partial_barrier_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.wave_size, wave_size);
  EXPECT_EQ(state.active_lanes, wave_size - kTailLaneCount);
  EXPECT_EQ(state.barrier_counts[0], kFinalBarrierTarget);
  EXPECT_EQ(state.barrier_counts[1], kFinalBarrierTarget);
  EXPECT_EQ(state.barrier_counts[2], kFinalBarrierTarget);
  for (uint32_t tid = 0u; tid < kThreadCount; ++tid)
    EXPECT_EQ(state.control[tid], 0x7b000000u | tid) << "thread=" << tid;

  for (uint32_t stage = 0u; stage < kStageCount; ++stage) {
    for (uint32_t lane = 0u; lane < kMaximumWaveSize; ++lane) {
      if (synchronize && lane < state.active_lanes) {
        expect_uint4_eq(state.observed[stage][lane], reduce_rank_filled_chunks(stage, lane), stage,
                        lane);
      } else if (lane >= state.active_lanes) {
        expect_uint4_eq(
            state.observed[stage][lane],
            make_uint4(kObservedCanary, kObservedCanary, kObservedCanary, kObservedCanary), stage,
            lane);
      }
    }
  }
}

TEST(ConSanDeviceRcclPartialBarrierTest, Correct) { run_rccl_partial_barrier(true); }

TEST(ConSanDeviceRcclPartialBarrierTest, Incorrect) { run_rccl_partial_barrier(false); }

} // namespace
