// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Repeated, multi-stream launch envelope for gfx11 loaded-image and report
/// generation identity. Every dispatch executes the same kernel object but
/// owns distinct output. This tests lifecycle isolation; it does not claim
/// that ConSan diagnoses races between separate kernels.

#include "consan_device_test_support.h"

#include <array>

namespace {

constexpr uint32_t kStreams = 4u;
constexpr uint32_t kRounds = 2u;
constexpr uint32_t kDispatches = kStreams * kRounds;
constexpr uint32_t kMaximumThreads = 64u;

struct DispatchState {
  uint32_t observed;
  uint32_t control[kMaximumThreads];
};

template <bool Synchronize>
__global__ void gfx11_repeated_image_for_consan(DispatchState *states, uint32_t dispatch) {
  __shared__ uint32_t payload;
  const uint32_t tid = threadIdx.x;
  const uint32_t wave_size = warpSize;
  if (tid == 0u)
    consan_device_test::store_lds_word(&payload, 0xd1500000u | dispatch);
  if constexpr (Synchronize)
    __syncthreads();
  if (tid == wave_size)
    states[dispatch].observed = consan_device_test::load_lds_word(&payload);
  states[dispatch].control[tid] = 0xd1000000u | (dispatch << 8u) | tid;
}

void run_gfx11_dispatch_identity(bool synchronize) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  ASSERT_LE(thread_count, kMaximumThreads);
  DispatchState *device_states = nullptr;
  HIP_ASSERT(hipMalloc(&device_states, sizeof(DispatchState) * kDispatches));
  HIP_ASSERT(hipMemset(device_states, 0, sizeof(DispatchState) * kDispatches));

  std::array<hipStream_t, kStreams> streams{};
  for (hipStream_t &stream : streams)
    HIP_ASSERT(hipStreamCreate(&stream));
  for (uint32_t round = 0; round < kRounds; ++round) {
    for (uint32_t stream = 0; stream < kStreams; ++stream) {
      const uint32_t dispatch = round * kStreams + stream;
      if (synchronize) {
        gfx11_repeated_image_for_consan<true>
            <<<1, thread_count, 0, streams[stream]>>>(device_states, dispatch);
      } else {
        gfx11_repeated_image_for_consan<false>
            <<<1, thread_count, 0, streams[stream]>>>(device_states, dispatch);
      }
      HIP_ASSERT(hipGetLastError());
    }
  }
  for (hipStream_t stream : streams)
    HIP_ASSERT(hipStreamSynchronize(stream));

  std::array<DispatchState, kDispatches> result{};
  HIP_ASSERT(hipMemcpy(result.data(), device_states, sizeof(result), hipMemcpyDeviceToHost));
  for (hipStream_t stream : streams)
    HIP_ASSERT(hipStreamDestroy(stream));
  HIP_ASSERT(hipFree(device_states));

  for (uint32_t dispatch = 0; dispatch < kDispatches; ++dispatch) {
    for (uint32_t tid = 0; tid < thread_count; ++tid) {
      EXPECT_EQ(result[dispatch].control[tid], 0xd1000000u | (dispatch << 8u) | tid)
          << "dispatch=" << dispatch << " thread=" << tid;
    }
    if (synchronize)
      EXPECT_EQ(result[dispatch].observed, 0xd1500000u | dispatch) << "dispatch=" << dispatch;
  }
}

TEST(ConSanDeviceGfx11DispatchIdentityTest, Correct) { run_gfx11_dispatch_identity(true); }
TEST(ConSanDeviceGfx11DispatchIdentityTest, Incorrect) { run_gfx11_dispatch_identity(false); }

} // namespace
