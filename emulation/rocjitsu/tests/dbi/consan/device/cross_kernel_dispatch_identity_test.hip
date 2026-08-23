// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Dispatch-local LDS lifetime distilled from batched Sharktank decode.
/// Distinct kernels launched on separate queues may reuse the same queue-local
/// AQL packet ID, LDS bytes, and local epoch numbers, but their records must
/// never be interpreted as one dispatch.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kMaximumThreads = 128u;
constexpr uint32_t kBatchedPairs = 32u;

struct DeviceState {
  uint32_t observed;
  uint32_t producer_control[kMaximumThreads];
  uint32_t consumer_control[kMaximumThreads];
};

__global__ __launch_bounds__(128) void dispatch_local_producer_for_consan(DeviceState *state) {
  __shared__ uint32_t payload;
  const uint32_t tid = threadIdx.x;
  if (tid == warpSize)
    consan_device_test::store_lds_word(&payload, 0xd1510001u);
  state->producer_control[tid] = 0xd1511000u | tid;
}

template <bool Synchronize>
__global__ __launch_bounds__(128) void dispatch_local_consumer_for_consan(DeviceState *state) {
  __shared__ uint32_t payload;
  const uint32_t tid = threadIdx.x;
  if (tid == 0u)
    consan_device_test::store_lds_word(&payload, 0xd1520002u);
  if constexpr (Synchronize)
    __syncthreads();
  if (tid == warpSize)
    state->observed = consan_device_test::load_lds_word(&payload);
  state->consumer_control[tid] = 0xd1521000u | tid;
}

void run_cross_kernel_dispatch_identity(bool synchronize) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  ASSERT_LE(thread_count, kMaximumThreads);

  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemset(device_state, 0, sizeof(DeviceState)));

  hipStream_t producer_stream = nullptr;
  hipStream_t consumer_stream = nullptr;
  HIP_ASSERT(hipStreamCreate(&producer_stream));
  HIP_ASSERT(hipStreamCreate(&consumer_stream));

  for (uint32_t pair = 0; pair < kBatchedPairs; ++pair) {
    dispatch_local_producer_for_consan<<<1, thread_count, 0, producer_stream>>>(device_state);
    HIP_ASSERT(hipGetLastError());
    if (synchronize) {
      dispatch_local_consumer_for_consan<true>
          <<<1, thread_count, 0, consumer_stream>>>(device_state);
    } else {
      dispatch_local_consumer_for_consan<false>
          <<<1, thread_count, 0, consumer_stream>>>(device_state);
    }
    HIP_ASSERT(hipGetLastError());
  }
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipStreamDestroy(producer_stream));
  HIP_ASSERT(hipStreamDestroy(consumer_stream));

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(result), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  for (uint32_t tid = 0; tid < thread_count; ++tid) {
    EXPECT_EQ(result.producer_control[tid], 0xd1511000u | tid) << "thread=" << tid;
    EXPECT_EQ(result.consumer_control[tid], 0xd1521000u | tid) << "thread=" << tid;
  }
  if (synchronize)
    EXPECT_EQ(result.observed, 0xd1520002u);
}

TEST(ConSanDeviceCrossKernelDispatchIdentityTest, Correct) {
  run_cross_kernel_dispatch_identity(true);
}

TEST(ConSanDeviceCrossKernelDispatchIdentityTest, Incorrect) {
  run_cross_kernel_dispatch_identity(false);
}

} // namespace
