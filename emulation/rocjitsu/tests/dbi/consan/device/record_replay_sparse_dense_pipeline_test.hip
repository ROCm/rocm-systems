// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Sparse-sampled Record/Replay over an access-dense, barriered LDS pipeline.
/// The shape distills the large static LDS inventories and repeated tile
/// publication in Qwen attention and TP2 matmul kernels. Correct/Incorrect
/// differ only in the publication barrier, while the dense warmup remains
/// ordered and result-checked on CDNA3/4/5 and RDNA3/4.

#include "consan_device_test_support.h"

namespace {

#if defined(RJ_CONSAN_SPARSE_DENSE_WAVE32)
constexpr uint32_t kWaveSize = 32u;
#else
constexpr uint32_t kWaveSize = 64u;
#endif
constexpr uint32_t kThreadCount = 2u * kWaveSize;
constexpr uint32_t kWorkgroupCount = 2u;

#if defined(RJ_CONSAN_SPARSE_DENSE_MODERN_LDS)
#define RJ_CONSAN_SPARSE_DENSE_STORE "ds_store_b32"
#define RJ_CONSAN_SPARSE_DENSE_LOAD "ds_load_b32"
#define RJ_CONSAN_SPARSE_DENSE_WAIT "s_wait_dscnt 0\n\t"
#else
#define RJ_CONSAN_SPARSE_DENSE_STORE "ds_write_b32"
#define RJ_CONSAN_SPARSE_DENSE_LOAD "ds_read_b32"
#define RJ_CONSAN_SPARSE_DENSE_WAIT "s_waitcnt lgkmcnt(0)\n\t"
#endif

struct DeviceState {
  uint32_t synchronize;
  uint32_t warmup[kThreadCount];
  uint32_t observed[kWaveSize];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ uint32_t lds_address(volatile uint32_t *address) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
}

__device__ __forceinline__ uint32_t dense_ordered_roundtrip(volatile uint32_t *address,
                                                            uint32_t value) {
  const uint32_t address_value = lds_address(address);
  uint32_t observed = 0u;
  asm volatile(".rept 320\n\t" RJ_CONSAN_SPARSE_DENSE_STORE
               " %1, %2\n\t" RJ_CONSAN_SPARSE_DENSE_LOAD " %0, %1\n\t" RJ_CONSAN_SPARSE_DENSE_WAIT
               ".endr\n\t"
               : "=v"(observed)
               : "v"(address_value), "v"(value)
               : "memory");
  return observed;
}

__device__ __forceinline__ void store_lds(volatile uint32_t *address, uint32_t value) {
  const uint32_t address_value = lds_address(address);
  asm volatile(RJ_CONSAN_SPARSE_DENSE_STORE " %0, %1\n\t"
               :
               : "v"(address_value), "v"(value)
               : "memory");
}

__device__ __forceinline__ uint32_t load_lds(volatile uint32_t *address) {
  const uint32_t address_value = lds_address(address);
  uint32_t value = 0u;
  asm volatile(RJ_CONSAN_SPARSE_DENSE_LOAD " %0, %1\n\t" RJ_CONSAN_SPARSE_DENSE_WAIT
               : "=v"(value)
               : "v"(address_value)
               : "memory");
  return value;
}

__global__ __launch_bounds__(kThreadCount) void record_replay_sparse_dense_pipeline_for_consan(
    DeviceState *states) {
  DeviceState *state = &states[blockIdx.x];
  __shared__ uint32_t warmup[kThreadCount];
  __shared__ uint32_t tile[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t wave = tid / kWaveSize;
  const uint32_t warmup_value = 0xd1000000u | tid;

  state->warmup[tid] = dense_ordered_roundtrip(&warmup[tid], warmup_value);
  if (wave == 0u)
    store_lds(&tile[lane], 0xd2000000u | lane);
  if (state->synchronize)
    __syncthreads();
  if (wave == 1u)
    state->observed[lane] = load_lds(&tile[lane]);
  state->control[tid] = 0xd3000000u | tid;
}

void run_record_replay_sparse_dense_pipeline(bool synchronize) {
  std::array<DeviceState, kWorkgroupCount> states{};
  for (DeviceState &state : states)
    state.synchronize = synchronize;
  DeviceState *device_states = nullptr;
  HIP_ASSERT(hipMalloc(&device_states, sizeof(states)));
  HIP_ASSERT(hipMemcpy(device_states, states.data(), sizeof(states), hipMemcpyHostToDevice));
  record_replay_sparse_dense_pipeline_for_consan<<<kWorkgroupCount, kThreadCount>>>(device_states);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(states.data(), device_states, sizeof(states), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_states));

  for (uint32_t workgroup = 0u; workgroup < kWorkgroupCount; ++workgroup) {
    const DeviceState &state = states[workgroup];
    for (uint32_t tid = 0u; tid < kThreadCount; ++tid) {
      EXPECT_EQ(state.warmup[tid], 0xd1000000u | tid)
          << "workgroup=" << workgroup << " thread=" << tid;
      EXPECT_EQ(state.control[tid], 0xd3000000u | tid)
          << "workgroup=" << workgroup << " thread=" << tid;
    }
    if (synchronize) {
      for (uint32_t lane = 0u; lane < kWaveSize; ++lane)
        EXPECT_EQ(state.observed[lane], 0xd2000000u | lane)
            << "workgroup=" << workgroup << " lane=" << lane;
    }
  }
}

TEST(ConSanDeviceRecordReplaySparseDensePipelineTest, Correct) {
  run_record_replay_sparse_dense_pipeline(true);
}

TEST(ConSanDeviceRecordReplaySparseDensePipelineTest, Incorrect) {
  run_record_replay_sparse_dense_pipeline(false);
}

} // namespace

#undef RJ_CONSAN_SPARSE_DENSE_STORE
#undef RJ_CONSAN_SPARSE_DENSE_LOAD
#undef RJ_CONSAN_SPARSE_DENSE_WAIT
