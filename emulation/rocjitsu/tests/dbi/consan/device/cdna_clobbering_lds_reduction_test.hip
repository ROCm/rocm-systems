// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Max-then-sum reduction distilled from PyTorch softmax kernels whose
/// compiler-generated LDS loads overwrite their own address VGPR. The paired
/// workloads differ only in the publication barrier. CDNA3/4 additionally
/// preserve a live AccVGPR value; wave32 RDNA3/4 and CDNA5 preserve an
/// independent ordinary VGPR across the same target-native tied load.

#include "consan_device_test_support.h"

namespace {

#if defined(RJ_CONSAN_CLOBBERING_REDUCTION_WAVE32)
constexpr uint32_t kWaveSize = 32u;
#else
constexpr uint32_t kWaveSize = 64u;
#endif
constexpr uint32_t kThreadCount = 2u * kWaveSize;

#if defined(RJ_CONSAN_CLOBBERING_REDUCTION_MODERN_LDS)
#define RJ_CONSAN_CLOBBERING_REDUCTION_LOAD "ds_load_b32 %0, %0\n\t"
#define RJ_CONSAN_CLOBBERING_REDUCTION_WAIT "s_wait_dscnt 0\n\t"
#else
#define RJ_CONSAN_CLOBBERING_REDUCTION_LOAD "ds_read_b32 %0, %0\n\t"
#define RJ_CONSAN_CLOBBERING_REDUCTION_WAIT "s_waitcnt lgkmcnt(0)\n\t"
#endif

struct DeviceState {
  uint32_t synchronize;
  uint32_t maximum;
  uint32_t denominator;
  uint32_t ordinary_live[kThreadCount];
  uint32_t accumulator_live[kThreadCount];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ uint32_t load_lds_clobbering_address(volatile uint32_t *address) {
  uint32_t address_and_value =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  // Tying the output to the address produces the compiler idiom at issue:
  //     ds_read_b32 vN, vN
  // The loaded value replaces the byte address in the same physical VGPR.
  asm volatile(RJ_CONSAN_CLOBBERING_REDUCTION_LOAD RJ_CONSAN_DEVICE_LARGE_CAVE
                   RJ_CONSAN_CLOBBERING_REDUCTION_WAIT
               : "+v"(address_and_value)
               :
               : "memory");
  return address_and_value;
}

__global__
__launch_bounds__(kThreadCount) void clobbering_lds_reduction_for_consan(DeviceState *state) {
  __shared__ volatile uint32_t partial[2];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t wave = tid / kWaveSize;
  const uint32_t sentinel = 0xc1000000u | tid;
  register uint32_t ordinary_live asm("v23") = sentinel;
#if defined(RJ_CONSAN_CLOBBERING_REDUCTION_WAVE32)
  register uint32_t accumulator_live_register asm("v24") = sentinel;
  asm volatile("" : "+v"(accumulator_live_register) : : "memory");
#else
  asm volatile("v_accvgpr_write_b32 a0, %0\n\t" : : "v"(ordinary_live) : "a0", "memory");
#endif

  if (lane == 0u) {
    consan_device_test::store_lds_word(&partial[wave], wave == 0u ? 8u : 32u);
  }
  if (state->synchronize != 0u)
    __syncthreads();

  if (tid == 0u) {
    const uint32_t first = load_lds_clobbering_address(&partial[0]);
    const uint32_t second = load_lds_clobbering_address(&partial[1]);
    state->maximum = first > second ? first : second;
    state->denominator = first + second;
  }
  uint32_t accumulator_live = 0u;
#if defined(RJ_CONSAN_CLOBBERING_REDUCTION_WAVE32)
  accumulator_live = accumulator_live_register;
#else
  asm volatile("v_accvgpr_read_b32 %0, a0\n\t" : "=v"(accumulator_live) : : "a0", "memory");
#endif
  state->ordinary_live[tid] = ordinary_live;
  state->accumulator_live[tid] = accumulator_live;
  state->control[tid] = 0xc10b0000u | tid;
}

void run_clobbering_lds_reduction(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  clobbering_lds_reduction_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0u; tid < kThreadCount; ++tid) {
    EXPECT_EQ(result.ordinary_live[tid], 0xc1000000u | tid) << "thread=" << tid;
    EXPECT_EQ(result.accumulator_live[tid], 0xc1000000u | tid) << "thread=" << tid;
    EXPECT_EQ(result.control[tid], 0xc10b0000u | tid) << "thread=" << tid;
  }
  if (synchronize) {
    EXPECT_EQ(result.maximum, 32u);
    EXPECT_EQ(result.denominator, 40u);
  }
}

TEST(ConSanDeviceClobberingLdsReductionTest, Correct) { run_clobbering_lds_reduction(true); }

TEST(ConSanDeviceClobberingLdsReductionTest, Incorrect) { run_clobbering_lds_reduction(false); }

} // namespace

#undef RJ_CONSAN_CLOBBERING_REDUCTION_LOAD
#undef RJ_CONSAN_CLOBBERING_REDUCTION_WAIT
