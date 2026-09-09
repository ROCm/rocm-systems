// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Compiler-generated regression for #11019. A wide scalar clobber raises the
/// gfx950 descriptor to 106 SGPRs without leaving a persistent dispatch-ID
/// pair. Record/Replay must compose its private dispatch capture with whatever
/// owner/epoch/workgroup representation persistent placement selects.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kMaxThreadCount = 128u;

struct DeviceState {
  uint32_t synchronize;
  float input[kMaxThreadCount];
  float output[kMaxThreadCount];
  uint32_t abi_control[kMaxThreadCount];
};

#if defined(__gfx1250__)
#define RJ_CONSAN_PRIVATE_ENTRY_WAIT "s_wait_dscnt 0\n\t"
#else
#define RJ_CONSAN_PRIVATE_ENTRY_WAIT "s_waitcnt lgkmcnt(0)\n\t"
#endif

// Preserve every portable ordinary SGPR across one LDS access. Together with
// forced VGPR spilling this selects private dispatch identity plus an
// entry-scalar preservation window, the minimal combination that exposed the
// queue-preload ABI regression in the tree-atomic-or E2E kernel.
#define RJ_CONSAN_PRIVATE_ENTRY_SCALAR_USES                                                      \
  ".irp r,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,"         \
  "29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,"         \
  "57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,"         \
  "85,86,87,88,89\n\t"                                                                         \
  "s_mov_b32 s\\r, s\\r\n\t"                                                                    \
  ".endr\n\t"

#define RJ_CONSAN_PRIVATE_ENTRY_SCALAR_CLOBBERS                                                  \
  "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27",  \
      "s28", "s29", "s30", "s31", "s36", "s37", "s38", "s39", "s40", "s41", "s42",       \
      "s43", "s44", "s45", "s46", "s47", "s48", "s49", "s50", "s51", "s52", "s53",       \
      "s54", "s55", "s56", "s57", "s58", "s59", "s60", "s61", "s62", "s63", "s64",       \
      "s65", "s66", "s67", "s68", "s69", "s70", "s71", "s72", "s73", "s74", "s75",       \
      "s76", "s77", "s78", "s79", "s80", "s81", "s82", "s83", "s84", "s85", "s86",       \
      "s87", "s88", "s89"

__global__
__launch_bounds__(kMaxThreadCount) void dispatch_id_sgpr_pressure_for_consan(DeviceState *state) {
  __shared__ float tile[64];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;

  if (wave == 0u)
    tile[lane] = state->input[lane];
  if (state->synchronize)
    __syncthreads();

  // Match the downstream reproducer. These declarations make the compiler
  // account for s16:s99 while leaving the registers dead at later sites; the
  // point is persistent dispatch identity, not site-local scalar liveness.
  asm volatile("" ::
                   : "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26",
                     "s27", "s28", "s29", "s30", "s31", "s32", "s33", "s34", "s35", "s36", "s37",
                     "s38", "s39", "s40", "s41", "s42", "s43", "s44", "s45", "s46", "s47", "s48",
                     "s49", "s50", "s51", "s52", "s53", "s54", "s55", "s56", "s57", "s58", "s59",
                     "s60", "s61", "s62", "s63", "s64", "s65", "s66", "s67", "s68", "s69", "s70",
                     "s71", "s72", "s73", "s74", "s75", "s76", "s77", "s78", "s79", "s80", "s81",
                     "s82", "s83", "s84", "s85", "s86", "s87", "s88", "s89", "s90", "s91", "s92",
                     "s93", "s94", "s95", "s96", "s97", "s98", "s99");

  float value = 0.0f;
  if (wave == 1u)
    value = tile[lane] + tile[(lane + 1u) % warpSize];
  __syncthreads();
  if (wave == 0u)
    tile[lane] = state->input[lane] + 1.0f;
  __syncthreads();
  if (wave == 1u)
    state->output[lane] = value + tile[(lane + 2u) % warpSize];
}

__global__ __launch_bounds__(kMaxThreadCount) void
private_entry_abi_spill_for_consan(DeviceState *state) {
  __shared__ volatile uint32_t tile[kMaxThreadCount];
  const uint32_t tid = threadIdx.x;
  const uint32_t address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&tile[tid]) & 0xffffu);
  const uint32_t expected = 0xea950000u | tid;
  uint32_t observed = 0u;
  asm volatile(RJ_CONSAN_PRIVATE_ENTRY_WAIT RJ_CONSAN_PRIVATE_ENTRY_SCALAR_USES
                   "ds_write_b32 %1, %2\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
                   "ds_read_b32 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
                   RJ_CONSAN_PRIVATE_ENTRY_WAIT RJ_CONSAN_PRIVATE_ENTRY_SCALAR_USES
               : "=v"(observed)
               : "v"(address), "v"(expected)
               : RJ_CONSAN_PRIVATE_ENTRY_SCALAR_CLOBBERS, "memory");
  state->abi_control[tid] = observed;
}

void run_dispatch_id_sgpr_pressure(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize = synchronize;
  for (uint32_t tid = 0u; tid < thread_count; ++tid)
    state.input[tid] = static_cast<float>(tid);

  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  dispatch_id_sgpr_pressure_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  if (synchronize) {
    for (uint32_t lane = 0u; lane < wave_size; ++lane) {
      const float expected = static_cast<float>(lane) +
                             static_cast<float>((lane + 1u) % wave_size) +
                             static_cast<float>((lane + 2u) % wave_size) + 1.0f;
      EXPECT_EQ(state.output[lane], expected) << "lane=" << lane;
    }
  }
}

TEST(ConSanDeviceDispatchIdSgprPressureTest, Correct) { run_dispatch_id_sgpr_pressure(true); }

TEST(ConSanDeviceDispatchIdSgprPressureTest, Incorrect) { run_dispatch_id_sgpr_pressure(false); }

TEST(ConSanDeviceDispatchIdSgprPressureTest, PrivateEntryAbiSpill) {
  DeviceState state{};
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  private_entry_abi_spill_for_consan<<<1, kMaxThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0u; tid < kMaxThreadCount; ++tid)
    EXPECT_EQ(state.abi_control[tid], 0xea950000u | tid) << "thread=" << tid;
}

} // namespace

#undef RJ_CONSAN_PRIVATE_ENTRY_SCALAR_CLOBBERS
#undef RJ_CONSAN_PRIVATE_ENTRY_SCALAR_USES
#undef RJ_CONSAN_PRIVATE_ENTRY_WAIT
