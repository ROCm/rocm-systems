// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// CDNA Stream-K last-arriver publication at the E2E failure resource shape:
/// all 256 ordinary VGPRs, a dynamic private frame, and LDS accesses in a
/// shared noinline helper. MFMA/AccVGPR boundary pressure is kept in the
/// adjacent MFMA pair because v255 and a live accumulator cannot coexist in
/// CDNA's unified physical register file.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kWaveSize = 64u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;

struct DeviceState {
  uint32_t synchronize;
  uint32_t turn;
  uint32_t handoff[2];
  uint32_t tickets[2];
  uint32_t observed;
  uint32_t private_value[kThreadCount];
};

__device__ __noinline__ void publish_partial(volatile uint32_t *address, uint32_t value) {
  consan_device_test::store_lds_word(address, value);
}

__device__ __noinline__ uint32_t consume_partial(volatile uint32_t *address) {
  return consan_device_test::load_lds_word(address);
}

__device__ __forceinline__ uint32_t fetch_add_acq_rel(uint32_t *address) {
  const uint32_t one = 1u;
  uint32_t old = 0u;
  asm volatile("buffer_wbl2 sc1\n\t"
               "s_waitcnt vmcnt(0)\n\t"
               "global_atomic_add %0, %1, %2, off sc0 sc1\n\t"
               "s_waitcnt vmcnt(0)\n\t"
               "buffer_inv sc1\n\t"
               "s_waitcnt vmcnt(0)\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               : "=&v"(old)
               : "v"(address), "v"(one)
               : "memory");
  return old;
}

__global__
__launch_bounds__(kThreadCount) void cdna_full_bank_streamk_for_consan(DeviceState *state) {
  __shared__ volatile uint32_t partials[2];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t wave = tid / kWaveSize;
  const uint32_t private_count = (tid & 7u) + 1u;
  auto *private_words =
      static_cast<volatile uint32_t *>(__builtin_alloca(private_count * sizeof(uint32_t)));
  for (uint32_t i = 0; i < private_count; ++i)
    private_words[i] = 0xc5000000u ^ (tid * 17u + i);
  const uint32_t selected = private_words[(tid * 3u + 1u) % private_count];

  asm volatile("v_mov_b32 v255, %0\n\t" : : "v"(selected) : "v255");

  if (lane == 0u) {
    volatile uint32_t *turn = &state->turn;
    while (*turn != wave) {
    }
    publish_partial(&partials[wave], 0x500u + 23u * wave);
    uint32_t *handoff = &state->handoff[0];
    if (state->synchronize == 0u && wave + 1u == 2u)
      handoff = &state->handoff[1];
    const uint32_t ticket = fetch_add_acq_rel(handoff);
    state->tickets[wave] = ticket;
    *turn = wave + 1u;
    if (ticket + 1u == 2u)
      state->observed = consume_partial(&partials[0]) + consume_partial(&partials[1]);
  }

  state->private_value[tid] = selected;
}

uint32_t expected_private_value(uint32_t tid) {
  const uint32_t count = (tid & 7u) + 1u;
  return 0xc5000000u ^ (tid * 17u + (tid * 3u + 1u) % count);
}

void run_cdna_full_bank_streamk(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  initial.handoff[1] = 1u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  cdna_full_bank_streamk_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  EXPECT_EQ(result.turn, 2u);
  EXPECT_EQ(result.handoff[0], synchronize ? 2u : 1u);
  EXPECT_EQ(result.handoff[1], synchronize ? 1u : 2u);
  EXPECT_EQ(result.tickets[0], 0u);
  EXPECT_EQ(result.tickets[1], 1u);
  for (uint32_t tid = 0; tid < kThreadCount; ++tid) {
    EXPECT_EQ(result.private_value[tid], expected_private_value(tid)) << "thread=" << tid;
  }
  if (synchronize)
    EXPECT_EQ(result.observed, 0xa17u);
}

TEST(ConSanDeviceCdnaFullBankStreamkTest, Correct) { run_cdna_full_bank_streamk(true); }
TEST(ConSanDeviceCdnaFullBankStreamkTest, Incorrect) { run_cdna_full_bank_streamk(false); }

} // namespace
