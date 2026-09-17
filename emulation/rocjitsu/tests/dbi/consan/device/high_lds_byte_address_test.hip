// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Regression for packed watchpoints aliasing LDS offsets separated by 64 KiB
// and turning a two-byte store into a three-byte access. Exercise the emitted
// probe, not just the host decoder; keep a real high-address race as a control.
#include "consan_device_test_support.h"

#include <dlfcn.h>

namespace {

struct HighLdsState {
  uint32_t base;
  uint32_t stride;
  uint16_t observed[128];
};

__global__ void high_lds_byte_address_for_consan(HighLdsState *state) {
  extern __shared__ uint16_t storage[];
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t wave = threadIdx.x / warpSize;
  const uint32_t offset = state->base + wave * state->stride + 2u * lane;
  const uint32_t address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&storage[offset / 2u]) & 0xfffffu);
  const uint32_t value = 1u + threadIdx.x;
#if defined(__gfx1250__)
  asm volatile("ds_store_b16 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
#else
  asm volatile("ds_write_b16 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
#endif
               :
               : "v"(address), "v"(value)
               : "memory");
  __syncthreads();
  state->observed[threadIdx.x] = storage[offset / 2u];
}

void run_high_lds_byte_address(bool race) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  hipDeviceProp_t properties{};
  HIP_ASSERT(hipGetDeviceProperties(&properties, 0));
  ASSERT_GE(properties.sharedMemPerBlock, wave_size == 32u ? 0x41000u : 0x21000u);
  for (uint32_t high : {0x10000u, 0x20000u, 0x40000u}) {
    if (high + 0x1000u > properties.sharedMemPerBlock)
      continue;
    HighLdsState state{};
    state.base = race ? high + 0x800u : 0x800u;
    state.stride = race ? 0u : high;
    HighLdsState *device = nullptr;
    HIP_ASSERT(hipMalloc(&device, sizeof(state)));
    HIP_ASSERT(hipMemcpy(device, &state, sizeof(state), hipMemcpyHostToDevice));
    high_lds_byte_address_for_consan<<<1, 2u * wave_size, high + 0x1000u>>>(device);
    HIP_ASSERT(hipGetLastError());
    HIP_ASSERT(hipDeviceSynchronize());
    HIP_ASSERT(hipMemcpy(&state, device, sizeof(state), hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(device));
    if (!race) {
      for (uint32_t tid = 0; tid < 2u * wave_size; ++tid)
        EXPECT_EQ(state.observed[tid], tid + 1u);
    }
  }
}

TEST(ConSanDeviceHighLdsByteAddressTest, Correct) { run_high_lds_byte_address(false); }
TEST(ConSanDeviceHighLdsByteAddressTest, Incorrect) { run_high_lds_byte_address(true); }

void run_high_lds_generation_rollover(bool race) {
  HIP_ASSERT(hipInit(0));
  using Advance = bool (*)(uint64_t);
  Advance advance = nullptr;
  // Baseline cases have no hook. Instrumented cases must expose the test API.
  if (std::getenv("HSA_TOOLS_LIB") != nullptr) {
    void *hook = dlopen(std::getenv("HSA_TOOLS_LIB"), RTLD_LAZY | RTLD_NOLOAD);
    ASSERT_NE(hook, nullptr) << dlerror();
    advance = reinterpret_cast<Advance>(dlsym(hook, "rj_dbi_test_advance_report_generation"));
    dlclose(hook); // ROCr retains its own reference throughout the test.
    ASSERT_NE(advance, nullptr);
    // Include a nonzero upper 32-bit word as well as compact-tag rollover.
    const uint64_t next = (uint64_t{1} << 32u) + (uint64_t{1} << 14u) + (race ? 1u : 0u);
    ASSERT_TRUE(advance(next - 1u));
    EXPECT_FALSE(advance(next - 2u)); // identities may never rewind
  }
  run_high_lds_byte_address(race);
  if (advance != nullptr) {
    // The code object's live allocation must prevent counter manipulation.
    EXPECT_FALSE(advance(uint64_t{1} << 40u));
  }
}

TEST(ConSanDeviceHighLdsGenerationRolloverTest, Correct) {
  run_high_lds_generation_rollover(false);
}
TEST(ConSanDeviceHighLdsGenerationRolloverTest, Incorrect) {
  run_high_lds_generation_rollover(true);
}

} // namespace
