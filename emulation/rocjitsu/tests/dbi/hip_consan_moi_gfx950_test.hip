// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_moi_gfx950_test.hip
/// @brief Focused native gfx950 ConSan MOI instrumentation tests.

#include <cstdint>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

namespace {

__global__ __launch_bounds__(64) void gfx950_moi_single_lds_store(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  const uint32_t value = 0x95020000u | tid;
  asm volatile("ds_write_b32 %0, %1\n\t"
               ".rept 128\n\t"
               "s_nop 0\n\t"
               ".endr\n\t"
               :
               : "v"(address), "v"(value)
               : "memory");
  __syncthreads();
  if (tid == 0)
    out[0] = lds[0];
}

__global__ __launch_bounds__(64) void gfx950_moi_forced_spill(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  const uint32_t value = 0x95030000u | tid;
  uint32_t victim0 = 0x10500000u ^ tid;
  uint32_t victim1 = 0x20500000u ^ (tid * 3u);
  uint32_t victim2 = 0x30500000u ^ (tid * 5u);
  uint32_t victim3 = 0x40500000u ^ (tid * 7u);
  asm volatile("ds_write_b32 %4, %5\n\t"
               ".rept 128\n\t"
               "s_nop 0\n\t"
               ".endr\n\t"
               : "+v"(victim0), "+v"(victim1), "+v"(victim2), "+v"(victim3)
               : "v"(address), "v"(value)
               : "memory");
  const uint32_t base = tid * 4u;
  out[base + 0u] = victim0;
  out[base + 1u] = victim1;
  out[base + 2u] = victim2;
  out[base + 3u] = victim3;
}

__global__ __launch_bounds__(64) void gfx950_standalone_report_publication(uint32_t *out) {
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  uint32_t *slot = out + tid;
  const uint32_t value = 0x95040000u ^ tid;
  asm volatile("flat_store_dword %0, %1\n\t"
               "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
               :
               : "v"(slot), "v"(value)
               : "memory");
}

__global__ __launch_bounds__(128) void gfx950_moi_identity_grid(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t local =
      static_cast<uint32_t>(threadIdx.x) +
      static_cast<uint32_t>(blockDim.x) *
          (static_cast<uint32_t>(threadIdx.y) +
           static_cast<uint32_t>(blockDim.y) * static_cast<uint32_t>(threadIdx.z));
  const uint32_t group = static_cast<uint32_t>(blockIdx.x) +
                         static_cast<uint32_t>(gridDim.x) *
                             (static_cast<uint32_t>(blockIdx.y) +
                              static_cast<uint32_t>(gridDim.y) * static_cast<uint32_t>(blockIdx.z));
  if ((local & 63u) == 0u) {
    const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
    const uint32_t value = 0x95050000u | (group << 4u) | (local >> 6u);
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (local == 0u)
    out[group] = 0x95050000u | group;
}

__global__ __launch_bounds__(128) void gfx950_moi_two_wave_race(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  if (tid == 0u) {
    const uint32_t value = 0x95060000u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (tid == 64u) {
    const uint32_t value = 0x95060040u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (tid == 0u)
    out[0] = 0x95060001u;
}

__global__ __launch_bounds__(64) void gfx950_moi_same_wave_program_order(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  if (tid == 0u) {
    uint32_t value = 0x95070000u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
    value = 0x95070001u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
    out[0] = 0x95070001u;
  }
}

__global__ __launch_bounds__(128) void gfx950_moi_static_sampled_selection(uint32_t *out) {
  __shared__ uint32_t lds[2];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  if (tid == 0u) {
    const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
    const uint32_t value = 0x95080000u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (tid == 64u) {
    const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[1]) & 0xffffu);
    const uint32_t value = 0x95080040u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (tid == 0u)
    out[0] = 0x95080001u;
}

TEST(ConSanMoiGfx950Test, SingleLdsStoreCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_single_lds_store<<<1, 64>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out & 0xffff0000u, 0x95020000u);
}

TEST(ConSanMoiGfx950Test, ForcedSpillPreservesLiveVgprs) {
  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kValuesPerThread = 4;
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kThreads * kValuesPerThread * sizeof(uint32_t)));
  gfx950_moi_forced_spill<<<1, kThreads>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint32_t> out(kThreads * kValuesPerThread);
  HIP_ASSERT(
      hipMemcpy(out.data(), device_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    const uint32_t base = tid * kValuesPerThread;
    EXPECT_EQ(out[base + 0u], 0x10500000u ^ tid) << "tid=" << tid;
    EXPECT_EQ(out[base + 1u], 0x20500000u ^ (tid * 3u)) << "tid=" << tid;
    EXPECT_EQ(out[base + 2u], 0x30500000u ^ (tid * 5u)) << "tid=" << tid;
    EXPECT_EQ(out[base + 3u], 0x40500000u ^ (tid * 7u)) << "tid=" << tid;
  }
}

TEST(ConSanMoiGfx950Test, StandaloneReportPublication) {
  constexpr uint32_t kThreads = 64;
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kThreads * sizeof(uint32_t)));
  gfx950_standalone_report_publication<<<1, kThreads>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint32_t> out(kThreads);
  HIP_ASSERT(
      hipMemcpy(out.data(), device_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  for (uint32_t tid = 0; tid < kThreads; ++tid)
    EXPECT_EQ(out[tid], 0x95040000u ^ tid) << "tid=" << tid;
}

TEST(ConSanMoiGfx950Test, StableThreeDimensionalIdentity) {
  constexpr uint32_t kGroups = 8;
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kGroups * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kGroups * sizeof(uint32_t)));
  gfx950_moi_identity_grid<<<dim3(2, 2, 2), dim3(8, 4, 4)>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint32_t> out(kGroups);
  HIP_ASSERT(
      hipMemcpy(out.data(), device_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  for (uint32_t group = 0; group < kGroups; ++group)
    EXPECT_EQ(out[group], 0x95050000u | group) << "group=" << group;
}

TEST(ConSanMoiGfx950Test, TwoWaveRaceCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_two_wave_race<<<1, 128>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out, 0x95060001u);
}

TEST(ConSanMoiGfx950Test, SameWaveProgramOrderCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_same_wave_program_order<<<1, 64>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out, 0x95070001u);
}

TEST(ConSanMoiGfx950Test, StaticSampledSelectionCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_static_sampled_selection<<<1, 128>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out, 0x95080001u);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}
