// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// These kernels are deliberately ill-formed. Their tests remain disabled by
// default and must only be launched through consan_fault_runner.py in
// destructive mode.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}

namespace {

#define HIP_ASSERT(command)                                                                        \
  do {                                                                                             \
    const hipError_t error = (command);                                                            \
    ASSERT_EQ(error, hipSuccess) << hipGetErrorString(error);                                      \
  } while (false)

#define RJ_CONSAN_MALFORMED_SYNC_NOP_CAVE                                                          \
  ".rept 128\n\t"                                                                                  \
  "s_nop 0\n\t"                                                                                    \
  ".endr\n\t"

__device__ __forceinline__ uint32_t malformed_sync_lds_store(uint32_t *lds, uint32_t value) {
  const auto address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds) & 0xFFFFu);
  asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_MALFORMED_SYNC_NOP_CAVE
               :
               : "v"(address), "v"(value)
               : "memory");
  return address;
}

__device__ __forceinline__ uint32_t malformed_sync_lds_load(uint32_t address) {
  uint32_t value = 0;
  asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_MALFORMED_SYNC_NOP_CAVE
               : "=v"(value)
               : "v"(address)
               : "memory");
  return value;
}

__global__ __launch_bounds__(64) void divergent_workgroup_barrier(uint32_t *out) {
  __shared__ uint32_t shared_value;
  const uint32_t address = malformed_sync_lds_store(&shared_value, 0xD100u);
  if (threadIdx.x < 32)
    __syncthreads();
  out[threadIdx.x] = malformed_sync_lds_load(address) + threadIdx.x;
}

__global__ __launch_bounds__(64) void unmatched_barrier_signal(uint32_t *out) {
  __shared__ uint32_t shared_value;
  const uint32_t address = malformed_sync_lds_store(&shared_value, 0x5100u);
  asm volatile("s_barrier_signal -1" ::: "memory");
  out[threadIdx.x] = malformed_sync_lds_load(address) + threadIdx.x;
}

__global__ __launch_bounds__(64) void unmatched_barrier_wait(uint32_t *out) {
  __shared__ uint32_t shared_value;
  const uint32_t address = malformed_sync_lds_store(&shared_value, 0xA100u);
  asm volatile("s_barrier_wait -1" ::: "memory");
  out[threadIdx.x] = malformed_sync_lds_load(address) + threadIdx.x;
}

template <typename Kernel> void dispatch_and_check(Kernel kernel, uint32_t base) {
  constexpr size_t kThreads = 64;
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kThreads * sizeof(uint32_t)));

  kernel<<<1, kThreads>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::array<uint32_t, kThreads> host_out{};
  HIP_ASSERT(hipMemcpy(host_out.data(), device_out, sizeof(host_out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  for (size_t i = 0; i < host_out.size(); ++i)
    EXPECT_EQ(host_out[i], base + i);
}

TEST(ConSanMalformedSyncTest, DISABLED_DivergentWorkgroupBarrier) {
  dispatch_and_check(divergent_workgroup_barrier, 0xD100u);
}

TEST(ConSanMalformedSyncTest, DISABLED_UnmatchedBarrierSignal) {
  dispatch_and_check(unmatched_barrier_signal, 0x5100u);
}

TEST(ConSanMalformedSyncTest, DISABLED_UnmatchedBarrierWait) {
  dispatch_and_check(unmatched_barrier_wait, 0xA100u);
}

} // namespace
