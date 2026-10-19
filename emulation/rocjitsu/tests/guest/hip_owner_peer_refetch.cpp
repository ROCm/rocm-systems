// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// A stock-driver, two-GPU coherence litmus. All four words occupy separate
// pages of one owner allocation. The requester uses the ordinary peer mapping;
// there is no guest-side control register or publication protocol.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kInitial = 0x13579bdfu;
constexpr uint32_t kUpdated = 0x2468ace0u;
constexpr size_t kPageBytes = 4096;

__device__ __forceinline__ uint32_t cacheable_owner_load(const uint32_t *owner) {
  // HIP volatile loads carry SCOPE_SYS and become UC in this model. Pin a
  // cacheable load and wait for it without invalidating the owner's cache.
  uint32_t value;
  asm volatile("global_load_b32 %0, %1, off" : "=v"(value) : "v"(owner) : "memory");
  asm volatile("s_wait_loadcnt 0" ::: "memory");
  return value;
}

__global__ void owner_read_wait_refetch(const uint32_t *owner, volatile uint32_t *ready,
                                        volatile uint32_t *done, uint32_t *observed) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;

  const uint32_t first = cacheable_owner_load(owner);
  const uint32_t second = cacheable_owner_load(owner);
  *ready = 1;
  while (*done == 0) {
  }
  const uint32_t fresh = cacheable_owner_load(owner);
  observed[0] = first;
  observed[1] = second;
  observed[2] = fresh;
}

__global__ void requester_write_signal(volatile uint32_t *owner, volatile uint32_t *ready,
                                       volatile uint32_t *done) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;

  while (*ready == 0) {
  }
  *owner = kUpdated;
  __threadfence_system();
  *done = 1;
}

bool step(const char *name, hipError_t status) {
  std::printf("owner-peer-refetch: %s status=%d (%s)\n", name, static_cast<int>(status),
              hipGetErrorString(status));
  std::fflush(stdout);
  return status == hipSuccess;
}

} // namespace

int main() {
  int count = 0;
  if (!step("hipGetDeviceCount", hipGetDeviceCount(&count)) || count != 2)
    return 1;

  uint8_t *allocation = nullptr;
  if (!step("hipSetDevice owner", hipSetDevice(0)) ||
      !step("hipMalloc owner pages", hipMalloc(&allocation, 4 * kPageBytes)) ||
      !step("hipMemset owner pages", hipMemset(allocation, 0, 4 * kPageBytes)))
    return 2;
  auto *owner = reinterpret_cast<uint32_t *>(allocation);
  auto *ready = reinterpret_cast<uint32_t *>(allocation + kPageBytes);
  auto *done = reinterpret_cast<uint32_t *>(allocation + 2 * kPageBytes);
  auto *observed = reinterpret_cast<uint32_t *>(allocation + 3 * kPageBytes);
  if (!step("seed owner", hipMemcpy(owner, &kInitial, sizeof(kInitial), hipMemcpyHostToDevice)))
    return 3;

  int can_access = 0;
  if (!step("hipDeviceCanAccessPeer requester,owner", hipDeviceCanAccessPeer(&can_access, 1, 0)) ||
      !can_access || !step("hipSetDevice requester", hipSetDevice(1)) ||
      !step("hipDeviceEnablePeerAccess owner", hipDeviceEnablePeerAccess(0, 0)))
    return 4;

  std::printf("owner-peer-refetch: owner=%p ready=%p done=%p observed=%p\n",
              static_cast<void *>(owner), static_cast<void *>(ready), static_cast<void *>(done),
              static_cast<void *>(observed));
  std::fflush(stdout);

  if (!step("hipSetDevice owner for launch", hipSetDevice(0)))
    return 5;
  hipLaunchKernelGGL(owner_read_wait_refetch, dim3(1), dim3(1), 0, 0, owner, ready, done, observed);
  if (!step("owner kernel launch", hipGetLastError()) ||
      !step("hipSetDevice requester for launch", hipSetDevice(1)))
    return 6;
  hipLaunchKernelGGL(requester_write_signal, dim3(1), dim3(1), 0, 0, owner, ready, done);
  if (!step("requester kernel launch", hipGetLastError()) ||
      !step("requester kernel synchronize", hipDeviceSynchronize()) ||
      !step("hipSetDevice owner for synchronize", hipSetDevice(0)) ||
      !step("owner kernel synchronize", hipDeviceSynchronize()))
    return 7;

  uint32_t values[3] = {};
  if (!step("copy observations",
            hipMemcpy(values, observed, sizeof(values), hipMemcpyDeviceToHost)))
    return 8;
  const bool pass = values[0] == kInitial && values[1] == kInitial && values[2] == kUpdated;
  std::printf("owner-peer-refetch: first=0x%08x hit=0x%08x fresh=0x%08x "
              "expected_old=0x%08x expected_new=0x%08x result=%s\n",
              values[0], values[1], values[2], kInitial, kUpdated, pass ? "PASS" : "FAIL");
  std::fflush(stdout);

  (void)hipSetDevice(1);
  (void)hipDeviceDisablePeerAccess(0);
  (void)hipSetDevice(0);
  (void)hipFree(allocation);
  return pass ? 0 : 9;
}
