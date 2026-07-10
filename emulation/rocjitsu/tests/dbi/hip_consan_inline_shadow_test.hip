// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_inline_shadow_test.hip
/// @brief Real-HIP smoke for ConSan MOI inline-shadow diagnostics.

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

#define RJ_CONSAN_INLINE_SHADOW_NOP_CAVE                                                           \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"                                                                                    \
  "s_nop 0\n\t"

namespace {

__device__ __forceinline__ void padded_ds_store_b32(uint32_t lds_addr, uint32_t value) {
  asm volatile("ds_store_b32 %0, %1\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               :
               : "v"(lds_addr), "v"(value)
               : "memory");
}

__device__ __forceinline__ uint32_t padded_ds_load_b32(uint32_t lds_addr) {
  uint32_t value = 0;
  asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_INLINE_SHADOW_NOP_CAVE
               : "=v"(value)
               : "v"(lds_addr)
               : "memory");
  return value;
}

__device__ __forceinline__ void padded_ds_store_b128(uint32_t lds_addr, uint4 value) {
  register uint32_t value0 asm("v20") = value.x;
  register uint32_t value1 asm("v21") = value.y;
  register uint32_t value2 asm("v22") = value.z;
  register uint32_t value3 asm("v23") = value.w;
  asm volatile("ds_store_b128 %0, v[20:23]\n\t" RJ_CONSAN_INLINE_SHADOW_NOP_CAVE
               :
               : "v"(lds_addr), "v"(value0), "v"(value1), "v"(value2), "v"(value3)
               : "memory");
}

__device__ __forceinline__ void release_flat_atomic(uint32_t *counter) {
  uint32_t one = 1;
  asm volatile("flat_atomic_add_u32 %0, %1 scope:SCOPE_DEV\n\t"
               "s_waitcnt vmcnt(0)\n\t"
               :
               : "v"(counter), "v"(one)
               : "memory");
}

__device__ __forceinline__ uint32_t acquire_flat_atomic(uint32_t *counter) {
  uint32_t old = 0;
  uint32_t one = 1;
  asm volatile("flat_atomic_add_u32 %0, %1, %2 th:TH_ATOMIC_RETURN scope:SCOPE_DEV\n\t"
               "s_waitcnt vmcnt(0)\n\t"
               : "=&v"(old)
               : "v"(counter), "v"(one)
               : "memory");
  return old;
}

__global__ __launch_bounds__(128) void padded_cross_wave_lds_store(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);

  if (tid == 0)
    lds[0] = 0x12345678u;
  __syncthreads();

  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  const uint32_t value = 0xA5A50000u | tid;
  asm volatile("ds_store_b32 %0, %1\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               "s_nop 0\n\t"
               :
               : "v"(lds_addr), "v"(value)
               : "memory");

  __syncthreads();
  if (tid == 0)
    out[0] = lds[0];
}

__global__ __launch_bounds__(64) void padded_cross_wave_lds_store_b128(uint32_t *out) {
  __shared__ uint32_t lds[4];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  const uint4 value =
      make_uint4(0xA0000000u | tid, 0xB0000000u | tid, 0xC0000000u | tid, 0xD0000000u | tid);
  padded_ds_store_b128(lds_addr, value);

  __syncthreads();
  if (tid == 0)
    out[0] = lds[0] ^ lds[1] ^ lds[2] ^ lds[3];
}

__global__ __launch_bounds__(128) void padded_barrier_ordered_lds_store(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);

  if (tid == 0)
    lds[0] = 0x12345678u;
  __syncthreads();

  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  if (tid < 32)
    padded_ds_store_b32(lds_addr, 0xCAFE0000u | tid);

  __syncthreads();

  if (tid >= 64 && tid < 96)
    padded_ds_store_b32(lds_addr, 0xFACE0000u | tid);

  __syncthreads();
  if (tid == 0)
    out[0] = lds[0];
}

__global__ __launch_bounds__(64) void padded_atomic_ordered_lds_accesses(uint32_t *out,
                                                                         uint32_t *release_counter,
                                                                         uint32_t *acquire_counter,
                                                                         uint32_t *flag) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (tid == 0) {
    padded_ds_store_b32(lds_addr, 0xA70C0001u);
    release_flat_atomic(release_counter);
    *flag = 1;
  }

  if (tid == 32) {
    volatile uint32_t *volatile_flag = flag;
    while (*volatile_flag == 0) {
    }
    const uint32_t old = acquire_flat_atomic(acquire_counter);
    out[0] = padded_ds_load_b32(lds_addr) ^ old;
  }
}

void free_device_ptr(uint32_t *ptr) {
  if (ptr != nullptr)
    (void)hipFree(ptr);
}

void run_padded_atomic_ordered_lds_accesses(bool same_atomic_address) {
  uint32_t *d_out = nullptr;
  uint32_t *d_release_counter = nullptr;
  uint32_t *d_acquire_counter = nullptr;
  uint32_t *d_flag = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_release_counter, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_acquire_counter, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_flag, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_release_counter, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_acquire_counter, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_flag, 0, sizeof(uint32_t)));

  uint32_t *acquire_counter = same_atomic_address ? d_release_counter : d_acquire_counter;
  padded_atomic_ordered_lds_accesses<<<1, 64>>>(d_out, d_release_counter, acquire_counter, d_flag);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_flag);
  free_device_ptr(d_acquire_counter);
  free_device_ptr(d_release_counter);
  free_device_ptr(d_out);

  EXPECT_NE(out, 0u);
}

} // namespace

TEST(ConSanInlineShadowTest, CrossWaveRacyStoreCompletes) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));

  padded_cross_wave_lds_store<<<1, 128>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);

  EXPECT_NE(out, 0u);
}

TEST(ConSanInlineShadowTest, BarrierOrderedCrossWaveStoreCompletes) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));

  padded_barrier_ordered_lds_store<<<1, 128>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);

  EXPECT_NE(out, 0u);
}

TEST(ConSanInlineShadowTest, CrossWaveRacyB128StoreCompletes) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));

  padded_cross_wave_lds_store_b128<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  free_device_ptr(d_out);
}

TEST(ConSanInlineShadowTest, SameAddressAtomicHandoffCompletes) {
  run_padded_atomic_ordered_lds_accesses(/*same_atomic_address=*/true);
}

TEST(ConSanInlineShadowTest, WrongAddressAtomicHandoffCompletes) {
  run_padded_atomic_ordered_lds_accesses(/*same_atomic_address=*/false);
}
