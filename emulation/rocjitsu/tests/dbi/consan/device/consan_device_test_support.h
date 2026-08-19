// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

// The prototype patcher needs nearby relocation space. This is fixture
// accommodation, not part of the behavioral contract: a replacement patcher
// may ignore it, and no test observes how this space is used.
#define RJ_CONSAN_DEVICE_CAVE                                                                      \
  ".rept 128\n\t"                                                                                  \
  "s_nop 0\n\t"                                                                                    \
  ".endr\n\t"

// Atomic-aware sampled and inline probes carry substantially more causal
// state than an access-only probe. Keep that prototype-patcher accommodation
// explicit in the few fixtures that need it.
#define RJ_CONSAN_DEVICE_LARGE_CAVE                                                                \
  ".rept 512\n\t"                                                                                  \
  "s_nop 0\n\t"                                                                                    \
  ".endr\n\t"

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    const hipError_t error = (call);                                                               \
    ASSERT_EQ(error, hipSuccess) << "HIP error: " << hipGetErrorString(error);                     \
  } while (0)

namespace consan_device_test {

// A small register-rich epilogue distilled from fused attention/GEMM kernels.
// Keeping the independent values live across one scheduling boundary models
// the handoff from an LDS stage to elementwise device code without prescribing
// any sanitizer implementation detail.
__host__ __device__ __forceinline__ uint32_t rotate_left(uint32_t value, uint32_t amount) {
  return (value << amount) | (value >> (32u - amount));
}

__device__ __forceinline__ uint32_t fused_epilogue_mix(const uint32_t *input, uint32_t lane,
                                                       uint32_t input_mask) {
  uint32_t values[8];
#pragma unroll
  for (uint32_t i = 0; i < 8u; ++i)
    values[i] = input[(lane + 13u * i) & input_mask] ^ (0x9e3779b9u * (i + 1u));
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  uint32_t result = 0u;
#pragma unroll
  for (uint32_t i = 0; i < 8u; ++i)
    result ^= rotate_left(values[i], i + 1u);
  return result;
}

inline uint32_t fused_epilogue_mix_reference(const uint32_t *input, uint32_t lane,
                                             uint32_t input_mask) {
  uint32_t result = 0u;
  for (uint32_t i = 0; i < 8u; ++i) {
    const uint32_t value = input[(lane + 13u * i) & input_mask] ^ (0x9e3779b9u * (i + 1u));
    result ^= rotate_left(value, i + 1u);
  }
  return result;
}

__device__ __forceinline__ void store_lds_word(volatile uint32_t *address, uint32_t value) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1250__)
  asm volatile("ds_write_b32 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(lds_address), "v"(value)
               : "memory");
#else
  asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(lds_address), "v"(value)
               : "memory");
#endif
}

__device__ __forceinline__ uint32_t load_lds_word(volatile uint32_t *address) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  uint32_t value = 0u;
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile("ds_read_b32 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_waitcnt lgkmcnt(0)\n\t"
               : "=v"(value)
               : "v"(lds_address)
               : "memory");
#elif defined(__gfx1250__)
  asm volatile("ds_read_b32 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_wait_dscnt 0\n\t"
               : "=v"(value)
               : "v"(lds_address)
               : "memory");
#elif defined(__gfx1201__)
  asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_wait_dscnt 0\n\t"
               : "=v"(value)
               : "v"(lds_address)
               : "memory");
#else
  asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_waitcnt lgkmcnt(0)\n\t"
               : "=v"(value)
               : "v"(lds_address)
               : "memory");
#endif
  return value;
}

inline uint32_t active_wave_size() {
  hipDeviceProp_t properties{};
  EXPECT_EQ(hipGetDeviceProperties(&properties, 0), hipSuccess);
  EXPECT_TRUE(properties.warpSize == 32 || properties.warpSize == 64);
  return static_cast<uint32_t>(properties.warpSize);
}

} // namespace consan_device_test

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return result;
}
