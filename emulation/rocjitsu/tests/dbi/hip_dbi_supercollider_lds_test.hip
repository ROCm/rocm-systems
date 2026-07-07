// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_dbi_supercollider_lds_test.hip
/// @brief Real-HIP smoke for SuperCollider-style DBI of a padded LDS store.

#include <cstdint>
#include <cstdlib>
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

namespace {

bool bool_env(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool expect_dbi_trap() { return bool_env("RJ_DBI_SC_EXPECT_TRAP"); }

bool require_dbi_patch() { return bool_env("RJ_DBI_SC_REQUIRE_PATCH"); }

__global__ __launch_bounds__(64) void padded_lds_store(uint32_t *out, uint32_t racy) {
  __shared__ uint32_t lds[64];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);

  if (tid == 0)
    lds[0] = 0x12345678u;
  __syncthreads();

  const auto clean_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[tid]) & 0xFFFFu);
  const auto racy_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  const uint32_t lds_addr = racy != 0 ? racy_addr : clean_addr;
  const uint32_t value = 0xC0FFEE00u | tid;
  asm volatile("ds_store_b32 %0, %1\n\t"
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
  if (racy != 0) {
    if (tid == 0)
      out[0] = lds[0];
  } else {
    out[tid] = lds[tid];
  }
}

void launch_padded_lds_store(uint32_t *d_out, uint32_t racy) {
  padded_lds_store<<<1, 64>>>(d_out, racy);
  HIP_ASSERT(hipGetLastError());
}

void free_device_ptr(uint32_t *ptr) {
  if (ptr != nullptr)
    (void)hipFree(ptr);
}

} // namespace

TEST(DbiSuperColliderLdsTest, PaddedCleanStoreCompletes) {
  constexpr size_t kValues = 64;
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kValues * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, kValues * sizeof(uint32_t)));

  launch_padded_lds_store(d_out, 0);
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);
  if (require_dbi_patch()) {
    free_device_ptr(d_out);
    return;
  }

  std::vector<uint32_t> out(kValues);
  HIP_ASSERT(hipMemcpy(out.data(), d_out, kValues * sizeof(uint32_t), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);

  for (uint32_t i = 0; i < kValues; ++i)
    EXPECT_EQ(out[i], 0xC0FFEE00u | i);
}

TEST(DbiSuperColliderLdsTest, PaddedRacyStoreCompletesOrTripsDbiTrap) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));

  launch_padded_lds_store(d_out, 1);
  const hipError_t sync = hipDeviceSynchronize();
  if (expect_dbi_trap()) {
    EXPECT_NE(sync, hipSuccess) << "expected DBI-injected s_trap to fail the dispatch";
    free_device_ptr(d_out);
    return;
  }

  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);

  EXPECT_NE(out, 0u);
}
