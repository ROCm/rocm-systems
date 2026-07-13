// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_lds_test.hip
/// @brief Real-HIP smoke for ConSan DBI of a padded LDS store.

#include <cstddef>
#include <cstdint>
#include <cstdio>
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

bool expect_dbi_trap() { return bool_env("RJ_CONSAN_EXPECT_TRAP"); }

bool require_dbi_patch() { return bool_env("RJ_CONSAN_REQUIRE_PATCH"); }

constexpr uint32_t kReportMarker = 424242u;

struct ReportEnvGuard {
  ~ReportEnvGuard() {
    (void)unsetenv("RJ_CONSAN_REPORT_BUFFER");
    (void)unsetenv("RJ_CONSAN_REPORT_MARKER");
  }
};

bool set_report_buffer_env(uint32_t *d_report) {
  char address[32];
  const int written =
      std::snprintf(address, sizeof(address), "0x%llx",
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_report)));
  if (written < 0 || static_cast<size_t>(written) >= sizeof(address)) {
    ADD_FAILURE() << "failed to format report-buffer address";
    return false;
  }
  if (setenv("RJ_CONSAN_REPORT_BUFFER", address, 1) != 0) {
    ADD_FAILURE() << "failed to set RJ_CONSAN_REPORT_BUFFER";
    return false;
  }
  if (setenv("RJ_CONSAN_REPORT_MARKER", "424242", 1) != 0) {
    ADD_FAILURE() << "failed to set RJ_CONSAN_REPORT_MARKER";
    return false;
  }
  return true;
}

__global__ __launch_bounds__(64) void padded_lds_store(uint32_t *out, uint32_t racy) {
  __shared__ uint32_t lds[64];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);

  const auto clean_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[tid]) & 0xFFFFu);
  const auto racy_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  const uint32_t lds_addr = racy != 0 ? racy_addr : clean_addr;
  const uint32_t value = 0xC0FFEE00u | tid;
  // Keep enough slack for the larger marker-buffer mismatch action.
#if defined(__gfx950__)
#define RJ_CONSAN_LDS_STORE_B32 "ds_write_b32"
#else
#define RJ_CONSAN_LDS_STORE_B32 "ds_store_b32"
#endif
  asm volatile(RJ_CONSAN_LDS_STORE_B32 " %0, %1\n\t"
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
#undef RJ_CONSAN_LDS_STORE_B32

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

TEST(ConSanLdsTest, PaddedCleanStoreCompletes) {
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

TEST(ConSanLdsTest, PaddedRacyStoreCompletesOrTripsDbiTrap) {
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

TEST(ConSanLdsTest, PaddedCleanStoreReportBufferStaysZero) {
  constexpr size_t kValues = 64;
  uint32_t *d_out = nullptr;
  uint32_t *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kValues * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, kValues * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, sizeof(uint32_t)));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_report_buffer_env(d_report));

  launch_padded_lds_store(d_out, 0);
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  uint32_t report = 0xFFFFFFFFu;
  HIP_ASSERT(hipMemcpy(&report, d_report, sizeof(report), hipMemcpyDeviceToHost));
  free_device_ptr(d_report);
  free_device_ptr(d_out);

  EXPECT_EQ(report, 0u);
}

TEST(ConSanLdsTest, PaddedRacyStoreReportsMarker) {
  uint32_t *d_out = nullptr;
  uint32_t *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, sizeof(uint32_t)));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_report_buffer_env(d_report));

  launch_padded_lds_store(d_out, 1);
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  uint32_t report = 0;
  HIP_ASSERT(hipMemcpy(&report, d_report, sizeof(report), hipMemcpyDeviceToHost));
  free_device_ptr(d_report);
  free_device_ptr(d_out);

  EXPECT_EQ(report, kReportMarker);
}
