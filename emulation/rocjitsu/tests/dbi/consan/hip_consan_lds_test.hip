// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_lds_test.hip
/// @brief Real-HIP smoke for ConSan DBI of a padded LDS store.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (const char *path = std::getenv("CONSAN_WORKLOAD_RESULT_PATH"))
    (void)std::remove(path);
  int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

#if defined(__gfx942__) || defined(__gfx950__)
#define RJ_CONSAN_TEST_WORKGROUP_BARRIER "s_barrier\n\t"
#define RJ_CONSAN_TEST_DS_STORE_B32 "ds_write_b32 %0, %1\n\t"
#define RJ_CONSAN_TEST_FLAT_STORE_B32 "flat_store_dword v[0:1], %0\n\t"
#else
#define RJ_CONSAN_TEST_WORKGROUP_BARRIER                                                           \
  "s_barrier_signal -1\n\t"                                                                        \
  "s_barrier_wait -1\n\t"
#define RJ_CONSAN_TEST_DS_STORE_B32 "ds_store_b32 %0, %1\n\t"
#define RJ_CONSAN_TEST_FLAT_STORE_B32 "flat_store_b32 v[0:1], %0\n\t"
#endif

#define RJ_CONSAN_TEST_NOP_CAVE                                                                    \
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

bool bool_env(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool expect_dbi_trap() { return bool_env("RJ_CONSAN_EXPECT_TRAP"); }

bool require_dbi_patch() { return bool_env("RJ_CONSAN_REQUIRE_PATCH"); }

constexpr uint32_t kReportMarker = 424242u;
constexpr uint32_t kMaxInstabilityLaunches = 100;

uint32_t instability_launch_budget() {
  const char *text = std::getenv("CONSAN_SC_INSTABILITY_LAUNCHES");
  if (text == nullptr)
    return 1;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (text[0] < '1' || text[0] > '9' || end == nullptr || *end != '\0' || value < 1 ||
      value > kMaxInstabilityLaunches) {
    ADD_FAILURE() << "CONSAN_SC_INSTABILITY_LAUNCHES must be an integer from 1 through "
                  << kMaxInstabilityLaunches << "; got '" << text << "'";
    return 0;
  }
  return static_cast<uint32_t>(value);
}

void write_instability_oracle(const char *outcome, uint32_t launches, uint32_t detections,
                              bool output_exact) {
  const char *path = std::getenv("CONSAN_WORKLOAD_RESULT_PATH");
  if (path == nullptr)
    return;
  std::ofstream output(path, std::ios::out | std::ios::trunc);
  output << "{\"schema_version\":1,\"oracle\":\"pass\",\"detail\":{"
         << "\"launch_budget\":" << launches << ",\"completed_launches\":" << launches
         << ",\"detection_count\":" << detections
         << ",\"output_exact\":" << (output_exact ? "true" : "false")
         << ",\"instability_outcome\":\"" << outcome << "\",\"sanitizer_proof\":false}}\n";
}

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

  if (tid == 0)
    lds[0] = 0x12345678u;
  asm volatile(RJ_CONSAN_TEST_WORKGROUP_BARRIER ::: "memory");

  const auto clean_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[tid]) & 0xFFFFu);
  const auto racy_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  const uint32_t lds_addr = racy != 0 ? racy_addr : clean_addr;
  const uint32_t value = 0xC0FFEE00u | tid;
  // Keep enough slack for the larger marker-buffer mismatch action.
  asm volatile(RJ_CONSAN_TEST_DS_STORE_B32 RJ_CONSAN_TEST_NOP_CAVE
               :
               : "v"(lds_addr), "v"(value)
               : "memory");

  asm volatile(RJ_CONSAN_TEST_WORKGROUP_BARRIER ::: "memory");
  if (racy != 0) {
    if (tid == 0)
      out[0] = lds[0];
  } else {
    out[tid] = lds[tid];
  }
}

__device__ __forceinline__ void padded_flat_lds_store(uint32_t value) {
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t"
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t" RJ_CONSAN_TEST_FLAT_STORE_B32 RJ_CONSAN_TEST_NOP_CAVE
               :
               : "v"(value)
               : "s14", "s15", "v0", "v1", "memory");
}

__global__ __launch_bounds__(128) void padded_cross_wave_flat_lds_store(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  constexpr uint32_t kStoredValue = 0xF1A7005Au;
  asm volatile("" : : "v"(&lds[0]) : "memory");
  // Keep the descriptor full so the regression exercises the spill-backed
  // resource plan instead of relying on target-specific unused high VGPRs.
  asm volatile("v_mov_b32 v255, 0" ::: "v255");
  padded_flat_lds_store(kStoredValue);
  __syncthreads();
  if (tid == 0)
    out[0] = lds[0];
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

TEST(ConSanLdsTest, PaddedCrossWaveFlatLdsStoreCompletes) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(*d_out)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(*d_out)));

  padded_cross_wave_flat_lds_store<<<1, 128>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  EXPECT_EQ(out, 0xF1A7005Au);
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
  const uint32_t launches = instability_launch_budget();
  ASSERT_NE(launches, 0u);
  uint32_t *d_out = nullptr;
  uint32_t *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kValues * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, sizeof(uint32_t)));
  ReportEnvGuard report_env;
  ASSERT_TRUE(set_report_buffer_env(d_report));
  std::vector<uint32_t> out(kValues);
  for (uint32_t launch = 0; launch < launches; ++launch) {
    HIP_ASSERT(hipMemset(d_out, 0, kValues * sizeof(uint32_t)));
    HIP_ASSERT(hipMemset(d_report, 0, sizeof(uint32_t)));
    launch_padded_lds_store(d_out, 0);
    const hipError_t sync = hipDeviceSynchronize();
    ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);
    uint32_t report = 0xFFFFFFFFu;
    HIP_ASSERT(hipMemcpy(&report, d_report, sizeof(report), hipMemcpyDeviceToHost));
    ASSERT_EQ(report, 0u) << "unexpected marker on launch " << launch;
    HIP_ASSERT(hipMemcpy(out.data(), d_out, kValues * sizeof(uint32_t), hipMemcpyDeviceToHost));
    for (uint32_t i = 0; i < kValues; ++i)
      ASSERT_EQ(out[i], 0xC0FFEE00u | i) << "output mismatch on launch " << launch;
  }
  free_device_ptr(d_report);
  free_device_ptr(d_out);
  write_instability_oracle("clean", launches, 0, true);
}

TEST(ConSanLdsTest, PaddedRacyStoreReportsMarker) {
  const uint32_t launches = instability_launch_budget();
  ASSERT_NE(launches, 0u);
  uint32_t *d_out = nullptr;
  uint32_t *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, sizeof(uint32_t)));
  ReportEnvGuard report_env;
  ASSERT_TRUE(set_report_buffer_env(d_report));
  uint32_t detections = 0;
  for (uint32_t launch = 0; launch < launches; ++launch) {
    HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
    HIP_ASSERT(hipMemset(d_report, 0, sizeof(uint32_t)));
    launch_padded_lds_store(d_out, 1);
    const hipError_t sync = hipDeviceSynchronize();
    ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);
    uint32_t report = 0;
    HIP_ASSERT(hipMemcpy(&report, d_report, sizeof(report), hipMemcpyDeviceToHost));
    if (report == kReportMarker)
      ++detections;
    else
      ASSERT_EQ(report, 0u) << "unexpected report value on launch " << launch;
  }
  free_device_ptr(d_report);
  free_device_ptr(d_out);
  EXPECT_GE(detections, 1u);
  if (detections >= 1)
    write_instability_oracle("marker", launches, detections, false);
}
