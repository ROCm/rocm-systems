// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_inline_shadow_test.hip
/// @brief Real-HIP smoke for ConSan MOI inline-shadow diagnostics.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // The focused clean controls publish evidence only after every in-process
  // launch has completed and matched exactly. Remove stale evidence first so
  // an early HIP/assertion failure cannot be mistaken for a current oracle.
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

constexpr uint32_t kMaxScCleanLaunches = 100;
constexpr uint32_t kScCompositeLaunches = 100;
constexpr uint32_t kMaxInlineVersionedLaunches = 100;
constexpr uint32_t kScCompositeReportMarker = 424242u;
constexpr uint32_t kBarrierCompositeCleanOutput = 0xB2200002u;
constexpr uint32_t kBarrierCompositeFaultAlternate = 0xA1100000u;

uint32_t sc_clean_launch_count() {
  const char *text = std::getenv("CONSAN_SC_CLEAN_LAUNCHES");
  if (text == nullptr)
    return 1;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (text[0] < '1' || text[0] > '9' || end == nullptr || *end != '\0' || value < 1 ||
      value > kMaxScCleanLaunches) {
    ADD_FAILURE() << "CONSAN_SC_CLEAN_LAUNCHES must be an integer from 1 through "
                  << kMaxScCleanLaunches << "; got '" << text << "'";
    return 0;
  }
  return static_cast<uint32_t>(value);
}

uint32_t inline_versioned_launch_count() {
  const char *text = std::getenv("CONSAN_INLINE_VERSIONED_LAUNCHES");
  if (text == nullptr)
    return 1;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (text[0] < '1' || text[0] > '9' || end == nullptr || *end != '\0' || value < 1 ||
      value > kMaxInlineVersionedLaunches) {
    ADD_FAILURE() << "CONSAN_INLINE_VERSIONED_LAUNCHES must be an integer from 1 through "
                  << kMaxInlineVersionedLaunches << "; got '" << text << "'";
    return 0;
  }
  return static_cast<uint32_t>(value);
}

void write_sc_clean_oracle(const char *control, uint32_t launches) {
  const char *path = std::getenv("CONSAN_WORKLOAD_RESULT_PATH");
  if (path == nullptr)
    return;
  std::ofstream output(path, std::ios::out | std::ios::trunc);
  output << "{\"schema_version\":1,\"oracle\":\"pass\",\"detail\":{"
         << "\"control\":\"" << control << "\",\"launches\":" << launches
         << ",\"output_exact\":true,\"consan_diagnostics\":0}}\n";
}

void write_inline_shadow_coalescing_oracle(const char *control, const char *access_shape,
                                           const char *address_topology, uint32_t actual,
                                           uint32_t expected, uint32_t active_lanes,
                                           uint32_t shadow_cells_per_lane,
                                           uint32_t distinct_shadow_cells) {
  const char *path = std::getenv("CONSAN_WORKLOAD_RESULT_PATH");
  if (path == nullptr)
    return;
  const bool passed = actual == expected;
  std::ofstream output(path, std::ios::out | std::ios::trunc);
  output << "{\"schema_version\":1,\"oracle\":\"" << (passed ? "pass" : "fail")
         << "\",\"detail\":{\"control\":\"" << control << "\",\"access_shape\":\"" << access_shape
         << "\",\"address_topology\":\"" << address_topology << "\",\"actual\":" << actual
         << ",\"expected\":" << expected << ",\"output_exact\":" << (passed ? "true" : "false")
         << ",\"active_lanes\":" << active_lanes
         << ",\"shadow_cells_per_lane\":" << shadow_cells_per_lane
         << ",\"distinct_shadow_cells\":" << distinct_shadow_cells << "}}\n";
}

void write_atomic_cas_oracle(const uint32_t (&actual)[3], const uint32_t (&expected)[3]) {
  const char *path = std::getenv("CONSAN_WORKLOAD_RESULT_PATH");
  if (path == nullptr)
    return;
  const bool passed =
      actual[0] == expected[0] && actual[1] == expected[1] && actual[2] == expected[2];
  std::ofstream output(path, std::ios::out | std::ios::trunc);
  output << "{\"schema_version\":1,\"oracle\":\"" << (passed ? "pass" : "fail")
         << "\",\"detail\":{\"actual\":[" << actual[0] << ',' << actual[1] << ',' << actual[2]
         << "],\"expected\":[" << expected[0] << ',' << expected[1] << ',' << expected[2]
         << "]}}\n";
}

uint32_t sc_composite_launch_count() {
  const char *text = std::getenv("CONSAN_SC_COMPOSITE_LAUNCHES");
  if (text == nullptr)
    return 1;
  if (std::strcmp(text, "100") != 0) {
    ADD_FAILURE() << "CONSAN_SC_COMPOSITE_LAUNCHES must equal " << kScCompositeLaunches << "; got '"
                  << text << "'";
    return 0;
  }
  return kScCompositeLaunches;
}

struct ScCompositeReportEnvGuard {
  bool active = false;
  ~ScCompositeReportEnvGuard() {
    if (!active)
      return;
    (void)unsetenv("RJ_CONSAN_REPORT_BUFFER");
    (void)unsetenv("RJ_CONSAN_REPORT_MARKER");
  }
};

bool set_sc_composite_report_buffer(uint32_t *d_report) {
  char address[32];
  const int written =
      std::snprintf(address, sizeof(address), "0x%llx",
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_report)));
  if (written < 0 || static_cast<size_t>(written) >= sizeof(address)) {
    ADD_FAILURE() << "failed to format SC composite report-buffer address";
    return false;
  }
  if (setenv("RJ_CONSAN_REPORT_BUFFER", address, 1) != 0 ||
      setenv("RJ_CONSAN_REPORT_MARKER", "424242", 1) != 0) {
    ADD_FAILURE() << "failed to set SC composite report-buffer environment";
    return false;
  }
  return true;
}

void write_barrier_composite_oracle(bool injected, uint32_t launches, uint32_t detections,
                                    uint32_t clean_outputs, uint32_t fault_alternate_outputs,
                                    uint32_t invalid_outputs) {
  const char *path = std::getenv("CONSAN_WORKLOAD_RESULT_PATH");
  if (path == nullptr)
    return;
  const bool clean_exact = !injected && clean_outputs == launches && detections == 0;
  const bool output_contract = invalid_outputs == 0 &&
                               clean_outputs + fault_alternate_outputs == launches &&
                               (!fault_alternate_outputs || injected);
  const bool passed = output_contract && (injected || clean_exact);
  const char *instability_outcome =
      detections != 0 ? "marker" : (fault_alternate_outputs != 0 ? "value_mismatch" : "clean");
  std::ofstream output(path, std::ios::out | std::ios::trunc);
  output << "{\"schema_version\":1,\"oracle\":\"" << (passed ? "pass" : "fail") << "\",\"detail\":{"
         << "\"control\":\"markerless-barrier-move-composite\","
         << "\"fault_injected\":" << (injected ? "true" : "false") << ','
         << "\"launch_budget\":" << launches << ",\"completed_launches\":" << launches
         << ",\"detection_count\":" << detections
         << ",\"output_exact\":" << (clean_exact ? "true" : "false")
         << ",\"output_contract_satisfied\":" << (output_contract ? "true" : "false")
         << ",\"instability_outcome\":\"" << instability_outcome << "\""
         << ",\"sanitizer_proof\":false"
         << ",\"clean_output_count\":" << clean_outputs
         << ",\"fault_alternate_output_count\":" << fault_alternate_outputs
         << ",\"invalid_output_count\":" << invalid_outputs
         << ",\"clean_expected\":" << kBarrierCompositeCleanOutput
         << ",\"fault_alternate\":" << kBarrierCompositeFaultAlternate << "}}\n";
}

bool strict_composite_bool_env(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr)
    return false;
  if (std::strcmp(value, "1") != 0) {
    ADD_FAILURE() << name << " must be absent or exactly '1' in SC composite mode; got '" << value
                  << "'";
    return false;
  }
  return true;
}

bool atomic_fault_injected() {
  if (strict_composite_bool_env("RJ_CONSAN_FAULT_DRY_RUN"))
    return false;
  const uint32_t enabled =
      (strict_composite_bool_env("RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS") ? 1u : 0u) +
      (strict_composite_bool_env("RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER") ? 1u : 0u) +
      (strict_composite_bool_env("RJ_CONSAN_FAULT_ATOMIC_WEAKEN_SCOPE") ? 1u : 0u);
  if (enabled > 1u)
    ADD_FAILURE() << "SC composite mode requires at most one atomic fault family";
  return enabled == 1u;
}

void write_atomic_composite_oracle(const char *control, bool injected, uint32_t launches,
                                   uint32_t detections, uint32_t clean_outputs,
                                   uint32_t fault_alternate_outputs, uint32_t invalid_outputs,
                                   uint32_t clean_expected, uint32_t fault_alternate,
                                   bool cas_control = false,
                                   uint32_t successful_cas_return_failures = 0,
                                   uint32_t failed_cas_return_failures = 0) {
  const char *path = std::getenv("CONSAN_WORKLOAD_RESULT_PATH");
  if (path == nullptr)
    return;
  const bool cas_contract = successful_cas_return_failures == 0 && failed_cas_return_failures == 0;
  const bool clean_exact = !injected && clean_outputs == launches && detections == 0;
  const bool output_contract = invalid_outputs == 0 && cas_contract &&
                               clean_outputs + fault_alternate_outputs == launches &&
                               (!fault_alternate_outputs || injected);
  const bool passed = output_contract && (injected || clean_exact);
  const char *instability_outcome =
      detections != 0 ? "marker" : (fault_alternate_outputs != 0 ? "value_mismatch" : "clean");
  std::ofstream output(path, std::ios::out | std::ios::trunc);
  output << "{\"schema_version\":1,\"oracle\":\"" << (passed ? "pass" : "fail")
         << "\",\"detail\":{\"control\":\"" << control
         << "\",\"fault_injected\":" << (injected ? "true" : "false")
         << ",\"launch_budget\":" << launches << ",\"completed_launches\":" << launches
         << ",\"detection_count\":" << detections
         << ",\"output_exact\":" << (clean_exact ? "true" : "false")
         << ",\"output_contract_satisfied\":" << (output_contract ? "true" : "false")
         << ",\"instability_outcome\":\"" << instability_outcome << "\""
         << ",\"sanitizer_proof\":false,\"clean_output_count\":" << clean_outputs
         << ",\"fault_alternate_output_count\":" << fault_alternate_outputs
         << ",\"invalid_output_count\":" << invalid_outputs
         << ",\"clean_expected\":" << clean_expected << ",\"fault_alternate\":" << fault_alternate;
  if (cas_control) {
    output << ",\"successful_cas_expected_return\":0"
           << ",\"successful_cas_return_failure_count\":" << successful_cas_return_failures
           << ",\"failed_cas_expected_return\":1"
           << ",\"failed_cas_return_failure_count\":" << failed_cas_return_failures;
  }
  output << "}}\n";
}

bool write_barrier_move_oracle(uint32_t actual) {
  constexpr uint32_t kPreWave0 = 0xA1100000u;
  constexpr uint32_t kPostWave2 = 0xB2200002u;
  const bool injected = std::getenv("RJ_CONSAN_FAULT_MOVE_BARRIER") != nullptr;
  const bool passed = actual == kPostWave2 || (injected && actual == kPreWave0);
  const char *path = std::getenv("CONSAN_WORKLOAD_RESULT_PATH");
  if (path != nullptr) {
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    output << "{\"schema_version\":1,\"oracle\":\"" << (passed ? "pass" : "fail")
           << "\",\"detail\":{\"control\":\"markerless-barrier-move\",\"fault_injected\":"
           << (injected ? "true" : "false") << ",\"actual\":" << actual
           << ",\"clean_expected\":" << kPostWave2 << ",\"fault_alternate\":" << kPreWave0
           << "}}\n";
  }
  return passed;
}

bool write_structured_divergent_barrier_oracle(uint32_t actual) {
  constexpr uint32_t kExpected = 0xD1A0001Fu;
  const bool passed = actual == kExpected;
  const char *path = std::getenv("CONSAN_WORKLOAD_RESULT_PATH");
  if (path != nullptr) {
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    output << "{\"schema_version\":1,\"oracle\":\"" << (passed ? "pass" : "fail")
           << "\",\"detail\":{\"control\":\"structured-divergent-barrier\",\"actual\":" << actual
           << ",\"expected\":" << kExpected << "}}\n";
  }
  return passed;
}

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
               "s_wait_dscnt 0\n\t"
               :
               : "v"(lds_addr), "v"(value)
               : "memory");
}

__device__ __forceinline__ uint32_t padded_ds_load_b32(uint32_t lds_addr) {
  uint32_t value = 0;
  asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_INLINE_SHADOW_NOP_CAVE "s_wait_dscnt 0\n\t"
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

__device__ __forceinline__ void padded_strong_group_flat_store_b32(uint32_t value) {
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t"
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t"
               "flat_store_b32 v[0:1], %0\n\t" RJ_CONSAN_INLINE_SHADOW_NOP_CAVE
               :
               : "v"(value)
               : "s14", "s15", "v0", "v1", "memory");
}

__device__ __forceinline__ void padded_strong_group_flat_store_b32_at(uint32_t lds_offset,
                                                                      uint32_t value) {
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t"
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t"
               "v_add_nc_u32 v0, v0, %0\n\t"
               "flat_store_b32 v[0:1], %1\n\t" RJ_CONSAN_INLINE_SHADOW_NOP_CAVE
               :
               : "v"(lds_offset), "v"(value)
               : "s14", "s15", "v0", "v1", "memory");
}

__device__ __forceinline__ void release_flat_atomic(uint32_t *counter) {
  uint32_t one = 1;
  asm volatile("s_wait_storecnt 0\n\t"
               "global_wb scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0\n\t"
               "flat_atomic_add_u32 %0, %1 scope:SCOPE_DEV\n\t"
               "s_waitcnt vmcnt(0)\n\t"
               :
               : "v"(counter), "v"(one)
               : "memory");
}

__device__ __forceinline__ void release_global_atomic(uint32_t *counter) {
  uint32_t one = 1;
  asm volatile("s_wait_storecnt 0\n\t"
               "global_wb scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0\n\t"
               "global_atomic_add_u32 %0, %1, off scope:SCOPE_DEV\n\t"
               "s_waitcnt vmcnt(0)\n\t"
               :
               : "v"(counter), "v"(one)
               : "memory");
}

__device__ __forceinline__ uint32_t acquire_flat_atomic(uint32_t *counter) {
  uint32_t old = 0;
  uint32_t one = 1;
  asm volatile("flat_atomic_add_u32 %0, %1, %2 th:TH_ATOMIC_RETURN scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0\n\t"
               "global_inv scope:SCOPE_DEV\n\t"
               "s_wait_loadcnt 0\n\t"
               : "=&v"(old)
               : "v"(counter), "v"(one)
               : "memory");
  return old;
}

__device__ __forceinline__ uint32_t acquire_global_atomic(uint32_t *counter) {
  uint32_t old = 0;
  uint32_t one = 1;
  asm volatile("global_atomic_add_u32 %0, %1, %2, off th:TH_ATOMIC_RETURN "
               "scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0\n\t"
               "global_inv scope:SCOPE_DEV\n\t"
               "s_wait_loadcnt 0\n\t"
               : "=&v"(old)
               : "v"(counter), "v"(one)
               : "memory");
  return old;
}

__device__ __forceinline__ void release_global_atomic_system(uint32_t *counter) {
  uint32_t one = 1;
  asm volatile("s_wait_storecnt 0\n\t"
               "global_wb scope:SCOPE_SYS\n\t"
               "s_wait_storecnt 0\n\t"
               "global_atomic_add_u32 %0, %1, off scope:SCOPE_SYS\n\t"
               "s_waitcnt vmcnt(0)\n\t"
               :
               : "v"(counter), "v"(one)
               : "memory");
}

__device__ __forceinline__ uint32_t acquire_global_atomic_system(uint32_t *counter) {
  uint32_t old = 0;
  uint32_t one = 1;
  asm volatile("global_atomic_add_u32 %0, %1, %2, off th:TH_ATOMIC_RETURN "
               "scope:SCOPE_SYS\n\t"
               "s_wait_storecnt 0\n\t"
               "global_inv scope:SCOPE_SYS\n\t"
               "s_wait_loadcnt 0\n\t"
               : "=&v"(old)
               : "v"(counter), "v"(one)
               : "memory");
  return old;
}

__device__ __forceinline__ uint32_t release_flat_cas(uint32_t *counter, uint32_t compare,
                                                     uint32_t replacement) {
  uint32_t old = 0;
  const uint2 values = make_uint2(replacement, compare);
  asm volatile("s_wait_storecnt 0\n\t"
               "global_wb scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0\n\t"
               "flat_atomic_cmpswap_b32 %0, %1, %2 th:TH_ATOMIC_RETURN scope:SCOPE_DEV\n\t"
               "s_wait_loadcnt 0\n\t"
               : "=&v"(old)
               : "v"(counter), "v"(values)
               : "memory");
  return old;
}

__device__ __forceinline__ uint32_t release_global_cas(uint32_t *counter, uint32_t compare,
                                                       uint32_t replacement) {
  uint32_t old = 0;
  const uint2 values = make_uint2(replacement, compare);
  asm volatile("s_wait_storecnt 0\n\t"
               "global_wb scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0\n\t"
               "global_atomic_cmpswap_b32 %0, %1, %2, off th:TH_ATOMIC_RETURN "
               "scope:SCOPE_DEV\n\t"
               "s_wait_loadcnt 0\n\t"
               : "=&v"(old)
               : "v"(counter), "v"(values)
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
  if (tid < 4)
    out[tid] = lds[tid];
}

__global__ __launch_bounds__(64) void padded_cross_wave_flat_lds_store(uint32_t *out) {
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  padded_strong_group_flat_store_b32(0xF1A70000u | tid);
  if (tid == 0)
    out[0] = 0xF1A70000u;
}

__global__ __launch_bounds__(32) void padded_coalesced_b32_same_address(uint32_t *out) {
  __shared__ uint32_t lds[32];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  padded_ds_store_b32(base, 0xB3200000u);
  if (tid == 0)
    out[0] = 0xB3200001u;
}

__global__ __launch_bounds__(32) void padded_coalesced_b32_divergent_address(uint32_t *out) {
  __shared__ uint32_t lds[32];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  padded_ds_store_b32(base + tid * sizeof(uint32_t), 0xB32D0000u | tid);
  if (tid == 0)
    out[0] = 0xB32D0001u;
}

__global__ __launch_bounds__(32) void padded_coalesced_b128_same_address(uint32_t *out) {
  __shared__ uint32_t lds[128];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  padded_ds_store_b128(base, make_uint4(1u, 2u, 3u, 4u));
  if (tid == 0)
    out[0] = 0xB1280001u;
}

__global__ __launch_bounds__(32) void padded_coalesced_b128_divergent_address(uint32_t *out) {
  __shared__ uint32_t lds[128];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  padded_ds_store_b128(base + tid * sizeof(uint4), make_uint4(tid, tid + 1u, tid + 2u, tid + 3u));
  if (tid == 0)
    out[0] = 0xB1280002u;
}

__global__ __launch_bounds__(32) void padded_coalesced_group_flat_same_address(uint32_t *out) {
  __shared__ uint32_t lds[32];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  asm volatile("" : : "v"(&lds[0]) : "memory");
  padded_strong_group_flat_store_b32(0xF1A70000u);
  if (tid == 0)
    out[0] = 0xF1A70001u;
}

__global__ __launch_bounds__(32) void padded_coalesced_group_flat_divergent_address(uint32_t *out) {
  __shared__ uint32_t lds[32];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  padded_strong_group_flat_store_b32_at(base + tid * sizeof(uint32_t), 0xF1AD0000u | tid);
  if (tid == 0)
    out[0] = 0xF1AD0001u;
}

__global__ __launch_bounds__(128) void padded_barrier_ordered_lds_store(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);

  if (tid == 0)
    lds[0] = 0x12345678u;
  __syncthreads();

  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  // Keep the control genuinely ordered: one representative in each of two
  // waves touches the shared word. A whole-wave same-address store would
  // itself contain unordered workitem pairs and would not be a clean barrier
  // fixture.
  if (tid == 0)
    padded_ds_store_b32(lds_addr, 0xCAFE0000u | tid);

  // Keep the fault target as one explicit, adjacent semantic barrier pair.
  // Compiler expansion of __syncthreads() may interleave bookkeeping that
  // makes an index-selected wait look like an ambiguous singleton.
  asm volatile("s_barrier_signal -1\n\t"
               "s_barrier_wait -1\n\t" ::
                   : "memory");

  if (tid == 64)
    padded_ds_store_b32(lds_addr, 0xFACE0000u | tid);

  // Test-only DBI destination for RJ_CONSAN_FAULT_MOVE_BARRIER. Keeping the
  // marker after the second access makes the moved barrier safe but too late.
  asm volatile("s_nop 42");
  asm volatile("s_nop 43");

  __syncthreads();
  if (tid == 0)
    out[0] = lds[0];
}

__global__ __launch_bounds__(128) void padded_markerless_barrier_move_control(uint32_t *out) {
  __shared__ uint32_t lds[4];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t wave_id = tid >> 5u;
  const auto base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  // Each wave initially owns one distinct LDS word.
  padded_ds_store_b32(base + wave_id * sizeof(uint32_t), 0xA1100000u | wave_id);

  // Deliberately adjacent so ConSan can qualify and relocate the whole pair.
  asm volatile("s_barrier_signal -1\n\t"
               "s_barrier_wait -1\n\t" ::
                   : "memory");

  // XOR-permute ownership. Post wave 2 now owns pre wave 0's address; moving
  // the middle pair across either real store removes their phase ordering.
  padded_ds_store_b32(base + (wave_id ^ 2u) * sizeof(uint32_t), 0xB2200000u | wave_id);

  asm volatile("s_barrier_signal -1\n\t"
               "s_barrier_wait -1\n\t" ::
                   : "memory");
  if (tid == 0)
    out[0] = lds[0];
}

// Purpose-built wave32/two-wave control for divergent barrier relocation.
// The optional first-wave store compiles to a structured EXEC diamond. The
// adjacent barrier pair remains uniform after EXEC reconvergence.
__global__ __launch_bounds__(64) void padded_structured_divergent_barrier_control(uint32_t *out) {
  __shared__ uint32_t lds[32];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (tid < 32)
    padded_ds_store_b32(address + tid * sizeof(uint32_t), 0xD1A00000u | tid);

  asm volatile("s_barrier_signal -1\n\t"
               "s_barrier_wait -1\n\t" ::
                   : "memory");

  if (tid == 32)
    out[0] = padded_ds_load_b32(address + 31u * sizeof(uint32_t));
}

__global__ __launch_bounds__(64) void padded_fence_barrier_ordered_lds_accesses(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (tid == 0)
    padded_ds_store_b32(lds_addr, 0xFECE0001u);
  __threadfence_block();
  // Keep the split barrier pair adjacent so the perturbation inventory can
  // prove and independently delay its release and acquire edges.
  asm volatile("s_barrier_signal -1\n\t"
               "s_barrier_wait -1\n\t" ::
                   : "memory");
  if (tid == 32)
    out[0] = padded_ds_load_b32(lds_addr);
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
    release_global_atomic(release_counter);
    *flag = 1;
  }

  if (tid == 32) {
    volatile uint32_t *volatile_flag = flag;
    while (*volatile_flag == 0) {
    }
    const uint32_t old = acquire_global_atomic(acquire_counter);
    out[0] = padded_ds_load_b32(lds_addr) ^ old;
  }
}

__global__ __launch_bounds__(64) void padded_atomic_system_scope_ordered_lds_accesses(
    uint32_t *out, uint32_t *release_counter, uint32_t *acquire_counter, uint32_t *flag) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (tid == 0) {
    padded_ds_store_b32(lds_addr, 0xA70C0001u);
    release_global_atomic_system(release_counter);
    *flag = 1;
  }

  if (tid == 32) {
    volatile uint32_t *volatile_flag = flag;
    while (*volatile_flag == 0) {
    }
    const uint32_t old = acquire_global_atomic_system(acquire_counter);
    out[0] = padded_ds_load_b32(lds_addr) ^ old;
  }
}

__global__ __launch_bounds__(64) void padded_atomic_collision_replaces_release(uint32_t *out,
                                                                               uint32_t *counters,
                                                                               uint32_t *flag) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (tid == 0) {
    padded_ds_store_b32(lds_addr, 0xC0111DE1u);
    release_flat_atomic(counters);
    // The inline table has 64 entries. A 256-byte address delta therefore
    // hashes to the same direct-mapped slot and replaces the first release.
    release_flat_atomic(counters + 64);
    *flag = 1;
  }

  if (tid == 32) {
    volatile uint32_t *volatile_flag = flag;
    while (*volatile_flag == 0) {
    }
    const uint32_t old = acquire_flat_atomic(counters);
    out[0] = padded_ds_load_b32(lds_addr) ^ old;
  }
}

__global__ __launch_bounds__(64) void padded_atomic_cas_failure_does_not_publish(uint32_t *out,
                                                                                 uint32_t *counters,
                                                                                 uint32_t *flag) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (tid == 0) {
    padded_ds_store_b32(lds_addr, 0xCA500001u);
    out[1] = release_flat_cas(counters, /*compare=*/0, /*replacement=*/1);
    // This address collides with counters[0], but the CAS fails because its
    // initial value is one. Failure must not evict the successful release.
    out[2] = release_flat_cas(counters + 64, /*compare=*/0, /*replacement=*/2);
    *flag = 1;
  }

  if (tid == 32) {
    volatile uint32_t *volatile_flag = flag;
    while (*volatile_flag == 0) {
    }
    const uint32_t old = acquire_flat_atomic(counters);
    out[0] = padded_ds_load_b32(lds_addr) ^ old;
  }
}

__global__ __launch_bounds__(64) void padded_vglobal_cas_failure_does_not_publish(
    uint32_t *out, uint32_t *counters, uint32_t *flag) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (tid == 0) {
    padded_ds_store_b32(lds_addr, 0xCA500001u);
    out[1] = release_global_cas(counters, /*compare=*/0, /*replacement=*/1);
    // Exercise success-predicated VGLOBAL publication: the colliding CAS fails
    // and therefore must not evict the successful release for counters[0].
    out[2] = release_global_cas(counters + 64, /*compare=*/0, /*replacement=*/2);
    *flag = 1;
  }

  if (tid == 32) {
    volatile uint32_t *volatile_flag = flag;
    while (*volatile_flag == 0) {
    }
    const uint32_t old = acquire_global_atomic(counters);
    out[0] = padded_ds_load_b32(lds_addr) ^ old;
  }
}

void free_device_ptr(uint32_t *ptr) {
  if (ptr != nullptr)
    (void)hipFree(ptr);
}

void run_padded_atomic_ordered_lds_accesses(bool same_atomic_address, bool system_scope = false) {
  const bool composite = std::getenv("CONSAN_SC_COMPOSITE_LAUNCHES") != nullptr;
  const uint32_t launches = composite ? sc_composite_launch_count() : sc_clean_launch_count();
  ASSERT_NE(launches, 0u);
  uint32_t *d_out = nullptr;
  uint32_t *d_release_counter = nullptr;
  uint32_t *d_acquire_counter = nullptr;
  uint32_t *d_flag = nullptr;
  uint32_t *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  // The adjacent word is a valid target for the +4-byte wrong-address fault.
  HIP_ASSERT(hipMalloc(&d_release_counter, 2 * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_acquire_counter, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_flag, sizeof(uint32_t)));
  if (composite)
    HIP_ASSERT(hipMalloc(&d_report, sizeof(uint32_t)));
  ScCompositeReportEnvGuard report_env{composite};
  if (composite)
    ASSERT_TRUE(set_sc_composite_report_buffer(d_report));
  uint32_t *acquire_counter = same_atomic_address ? d_release_counter : d_acquire_counter;
  const bool injected_wrong_address =
      std::getenv("RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS") != nullptr &&
      std::getenv("RJ_CONSAN_FAULT_DRY_RUN") == nullptr;
  const bool injected = composite && atomic_fault_injected();
  const uint32_t clean_expected = same_atomic_address ? 0xA70C0000u : 0xA70C0001u;
  const uint32_t fault_alternate = same_atomic_address ? 0xA70C0001u : 0xA70C0000u;
  const uint32_t legacy_expected =
      same_atomic_address && !injected_wrong_address ? 0xA70C0000u : 0xA70C0001u;
  uint32_t detections = 0;
  uint32_t clean_outputs = 0;
  uint32_t fault_alternate_outputs = 0;
  uint32_t invalid_outputs = 0;
  for (uint32_t launch = 0; launch < launches; ++launch) {
    HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
    HIP_ASSERT(hipMemset(d_release_counter, 0, 2 * sizeof(uint32_t)));
    HIP_ASSERT(hipMemset(d_acquire_counter, 0, sizeof(uint32_t)));
    HIP_ASSERT(hipMemset(d_flag, 0, sizeof(uint32_t)));
    if (composite)
      HIP_ASSERT(hipMemset(d_report, 0, sizeof(uint32_t)));
    if (system_scope) {
      padded_atomic_system_scope_ordered_lds_accesses<<<1, 64>>>(d_out, d_release_counter,
                                                                 acquire_counter, d_flag);
    } else {
      padded_atomic_ordered_lds_accesses<<<1, 64>>>(d_out, d_release_counter, acquire_counter,
                                                    d_flag);
    }
    HIP_ASSERT(hipGetLastError());
    const hipError_t sync = hipDeviceSynchronize();
    ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

    uint32_t out = 0;
    HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
    if (composite) {
      if (out == clean_expected)
        ++clean_outputs;
      else if (out == fault_alternate && injected)
        ++fault_alternate_outputs;
      else
        ++invalid_outputs;
      uint32_t report = 0;
      HIP_ASSERT(hipMemcpy(&report, d_report, sizeof(report), hipMemcpyDeviceToHost));
      ASSERT_TRUE(report == 0u || report == kScCompositeReportMarker)
          << "invalid SC marker value on launch " << launch << ": " << report;
      detections += report == kScCompositeReportMarker ? 1u : 0u;
    } else if (out != legacy_expected) {
      free_device_ptr(d_flag);
      free_device_ptr(d_acquire_counter);
      free_device_ptr(d_release_counter);
      free_device_ptr(d_out);
      ADD_FAILURE() << "launch " << launch << " output mismatch: expected " << legacy_expected
                    << ", observed " << out;
      return;
    }
  }
  free_device_ptr(d_report);
  free_device_ptr(d_flag);
  free_device_ptr(d_acquire_counter);
  free_device_ptr(d_release_counter);
  free_device_ptr(d_out);
  if (composite) {
    write_atomic_composite_oracle("release-acquire-atomic-rmw-composite", injected, launches,
                                  detections, clean_outputs, fault_alternate_outputs,
                                  invalid_outputs, clean_expected, fault_alternate);
    EXPECT_EQ(invalid_outputs, 0u);
    if (!injected) {
      EXPECT_EQ(clean_outputs, launches);
      EXPECT_EQ(detections, 0u);
    }
  } else {
    write_sc_clean_oracle(system_scope ? "system-release-acquire-atomic-handoff"
                                       : "release-acquire-atomic-handoff",
                          launches);
  }
}

void run_padded_atomic_collision_replaces_release() {
  uint32_t *d_out = nullptr;
  uint32_t *d_counters = nullptr;
  uint32_t *d_flag = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_counters, 65 * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_flag, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_counters, 0, 65 * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_flag, 0, sizeof(uint32_t)));

  padded_atomic_collision_replaces_release<<<1, 64>>>(d_out, d_counters, d_flag);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_flag);
  free_device_ptr(d_counters);
  free_device_ptr(d_out);
  EXPECT_EQ(out, 0xC0111DE0u);
}

void run_padded_atomic_cas_failure_does_not_publish(bool vglobal = false) {
  const bool composite = std::getenv("CONSAN_SC_COMPOSITE_LAUNCHES") != nullptr;
  const uint32_t launches = composite ? sc_composite_launch_count() : 1u;
  ASSERT_NE(launches, 0u);
  uint32_t *d_out = nullptr;
  uint32_t *d_counters = nullptr;
  uint32_t *d_flag = nullptr;
  uint32_t *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, 3 * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_counters, 65 * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_flag, sizeof(uint32_t)));
  if (composite)
    HIP_ASSERT(hipMalloc(&d_report, sizeof(uint32_t)));
  ScCompositeReportEnvGuard report_env{composite};
  if (composite)
    ASSERT_TRUE(set_sc_composite_report_buffer(d_report));
  const bool injected = composite && atomic_fault_injected();
  constexpr uint32_t kCleanExpected = 0xCA500000u;
  constexpr uint32_t kFaultAlternate = 0xCA500001u;
  uint32_t detections = 0;
  uint32_t clean_outputs = 0;
  uint32_t fault_alternate_outputs = 0;
  uint32_t invalid_outputs = 0;
  uint32_t successful_cas_return_failures = 0;
  uint32_t failed_cas_return_failures = 0;
  uint32_t last_out[3]{};
  for (uint32_t launch = 0; launch < launches; ++launch) {
    HIP_ASSERT(hipMemset(d_out, 0, 3 * sizeof(uint32_t)));
    HIP_ASSERT(hipMemset(d_counters, 0, 65 * sizeof(uint32_t)));
    HIP_ASSERT(hipMemset(d_flag, 0, sizeof(uint32_t)));
    if (composite)
      HIP_ASSERT(hipMemset(d_report, 0, sizeof(uint32_t)));
    const uint32_t one = 1;
    HIP_ASSERT(hipMemcpy(d_counters + 64, &one, sizeof(one), hipMemcpyHostToDevice));
    if (vglobal)
      padded_vglobal_cas_failure_does_not_publish<<<1, 64>>>(d_out, d_counters, d_flag);
    else
      padded_atomic_cas_failure_does_not_publish<<<1, 64>>>(d_out, d_counters, d_flag);
    HIP_ASSERT(hipGetLastError());
    const hipError_t sync = hipDeviceSynchronize();
    ASSERT_EQ(sync, hipSuccess) << "HIP error on launch " << launch << ": "
                                << hipGetErrorString(sync);
    HIP_ASSERT(hipMemcpy(last_out, d_out, sizeof(last_out), hipMemcpyDeviceToHost));
    if (composite) {
      if (last_out[0] == kCleanExpected)
        ++clean_outputs;
      else if (last_out[0] == kFaultAlternate && injected)
        ++fault_alternate_outputs;
      else
        ++invalid_outputs;
      successful_cas_return_failures += last_out[1] != 0u ? 1u : 0u;
      failed_cas_return_failures += last_out[2] != 1u ? 1u : 0u;
      uint32_t report = 0;
      HIP_ASSERT(hipMemcpy(&report, d_report, sizeof(report), hipMemcpyDeviceToHost));
      ASSERT_TRUE(report == 0u || report == kScCompositeReportMarker)
          << "invalid SC marker value on launch " << launch << ": " << report;
      detections += report == kScCompositeReportMarker ? 1u : 0u;
    }
  }
  free_device_ptr(d_report);
  free_device_ptr(d_flag);
  free_device_ptr(d_counters);
  free_device_ptr(d_out);
  const bool injected_wrong_address =
      std::getenv("RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS") != nullptr &&
      std::getenv("RJ_CONSAN_FAULT_DRY_RUN") == nullptr;
  const uint32_t expected[3] = {injected_wrong_address ? kFaultAlternate : kCleanExpected, 0u, 1u};
  if (composite) {
    write_atomic_composite_oracle("successful-returning-cas-composite", injected, launches,
                                  detections, clean_outputs, fault_alternate_outputs,
                                  invalid_outputs, kCleanExpected, kFaultAlternate, true,
                                  successful_cas_return_failures, failed_cas_return_failures);
    EXPECT_EQ(invalid_outputs, 0u);
    EXPECT_EQ(successful_cas_return_failures, 0u);
    EXPECT_EQ(failed_cas_return_failures, 0u);
    if (!injected) {
      EXPECT_EQ(clean_outputs, launches);
      EXPECT_EQ(detections, 0u);
    }
  } else {
    write_atomic_cas_oracle(last_out, expected);
    EXPECT_EQ(last_out[0], expected[0]);
    EXPECT_EQ(last_out[1], expected[1]);
    EXPECT_EQ(last_out[2], expected[2]);
  }
}

} // namespace

TEST(ConSanInlineShadowTest, CrossWaveRacyStoreCompletes) {
  const uint32_t launches = inline_versioned_launch_count();
  ASSERT_NE(launches, 0u);
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  for (uint32_t launch = 0; launch < launches; ++launch) {
    SCOPED_TRACE(::testing::Message() << "launch " << launch);
    HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
    padded_cross_wave_lds_store<<<1, 128>>>(d_out);
    HIP_ASSERT(hipGetLastError());
    const hipError_t sync = hipDeviceSynchronize();
    ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

    uint32_t out = 0;
    HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
    EXPECT_EQ(out & 0xFFFFFF80u, 0xA5A50000u);
    EXPECT_LE(out & 0x7Fu, 127u);
  }
  free_device_ptr(d_out);
}

TEST(ConSanInlineShadowTest, BarrierOrderedCrossWaveStoreCompletes) {
  const uint32_t launches = inline_versioned_launch_count();
  ASSERT_NE(launches, 0u);
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  for (uint32_t launch = 0; launch < launches; ++launch) {
    SCOPED_TRACE(::testing::Message() << "launch " << launch);
    HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
    padded_barrier_ordered_lds_store<<<1, 128>>>(d_out);
    HIP_ASSERT(hipGetLastError());
    const hipError_t sync = hipDeviceSynchronize();
    ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

    uint32_t out = 0;
    HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
    EXPECT_EQ(out, 0xFACE0040u);
  }
  free_device_ptr(d_out);
}

TEST(ConSanInlineShadowTest, MarkerlessBarrierMoveControlPreservesExactOutput) {
  const bool composite = std::getenv("CONSAN_SC_COMPOSITE_LAUNCHES") != nullptr;
  const bool injected = std::getenv("RJ_CONSAN_FAULT_MOVE_BARRIER") != nullptr;
  const uint32_t launches = sc_composite_launch_count();
  ASSERT_NE(launches, 0u);
  uint32_t *d_out = nullptr;
  uint32_t *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  if (composite)
    HIP_ASSERT(hipMalloc(&d_report, sizeof(uint32_t)));
  ScCompositeReportEnvGuard report_env{composite};
  if (composite)
    ASSERT_TRUE(set_sc_composite_report_buffer(d_report));
  uint32_t detections = 0;
  uint32_t clean_outputs = 0;
  uint32_t fault_alternate_outputs = 0;
  uint32_t invalid_outputs = 0;
  uint32_t last_output = 0;
  for (uint32_t launch = 0; launch < launches; ++launch) {
    HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
    if (composite)
      HIP_ASSERT(hipMemset(d_report, 0, sizeof(uint32_t)));
    padded_markerless_barrier_move_control<<<1, 128>>>(d_out);
    HIP_ASSERT(hipGetLastError());
    const hipError_t sync = hipDeviceSynchronize();
    ASSERT_EQ(sync, hipSuccess) << "HIP error on launch " << launch << ": "
                                << hipGetErrorString(sync);
    HIP_ASSERT(hipMemcpy(&last_output, d_out, sizeof(last_output), hipMemcpyDeviceToHost));
    if (last_output == kBarrierCompositeCleanOutput)
      ++clean_outputs;
    else if (last_output == kBarrierCompositeFaultAlternate && injected)
      ++fault_alternate_outputs;
    else
      ++invalid_outputs;
    if (composite) {
      uint32_t report = 0;
      HIP_ASSERT(hipMemcpy(&report, d_report, sizeof(report), hipMemcpyDeviceToHost));
      ASSERT_TRUE(report == 0u || report == kScCompositeReportMarker)
          << "invalid SC marker value on launch " << launch << ": " << report;
      detections += report == kScCompositeReportMarker ? 1u : 0u;
    }
  }
  if (d_report != nullptr)
    (void)hipFree(d_report);
  free_device_ptr(d_out);
  if (composite) {
    write_barrier_composite_oracle(injected, launches, detections, clean_outputs,
                                   fault_alternate_outputs, invalid_outputs);
    EXPECT_EQ(invalid_outputs, 0u);
    if (!injected) {
      EXPECT_EQ(clean_outputs, launches);
      EXPECT_EQ(detections, 0u);
    }
  } else {
    EXPECT_TRUE(write_barrier_move_oracle(last_output));
    if (!injected)
      EXPECT_EQ(last_output, kBarrierCompositeCleanOutput);
  }
}

TEST(ConSanInlineShadowTest, StructuredDivergentBarrierControlPreservesExactOutput) {
  constexpr uint32_t kExpected = 0xD1A0001Fu;
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));

  padded_structured_divergent_barrier_control<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  EXPECT_TRUE(write_structured_divergent_barrier_oracle(out));
  EXPECT_EQ(out, kExpected);
}

TEST(ConSanInlineShadowTest, FenceAndBarrierOrderedAccessesPreserveExactOutput) {
  const uint32_t launches = sc_clean_launch_count();
  ASSERT_NE(launches, 0u);
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  constexpr uint32_t kExpected = 0xFECE0001u;
  for (uint32_t launch = 0; launch < launches; ++launch) {
    HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
    padded_fence_barrier_ordered_lds_accesses<<<1, 64>>>(d_out);
    HIP_ASSERT(hipGetLastError());
    HIP_ASSERT(hipDeviceSynchronize());
    uint32_t out = 0;
    HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
    if (out != kExpected) {
      free_device_ptr(d_out);
      ADD_FAILURE() << "launch " << launch << " output mismatch: expected " << kExpected
                    << ", observed " << out;
      return;
    }
  }
  free_device_ptr(d_out);
  write_sc_clean_oracle("two-wave-fence-barrier", launches);
}

TEST(ConSanInlineShadowTest, CrossWaveRacyB128StoreCompletes) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, 4 * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, 4 * sizeof(uint32_t)));

  padded_cross_wave_lds_store_b128<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  uint32_t out[4]{};
  HIP_ASSERT(hipMemcpy(out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  const uint32_t lane = out[0] & 0x3Fu;
  EXPECT_EQ(out[0], 0xA0000000u | lane);
  EXPECT_EQ(out[1], 0xB0000000u | lane);
  EXPECT_EQ(out[2], 0xC0000000u | lane);
  EXPECT_EQ(out[3], 0xD0000000u | lane);
}

TEST(ConSanInlineShadowTest, CrossWaveRacyFlatLdsStoreCompletes) {
  uint32_t *out = nullptr;
  HIP_ASSERT(hipMalloc(&out, sizeof(*out)));
  HIP_ASSERT(hipMemset(out, 0, sizeof(*out)));
  padded_cross_wave_flat_lds_store<<<1, 64>>>(out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t host_out = 0;
  HIP_ASSERT(hipMemcpy(&host_out, out, sizeof(host_out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(out));
  EXPECT_EQ(host_out, 0xF1A70000u);
}

namespace {

enum class CoalescingControl {
  B32Same,
  B32Divergent,
  B128Same,
  B128Divergent,
  GroupFlatSame,
  GroupFlatDivergent,
};

void run_inline_shadow_coalescing_control(CoalescingControl control, const char *name,
                                          const char *shape, const char *topology,
                                          uint32_t expected, uint32_t cells_per_lane,
                                          uint32_t distinct_cells) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  switch (control) {
  case CoalescingControl::B32Same:
    padded_coalesced_b32_same_address<<<1, 32>>>(d_out);
    break;
  case CoalescingControl::B32Divergent:
    padded_coalesced_b32_divergent_address<<<1, 32>>>(d_out);
    break;
  case CoalescingControl::B128Same:
    padded_coalesced_b128_same_address<<<1, 32>>>(d_out);
    break;
  case CoalescingControl::B128Divergent:
    padded_coalesced_b128_divergent_address<<<1, 32>>>(d_out);
    break;
  case CoalescingControl::GroupFlatSame:
    padded_coalesced_group_flat_same_address<<<1, 32>>>(d_out);
    break;
  case CoalescingControl::GroupFlatDivergent:
    padded_coalesced_group_flat_divergent_address<<<1, 32>>>(d_out);
    break;
  }
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t actual = 0;
  HIP_ASSERT(hipMemcpy(&actual, d_out, sizeof(actual), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(d_out));
  write_inline_shadow_coalescing_oracle(name, shape, topology, actual, expected, 32u,
                                        cells_per_lane, distinct_cells);
  EXPECT_EQ(actual, expected);
}

} // namespace

TEST(ConSanInlineShadowTest, CoalescedB32SameAddressPreservesExactOutput) {
  run_inline_shadow_coalescing_control(CoalescingControl::B32Same, "b32-same-address",
                                       "native-b32-single-cell", "same-address", 0xB3200001u, 1u,
                                       1u);
}

TEST(ConSanInlineShadowTest, CoalescedB32DivergentAddressPreservesExactOutput) {
  run_inline_shadow_coalescing_control(CoalescingControl::B32Divergent, "b32-divergent-address",
                                       "native-b32-single-cell", "divergent-address", 0xB32D0001u,
                                       1u, 32u);
}

TEST(ConSanInlineShadowTest, CoalescedB128SameAddressPreservesExactOutput) {
  run_inline_shadow_coalescing_control(CoalescingControl::B128Same, "b128-same-address",
                                       "native-b128-multi-cell", "same-address", 0xB1280001u, 4u,
                                       4u);
}

TEST(ConSanInlineShadowTest, CoalescedB128DivergentAddressPreservesExactOutput) {
  run_inline_shadow_coalescing_control(CoalescingControl::B128Divergent, "b128-divergent-address",
                                       "native-b128-multi-cell", "divergent-address", 0xB1280002u,
                                       4u, 128u);
}

TEST(ConSanInlineShadowTest, CoalescedGroupFlatSameAddressPreservesExactOutput) {
  run_inline_shadow_coalescing_control(CoalescingControl::GroupFlatSame,
                                       "group-flat-b32-same-address", "normalized-group-flat-b32",
                                       "same-address", 0xF1A70001u, 1u, 1u);
}

TEST(ConSanInlineShadowTest, CoalescedGroupFlatDivergentAddressPreservesExactOutput) {
  run_inline_shadow_coalescing_control(
      CoalescingControl::GroupFlatDivergent, "group-flat-b32-divergent-address",
      "normalized-group-flat-b32", "divergent-address", 0xF1AD0001u, 1u, 32u);
}

TEST(ConSanInlineShadowTest, SameAddressAtomicHandoffCompletes) {
  run_padded_atomic_ordered_lds_accesses(/*same_atomic_address=*/true);
}

TEST(ConSanInlineShadowTest, WrongAddressAtomicHandoffCompletes) {
  run_padded_atomic_ordered_lds_accesses(/*same_atomic_address=*/false);
}

TEST(ConSanInlineShadowTest, SystemScopeSameAddressAtomicHandoffCompletes) {
  run_padded_atomic_ordered_lds_accesses(/*same_atomic_address=*/true,
                                         /*system_scope=*/true);
}

TEST(ConSanInlineShadowTest, CollidingReleaseReplacementCompletes) {
  run_padded_atomic_collision_replaces_release();
}

TEST(ConSanInlineShadowTest, FailedCasDoesNotReplaceReleaseCompletes) {
  run_padded_atomic_cas_failure_does_not_publish();
}

TEST(ConSanInlineShadowTest, VglobalFailedCasDoesNotReplaceReleaseCompletes) {
  run_padded_atomic_cas_failure_does_not_publish(/*vglobal=*/true);
}
