// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_lds_test.hip
/// @brief Real-HIP smoke for ConSan DBI of a padded LDS store.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <hip/hip_runtime.h>
#include <string_view>
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
#define RJ_CONSAN_TEST_DS_LOAD_B32 "ds_read_b32 %[loaded], %[lds_addr]\n\t"
#define RJ_CONSAN_TEST_FLAT_STORE_B32 "flat_store_dword v[0:1], %0\n\t"
#define RJ_CONSAN_TEST_FLAT_STORE_B32_NAMED "flat_store_dword v[0:1], %[value]\n\t"
#define RJ_CONSAN_TEST_FLAT_STORE_LOW_B8 "flat_store_byte v[0:1], %0\n\t"
#define RJ_CONSAN_TEST_V_ADD_U32 "v_add_u32 v0, v0, %[offset]\n\t"
#define RJ_CONSAN_TEST_FLAT_STORE_HIGH_B8 "flat_store_byte_d16_hi v[0:1], %[value]\n\t"
#define RJ_CONSAN_TEST_FLAT_STORE_HIGH_B16 "flat_store_short_d16_hi v[0:1], %0\n\t"
#define RJ_CONSAN_TEST_FLAT_LOAD_HIGH_B16 "flat_load_short_d16_hi %0, v[0:1]\n\t"
#define RJ_CONSAN_TEST_FLAT_LOAD_HIGH_I8 "flat_load_sbyte_d16_hi %0, v[0:1]\n\t"
#define RJ_CONSAN_TEST_FLAT_LOAD_WAIT "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
#define RJ_CONSAN_TEST_SET_VCC_TRUE "v_cmp_eq_u32_e32 vcc, %[value], %[value]\n\t"
#else
#define RJ_CONSAN_TEST_WORKGROUP_BARRIER                                                           \
  "s_barrier_signal -1\n\t"                                                                        \
  "s_barrier_wait -1\n\t"
#define RJ_CONSAN_TEST_DS_STORE_B32 "ds_store_b32 %0, %1\n\t"
#define RJ_CONSAN_TEST_DS_LOAD_B32 "ds_load_b32 %[loaded], %[lds_addr]\n\t"
#define RJ_CONSAN_TEST_FLAT_STORE_B32 "flat_store_b32 v[0:1], %0\n\t"
#define RJ_CONSAN_TEST_FLAT_STORE_B32_NAMED "flat_store_b32 v[0:1], %[value]\n\t"
#define RJ_CONSAN_TEST_FLAT_STORE_LOW_B8 "flat_store_b8 v[0:1], %0\n\t"
#define RJ_CONSAN_TEST_V_ADD_U32 "v_add_nc_u32 v0, v0, %[offset]\n\t"
#define RJ_CONSAN_TEST_FLAT_STORE_HIGH_B8 "flat_store_d16_hi_b8 v[0:1], %[value]\n\t"
#define RJ_CONSAN_TEST_FLAT_STORE_HIGH_B16 "flat_store_d16_hi_b16 v[0:1], %0\n\t"
#define RJ_CONSAN_TEST_FLAT_LOAD_HIGH_B16 "flat_load_d16_hi_b16 %0, v[0:1]\n\t"
#define RJ_CONSAN_TEST_FLAT_LOAD_HIGH_I8 "flat_load_d16_hi_i8 %0, v[0:1]\n\t"
#define RJ_CONSAN_TEST_FLAT_LOAD_WAIT "s_wait_loadcnt_dscnt 0\n\t"
#define RJ_CONSAN_TEST_SET_VCC_TRUE "v_cmp_eq_u32_e32 %[value], %[value]\n\t"
#endif

#if defined(__gfx942__) || defined(__gfx950__)
#define RJ_CONSAN_TEST_KEEP_HIGH_ORDINARY_SGPRS_LIVE ""
#else
#define RJ_CONSAN_TEST_KEEP_HIGH_ORDINARY_SGPRS_LIVE                                               \
  "s_mov_b64 s[102:103], s[102:103]\n\t"                                                           \
  "s_mov_b64 s[104:105], s[104:105]\n\t"
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

#define RJ_CONSAN_TEST_KEEP_ORDINARY_SGPRS_LIVE                                                    \
  "s_mov_b64 s[0:1], s[0:1]\n\t"                                                                   \
  "s_mov_b64 s[2:3], s[2:3]\n\t"                                                                   \
  "s_mov_b64 s[4:5], s[4:5]\n\t"                                                                   \
  "s_mov_b64 s[6:7], s[6:7]\n\t"                                                                   \
  "s_mov_b64 s[8:9], s[8:9]\n\t"                                                                   \
  "s_mov_b64 s[10:11], s[10:11]\n\t"                                                               \
  "s_mov_b64 s[12:13], s[12:13]\n\t"                                                               \
  "s_mov_b64 s[14:15], s[14:15]\n\t"                                                               \
  "s_mov_b64 s[16:17], s[16:17]\n\t"                                                               \
  "s_mov_b64 s[18:19], s[18:19]\n\t"                                                               \
  "s_mov_b64 s[20:21], s[20:21]\n\t"                                                               \
  "s_mov_b64 s[22:23], s[22:23]\n\t"                                                               \
  "s_mov_b64 s[24:25], s[24:25]\n\t"                                                               \
  "s_mov_b64 s[26:27], s[26:27]\n\t"                                                               \
  "s_mov_b64 s[28:29], s[28:29]\n\t"                                                               \
  "s_mov_b64 s[30:31], s[30:31]\n\t"                                                               \
  "s_mov_b64 s[32:33], s[32:33]\n\t"                                                               \
  "s_mov_b64 s[34:35], s[34:35]\n\t"                                                               \
  "s_mov_b64 s[36:37], s[36:37]\n\t"                                                               \
  "s_mov_b64 s[38:39], s[38:39]\n\t"                                                               \
  "s_mov_b64 s[40:41], s[40:41]\n\t"                                                               \
  "s_mov_b64 s[42:43], s[42:43]\n\t"                                                               \
  "s_mov_b64 s[44:45], s[44:45]\n\t"                                                               \
  "s_mov_b64 s[46:47], s[46:47]\n\t"                                                               \
  "s_mov_b64 s[48:49], s[48:49]\n\t"                                                               \
  "s_mov_b64 s[50:51], s[50:51]\n\t"                                                               \
  "s_mov_b64 s[52:53], s[52:53]\n\t"                                                               \
  "s_mov_b64 s[54:55], s[54:55]\n\t"                                                               \
  "s_mov_b64 s[56:57], s[56:57]\n\t"                                                               \
  "s_mov_b64 s[58:59], s[58:59]\n\t"                                                               \
  "s_mov_b64 s[60:61], s[60:61]\n\t"                                                               \
  "s_mov_b64 s[62:63], s[62:63]\n\t"                                                               \
  "s_mov_b64 s[64:65], s[64:65]\n\t"                                                               \
  "s_mov_b64 s[66:67], s[66:67]\n\t"                                                               \
  "s_mov_b64 s[68:69], s[68:69]\n\t"                                                               \
  "s_mov_b64 s[70:71], s[70:71]\n\t"                                                               \
  "s_mov_b64 s[72:73], s[72:73]\n\t"                                                               \
  "s_mov_b64 s[74:75], s[74:75]\n\t"                                                               \
  "s_mov_b64 s[76:77], s[76:77]\n\t"                                                               \
  "s_mov_b64 s[78:79], s[78:79]\n\t"                                                               \
  "s_mov_b64 s[80:81], s[80:81]\n\t"                                                               \
  "s_mov_b64 s[82:83], s[82:83]\n\t"                                                               \
  "s_mov_b64 s[84:85], s[84:85]\n\t"                                                               \
  "s_mov_b64 s[86:87], s[86:87]\n\t"                                                               \
  "s_mov_b64 s[88:89], s[88:89]\n\t"                                                               \
  "s_mov_b64 s[90:91], s[90:91]\n\t"                                                               \
  "s_mov_b64 s[92:93], s[92:93]\n\t"                                                               \
  "s_mov_b64 s[94:95], s[94:95]\n\t"                                                               \
  "s_mov_b64 s[96:97], s[96:97]\n\t"                                                               \
  "s_mov_b64 s[98:99], s[98:99]\n\t"                                                               \
  "s_mov_b64 s[100:101], s[100:101]\n\t" RJ_CONSAN_TEST_KEEP_HIGH_ORDINARY_SGPRS_LIVE

#define RJ_CONSAN_TEST_KEEP_ODD_VGPRS_LIVE                                                         \
  ".irp r,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,"        \
  "59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,"        \
  "113,115,117,119,121,123,125,127,129,131,133,135,137,139,141,143,145,147,149,151,153,155,"       \
  "157,159,161,163,165,167,169,171,173,175,177,179,181,183,185,187,189,191,193,195,197,199,"       \
  "201,203,205,207,209,211,213,215,217,219,221,223,225,227,229,231,233,235,237,239,241,243,"       \
  "245,247,249,251,253,255\n\t"                                                                     \
  "v_mov_b32 v\\r, v\\r\n\t"                                                                        \
  ".endr\n\t"

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

__global__ void
padded_zero_exec_flat_lds_store_preserves_scalar_pair_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  uint32_t before_lo = 0;
  uint32_t before_hi = 0;
  uint32_t after_lo = 0;
  uint32_t after_hi = 0;
  const uint32_t value = 0x51A7C0DEu;
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t"
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t"
               "v_mov_b32 %[before_lo], s2\n\t"
               "v_mov_b32 %[before_hi], s3\n\t"
               "s_mov_b64 vcc, exec\n\t"
               "s_mov_b64 exec, 0\n\t" RJ_CONSAN_TEST_FLAT_STORE_B32_NAMED RJ_CONSAN_TEST_NOP_CAVE
               "s_mov_b64 exec, vcc\n\t" RJ_CONSAN_TEST_KEEP_ORDINARY_SGPRS_LIVE
               "v_mov_b32 %[after_lo], s2\n\t"
               "v_mov_b32 %[after_hi], s3\n\t"
               : [before_lo] "=&v"(before_lo), [before_hi] "=&v"(before_hi),
                 [after_lo] "=&v"(after_lo), [after_hi] "=&v"(after_hi)
               : [value] "v"(value)
               : "s14", "s15", "v0", "v1", "vcc", "memory");
  out[0] = before_lo;
  out[1] = before_hi;
  out[2] = after_lo;
  out[3] = after_hi;
  out[4] = lds[0];
}

__global__ void
padded_full_pressure_flat_lds_store_preserves_registers_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  uint32_t scalar_before_lo = 0;
  uint32_t scalar_before_hi = 0;
  uint32_t scalar_after_lo = 0;
  uint32_t scalar_after_hi = 0;
  uint32_t vector_after_three = 0;
  uint32_t vector_after_five = 0;
  const uint32_t value = 0x51A7C0DEu;
  asm volatile("" : : "v"(&lds[0]) : "memory");
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t"
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t"
               "v_mov_b32 v3, 0x13579bdf\n\t"
               "v_mov_b32 v5, 0x2468ace0\n\t"
               "v_mov_b32 %[scalar_before_lo], s2\n\t"
               "v_mov_b32 %[scalar_before_hi], s3\n\t" RJ_CONSAN_TEST_FLAT_STORE_B32_NAMED
                   RJ_CONSAN_TEST_NOP_CAVE RJ_CONSAN_TEST_KEEP_ORDINARY_SGPRS_LIVE
                       RJ_CONSAN_TEST_KEEP_ODD_VGPRS_LIVE
               "v_mov_b32 %[scalar_after_lo], s2\n\t"
               "v_mov_b32 %[scalar_after_hi], s3\n\t"
               "v_mov_b32 %[vector_after_three], v3\n\t"
               "v_mov_b32 %[vector_after_five], v5\n\t"
               : [scalar_before_lo] "=&v"(scalar_before_lo),
                 [scalar_before_hi] "=&v"(scalar_before_hi),
                 [scalar_after_lo] "=&v"(scalar_after_lo),
                 [scalar_after_hi] "=&v"(scalar_after_hi),
                 [vector_after_three] "=&v"(vector_after_three),
                 [vector_after_five] "=&v"(vector_after_five)
               : [value] "v"(value)
               : "s14", "s15", "v0", "v1", "v3", "v5", "memory");
  out[0] = scalar_before_lo;
  out[1] = scalar_before_hi;
  out[2] = scalar_after_lo;
  out[3] = scalar_after_hi;
  out[4] = vector_after_three;
  out[5] = vector_after_five;
}

__global__ __launch_bounds__(64) void
padded_dynamic_stack_full_pressure_flat_lds_store_preserves_registers_for_instrumentation(
    uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const size_t private_count = (tid & 7u) + 1u;
  auto *private_words =
      static_cast<volatile uint32_t *>(__builtin_alloca(private_count * sizeof(uint32_t)));
  const uint32_t private_index = tid % private_count;
  const uint32_t private_value = 0xD1A60000u ^ tid;
  private_words[private_index] = private_value;

  uint32_t scalar_before_lo = 0;
  uint32_t scalar_before_hi = 0;
  uint32_t scalar_after_lo = 0;
  uint32_t scalar_after_hi = 0;
  uint32_t vector_after_three = 0;
  uint32_t vector_after_five = 0;
  uint32_t vcc_survived = 0;
  uint32_t scc_survived = 0;
  const uint32_t value = 0x51A7C0DEu;
  asm volatile("" : : "v"(&lds[0]) : "memory");
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t"
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t"
               "v_mov_b32 v3, 0x13579bdf\n\t"
               "v_mov_b32 v5, 0x2468ace0\n\t"
               "v_mov_b32 %[scalar_before_lo], s2\n\t"
               "v_mov_b32 %[scalar_before_hi], s3\n\t"
               "s_cmp_eq_u32 s2, s2\n\t" RJ_CONSAN_TEST_SET_VCC_TRUE
                   RJ_CONSAN_TEST_FLAT_STORE_B32_NAMED RJ_CONSAN_TEST_NOP_CAVE
                       RJ_CONSAN_TEST_KEEP_ORDINARY_SGPRS_LIVE RJ_CONSAN_TEST_KEEP_ODD_VGPRS_LIVE
               "v_mov_b32 %[scalar_after_lo], s2\n\t"
               "v_mov_b32 %[scalar_after_hi], s3\n\t"
               "v_mov_b32 %[vector_after_three], v3\n\t"
               "v_mov_b32 %[vector_after_five], v5\n\t"
               "s_cbranch_vccz 1f\n\t"
               "v_mov_b32 %[vcc_survived], 1\n\t"
               "1:\n\t"
               "s_cbranch_scc0 2f\n\t"
               "v_mov_b32 %[scc_survived], 1\n\t"
               "2:\n\t"
               : [scalar_before_lo] "=&v"(scalar_before_lo),
                 [scalar_before_hi] "=&v"(scalar_before_hi),
                 [scalar_after_lo] "=&v"(scalar_after_lo),
                 [scalar_after_hi] "=&v"(scalar_after_hi),
                 [vector_after_three] "=&v"(vector_after_three),
                 [vector_after_five] "=&v"(vector_after_five),
                 [vcc_survived] "+v"(vcc_survived), [scc_survived] "+v"(scc_survived)
               : [value] "v"(value)
               : "s14", "s15", "v0", "v1", "v3", "v5", "vcc", "memory");
  out[0] = scalar_before_lo;
  out[1] = scalar_before_hi;
  out[2] = scalar_after_lo;
  out[3] = scalar_after_hi;
  out[4] = vector_after_three;
  out[5] = vector_after_five;
  out[6] = private_words[private_index];
  out[7] = vcc_survived;
  out[8] = scc_survived;
}

__global__ void
padded_zero_exec_native_lds_load_preserves_allocated_scalars_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  uint32_t loaded = 0;
  uint32_t before[8] = {};
  uint32_t after[8] = {};
  const uint32_t lds_addr = 0;
  asm volatile("v_mov_b32 %[before0], s0\n\t"
               "v_mov_b32 %[before1], s1\n\t"
               "v_mov_b32 %[before2], s2\n\t"
               "v_mov_b32 %[before3], s3\n\t"
               "v_mov_b32 %[before4], s4\n\t"
               "v_mov_b32 %[before5], s5\n\t"
               "v_mov_b32 %[before6], s6\n\t"
               "v_mov_b32 %[before7], s7\n\t"
               "s_mov_b64 vcc, exec\n\t"
               "s_mov_b64 exec, 0\n\t" RJ_CONSAN_TEST_DS_LOAD_B32 RJ_CONSAN_TEST_NOP_CAVE
               "s_mov_b64 exec, vcc\n\t" RJ_CONSAN_TEST_KEEP_ORDINARY_SGPRS_LIVE
               "v_mov_b32 %[after0], s0\n\t"
               "v_mov_b32 %[after1], s1\n\t"
               "v_mov_b32 %[after2], s2\n\t"
               "v_mov_b32 %[after3], s3\n\t"
               "v_mov_b32 %[after4], s4\n\t"
               "v_mov_b32 %[after5], s5\n\t"
               "v_mov_b32 %[after6], s6\n\t"
               "v_mov_b32 %[after7], s7\n\t"
               : [loaded] "=&v"(loaded), [before0] "=&v"(before[0]), [before1] "=&v"(before[1]),
                 [before2] "=&v"(before[2]), [before3] "=&v"(before[3]), [before4] "=&v"(before[4]),
                 [before5] "=&v"(before[5]), [before6] "=&v"(before[6]), [before7] "=&v"(before[7]),
                 [after0] "=&v"(after[0]), [after1] "=&v"(after[1]), [after2] "=&v"(after[2]),
                 [after3] "=&v"(after[3]), [after4] "=&v"(after[4]), [after5] "=&v"(after[5]),
                 [after6] "=&v"(after[6]), [after7] "=&v"(after[7])
               : [lds_addr] "v"(lds_addr)
               : "vcc", "memory");
  for (uint32_t index = 0; index < 8; ++index) {
    out[index] = before[index];
    out[index + 8] = after[index];
  }
  out[16] = loaded;
  asm volatile("" : : "v"(&lds[0]) : "memory");
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

__device__ __forceinline__ uint32_t padded_flat_high_b16_load() {
  uint32_t value = 0x00005AA5u;
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t"
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t" RJ_CONSAN_TEST_FLAT_LOAD_HIGH_B16 RJ_CONSAN_TEST_NOP_CAVE
                   RJ_CONSAN_TEST_FLAT_LOAD_WAIT
               : "+v"(value)
               :
               : "s14", "s15", "v0", "v1", "memory");
  return value;
}

__device__ __forceinline__ uint32_t padded_flat_high_i8_load() {
  uint32_t value = 0x00005AA5u;
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t"
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t" RJ_CONSAN_TEST_FLAT_LOAD_HIGH_I8 RJ_CONSAN_TEST_NOP_CAVE
                   RJ_CONSAN_TEST_FLAT_LOAD_WAIT
               : "+v"(value)
               :
               : "s14", "s15", "v0", "v1", "memory");
  return value;
}

__global__ void padded_flat_high_d16_loads_kernel_for_instrumentation(uint32_t *out) {
  __shared__ uint16_t lds[1];
  if (threadIdx.x == 0)
    lds[0] = 0xBEEFu;
  asm volatile("" : : "v"(&lds[0]) : "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    // Declare enough architecturally allocated VGPRs for the duplicate value
    // and independently normalized high half selected by automatic liveness,
    // including the ordinary report-buffer action used by the production hook.
    asm volatile("v_mov_b32 v8, 0" ::: "v8");
    out[0] = padded_flat_high_b16_load();
    out[1] = padded_flat_high_i8_load();
  }
}

__device__ __forceinline__ uint32_t padded_flat_high_b8_store(uint32_t value) {
  uint32_t vcc_survived = 0;
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t"
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t" RJ_CONSAN_TEST_SET_VCC_TRUE
                   RJ_CONSAN_TEST_FLAT_STORE_HIGH_B8 RJ_CONSAN_TEST_NOP_CAVE "s_cbranch_vccz 1f\n\t"
               "v_mov_b32 %[survived], 1\n\t"
               "1:\n\t"
               : [survived] "+v"(vcc_survived)
               : [value] "v"(value)
               : "s14", "s15", "v0", "v1", "vcc", "memory");
  return vcc_survived;
}

__device__ __forceinline__ void padded_flat_low_b8_store(uint32_t value) {
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t"
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t" RJ_CONSAN_TEST_FLAT_STORE_LOW_B8 RJ_CONSAN_TEST_NOP_CAVE
               :
               : "v"(value)
               : "s14", "s15", "v0", "v1", "memory");
}

__device__ __forceinline__ void padded_flat_low_b8_store_at(uint32_t byte_offset, uint32_t value) {
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t"
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t" RJ_CONSAN_TEST_V_ADD_U32
                   RJ_CONSAN_TEST_FLAT_STORE_LOW_B8 RJ_CONSAN_TEST_NOP_CAVE
               :
               : [offset] "v"(byte_offset), "v"(value)
               : "s14", "s15", "v0", "v1", "vcc", "memory");
}

__device__ __forceinline__ void padded_exact_byte_pair(bool adjacent) {
  // CDNA cannot grow an ordinary VGPR allocation through the accumulator
  // boundary. Keep one otherwise dead ordinary register declared so the
  // production liveness planner has a legal instrumentation window on every
  // supported target.
  asm volatile("v_mov_b32 v20, 0" ::: "v20");
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  if (tid != 0 && tid != 64)
    return;
  const uint32_t byte_offset = adjacent && tid == 64 ? 1u : 0u;
  padded_flat_low_b8_store_at(byte_offset, tid == 0 ? 0x11u : 0x22u);
}

__global__ __launch_bounds__(128) void padded_inline_exact_local_adjacent_bytes(uint32_t *out) {
  __shared__ uint32_t lds[1];
  asm volatile("" : : "v"(&lds[0]) : "memory");
  padded_exact_byte_pair(/*adjacent=*/true);
  __syncthreads();
  if (threadIdx.x == 0)
    out[0] = 0x1a11u;
}

__global__ __launch_bounds__(128) void padded_inline_exact_local_overlapping_byte(uint32_t *out) {
  __shared__ uint32_t lds[1];
  asm volatile("" : : "v"(&lds[0]) : "memory");
  padded_exact_byte_pair(/*adjacent=*/false);
  __syncthreads();
  if (threadIdx.x == 0)
    out[0] = 0x1a22u;
}

__global__ __launch_bounds__(128) void padded_inline_exact_external_adjacent_bytes(uint32_t *out) {
  extern __shared__ uint32_t lds[];
  asm volatile("" : : "v"(&lds[0]) : "memory");
  padded_exact_byte_pair(/*adjacent=*/true);
  __syncthreads();
  if (threadIdx.x == 0)
    out[0] = 0x1e11u;
}

__global__ __launch_bounds__(128) void padded_inline_exact_external_overlapping_byte(uint32_t *out) {
  extern __shared__ uint32_t lds[];
  asm volatile("" : : "v"(&lds[0]) : "memory");
  padded_exact_byte_pair(/*adjacent=*/false);
  __syncthreads();
  if (threadIdx.x == 0)
    out[0] = 0x1e22u;
}

__device__ __forceinline__ void padded_flat_high_b16_store(uint32_t value) {
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t"
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t" RJ_CONSAN_TEST_FLAT_STORE_HIGH_B16 RJ_CONSAN_TEST_NOP_CAVE
               :
               : "v"(value)
               : "s14", "s15", "v0", "v1", "memory");
}

__global__ void padded_flat_high_d16_store_b8_kernel_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  uint32_t vcc_survived = 0;
  if (threadIdx.x == 0)
    lds[0] = 0xAAAA5555u;
  asm volatile("" : : "v"(&lds[0]) : "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    // The production liveness planner chooses the readback, normalization, and
    // report scratch from the declared allocation.
    asm volatile("v_mov_b32 v8, 0" ::: "v8");
    vcc_survived = padded_flat_high_b8_store(0x00EF0000u);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    out[0] = lds[0];
    out[2] = vcc_survived;
  }
}

__global__ void padded_flat_high_d16_store_b16_kernel_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  if (threadIdx.x == 0)
    lds[0] = 0xAAAA5555u;
  asm volatile("" : : "v"(&lds[0]) : "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("v_mov_b32 v8, 0" ::: "v8");
    padded_flat_high_b16_store(0xBEEF0000u);
  }
  __syncthreads();
  if (threadIdx.x == 0)
    out[1] = lds[0];
}

__global__ void padded_flat_high_d16_store_low_b8_kernel_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  if (threadIdx.x == 0)
    lds[0] = 0xAAAA5555u;
  asm volatile("" : : "v"(&lds[0]) : "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("v_mov_b32 v8, 0" ::: "v8");
    padded_flat_low_b8_store(0x000000EFu);
  }
  __syncthreads();
  if (threadIdx.x == 0)
    out[3] = lds[0];
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

TEST(ConSanLdsTest, ZeroExecFlatSpillPreservesLiveScalarPair) {
  constexpr size_t kValues = 5;
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kValues * sizeof(*d_out)));
  HIP_ASSERT(hipMemset(d_out, 0, kValues * sizeof(*d_out)));

  padded_zero_exec_flat_lds_store_preserves_scalar_pair_for_instrumentation<<<1, 1>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::array<uint32_t, kValues> out{};
  HIP_ASSERT(hipMemcpy(out.data(), d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  EXPECT_EQ(out[0], out[2]);
  EXPECT_EQ(out[1], out[3]);
}

TEST(ConSanLdsTest, FullPressureFlatSpillPreservesLiveRegisterFiles) {
  constexpr size_t kValues = 6;
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kValues * sizeof(*d_out)));
  HIP_ASSERT(hipMemset(d_out, 0, kValues * sizeof(*d_out)));

  padded_full_pressure_flat_lds_store_preserves_registers_for_instrumentation<<<1, 1>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::array<uint32_t, kValues> out{};
  HIP_ASSERT(hipMemcpy(out.data(), d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  EXPECT_EQ(out[0], out[2]);
  EXPECT_EQ(out[1], out[3]);
  EXPECT_EQ(out[4], 0x13579BDFu);
  EXPECT_EQ(out[5], 0x2468ACE0u);
}

TEST(ConSanLdsTest, DynamicStackFullPressureFlatSpillPreservesLiveRegistersAndPrivateValues) {
  constexpr size_t kValues = 9;
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kValues * sizeof(*d_out)));
  HIP_ASSERT(hipMemset(d_out, 0, kValues * sizeof(*d_out)));

  padded_dynamic_stack_full_pressure_flat_lds_store_preserves_registers_for_instrumentation<<<1, 1>>>(
      d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::array<uint32_t, kValues> out{};
  HIP_ASSERT(hipMemcpy(out.data(), d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  EXPECT_EQ(out[0], out[2]);
  EXPECT_EQ(out[1], out[3]);
  EXPECT_EQ(out[4], 0x13579BDFu);
  EXPECT_EQ(out[5], 0x2468ACE0u);
  EXPECT_EQ(out[6], 0xD1A60000u);
  EXPECT_EQ(out[7], 1u);
  EXPECT_EQ(out[8], 1u);
}

TEST(ConSanLdsTest, ZeroExecNativeLdsSpillPreservesAllocatedScalars) {
  constexpr size_t kValues = 17;
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kValues * sizeof(*d_out)));
  HIP_ASSERT(hipMemset(d_out, 0, kValues * sizeof(*d_out)));

  padded_zero_exec_native_lds_load_preserves_allocated_scalars_for_instrumentation<<<1, 1>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::array<uint32_t, kValues> out{};
  HIP_ASSERT(hipMemcpy(out.data(), d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  for (size_t index = 0; index < 8; ++index)
    EXPECT_EQ(out[index], out[index + 8]) << "s" << index;
}

TEST(ConSanLdsTest, PaddedHighHalfD16FlatLoadsProduceExpectedResults) {
  constexpr size_t kValues = 2;
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kValues * sizeof(*d_out)));
  HIP_ASSERT(hipMemset(d_out, 0, kValues * sizeof(*d_out)));

  padded_flat_high_d16_loads_kernel_for_instrumentation<<<1, 1>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::array<uint32_t, kValues> out{};
  HIP_ASSERT(hipMemcpy(out.data(), d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  hipDeviceProp_t properties{};
  HIP_ASSERT(hipGetDeviceProperties(&properties, 0));
  const std::string_view arch(properties.gcnArchName);
  const bool cdna_sram_ecc = arch.starts_with("gfx942") || arch.starts_with("gfx950");
  const uint32_t expected_b16 = cdna_sram_ecc ? 0xBEEF0000u : 0xBEEF5AA5u;
  const uint32_t expected_i8 = cdna_sram_ecc ? 0xFFEF0000u : 0xFFEF5AA5u;
  // SRAM-ECC CDNA execution writes the selected half and clears the other;
  // RDNA preserves the untouched half. Instrumentation must retain either
  // target's native behavior, including signed-byte extension.
  EXPECT_EQ(out[0], expected_b16);
  EXPECT_EQ(out[1], expected_i8);
}

TEST(ConSanLdsTest, PaddedHighHalfD16FlatStoresProduceExpectedResults) {
  constexpr size_t kValues = 4;
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kValues * sizeof(*d_out)));
  HIP_ASSERT(hipMemset(d_out, 0, kValues * sizeof(*d_out)));

  padded_flat_high_d16_store_b8_kernel_for_instrumentation<<<1, 1>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  padded_flat_high_d16_store_b16_kernel_for_instrumentation<<<1, 1>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  padded_flat_high_d16_store_low_b8_kernel_for_instrumentation<<<1, 1>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::array<uint32_t, kValues> out{};
  HIP_ASSERT(hipMemcpy(out.data(), d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  EXPECT_EQ(out[0], 0xAAAA55EFu);
  EXPECT_EQ(out[1], 0xAAAABEEFu);
  EXPECT_EQ(out[2], 1u);
  EXPECT_EQ(out[3], 0xAAAA55EFu);
}

TEST(ConSanLdsTest, InlineExactLocalAdjacentBytesStayIndependent) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(*d_out)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(*d_out)));
  padded_inline_exact_local_adjacent_bytes<<<1, 128>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  EXPECT_EQ(out, 0x1a11u);
}

TEST(ConSanLdsTest, InlineExactLocalOverlappingByteCompletes) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(*d_out)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(*d_out)));
  padded_inline_exact_local_overlapping_byte<<<1, 128>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  EXPECT_EQ(out, 0x1a22u);
}

TEST(ConSanLdsTest, InlineExactExternalAdjacentBytesStayIndependent) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(*d_out)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(*d_out)));
  padded_inline_exact_external_adjacent_bytes<<<1, 128, sizeof(uint32_t)>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  EXPECT_EQ(out, 0x1e11u);
}

TEST(ConSanLdsTest, InlineExactExternalOverlappingByteCompletes) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(*d_out)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(*d_out)));
  padded_inline_exact_external_overlapping_byte<<<1, 128, sizeof(uint32_t)>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  EXPECT_EQ(out, 0x1e22u);
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
