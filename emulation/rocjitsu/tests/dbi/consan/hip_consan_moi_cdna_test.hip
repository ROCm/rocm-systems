// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_moi_cdna_test.hip
/// @brief CDNA3/CDNA4 smoke tests for hook-driven ConSan MOI instrumentation.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

#include "hip_consan_address_group_test.h"
#include "rocjitsu/code/patch/consan/consan_moi_abi.h"

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    const hipError_t err = (call);                                                                 \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

#define RJ_CONSAN_MOI_RECORD_NOP_CAVE                                                              \
  ".rept 128\n\t"                                                                                  \
  "s_nop 0\n\t"                                                                                    \
  ".endr\n\t"

namespace {

using MoiAccessRecord = rocjitsu::ConSanMoiAccessRecord;
using MoiAtomicRecord = rocjitsu::ConSanMoiAtomicRecord;
using MoiBarrierRecord = rocjitsu::ConSanMoiBarrierRecord;
using MoiFenceRecord = rocjitsu::ConSanMoiFenceRecord;
using MoiReportHeader = rocjitsu::ConSanMoiReportHeader;
using MoiSampledCausalWindow = rocjitsu::ConSanMoiSampledCausalWindow;
using MoiSampledPendingAcquireSlot = rocjitsu::ConSanMoiSampledPendingAcquireSlot;
using MoiSampledSyncMetadataPacked = rocjitsu::ConSanMoiSampledSyncMetadataPacked;

constexpr size_t moi_report_buffer_bytes(uint32_t access_record_capacity,
                                         uint32_t barrier_record_capacity = 0,
                                         uint32_t atomic_record_capacity = 0,
                                         uint32_t fence_record_capacity = 0) {
  return sizeof(MoiReportHeader) + access_record_capacity * sizeof(MoiAccessRecord) +
         barrier_record_capacity * sizeof(MoiBarrierRecord) +
         atomic_record_capacity * sizeof(MoiAtomicRecord) +
         fence_record_capacity * sizeof(MoiFenceRecord);
}

constexpr size_t moi_sampled_report_buffer_bytes(uint32_t capacity) {
  return sizeof(MoiReportHeader) +
         capacity * (sizeof(MoiSampledCausalWindow) + sizeof(uint64_t) +
                     sizeof(MoiSampledSyncMetadataPacked) + sizeof(MoiSampledPendingAcquireSlot));
}

struct SampledEntry {
  bool valid;
  uint32_t kind;
  uint32_t epoch;
  uint32_t generation;
  uint32_t cell_count;
};

SampledEntry sampled_entry(const std::vector<uint8_t> &report, uint32_t capacity, uint32_t index) {
  uint64_t value = 0;
  const size_t offset = sizeof(MoiReportHeader) + capacity * sizeof(MoiSampledCausalWindow) +
                        static_cast<size_t>(index) * sizeof(value);
  std::memcpy(&value, report.data() + offset, sizeof(value));
  return {
      (value & 1u) != 0,
      static_cast<uint32_t>((value >> 2u) & 0x7u),
      static_cast<uint32_t>((value >> 15u) & 0x3ffu),
      static_cast<uint32_t>((value >> 25u) & 0xfffffu),
      static_cast<uint32_t>(((value >> 59u) & 0x1fu) + 1u),
  };
}

struct ReportEnvGuard {
  ~ReportEnvGuard() {
    (void)unsetenv("RJ_CONSAN_MOI_REPORT_BUFFER");
    (void)unsetenv("RJ_CONSAN_MOI_REPORT_BUFFER_SIZE");
  }
};

bool set_moi_report_buffer_env(void *report, size_t report_bytes) {
  char address[32];
  const int address_written =
      std::snprintf(address, sizeof(address), "0x%llx",
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(report)));
  if (address_written < 0 || static_cast<size_t>(address_written) >= sizeof(address))
    return false;

  char size[32];
  const int size_written =
      std::snprintf(size, sizeof(size), "%llu", static_cast<unsigned long long>(report_bytes));
  if (size_written < 0 || static_cast<size_t>(size_written) >= sizeof(size))
    return false;

  return setenv("RJ_CONSAN_MOI_REPORT_BUFFER", address, 1) == 0 &&
         setenv("RJ_CONSAN_MOI_REPORT_BUFFER_SIZE", size, 1) == 0;
}

__attribute__((amdgpu_num_vgpr(256))) __global__
__launch_bounds__(64) void moi_cdna_forced_spill_test_kernel_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  uint32_t values[8] = {
      0x11000000u ^ tid, 0x22000000u ^ tid, 0x33000000u ^ tid, 0x44000000u ^ tid,
      0x55000000u ^ tid, 0x66000000u ^ tid, 0x77000000u ^ tid, 0x88000000u ^ tid,
  };

  asm volatile("v_mov_b32 v255, 0" ::: "v255");
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  asm volatile("ds_write_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
               :
               : "v"(lds_addr), "v"(values[0])
               : "memory");
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));

  const uint32_t output_base = tid * 8u;
  for (uint32_t i = 0; i < 8u; ++i)
    out[output_base + i] = values[i];
}

__attribute__((amdgpu_num_vgpr(256))) __global__
__launch_bounds__(64) void moi_cdna_barrier_forced_spill_kernel_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  uint32_t values[8] = {
      0x91000000u ^ tid, 0x92000000u ^ tid, 0x93000000u ^ tid, 0x94000000u ^ tid,
      0x95000000u ^ tid, 0x96000000u ^ tid, 0x97000000u ^ tid, 0x98000000u ^ tid,
  };

  asm volatile("v_mov_b32 v255, 0" ::: "v255");
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  asm volatile("ds_write_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
               :
               : "v"(lds_addr), "v"(values[0])
               : "memory");
  asm volatile("s_barrier" ::: "memory");
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));

  const uint32_t output_base = tid * 8u;
  for (uint32_t i = 0; i < 8u; ++i)
    out[output_base + i] = values[i];
}

__attribute__((amdgpu_num_vgpr(256))) __global__ __launch_bounds__(
    64) void moi_cdna_atomic_forced_spill_kernel_for_instrumentation(
    uint32_t *vector_atomic_value, uint32_t *scalar_atomic_value, uint32_t *out) {
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  uint32_t values[8] = {
      0xa1000000u ^ tid, 0xa2000000u ^ tid, 0xa3000000u ^ tid, 0xa4000000u ^ tid,
      0xa5000000u ^ tid, 0xa6000000u ^ tid, 0xa7000000u ^ tid, 0xa8000000u ^ tid,
  };
  constexpr uint32_t kIncrement = 1;
  constexpr uint32_t kZeroOffset = 0;

  asm volatile("v_mov_b32 v255, 0" ::: "v255");
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  if (tid == 0) {
    asm volatile("buffer_wbl2 sc1\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 "global_atomic_add %0, %1, off sc1\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 "buffer_wbl2 sc1\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 "global_atomic_add %2, %1, %3 sc1\n\t"
                 "s_waitcnt vmcnt(0)\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
                 :
                 : "v"(vector_atomic_value), "v"(kIncrement), "v"(kZeroOffset),
                   "s"(scalar_atomic_value)
                 : "memory");
  }
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));

  const uint32_t output_base = tid * 8u;
  for (uint32_t i = 0; i < 8u; ++i)
    out[output_base + i] = values[i];
}

__global__ __launch_bounds__(
    64) void moi_cdna_dynamic_stack_inline_spill_kernel_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  const size_t private_count = (tid & 7u) + 1u;
  auto *private_words =
      static_cast<volatile uint32_t *>(__builtin_alloca(private_count * sizeof(uint32_t)));
  const uint32_t private_index = tid % private_count;
  const uint32_t value = 0xd9500000u ^ tid;
  // Keep enough descriptor-backed VGPRs for Inline Shadow's 16-register
  // forced-spill recipe without exhausting the file and forcing persistent
  // owner state into private memory.
  asm volatile("v_mov_b32 v31, 0" ::: "v31");
  private_words[private_index] = value;
  asm volatile("ds_write_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
               :
               : "v"(lds_addr), "v"(value)
               : "memory");
  out[tid] = private_words[private_index];
}

__device__ __noinline__ uint32_t moi_cdna_shared_lds_helper_for_instrumentation(uint32_t lds_addr,
                                                                                uint32_t value) {
  asm volatile("ds_write_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
               :
               : "v"(lds_addr), "v"(value)
               : "memory");
  return value ^ 0x51a7ed00u;
}

__attribute__((amdgpu_num_vgpr(256))) __global__
__launch_bounds__(64) void moi_cdna_shared_helper_caller_a(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  asm volatile("v_mov_b32 v255, 0" ::: "v255");
  out[tid] = moi_cdna_shared_lds_helper_for_instrumentation(lds_addr, 0xa0000000u | tid);
}

__attribute__((amdgpu_num_vgpr(256))) __global__
__launch_bounds__(64) void moi_cdna_shared_helper_caller_b(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  asm volatile("v_mov_b32 v255, 0" ::: "v255");
  out[tid] = moi_cdna_shared_lds_helper_for_instrumentation(lds_addr, 0xb0000000u | tid);
}

TEST(ConSanMoiCdnaTest, ForcedSpillRecordReplayPreservesLiveVgprsAndWritesRecord) {
  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kValuesPerThread = 8;
  constexpr uint32_t kRecordSlots = 1;
  constexpr size_t kReportBytes = moi_report_buffer_bytes(kRecordSlots);
  uint32_t *out_device = nullptr;
  void *report_device = nullptr;
  HIP_ASSERT(hipMalloc(&out_device, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&report_device, kReportBytes));
  HIP_ASSERT(hipMemset(out_device, 0, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(report_device, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(report_device, kReportBytes));
  moi_cdna_forced_spill_test_kernel_for_instrumentation<<<1, kThreads>>>(out_device);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(kThreads * kValuesPerThread);
  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(
      hipMemcpy(out.data(), out_device, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(report.data(), report_device, report.size(), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(report_device));
  HIP_ASSERT(hipFree(out_device));

  constexpr uint32_t kBases[kValuesPerThread] = {
      0x11000000u, 0x22000000u, 0x33000000u, 0x44000000u,
      0x55000000u, 0x66000000u, 0x77000000u, 0x88000000u,
  };
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    for (uint32_t i = 0; i < kValuesPerThread; ++i)
      EXPECT_EQ(out[tid * kValuesPerThread + i], kBases[i] ^ tid)
          << "tid=" << tid << " value=" << i;
  }

  MoiAccessRecord record{};
  std::memcpy(&record, report.data() + sizeof(MoiReportHeader), sizeof(record));
  EXPECT_EQ(record.access_kind, 2u);
  EXPECT_EQ(record.lds_byte_count, sizeof(uint32_t));
  EXPECT_EQ(record.cell_count, 1u);
}

TEST(ConSanMoiCdnaTest, SecondAddressGroupReportsExactLaneConflict) {
  hipDeviceProp_t properties{};
  HIP_ASSERT(hipGetDeviceProperties(&properties, 0));
  ASSERT_TRUE(properties.warpSize == 32 || properties.warpSize == 64);

  uint32_t *out_device = nullptr;
  HIP_ASSERT(hipMalloc(&out_device, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(out_device, 0xff, sizeof(uint32_t)));
  moi_second_address_group_race_for_instrumentation<<<1, properties.warpSize>>>(out_device);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  uint32_t out = UINT32_MAX;
  HIP_ASSERT(hipMemcpy(&out, out_device, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(out_device));
  EXPECT_EQ(out, 0u);
}

TEST(ConSanMoiCdnaTest, BarrierForcedSpillPreservesLiveVgprsAndWritesRecords) {
  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kValuesPerThread = 8;
  constexpr uint32_t kAccessSlots = 1;
  constexpr uint32_t kBarrierSlots = 2;
  constexpr size_t kReportBytes = moi_report_buffer_bytes(kAccessSlots, kBarrierSlots);
  uint32_t *out_device = nullptr;
  void *report_device = nullptr;
  HIP_ASSERT(hipMalloc(&out_device, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&report_device, kReportBytes));
  HIP_ASSERT(hipMemset(out_device, 0, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(report_device, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(report_device, kReportBytes));
  moi_cdna_barrier_forced_spill_kernel_for_instrumentation<<<1, kThreads>>>(out_device);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(kThreads * kValuesPerThread);
  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(
      hipMemcpy(out.data(), out_device, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(report.data(), report_device, report.size(), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(report_device));
  HIP_ASSERT(hipFree(out_device));

  constexpr uint32_t kBases[kValuesPerThread] = {
      0x91000000u, 0x92000000u, 0x93000000u, 0x94000000u,
      0x95000000u, 0x96000000u, 0x97000000u, 0x98000000u,
  };
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    for (uint32_t i = 0; i < kValuesPerThread; ++i)
      EXPECT_EQ(out[tid * kValuesPerThread + i], kBases[i] ^ tid)
          << "tid=" << tid << " value=" << i;
  }

  MoiReportHeader header{};
  std::memcpy(&header, report.data(), sizeof(header));
  EXPECT_EQ(header.access_record_count, 1u);
  EXPECT_GE(header.barrier_record_count, 1u);
  EXPECT_LE(header.barrier_record_count, kBarrierSlots);
  MoiAccessRecord access{};
  std::memcpy(&access, report.data() + sizeof(MoiReportHeader), sizeof(access));
  EXPECT_EQ(access.access_kind, 2u);
  std::vector<MoiBarrierRecord> barriers(kBarrierSlots);
  std::memcpy(barriers.data(), report.data() + sizeof(MoiReportHeader) + sizeof(MoiAccessRecord),
              barriers.size() * sizeof(MoiBarrierRecord));
  EXPECT_TRUE(std::any_of(barriers.begin(), barriers.end(), [](const MoiBarrierRecord &barrier) {
    return barrier.event_index != 0u;
  }));
}

TEST(ConSanMoiCdnaTest, AtomicForcedSpillExecutesVglobalAddressForms) {
  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kValuesPerThread = 8;
  constexpr uint32_t kAtomicSlots = 2;
  uint32_t *atomic_device = nullptr;
  uint32_t *out_device = nullptr;
  HIP_ASSERT(hipMalloc(&atomic_device, kAtomicSlots * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&out_device, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(atomic_device, 0, kAtomicSlots * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(out_device, 0, kThreads * kValuesPerThread * sizeof(uint32_t)));
  moi_cdna_atomic_forced_spill_kernel_for_instrumentation<<<1, kThreads>>>(
      atomic_device, atomic_device + 1, out_device);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::array<uint32_t, kAtomicSlots> atomic_values{};
  std::vector<uint32_t> out(kThreads * kValuesPerThread);
  HIP_ASSERT(hipMemcpy(atomic_values.data(), atomic_device, sizeof(atomic_values),
                       hipMemcpyDeviceToHost));
  HIP_ASSERT(
      hipMemcpy(out.data(), out_device, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(out_device));
  HIP_ASSERT(hipFree(atomic_device));

  EXPECT_EQ(atomic_values[0], 1u);
  EXPECT_EQ(atomic_values[1], 1u);
  constexpr uint32_t kBases[kValuesPerThread] = {
      0xa1000000u, 0xa2000000u, 0xa3000000u, 0xa4000000u,
      0xa5000000u, 0xa6000000u, 0xa7000000u, 0xa8000000u,
  };
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    for (uint32_t i = 0; i < kValuesPerThread; ++i)
      EXPECT_EQ(out[tid * kValuesPerThread + i], kBases[i] ^ tid)
          << "tid=" << tid << " value=" << i;
  }

}

TEST(ConSanMoiCdnaTest, DynamicStackInlineSpillPreservesPrivateValues) {
  constexpr uint32_t kThreads = 64;
  uint32_t *out_device = nullptr;
  HIP_ASSERT(hipMalloc(&out_device, kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(out_device, 0, kThreads * sizeof(uint32_t)));

  moi_cdna_dynamic_stack_inline_spill_kernel_for_instrumentation<<<1, kThreads>>>(out_device);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(kThreads);
  HIP_ASSERT(
      hipMemcpy(out.data(), out_device, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(out_device));

  for (uint32_t tid = 0; tid < kThreads; ++tid)
    EXPECT_EQ(out[tid], 0xd9500000u ^ tid) << "tid=" << tid;
}

TEST(ConSanMoiCdnaTest, ForcedSpillSampledPreservesLiveVgprsAndWritesEntry) {
  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kValuesPerThread = 8;
  constexpr uint32_t kSampledSlots = 1;
  constexpr size_t kReportBytes = moi_sampled_report_buffer_bytes(kSampledSlots);
  uint32_t *out_device = nullptr;
  void *report_device = nullptr;
  HIP_ASSERT(hipMalloc(&out_device, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&report_device, kReportBytes));
  HIP_ASSERT(hipMemset(out_device, 0, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(report_device, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(report_device, kReportBytes));
  moi_cdna_forced_spill_test_kernel_for_instrumentation<<<1, kThreads>>>(out_device);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(kThreads * kValuesPerThread);
  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(
      hipMemcpy(out.data(), out_device, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(report.data(), report_device, report.size(), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(report_device));
  HIP_ASSERT(hipFree(out_device));

  constexpr uint32_t kBases[kValuesPerThread] = {
      0x11000000u, 0x22000000u, 0x33000000u, 0x44000000u,
      0x55000000u, 0x66000000u, 0x77000000u, 0x88000000u,
  };
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    for (uint32_t i = 0; i < kValuesPerThread; ++i)
      EXPECT_EQ(out[tid * kValuesPerThread + i], kBases[i] ^ tid)
          << "tid=" << tid << " value=" << i;
  }

  const SampledEntry entry = sampled_entry(report, kSampledSlots, 0);
  ASSERT_TRUE(entry.valid);
  EXPECT_EQ(entry.kind, 2u);
  EXPECT_EQ(entry.cell_count, 1u);

  MoiSampledCausalWindow window{};
  std::memcpy(&window, report.data() + sizeof(MoiReportHeader), sizeof(window));
  EXPECT_EQ(window.publication_state, 2u);
  EXPECT_EQ(window.first_entry, 0u);
  EXPECT_EQ(window.entry_count, 1u);
  EXPECT_EQ(window.workgroup_x, 0u);
  EXPECT_EQ(window.workgroup_y, 0u);
  EXPECT_EQ(window.workgroup_z, 0u);
  EXPECT_EQ(window.epoch, entry.epoch);
  EXPECT_EQ(window.generation & 3u, entry.generation);
}

TEST(ConSanMoiCdnaTest, ForcedSpillInlineShadowPreservesLiveVgprs) {
  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kValuesPerThread = 8;
  uint32_t *out_device = nullptr;
  HIP_ASSERT(hipMalloc(&out_device, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(out_device, 0, kThreads * kValuesPerThread * sizeof(uint32_t)));

  moi_cdna_forced_spill_test_kernel_for_instrumentation<<<1, kThreads>>>(out_device);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(kThreads * kValuesPerThread);
  HIP_ASSERT(
      hipMemcpy(out.data(), out_device, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(out_device));

  constexpr uint32_t kBases[kValuesPerThread] = {
      0x11000000u, 0x22000000u, 0x33000000u, 0x44000000u,
      0x55000000u, 0x66000000u, 0x77000000u, 0x88000000u,
  };
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    for (uint32_t i = 0; i < kValuesPerThread; ++i)
      EXPECT_EQ(out[tid * kValuesPerThread + i], kBases[i] ^ tid)
          << "tid=" << tid << " value=" << i;
  }
}

TEST(ConSanMoiCdnaTest, SharedHelperForcedSpillPreservesBothCallers) {
  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kRecordSlots = 2;
  constexpr size_t kReportBytes = moi_report_buffer_bytes(kRecordSlots);
  uint32_t *out_device = nullptr;
  void *report_device = nullptr;
  HIP_ASSERT(hipMalloc(&out_device, 2u * kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&report_device, kReportBytes));
  HIP_ASSERT(hipMemset(out_device, 0, 2u * kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(report_device, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(report_device, kReportBytes));
  moi_cdna_shared_helper_caller_a<<<1, kThreads>>>(out_device);
  HIP_ASSERT(hipGetLastError());
  moi_cdna_shared_helper_caller_b<<<1, kThreads>>>(out_device + kThreads);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(2u * kThreads);
  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(
      hipMemcpy(out.data(), out_device, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(report.data(), report_device, report.size(), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(report_device));
  HIP_ASSERT(hipFree(out_device));

  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    EXPECT_EQ(out[tid], (0xa0000000u | tid) ^ 0x51a7ed00u) << "caller A tid=" << tid;
    EXPECT_EQ(out[kThreads + tid], (0xb0000000u | tid) ^ 0x51a7ed00u) << "caller B tid=" << tid;
  }
  std::vector<MoiAccessRecord> records(kRecordSlots);
  std::memcpy(records.data(), report.data() + sizeof(MoiReportHeader),
              records.size() * sizeof(MoiAccessRecord));
  EXPECT_TRUE(std::any_of(records.begin(), records.end(),
                          [](const MoiAccessRecord &record) { return record.access_kind == 2u; }));
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return result;
}
