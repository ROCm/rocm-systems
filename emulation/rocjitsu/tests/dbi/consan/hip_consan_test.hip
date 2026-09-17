// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_test.hip
/// @brief Real-HIP smoke tests for ConSan DBI report-buffer instrumentation.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "hip_consan_address_group_test.h"
#include "rocjitsu/code/patch/consan/consan_abi.h"

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

#define RJ_CONSAN_NOP_CAVE                                                                         \
  ".rept 128\n\t"                                                                                  \
  "s_nop 0\n\t"                                                                                    \
  ".endr\n\t"

#include "hip_consan_empty_exec_scalar_spill.h"

#if defined(__gfx1250__)
#define RJ_CONSAN_WAIT_ATOMIC_STORE "s_wait_storecnt 0\n\t"
#else
#define RJ_CONSAN_WAIT_ATOMIC_STORE "s_waitcnt vmcnt(0)\n\t"
#endif

namespace {

using ReportHeader = rocjitsu::consan::ReportHeader;
using CausalWindow = rocjitsu::consan::CausalWindow;
using PendingAcquireSlot = rocjitsu::consan::PendingAcquireSlot;
using SyncMetadataPacked = rocjitsu::consan::SyncMetadataPacked;

constexpr uint32_t kReportMagic = rocjitsu::consan::kReportMagic;
constexpr uint32_t kReportAbiVersion = rocjitsu::consan::kReportAbiVersion;
constexpr uint32_t kAccessKindRead = 1;
constexpr uint32_t kAccessKindWrite = 2;

struct Entry {
  bool valid = false;
  bool consumed = false;
  uint32_t kind = 0;
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
  uint32_t generation = 0;
  uint32_t start_byte = 0;
  uint32_t byte_count = 0;
};

constexpr uint32_t kWindowEmpty = 0;
constexpr uint32_t kWindowPublishing = 1;
constexpr uint32_t kWindowReady = 2;

constexpr size_t direct_report_buffer_bytes(uint32_t capacity) {
  return sizeof(ReportHeader) +
         capacity * (sizeof(CausalWindow) + sizeof(uint64_t) + sizeof(SyncMetadataPacked) +
                     sizeof(PendingAcquireSlot));
}

constexpr size_t direct_entries_offset(uint32_t capacity) {
  return sizeof(ReportHeader) + capacity * sizeof(CausalWindow);
}

Entry decode_entry(uint64_t value);

CausalWindow read_window(const std::vector<uint8_t> &report, uint32_t index) {
  CausalWindow window;
  std::memcpy(&window,
              report.data() + sizeof(ReportHeader) +
                  static_cast<size_t>(index) * sizeof(CausalWindow),
              sizeof(window));
  return window;
}

Entry read_entry(const std::vector<uint8_t> &report, uint32_t capacity, uint32_t index) {
  uint64_t raw_entry = 0;
  std::memcpy(&raw_entry,
              report.data() + direct_entries_offset(capacity) +
                  static_cast<size_t>(index) * sizeof(uint64_t),
              sizeof(raw_entry));
  return decode_entry(raw_entry);
}

Entry decode_entry(uint64_t value) {
  return Entry{
      (value & 1u) != 0,
      (value & 2u) != 0,
      static_cast<uint32_t>((value >> 2u) & 0x7u),
      static_cast<uint32_t>((value >> 5u) & 0x3ffu),
      static_cast<uint32_t>((value >> 15u) & 0x3ffu),
      static_cast<uint32_t>((value >> 25u) & 0x3ffffu),
      static_cast<uint32_t>((value >> 43u) & 0xffffu),
      static_cast<uint32_t>(((value >> 59u) & 0x1fu) + 1u),
  };
}

struct ReportEnvGuard {
  ~ReportEnvGuard() {
    (void)unsetenv("RJ_CONSAN_REPORT_BUFFER");
    (void)unsetenv("RJ_CONSAN_REPORT_BUFFER_SIZE");
  }
};

bool set_report_buffer_env(void *d_report, size_t report_bytes) {
  char address[32];
  const int address_written =
      std::snprintf(address, sizeof(address), "0x%llx",
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_report)));
  if (address_written < 0 || static_cast<size_t>(address_written) >= sizeof(address)) {
    ADD_FAILURE() << "failed to format ConSan report-buffer address";
    return false;
  }
  char size[32];
  const int size_written =
      std::snprintf(size, sizeof(size), "%llu", static_cast<unsigned long long>(report_bytes));
  if (size_written < 0 || static_cast<size_t>(size_written) >= sizeof(size)) {
    ADD_FAILURE() << "failed to format ConSan report-buffer size";
    return false;
  }
  if (setenv("RJ_CONSAN_REPORT_BUFFER", address, 1) != 0) {
    ADD_FAILURE() << "failed to set RJ_CONSAN_REPORT_BUFFER";
    return false;
  }
  if (setenv("RJ_CONSAN_REPORT_BUFFER_SIZE", size, 1) != 0) {
    ADD_FAILURE() << "failed to set RJ_CONSAN_REPORT_BUFFER_SIZE";
    return false;
  }
  return true;
}

bool access_kinds_conflict(uint32_t current, uint32_t prior) {
  if (current == 0 || prior == 0)
    return false;
  if (current == kAccessKindRead)
    return prior != kAccessKindRead;
  return true;
}

bool entries_conflict(const Entry &current, const Entry &prior) {
  const uint32_t current_end = current.start_byte + current.byte_count;
  const uint32_t prior_end = prior.start_byte + prior.byte_count;
  return current.valid && prior.valid && !prior.consumed && current.epoch == prior.epoch &&
         current.generation == prior.generation && current.owner_id != prior.owner_id &&
         access_kinds_conflict(current.kind, prior.kind) && current.start_byte < prior_end &&
         prior.start_byte < current_end;
}

__global__ __launch_bounds__(
    64) void forced_spill_preserves_live_vgprs_test_kernel_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  uint32_t values[8] = {
      0x11000000u ^ tid, 0x22000000u ^ tid, 0x33000000u ^ tid, 0x44000000u ^ tid,
      0x55000000u ^ tid, 0x66000000u ^ tid, 0x77000000u ^ tid, 0x88000000u ^ tid,
  };

  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
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
__launch_bounds__(64) void relocated_b128_load_after_barrier_for_instrumentation(uint32_t *out) {
  __shared__ uint4 lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  if (tid == 0)
    lds[0] = make_uint4(0x10203040u, 0x50607080u, 0x90A0B0C0u, 0xD0E0F000u);
  __syncthreads();

  // Keep the descriptor full and omit a local NOP cave so the access must use
  // the spill-backed appended-body path exercised by the Tensile regression.
  asm volatile("v_mov_b32 v255, 0" ::: "v255");
  if (tid == 0) {
    const uint4 loaded = lds[0];
    out[0] = loaded.x;
    out[1] = loaded.y;
    out[2] = loaded.z;
    out[3] = loaded.w;
  }
  __syncthreads();
}

__attribute__((amdgpu_num_vgpr(256))) __global__
__launch_bounds__(64) void full_vgpr_file_test_kernel_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  const uint32_t value = 0xF0110000u ^ tid;
  asm volatile("v_mov_b32 v255, 0" ::: "v255");
  asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
               :
               : "v"(lds_addr), "v"(value)
               : "memory");
  out[tid] = value;
}

__device__ __noinline__ uint32_t shared_noinline_lds_helper_for_instrumentation(uint32_t lds_addr,
                                                                                uint32_t value) {
  asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
               :
               : "v"(lds_addr), "v"(value)
               : "memory");
  return value ^ 0x51A7ED00u;
}

__global__ __launch_bounds__(64) void shared_helper_caller_a(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  out[tid] = shared_noinline_lds_helper_for_instrumentation(lds_addr, 0xA0000000u | tid);
}

__global__ __launch_bounds__(64) void shared_helper_caller_b(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  out[tid] = shared_noinline_lds_helper_for_instrumentation(lds_addr, 0xB0000000u | tid);
}

__device__ __noinline__ void shared_noinline_atomic_add_for_instrumentation(uint32_t *counter,
                                                                            uint32_t lds_addr) {
  uint32_t one = 1;
  // Give ConSan an LDS producer whose release must be instrumented.
  asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE : : "v"(lds_addr), "v"(one) : "memory");
  asm volatile("s_wait_storecnt 0\n\t"
               "global_wb scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0\n\t"
               "flat_atomic_add_u32 %0, %1 scope:SCOPE_DEV\n\t" RJ_CONSAN_WAIT_ATOMIC_STORE
               :
               : "v"(counter), "v"(one)
               : "memory");
}

__global__ __launch_bounds__(64) void shared_atomic_helper_caller_a(uint32_t *counter,
                                                                    uint32_t *out) {
  __shared__ uint32_t lds;
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds) & 0xFFFFu);
  // Force a real 32-VGPR descriptor allocation. The amdgpu_num_vgpr attribute
  // is only an upper bound and does not create a spillable architectural
  // window for this otherwise tiny caller.
  asm volatile("v_mov_b32 v31, 0" ::: "v31");
  if (threadIdx.x == 0) {
    shared_noinline_atomic_add_for_instrumentation(counter, lds_addr);
    out[0] = 0xA70A1C01u;
  }
}

__global__ __launch_bounds__(64) void shared_atomic_helper_caller_b(uint32_t *counter,
                                                                    uint32_t *out) {
  __shared__ uint32_t lds;
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds) & 0xFFFFu);
  asm volatile("v_mov_b32 v31, 0" ::: "v31");
  if (threadIdx.x == 0) {
    shared_noinline_atomic_add_for_instrumentation(counter, lds_addr);
    out[0] = 0xB70B1C02u;
  }
}

__global__ __launch_bounds__(64) void two_wave_lds_accesses(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (tid < 32) {
    const uint32_t value = 0xABCD0000u | tid;
    asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
                 :
                 : "v"(lds_addr), "v"(value)
                 : "memory");
  }
  if (tid >= 32) {
    uint32_t loaded = 0;
    asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
                 : "=v"(loaded)
                 : "v"(lds_addr)
                 : "memory");
    if (tid == 32)
      out[0] = loaded;
  }
}

__device__ __noinline__ void native_b96_aliasing_access_for_instrumentation(uint32_t lds_addr,
                                                                            uint32_t *out,
                                                                            bool barrier_ordered) {
  constexpr uint32_t kNew0 = 0x13579BDFu;
  constexpr uint32_t kNew1 = 0x2468ACE0u;
  constexpr uint32_t kNew2 = 0xA5A55A5Au;
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  if (tid < (barrier_ordered ? 1u : 32u)) {
    // Keep the data tuple and address in fixed, nonoverlapping VGPRs so this
    // emits the native three-dword store on both RDNA4 targets.
    asm volatile("v_mov_b32 v4, %0\n\t"
                 "v_mov_b32 v5, %1\n\t"
                 "v_mov_b32 v6, %2\n\t"
                 "v_mov_b32 v7, %3\n\t"
                 "ds_store_b96 v7, v[4:6]\n\t" RJ_CONSAN_NOP_CAVE
                 :
                 : "v"(kNew0), "v"(kNew1), "v"(kNew2), "v"(lds_addr)
                 : "v4", "v5", "v6", "v7", "memory");
  }
  if (barrier_ordered)
    __syncthreads();
  if (tid >= 32) {
    uint32_t loaded0 = 0;
    uint32_t loaded1 = 0;
    uint32_t loaded2 = 0;
    // Deliberately alias the address and destination at v4. The ConSan
    // trampoline must snapshot the address before executing this load.
    asm volatile("v_mov_b32 v4, %3\n\t"
                 "ds_load_b96 v[4:6], v4\n\t" RJ_CONSAN_NOP_CAVE "s_wait_dscnt 0\n\t"
                 "v_mov_b32 %0, v4\n\t"
                 "v_mov_b32 %1, v5\n\t"
                 "v_mov_b32 %2, v6\n\t"
                 : "=v"(loaded0), "=v"(loaded1), "=v"(loaded2)
                 : "v"(lds_addr)
                 : "v4", "v5", "v6", "memory");
    if (tid == 32) {
      out[0] = loaded0;
      out[1] = loaded1;
      out[2] = loaded2;
    }
  }
}

__global__ __launch_bounds__(64) void native_b96_racy_aliasing_load(uint32_t *out) {
  __shared__ uint32_t lds[3];
  if (threadIdx.x == 0) {
    lds[0] = 0x01020304u;
    lds[1] = 0x11223344u;
    lds[2] = 0x55667788u;
  }
  __syncthreads();
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  native_b96_aliasing_access_for_instrumentation(lds_addr, out, /*barrier_ordered=*/false);
}

__global__ __launch_bounds__(64) void native_b96_barrier_ordered_aliasing_load(uint32_t *out) {
  __shared__ uint32_t lds[3];
  if (threadIdx.x == 0) {
    lds[0] = 0x01020304u;
    lds[1] = 0x11223344u;
    lds[2] = 0x55667788u;
  }
  __syncthreads();
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  native_b96_aliasing_access_for_instrumentation(lds_addr, out, /*barrier_ordered=*/true);
}

__global__ __launch_bounds__(64) void gfx1201_vgpr_scratch_roundtrip(uint32_t *out) {
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  volatile uint32_t private_words[4];
  private_words[tid & 3u] = tid;

  uint32_t value = 0x5A170000u ^ tid;
  asm volatile("scratch_store_b32 off, %0, off offset:0\n\t"
               "s_wait_storecnt 0\n\t"
               "v_mov_b32 %0, 0\n\t"
               "scratch_load_b32 %0, off, off offset:0\n\t"
               "s_wait_loadcnt 0\n\t"
               : "+v"(value)
               :
               : "memory");
  out[tid] = value;
}

__global__
__launch_bounds__(64) void static_private_stack_test_kernel_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  volatile uint32_t private_words[8];
  const uint32_t private_index = tid & 7u;
  const uint32_t value = 0x57AC0000u ^ tid;
  private_words[private_index] = value;
  asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
               :
               : "v"(lds_addr), "v"(value)
               : "memory");
  out[tid] = private_words[private_index];
}

__global__
__launch_bounds__(64) void dynamic_private_stack_test_kernel_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  const size_t private_count = (tid & 7u) + 1u;
  auto *private_words =
      static_cast<volatile uint32_t *>(__builtin_alloca(private_count * sizeof(uint32_t)));
  const uint32_t private_index = tid % private_count;
  const uint32_t value = 0xD1A60000u ^ tid;
  private_words[private_index] = value;
  asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
               :
               : "v"(lds_addr), "v"(value)
               : "memory");
  out[tid] = private_words[private_index];
}

__global__
__launch_bounds__(64) void equivalent_lds_accesses_in_different_workgroups(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (blockIdx.x == 0 && tid < 32) {
    const uint32_t value = 0xCAFE0000u | tid;
    asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
                 :
                 : "v"(lds_addr), "v"(value)
                 : "memory");
  }
  if (blockIdx.x == 1 && tid >= 32) {
    uint32_t loaded = 0;
    asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
                 : "=v"(loaded)
                 : "v"(lds_addr)
                 : "memory");
    if (tid == 32)
      out[0] = loaded;
  }
}

__global__
__launch_bounds__(64) void equivalent_lds_accesses_in_different_2d_workgroups(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (blockIdx.y == 0 && tid < 32) {
    const uint32_t value = 0x2D000000u | tid;
    asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
                 :
                 : "v"(lds_addr), "v"(value)
                 : "memory");
  }
  if (blockIdx.y == 1 && tid >= 32) {
    uint32_t loaded = 0;
    asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
                 : "=v"(loaded)
                 : "v"(lds_addr)
                 : "memory");
    if (tid == 32)
      out[0] = loaded;
  }
}

__global__
__launch_bounds__(64) void equivalent_lds_accesses_in_different_3d_workgroups(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (blockIdx.z == 0 && tid < 32) {
    const uint32_t value = 0x3D000000u | tid;
    asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
                 :
                 : "v"(lds_addr), "v"(value)
                 : "memory");
  }
  if (blockIdx.z == 1 && tid >= 32) {
    uint32_t loaded = 0;
    asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
                 : "=v"(loaded)
                 : "v"(lds_addr)
                 : "memory");
    if (tid == 32)
      out[0] = loaded;
  }
}

__global__
__launch_bounds__(64) void repeated_dispatch_saturation_preserves_output_for_instrumentation(
    uint32_t *out, uint32_t output_index, uint32_t expected) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (tid == 0) {
    asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
                 :
                 : "v"(lds_addr), "v"(expected)
                 : "memory");
  }
  __syncthreads();
  if (tid == 32 && blockIdx.x == 0) {
    uint32_t loaded = 0;
    asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE "s_wait_dscnt 0\n\t"
                 : "=v"(loaded)
                 : "v"(lds_addr)
                 : "memory");
    out[output_index] = loaded;
  }
}

#if defined(RJ_CONSAN_ENABLE_STAGED_DSCNT_TEST)
__global__
__launch_bounds__(64) void staged_dscnt_load_preserves_output_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[64];
  constexpr uint32_t kLdsValue = 0x10203040u;
  lds[threadIdx.x] = kLdsValue;
  __syncthreads();

  const auto lds_addr =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[threadIdx.x]) & 0xFFFFu);
  uint32_t sum;
  // Keep three addtid loads outstanding ahead of one ordinary LDS load, then
  // consume the four results with descending counter waits. ConSan only
  // supports the ordinary load, but relocating it must preserve its position
  // in the shared DS completion stream.
  // DS_LOAD_ADDTID_B32 uses M0[15:0] as its byte base. Make that base zero
  // explicitly instead of relying on truncation of a nonzero literal.
  asm volatile("s_mov_b32 m0, 0\n\t"
               "v_mov_b32 v8, %1\n\t"
               "ds_load_addtid_b32 v0\n\t"
               "ds_load_addtid_b32 v1\n\t"
               "ds_load_addtid_b32 v2\n\t"
               "s_cmp_eq_u32 0, 0\n\t"
               "ds_load_b32 v3, v8\n\t"
               "s_wait_dscnt 3\n\t"
               "v_mov_b32 v5, v0\n\t"
               "s_wait_dscnt 2\n\t"
               "v_add_nc_u32 v5, v1, v5\n\t"
               "s_wait_dscnt 1\n\t"
               "v_add_nc_u32 v5, v2, v5\n\t"
               "s_wait_dscnt 0\n\t"
               "v_add_nc_u32 v5, v3, v5\n\t"
               "s_cbranch_scc1 1f\n\t"
               "v_mov_b32 v5, 0\n\t"
               "1:\n\t"
               "v_mov_b32 %0, v5\n\t"
               : "=v"(sum)
               : "v"(lds_addr)
               : "v0", "v1", "v2", "v3", "v5", "v8", "memory");
  out[threadIdx.x] = sum;
}
#endif

__global__ __launch_bounds__(64) void atomic_handoff_lds_accesses(uint32_t *out,
                                                                  uint32_t *release_counter,
                                                                  uint32_t *acquire_counter,
                                                                  uint32_t *flag) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (tid == 0) {
    constexpr uint32_t value = 0xA70C0001u;
    uint32_t one = 1;
    asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
                 :
                 : "v"(lds_addr), "v"(value)
                 : "memory");
    asm volatile("s_wait_storecnt 0\n\t"
                 "global_wb scope:SCOPE_DEV\n\t"
                 "s_wait_storecnt 0\n\t"
                 "flat_atomic_add_u32 %0, %1 scope:SCOPE_DEV\n\t" RJ_CONSAN_WAIT_ATOMIC_STORE
                 :
                 : "v"(release_counter), "v"(one)
                 : "memory");
    *flag = 1;
  }

  if (tid == 32) {
    volatile uint32_t *volatile_flag = flag;
    while (*volatile_flag == 0) {
    }
    uint32_t one = 1;
    uint32_t old = 0;
    asm volatile("flat_atomic_add_u32 %0, %1, %2 th:TH_ATOMIC_RETURN scope:SCOPE_DEV\n\t"
                 "s_wait_storecnt 0\n\t"
                 "global_inv scope:SCOPE_DEV\n\t"
                 "s_wait_loadcnt 0\n\t"
                 : "=&v"(old)
                 : "v"(acquire_counter), "v"(one)
                 : "memory");
    uint32_t loaded = 0;
    asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_NOP_CAVE
                 : "=v"(loaded)
                 : "v"(lds_addr)
                 : "memory");
    out[0] = loaded ^ old;
  }
}

void free_device_ptr(uint32_t *ptr) {
  if (ptr != nullptr)
    (void)hipFree(ptr);
}

} // namespace

TEST(ConSanSpillHipTest, Gfx1201VgprScratchRoundTrip) {
  constexpr uint32_t kThreads = 64;
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, kThreads * sizeof(uint32_t)));

  gfx1201_vgpr_scratch_roundtrip<<<1, kThreads>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(kThreads);
  HIP_ASSERT(hipMemcpy(out.data(), d_out, kThreads * sizeof(uint32_t), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  for (uint32_t tid = 0; tid < kThreads; ++tid)
    EXPECT_EQ(out[tid], 0x5A170000u ^ tid) << "tid=" << tid;
}

TEST(ConSanHipTest, RelocatedB128LoadAfterBarrierPreservesLoadedValue) {
  constexpr std::array<uint32_t, 4> kExpected = {
      0x10203040u,
      0x50607080u,
      0x90A0B0C0u,
      0xD0E0F000u,
  };
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kExpected.size() * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, kExpected.size() * sizeof(uint32_t)));

  relocated_b128_load_after_barrier_for_instrumentation<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::array<uint32_t, 4> out{};
  HIP_ASSERT(hipMemcpy(out.data(), d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  EXPECT_EQ(out, kExpected);
}

TEST(ConSanHipTest, ForcedSpillPreservesLiveVgprsAndWritesEntry) {
  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kValuesPerThread = 8;
  constexpr uint32_t kSlots = 1;
  constexpr size_t kReportBytes = direct_report_buffer_bytes(kSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_report_buffer_env(d_report, kReportBytes));
  forced_spill_preserves_live_vgprs_test_kernel_for_instrumentation<<<1, kThreads>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(kThreads * kValuesPerThread);
  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(out.data(), d_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  constexpr uint32_t kBases[kValuesPerThread] = {
      0x11000000u, 0x22000000u, 0x33000000u, 0x44000000u,
      0x55000000u, 0x66000000u, 0x77000000u, 0x88000000u,
  };
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    for (uint32_t i = 0; i < kValuesPerThread; ++i)
      EXPECT_EQ(out[tid * kValuesPerThread + i], kBases[i] ^ tid)
          << "tid=" << tid << " value=" << i;
  }

  std::optional<Entry> live_entry;
  for (uint32_t i = 0; i < kSlots; ++i) {
    const Entry entry = read_entry(report, kSlots, i);
    if (entry.valid) {
      live_entry = entry;
      break;
    }
  }
  ASSERT_TRUE(live_entry);
  EXPECT_EQ(live_entry->kind, kAccessKindWrite);
  EXPECT_EQ(live_entry->byte_count, sizeof(uint32_t));
  const CausalWindow window = read_window(report, 0);
  EXPECT_EQ(window.publication_state, kWindowReady);
  EXPECT_EQ(window.first_entry, 0u);
  EXPECT_EQ(window.entry_count, 1u);
  EXPECT_EQ(window.workgroup_x, 0u);
  EXPECT_EQ(window.workgroup_y, 0u);
  EXPECT_EQ(window.workgroup_z, 0u);
  EXPECT_EQ(window.epoch, live_entry->epoch);
  EXPECT_EQ(window.generation & 3u, live_entry->generation);
  ReportHeader header;
  std::memcpy(&header, report.data(), sizeof(header));
  EXPECT_EQ(header.causal_window_count, 1u);
  EXPECT_EQ(header.dropped_window_count, 0u);
}

TEST(ConSanHipTest, ZeroExecScalarSpillCorrectPreservesLiveScalars) {
  uint32_t *out_device = nullptr;
  HIP_ASSERT(hipMalloc(&out_device, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(out_device, 0, sizeof(uint32_t)));

  zero_exec_scalar_spill_correct_kernel_for_instrumentation<<<1, 64>>>(out_device);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, out_device, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(out_device));
  EXPECT_EQ(out, 0x2468ace0u);
}

TEST(ConSanHipTest, NonemptyScalarSpillIncorrectReportsRace) {
  const uint32_t active_wave_count = kNonemptyScalarSpillWaveCount;
  uint32_t *out_device = nullptr;
  HIP_ASSERT(hipMalloc(&out_device, kNonemptyScalarSpillWaveCount * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(out_device, 0, kNonemptyScalarSpillWaveCount * sizeof(uint32_t)));

  nonempty_scalar_spill_incorrect_kernel_for_instrumentation<<<1,
                                                               kNonemptyScalarSpillThreadCount>>>(
      out_device, active_wave_count);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::array<uint32_t, kNonemptyScalarSpillWaveCount> out{};
  HIP_ASSERT(hipMemcpy(out.data(), out_device, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(out_device));
  constexpr uint32_t expected = 0x2468ace0u;
  for (uint32_t wave = 0u; wave < kNonemptyScalarSpillWaveCount; ++wave)
    EXPECT_EQ(out[wave], wave < active_wave_count ? expected : 0u) << "wave=" << wave;
}

TEST(ConSanHipTest, FullVgprFileForcedSpillPreservesOutput) {
  constexpr uint32_t kThreads = 64;
  constexpr size_t kReportBytes = direct_report_buffer_bytes(1);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_report_buffer_env(d_report, kReportBytes));
  full_vgpr_file_test_kernel_for_instrumentation<<<1, kThreads>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(kThreads);
  HIP_ASSERT(hipMemcpy(out.data(), d_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);
  for (uint32_t tid = 0; tid < kThreads; ++tid)
    EXPECT_EQ(out[tid], 0xF0110000u ^ tid) << "tid=" << tid;
}

TEST(ConSanHipTest, SharedNoinlineHelperPreservesBothCallers) {
  constexpr uint32_t kThreads = 64;
  constexpr size_t kReportBytes = direct_report_buffer_bytes(2);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, 2 * kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, 2 * kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_report_buffer_env(d_report, kReportBytes));
  shared_helper_caller_a<<<1, kThreads>>>(d_out);
  shared_helper_caller_b<<<1, kThreads>>>(d_out + kThreads);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(2 * kThreads);
  HIP_ASSERT(hipMemcpy(out.data(), d_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    EXPECT_EQ(out[tid], (0xA0000000u | tid) ^ 0x51A7ED00u) << "caller A tid=" << tid;
    EXPECT_EQ(out[kThreads + tid], (0xB0000000u | tid) ^ 0x51A7ED00u) << "caller B tid=" << tid;
  }
}

TEST(ConSanHipTest, SharedAtomicNoinlineHelperPreservesBothCallers) {
  uint32_t *d_counters = nullptr;
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_counters, 2 * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_out, 2 * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_counters, 0, 2 * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, 2 * sizeof(uint32_t)));

  shared_atomic_helper_caller_a<<<1, 64>>>(d_counters, d_out);
  shared_atomic_helper_caller_b<<<1, 64>>>(d_counters + 1, d_out + 1);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  uint32_t counters[2] = {};
  uint32_t out[2] = {};
  HIP_ASSERT(hipMemcpy(counters, d_counters, sizeof(counters), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  free_device_ptr(d_counters);
  EXPECT_EQ(counters[0], 1u);
  EXPECT_EQ(counters[1], 1u);
  EXPECT_EQ(out[0], 0xA70A1C01u);
  EXPECT_EQ(out[1], 0xB70B1C02u);
}

TEST(ConSanHipTest, StaticPrivateStackForcedSpillPreservesOutput) {
  constexpr uint32_t kThreads = 64;
  constexpr size_t kReportBytes = direct_report_buffer_bytes(1);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_report_buffer_env(d_report, kReportBytes));
  static_private_stack_test_kernel_for_instrumentation<<<1, kThreads>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(kThreads);
  HIP_ASSERT(hipMemcpy(out.data(), d_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);
  for (uint32_t tid = 0; tid < kThreads; ++tid)
    EXPECT_EQ(out[tid], 0x57AC0000u ^ tid) << "tid=" << tid;
}

TEST(ConSanHipTest, DynamicPrivateStackPreservesOutput) {
  constexpr uint32_t kThreads = 64;
  // This fixture exercises ConSan runtime validation.
  // Provision the larger access-only ABI rather than coupling the kernel test
  // to whichever mode happens to run it.
  constexpr size_t kReportBytes = direct_report_buffer_bytes(/*access_capacity=*/1);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_report_buffer_env(d_report, kReportBytes));
  dynamic_private_stack_test_kernel_for_instrumentation<<<1, kThreads>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(kThreads);
  HIP_ASSERT(hipMemcpy(out.data(), d_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);
  for (uint32_t tid = 0; tid < kThreads; ++tid)
    EXPECT_EQ(out[tid], 0xD1A60000u ^ tid) << "tid=" << tid;
}

void run_native_b96_aliasing_load(bool barrier_ordered, std::array<uint32_t, 3> &out) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, 3u * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, 3u * sizeof(uint32_t)));

  if (barrier_ordered)
    native_b96_barrier_ordered_aliasing_load<<<1, 64>>>(d_out);
  else
    native_b96_racy_aliasing_load<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(out.data(), d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
}

TEST(ConSanHipTest, NativeB96RacyAliasingLoadCompletesWithValidTupleWords) {
  constexpr std::array old_values = {0x01020304u, 0x11223344u, 0x55667788u};
  constexpr std::array new_values = {0x13579BDFu, 0x2468ACE0u, 0xA5A55A5Au};
  std::array<uint32_t, 3> out{};
  ASSERT_NO_FATAL_FAILURE(run_native_b96_aliasing_load(/*barrier_ordered=*/false, out));
  for (size_t i = 0; i < out.size(); ++i)
    EXPECT_TRUE(out[i] == old_values[i] || out[i] == new_values[i]) << "word=" << i;
}

TEST(ConSanHipTest, NativeB96BarrierOrderedAliasingLoadPreservesOutput) {
  constexpr std::array expected = {0x13579BDFu, 0x2468ACE0u, 0xA5A55A5Au};
  std::array<uint32_t, 3> out{};
  ASSERT_NO_FATAL_FAILURE(run_native_b96_aliasing_load(/*barrier_ordered=*/true, out));
  EXPECT_EQ(out, expected);
}

TEST(ConSanHipTest, TwoWaveAccessesRunWithAutoReportBuffer) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));

  two_wave_lds_accesses<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  EXPECT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  free_device_ptr(d_out);
}

TEST(ConSanHipTest, SecondAddressGroupReportsExactLaneConflict) {
  hipDeviceProp_t properties{};
  HIP_ASSERT(hipGetDeviceProperties(&properties, 0));
  ASSERT_TRUE(properties.warpSize == 32 || properties.warpSize == 64);

  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0xff, sizeof(uint32_t)));
  second_address_group_race_for_instrumentation<<<1, properties.warpSize>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  uint32_t out = std::numeric_limits<uint32_t>::max();
  HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  EXPECT_EQ(out, 0u);
}

TEST(ConSanHipTest, DisjointLaneB128StoresDoNotConflict) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  disjoint_lane_b128_stores_for_instrumentation<<<1, 32>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  free_device_ptr(d_out);
}

TEST(ConSanHipTest, RepeatedDispatchesRetainAutomaticRecords) {
  constexpr uint32_t kDispatchCount = 4u;
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));

  for (uint32_t dispatch = 0; dispatch < kDispatchCount; ++dispatch) {
    two_wave_lds_accesses<<<1, 64>>>(d_out);
    HIP_ASSERT(hipGetLastError());
    const hipError_t sync = hipDeviceSynchronize();
    ASSERT_EQ(sync, hipSuccess) << "dispatch " << dispatch
                                << " HIP error: " << hipGetErrorString(sync);
  }

  free_device_ptr(d_out);
}

TEST(ConSanHipTest, SaturationPreservesRepeatedDispatchOutput) {
  constexpr uint32_t kDispatchCount = 10u;
  constexpr uint32_t kWorkgroupCount = 10u;
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kDispatchCount * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, kDispatchCount * sizeof(uint32_t)));

  for (uint32_t dispatch = 0; dispatch < kDispatchCount; ++dispatch) {
    const uint32_t expected = 0x5A170000u ^ dispatch;
    repeated_dispatch_saturation_preserves_output_for_instrumentation<<<kWorkgroupCount, 64>>>(
        d_out, dispatch, expected);
    HIP_ASSERT(hipGetLastError());
  }
  // Ten workgroups exceed the eight causal-window banks even when a target
  // uses its stable code-object identity fallback. Keep every packet live
  // until the queue has observed all dispatches so targets with hardware
  // dispatch identity exercise that dimension too.
  HIP_ASSERT(hipDeviceSynchronize());

  std::array<uint32_t, kDispatchCount> out{};
  HIP_ASSERT(hipMemcpy(out.data(), d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  for (uint32_t dispatch = 0; dispatch < kDispatchCount; ++dispatch)
    EXPECT_EQ(out[dispatch], 0x5A170000u ^ dispatch) << "dispatch=" << dispatch;
}

#if defined(RJ_CONSAN_ENABLE_STAGED_DSCNT_TEST)
TEST(ConSanHipTest, StagedDscntLoadPreservesOutput) {
  constexpr uint32_t kExpected = 4u * 0x10203040u;
  std::array<uint32_t, 64> out{};
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, out.size() * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, out.size() * sizeof(uint32_t)));

  staged_dscnt_load_preserves_output_for_instrumentation<<<1, out.size()>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(out.data(), d_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);

  for (uint32_t lane = 0; lane < out.size(); ++lane)
    EXPECT_EQ(out[lane], kExpected) << "lane=" << lane;
}
#endif

TEST(ConSanHipTest, BackendWritesDirectEntries) {
  constexpr uint32_t kSlots = 2;
  constexpr size_t kReportBytes = direct_report_buffer_bytes(kSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_report_buffer_env(d_report, kReportBytes));

  two_wave_lds_accesses<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  ReportHeader header;
  std::memcpy(&header, report.data(), sizeof(header));

  EXPECT_EQ(header.event_counter, 0u);
  EXPECT_EQ(header.causal_window_count, 2u);
  EXPECT_EQ(header.dropped_window_count, 0u);

  std::vector<Entry> live_entries;
  for (uint32_t i = 0; i < kSlots; ++i) {
    Entry entry = read_entry(report, kSlots, i);
    if (entry.valid)
      live_entries.push_back(entry);
  }

  ASSERT_EQ(live_entries.size(), 2u);
  for (uint32_t i = 0; i < kSlots; ++i) {
    const CausalWindow window = read_window(report, i);
    const Entry entry = read_entry(report, kSlots, i);
    EXPECT_EQ(window.publication_state, kWindowReady) << "slot=" << i;
    EXPECT_EQ(window.first_entry, i) << "slot=" << i;
    EXPECT_EQ(window.entry_count, 1u) << "slot=" << i;
    EXPECT_EQ(window.workgroup_x, 0u) << "slot=" << i;
    EXPECT_EQ(window.workgroup_y, 0u) << "slot=" << i;
    EXPECT_EQ(window.workgroup_z, 0u) << "slot=" << i;
    EXPECT_EQ(window.epoch, entry.epoch) << "slot=" << i;
    EXPECT_EQ(window.generation & 3u, entry.generation) << "slot=" << i;
  }
  std::stable_sort(live_entries.begin(), live_entries.end(),
                   [](const Entry &lhs, const Entry &rhs) { return lhs.kind < rhs.kind; });
  EXPECT_EQ(live_entries[0].kind, kAccessKindRead);
  EXPECT_EQ(live_entries[1].kind, kAccessKindWrite);
  EXPECT_FALSE(live_entries[0].consumed);
  EXPECT_FALSE(live_entries[1].consumed);
  EXPECT_NE(live_entries[0].owner_id, live_entries[1].owner_id);
  EXPECT_EQ(live_entries[0].epoch, 0u);
  EXPECT_EQ(live_entries[1].epoch, 0u);
  EXPECT_EQ(live_entries[0].generation, 0u);
  EXPECT_EQ(live_entries[1].generation, 0u);
  EXPECT_EQ(live_entries[0].start_byte, live_entries[1].start_byte);
  EXPECT_EQ(live_entries[0].byte_count, sizeof(uint32_t));
  EXPECT_EQ(live_entries[1].byte_count, sizeof(uint32_t));
  EXPECT_TRUE(entries_conflict(live_entries[1], live_entries[0]));
}

TEST(ConSanHipTest, RuntimeBackendRetainsAllWavesForSelectedAddress) {
  constexpr uint32_t kSlots = 2;
  constexpr size_t kReportBytes = direct_report_buffer_bytes(kSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_report_buffer_env(d_report, kReportBytes));

  // ConSan mode rotates the selected residue with the physical dispatch ID.
  // Launch one complete stride of workgroups so exactly one workgroup is
  // selected regardless of the runtime's initial dispatch-ID parity.
  two_wave_lds_accesses<<<2, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  std::vector<Entry> live_entries;
  for (uint32_t i = 0; i < kSlots; ++i) {
    Entry entry = read_entry(report, kSlots, i);
    if (entry.valid)
      live_entries.push_back(entry);
  }

  ASSERT_EQ(live_entries.size(), 2u);
  EXPECT_NE(live_entries[0].owner_id, live_entries[1].owner_id);
  EXPECT_EQ(live_entries[0].start_byte, live_entries[1].start_byte);
  for (const Entry &entry : live_entries) {
    EXPECT_EQ(entry.epoch, 0u);
    EXPECT_EQ(entry.byte_count, sizeof(uint32_t));
  }
  size_t ready_windows = 0;
  for (uint32_t i = 0; i < kSlots; ++i) {
    const CausalWindow window = read_window(report, i);
    const Entry entry = read_entry(report, kSlots, i);
    if (window.publication_state == kWindowReady) {
      ++ready_windows;
      EXPECT_TRUE(entry.valid) << "slot=" << i;
      EXPECT_EQ(window.first_entry, i) << "slot=" << i;
      EXPECT_EQ(window.entry_count, 1u) << "slot=" << i;
      EXPECT_LT(window.workgroup_x, 2u) << "slot=" << i;
      EXPECT_EQ(window.workgroup_y, 0u) << "slot=" << i;
      EXPECT_EQ(window.workgroup_z, 0u) << "slot=" << i;
      EXPECT_EQ(window.epoch, entry.epoch) << "slot=" << i;
      EXPECT_EQ(window.generation & 3u, entry.generation) << "slot=" << i;
    } else {
      EXPECT_EQ(window.publication_state, kWindowEmpty) << "slot=" << i;
      EXPECT_FALSE(entry.valid) << "slot=" << i;
    }
  }
  EXPECT_EQ(ready_windows, 2u);
  const CausalWindow first_window = read_window(report, 0);
  const CausalWindow second_window = read_window(report, 1);
  EXPECT_EQ(first_window.workgroup_x, second_window.workgroup_x);
  EXPECT_LT(first_window.workgroup_x, 2u);
  ReportHeader header;
  std::memcpy(&header, report.data(), sizeof(header));
  EXPECT_EQ(header.causal_window_count, 2u);
  EXPECT_EQ(header.dropped_window_count, 0u);
}

TEST(ConSanHipTest, InKernelCheckerReportsConflictImmediately) {
  constexpr uint32_t kSlots = 2;
  constexpr size_t kReportBytes = direct_report_buffer_bytes(kSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_report_buffer_env(d_report, kReportBytes));

  two_wave_lds_accesses<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  ReportHeader header;
  HIP_ASSERT(hipMemcpy(&header, d_report, sizeof(header), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  EXPECT_GT(header.event_counter, 0u);
}

void check_workgroup_isolation(unsigned axis) {
  constexpr uint32_t kSlots = 2;
  constexpr size_t kReportBytes = direct_report_buffer_bytes(kSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportHeader initial_header;
  initial_header.header_size = sizeof(ReportHeader);
  initial_header.watchpoint_capacity = kSlots;
  initial_header.causal_window_capacity = kSlots;
  HIP_ASSERT(hipMemcpy(d_report, &initial_header, sizeof(initial_header), hipMemcpyHostToDevice));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_report_buffer_env(d_report, kReportBytes));

  if (axis == 0)
    equivalent_lds_accesses_in_different_workgroups<<<2, 64>>>(d_out);
  else if (axis == 1)
    equivalent_lds_accesses_in_different_2d_workgroups<<<dim3(1, 2, 1), 64>>>(d_out);
  else
    equivalent_lds_accesses_in_different_3d_workgroups<<<dim3(1, 1, 2), 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  ReportHeader header;
  std::memcpy(&header, report.data(), sizeof(header));
  ASSERT_EQ(header.magic, kReportMagic);
  ASSERT_EQ(header.abi_version, kReportAbiVersion);
  ASSERT_EQ(header.header_size, sizeof(ReportHeader));
  ASSERT_EQ(header.generation, initial_header.generation);
  ASSERT_EQ(header.dispatch_id, initial_header.dispatch_id);

  EXPECT_EQ(header.watchpoint_capacity, kSlots);
  EXPECT_EQ(header.causal_window_capacity, kSlots);
  EXPECT_EQ(header.causal_window_count, 2u);
  EXPECT_EQ(header.malformed_window_count, 0u);
  EXPECT_EQ(header.dropped_window_count, 0u);

  const auto coordinate = [axis](const CausalWindow &w) {
    return axis == 0 ? w.workgroup_x : axis == 1 ? w.workgroup_y : w.workgroup_z;
  };
  struct PublishedSample {
    CausalWindow window;
    Entry entry;
  };
  std::vector<PublishedSample> published;
  for (uint32_t i = 0; i < kSlots; ++i) {
    const CausalWindow window = read_window(report, i);
    const Entry entry = read_entry(report, kSlots, i);
    if (window.publication_state == kWindowReady) {
      EXPECT_TRUE(entry.valid) << "slot=" << i;
      EXPECT_EQ(window.first_entry, i) << "slot=" << i;
      EXPECT_EQ(window.entry_count, 1u) << "slot=" << i;
      if (axis != 0)
        EXPECT_EQ(window.workgroup_x, 0u);
      if (axis != 1)
        EXPECT_EQ(window.workgroup_y, 0u);
      if (axis != 2)
        EXPECT_EQ(window.workgroup_z, 0u);
      EXPECT_EQ(window.generation, header.generation) << "slot=" << i;
      EXPECT_EQ(window.epoch, entry.epoch) << "slot=" << i;
      EXPECT_EQ(window.generation & 3u, entry.generation) << "slot=" << i;
      EXPECT_EQ(window.cluster_workgroup_id, 0u) << "slot=" << i;
      EXPECT_EQ(entry.byte_count, sizeof(uint32_t)) << "slot=" << i;
      if (entry.valid)
        published.push_back({window, entry});
    } else {
      EXPECT_EQ(window.publication_state, kWindowEmpty) << "slot=" << i;
      EXPECT_FALSE(entry.valid) << "slot=" << i;
      EXPECT_EQ(window.generation, 0u) << "slot=" << i;
      EXPECT_EQ(window.dispatch_id, 0u) << "slot=" << i;
      EXPECT_EQ(window.workgroup_x, 0u) << "slot=" << i;
      if (axis != 0)
        EXPECT_EQ(window.workgroup_x, 0u);
      if (axis != 1)
        EXPECT_EQ(window.workgroup_y, 0u);
      if (axis != 2)
        EXPECT_EQ(window.workgroup_z, 0u);
      EXPECT_EQ(window.epoch, 0u) << "slot=" << i;
      EXPECT_EQ(window.first_entry, 0u) << "slot=" << i;
      EXPECT_EQ(window.entry_count, 0u) << "slot=" << i;
      EXPECT_EQ(window.cluster_workgroup_id, 0u) << "slot=" << i;
      EXPECT_FALSE(entry.consumed) << "slot=" << i;
      EXPECT_EQ(entry.kind, 0u) << "slot=" << i;
      EXPECT_EQ(entry.owner_id, 0u) << "slot=" << i;
      EXPECT_EQ(entry.epoch, 0u) << "slot=" << i;
      EXPECT_EQ(entry.generation, 0u) << "slot=" << i;
      EXPECT_EQ(entry.start_byte, 0u) << "slot=" << i;
      EXPECT_EQ(entry.byte_count, 1u) << "slot=" << i;
    }
  }
  ASSERT_EQ(published.size(), 2u);
  std::stable_sort(published.begin(), published.end(),
                   [&](const PublishedSample &lhs, const PublishedSample &rhs) {
                     return coordinate(lhs.window) < coordinate(rhs.window);
                   });
  EXPECT_EQ(coordinate(published[0].window), 0u);
  EXPECT_EQ(published[0].entry.kind, kAccessKindWrite);
  EXPECT_EQ(coordinate(published[1].window), 1u);
  EXPECT_EQ(published[1].entry.kind, kAccessKindRead);
  EXPECT_NE(published[0].window.dispatch_id, 0u);
  EXPECT_EQ(published[0].window.dispatch_id, published[1].window.dispatch_id);
  EXPECT_EQ(published[0].entry.owner_id, 0u);
  EXPECT_EQ(published[1].entry.owner_id, 1u);
  EXPECT_EQ(published[0].entry.epoch, 0u);
  EXPECT_EQ(published[1].entry.epoch, 0u);
  EXPECT_EQ(published[0].entry.start_byte, 0u);
  EXPECT_EQ(published[0].entry.start_byte, published[1].entry.start_byte);
  // The entry-only predicate intentionally sees the same-byte-range write/read
  // conflict. The checker keeps this clean by partitioning entries by the exact
  // workgroup descriptor carried by their enclosing causal windows.
  EXPECT_TRUE(entries_conflict(published[0].entry, published[1].entry));
  EXPECT_EQ(header.causal_window_count, 2u);
  EXPECT_EQ(header.dropped_window_count, 0u);
  std::printf("SA2C sampled evidence ready_windows=%zu live_entries=%zu workgroups=0,1 "
              "dropped_windows=%u\n",
              published.size(), published.size(), header.dropped_window_count);
}

TEST(ConSanHipTest, DirectRetainsExactMultiWorkgroupWindows) { check_workgroup_isolation(0); }

TEST(ConSanHipTest, EquivalentAccessesInDifferent2DWorkgroupsDoNotConflict) {
  check_workgroup_isolation(1);
}

TEST(ConSanHipTest, EquivalentAccessesInDifferent3DWorkgroupsDoNotConflict) {
  check_workgroup_isolation(2);
}

TEST(ConSanHipTest, OneSlotPreservesBufferBounds) {
  constexpr uint32_t kAccessRecords = 1;
  constexpr size_t kReportBytes = direct_report_buffer_bytes(kAccessRecords);
  constexpr size_t kSentinelBytes = 64;
  constexpr uint8_t kSentinel = 0xA5;
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes + kSentinelBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));
  auto *d_report_bytes = static_cast<uint8_t *>(d_report);
  HIP_ASSERT(hipMemset(d_report_bytes + kReportBytes, kSentinel, kSentinelBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_report_buffer_env(d_report, kReportBytes));

  two_wave_lds_accesses<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  std::vector<uint8_t> report(kReportBytes + kSentinelBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  ReportHeader header;
  std::memcpy(&header, report.data(), sizeof(header));
  EXPECT_EQ(header.causal_window_count, 1u);
  EXPECT_EQ(header.malformed_window_count, 0u);
  const auto entry = read_entry(report, kAccessRecords, 0);
  EXPECT_TRUE(entry.valid);
  EXPECT_EQ(entry.byte_count, sizeof(uint32_t));

  for (size_t i = 0; i < kSentinelBytes; ++i)
    EXPECT_EQ(report[kReportBytes + i], kSentinel) << "sentinel byte " << i << " was modified";
}

TEST(ConSanHipTest, AtomicHandoffRunsWithAutoReportBuffer) {
  uint32_t *d_out = nullptr;
  uint32_t *d_counter = nullptr;
  uint32_t *d_flag = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_counter, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_flag, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_counter, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_flag, 0, sizeof(uint32_t)));

  atomic_handoff_lds_accesses<<<1, 64>>>(d_out, d_counter, d_counter, d_flag);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  free_device_ptr(d_flag);
  free_device_ptr(d_counter);
  free_device_ptr(d_out);
  EXPECT_EQ(out, 0xA70C0000u);
}
