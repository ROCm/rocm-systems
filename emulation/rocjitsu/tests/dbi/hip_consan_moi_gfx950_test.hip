// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_moi_gfx950_test.hip
/// @brief Focused native gfx950 ConSan MOI instrumentation tests.

#include <cstdint>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

namespace {

__device__ __forceinline__ void gfx950_padded_strong_group_flat_store(uint32_t value) {
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t"
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t"
               "flat_store_dword v[0:1], %0\n\t"
               ".rept 128\n\t"
               "s_nop 0\n\t"
               ".endr\n\t"
               :
               : "v"(value)
               : "s14", "s15", "v0", "v1", "memory");
}

__global__ __launch_bounds__(64) void gfx950_moi_single_lds_store(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  const uint32_t value = 0x95020000u | tid;
  asm volatile("ds_write_b32 %0, %1\n\t"
               ".rept 128\n\t"
               "s_nop 0\n\t"
               ".endr\n\t"
               :
               : "v"(address), "v"(value)
               : "memory");
  __syncthreads();
  if (tid == 0)
    out[0] = lds[0];
}

__global__ __launch_bounds__(64) void gfx950_moi_forced_spill(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  const uint32_t value = 0x95030000u | tid;
  uint32_t victim0 = 0x10500000u ^ tid;
  uint32_t victim1 = 0x20500000u ^ (tid * 3u);
  uint32_t victim2 = 0x30500000u ^ (tid * 5u);
  uint32_t victim3 = 0x40500000u ^ (tid * 7u);
  asm volatile("ds_write_b32 %4, %5\n\t"
               ".rept 128\n\t"
               "s_nop 0\n\t"
               ".endr\n\t"
               : "+v"(victim0), "+v"(victim1), "+v"(victim2), "+v"(victim3)
               : "v"(address), "v"(value)
               : "memory");
  const uint32_t base = tid * 4u;
  out[base + 0u] = victim0;
  out[base + 1u] = victim1;
  out[base + 2u] = victim2;
  out[base + 3u] = victim3;
}

__global__ __launch_bounds__(64) void gfx950_moi_forced_sampled_spill(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  uint32_t values[8] = {
      0x11500000u ^ tid, 0x22500000u ^ tid, 0x33500000u ^ tid, 0x44500000u ^ tid,
      0x55500000u ^ tid, 0x66500000u ^ tid, 0x77500000u ^ tid, 0x88500000u ^ tid,
  };
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  asm volatile("ds_write_b32 %0, %1\n\t"
               ".rept 128\n\t"
               "s_nop 0\n\t"
               ".endr\n\t"
               :
               : "v"(address), "v"(values[0])
               : "memory");
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  const uint32_t base = tid * 8u;
  for (uint32_t i = 0; i < 8u; ++i)
    out[base + i] = values[i];
}

__global__ __launch_bounds__(128) void gfx950_moi_forced_inline_spill(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  uint32_t values[8] = {
      0x12500000u ^ tid, 0x23500000u ^ tid, 0x34500000u ^ tid, 0x45500000u ^ tid,
      0x56500000u ^ tid, 0x67500000u ^ tid, 0x78500000u ^ tid, 0x89500000u ^ tid,
  };
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  if (tid == 0u || tid == 64u) {
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(values[0])
                 : "memory");
  }
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  const uint32_t base = tid * 8u;
  for (uint32_t i = 0; i < 8u; ++i)
    out[base + i] = values[i];
}

__global__ __launch_bounds__(64) void gfx950_standalone_report_publication(uint32_t *out) {
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  uint32_t *slot = out + tid;
  const uint32_t value = 0x95040000u ^ tid;
  asm volatile("flat_store_dword %0, %1\n\t"
               "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
               :
               : "v"(slot), "v"(value)
               : "memory");
}

__global__ __launch_bounds__(128) void gfx950_moi_identity_grid(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t local =
      static_cast<uint32_t>(threadIdx.x) +
      static_cast<uint32_t>(blockDim.x) *
          (static_cast<uint32_t>(threadIdx.y) +
           static_cast<uint32_t>(blockDim.y) * static_cast<uint32_t>(threadIdx.z));
  const uint32_t group = static_cast<uint32_t>(blockIdx.x) +
                         static_cast<uint32_t>(gridDim.x) *
                             (static_cast<uint32_t>(blockIdx.y) +
                              static_cast<uint32_t>(gridDim.y) * static_cast<uint32_t>(blockIdx.z));
  if ((local & 63u) == 0u) {
    const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
    const uint32_t value = 0x95050000u | (group << 4u) | (local >> 6u);
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (local == 0u)
    out[group] = 0x95050000u | group;
}

__global__ __launch_bounds__(128) void gfx950_moi_1d_identity_grid(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t local = static_cast<uint32_t>(threadIdx.x);
  const uint32_t group = static_cast<uint32_t>(blockIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  if (local == 0u) {
    const uint32_t value = 0x95100000u | group;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (local == 64u) {
    const uint32_t value = 0x95100040u | group;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (local == 0u)
    out[group] = 0x95100000u | group;
}

__global__ __launch_bounds__(128) void gfx950_moi_2d_identity_grid(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t local = static_cast<uint32_t>(threadIdx.x);
  const uint32_t group = static_cast<uint32_t>(blockIdx.x) +
                         static_cast<uint32_t>(gridDim.x) * static_cast<uint32_t>(blockIdx.y);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  if (local == 0u) {
    const uint32_t value = 0x95200000u | group;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (local == 64u) {
    const uint32_t value = 0x95200040u | group;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (local == 0u)
    out[group] = 0x95200000u | group;
}

__global__
__launch_bounds__(128) void gfx950_moi_no_coordinate_identity_grid(uint32_t *completion_count) {
  __shared__ uint32_t lds[1];
  const uint32_t local = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  if (local == 0u) {
    const uint32_t value = 0x95300000u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (local == 64u) {
    const uint32_t value = 0x95300040u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (local == 0u)
    atomicAdd(completion_count, 1u);
}

// Deliberately does not reference blockIdx or gridDim. The original descriptor
// therefore has no semantic need for any workgroup-ID system SGPR, while the
// launch still contains eight independent workgroups. Lane zero obtains a
// unique global ticket and broadcasts it with LDS atomics, which ConSan
// intentionally excludes from ordinary access instrumentation. Ticket parity
// selects either Wave64 owner zero or owner one to perform the sole ordinary
// LDS write in that group. Correct workgroup isolation is clean; substituting
// zero for every workgroup coordinate aliases alternating owners and reports a
// false conflict.
__global__
__launch_bounds__(128) void gfx950_moi_no_coordinate_clean_grid(uint32_t *completion_count) {
  __shared__ uint32_t ticket_slot;
  __shared__ uint32_t target[1];
  const uint32_t local = static_cast<uint32_t>(threadIdx.x);
  if (local == 0u)
    atomicExch(&ticket_slot, atomicAdd(completion_count, 1u));
  __syncthreads();
  if (local == 0u || local == 64u) {
    const uint32_t ticket = atomicAdd(&ticket_slot, 0u);
    const uint32_t owner = local >> 6u;
    if ((ticket & 1u) != owner)
      return;
    const uint32_t address =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&target[0]) & 0xffffu);
    const uint32_t value = 0x95310000u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
}

__global__ __launch_bounds__(128) void gfx950_moi_two_wave_race(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  if (tid == 0u) {
    const uint32_t value = 0x95060000u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (tid == 64u) {
    const uint32_t value = 0x95060040u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (tid == 0u)
    out[0] = 0x95060001u;
}

__global__ __launch_bounds__(128) void gfx950_moi_two_wave_group_flat_race(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  asm volatile("" : : "v"(&lds[0]));
  if (tid == 0u || tid == 64u)
    gfx950_padded_strong_group_flat_store(0x95070000u | tid);
  if (tid == 0u)
    out[0] = 0x95070001u;
}

__global__ __launch_bounds__(128) void gfx950_moi_two_wave_read_read(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  if (tid == 0u || tid == 64u) {
    uint32_t value = 0;
    asm volatile("ds_read_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 : "=v"(value)
                 : "v"(address)
                 : "memory");
    out[tid >> 6u] = value;
  }
}

__global__ __launch_bounds__(128) void gfx950_moi_two_wave_b128_race(uint32_t *out) {
  __shared__ uint32_t lds[4];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  if (tid == 0u || tid == 64u) {
    register uint32_t value0 asm("v20") = 0x95080000u | tid;
    register uint32_t value1 asm("v21") = 0x95081000u | tid;
    register uint32_t value2 asm("v22") = 0x95082000u | tid;
    register uint32_t value3 asm("v23") = 0x95083000u | tid;
    asm volatile("ds_write_b128 %0, v[20:23]\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value0), "v"(value1), "v"(value2), "v"(value3)
                 : "memory");
  }
  if (tid == 0u)
    out[0] = 0x95080001u;
}

__global__ __launch_bounds__(128) void gfx950_moi_inline_diagnostic_overflow(uint32_t *out) {
  __shared__ uint32_t lds[5];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  const uint32_t address4 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[4]) & 0xffffu);
  if (tid == 0u || tid == 64u) {
    register uint32_t value0 asm("v20") = 0x950a0000u | tid;
    register uint32_t value1 asm("v21") = 0x950a1000u | tid;
    register uint32_t value2 asm("v22") = 0x950a2000u | tid;
    register uint32_t value3 asm("v23") = 0x950a3000u | tid;
    const uint32_t value4 = 0x950a4000u | tid;
    asm volatile("ds_write_b128 %0, v[20:23]\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address0), "v"(value0), "v"(value1), "v"(value2), "v"(value3)
                 : "memory");
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address4), "v"(value4)
                 : "memory");
  }
  if (tid == 0u)
    out[0] = 0x950a0001u;
}

__global__ __launch_bounds__(128) void gfx950_moi_barrier_ordered_two_wave(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  if (tid == 0u) {
    const uint32_t value = 0x95090000u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  __syncthreads();
  if (tid == 64u) {
    const uint32_t value = 0x95090040u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (tid == 0u)
    out[0] = 0x95090001u;
}

__global__ __launch_bounds__(128) void gfx950_moi_forced_barrier_spill(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  uint32_t values[8] = {
      0x13500000u ^ tid, 0x24500000u ^ tid, 0x35500000u ^ tid, 0x46500000u ^ tid,
      0x57500000u ^ tid, 0x68500000u ^ tid, 0x79500000u ^ tid, 0x8a500000u ^ tid,
  };
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  if (tid == 0u) {
    asm volatile("s_nop 0\n\t"
                 "ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(values[0])
                 : "memory");
  }
  __syncthreads();
  if (tid == 64u) {
    asm volatile("s_nop 0\n\t"
                 "ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(values[1])
                 : "memory");
  }
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  const uint32_t base = tid * 8u;
  for (uint32_t i = 0; i < 8u; ++i)
    out[base + i] = values[i];
}

__global__ __launch_bounds__(64) void gfx950_moi_same_wave_program_order(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  if (tid == 0u) {
    uint32_t value = 0x95070000u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
    value = 0x95070001u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
    out[0] = 0x95070001u;
  }
}

__global__ __launch_bounds__(64) void gfx950_moi_same_wave_v0_clobber(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  if (tid == 0u) {
    const uint32_t first = 0x950b0000u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(first)
                 : "memory");
    // The entry prologue must have snapshotted the logical Wave64 owner. An
    // access probe that re-derives it from mutable guest v0 observes owner two
    // here and falsely conflicts with the first access from owner zero.
    asm volatile("v_mov_b32 v0, 0x80\n\t" : : : "v0");
    const uint32_t second = 0x950b0001u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(second)
                 : "memory");
    out[0] = second;
  }
}

__global__ __launch_bounds__(64) void gfx950_moi_inline_multidim_same_wave(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t local =
      static_cast<uint32_t>(threadIdx.x) +
      static_cast<uint32_t>(blockDim.x) *
          (static_cast<uint32_t>(threadIdx.y) +
           static_cast<uint32_t>(blockDim.y) * static_cast<uint32_t>(threadIdx.z));
  const uint32_t second_local = blockDim.z == 1u ? static_cast<uint32_t>(blockDim.x)
                                                 : static_cast<uint32_t>(blockDim.x * blockDim.y);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  if (local == 0u) {
    const uint32_t first = 0x950c0000u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(first)
                 : "memory");
  }
  if (local == second_local) {
    const uint32_t second = 0x950c0001u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(second)
                 : "memory");
  }
  if (local == 0u)
    out[0] = 0x950c0002u;
}

__global__ __launch_bounds__(128) void gfx950_moi_static_sampled_selection(uint32_t *out) {
  __shared__ uint32_t lds[2];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  if (tid == 0u) {
    const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
    const uint32_t value = 0x95080000u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (tid == 64u) {
    const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[1]) & 0xffffu);
    const uint32_t value = 0x95080040u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
  }
  if (tid == 0u)
    out[0] = 0x95080001u;
}

__global__ __launch_bounds__(128) void gfx950_moi_atomic_handoff(uint32_t *out,
                                                                 uint32_t *release_counter,
                                                                 uint32_t *acquire_counter,
                                                                 uint32_t *flag) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  if (tid == 0u) {
    const uint32_t value = 0x95090001u;
    const uint32_t one = 1u;
    asm volatile("ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(value)
                 : "memory");
    asm volatile("flat_atomic_add %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 :
                 : "v"(release_counter), "v"(one)
                 : "memory");
    *flag = 1u;
  }
  if (tid == 64u) {
    volatile uint32_t *volatile_flag = flag;
    while (*volatile_flag == 0u) {
    }
    const uint32_t one = 1u;
    uint32_t old = 0u;
    asm volatile("flat_atomic_add %0, %1, %2 sc0\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 : "=&v"(old)
                 : "v"(acquire_counter), "v"(one)
                 : "memory");
    uint32_t loaded = 0u;
    asm volatile("ds_read_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 : "=v"(loaded)
                 : "v"(address)
                 : "memory");
    out[0] = loaded;
    out[1] = old;
  }
}

__global__ __launch_bounds__(128) void gfx950_moi_forced_atomic_spill(uint32_t *out,
                                                                      uint32_t *release_counter,
                                                                      uint32_t *acquire_counter,
                                                                      uint32_t *flag) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  uint32_t values[8] = {
      0x14500000u ^ tid, 0x25500000u ^ tid, 0x36500000u ^ tid, 0x47500000u ^ tid,
      0x58500000u ^ tid, 0x69500000u ^ tid, 0x7a500000u ^ tid, 0x8b500000u ^ tid,
  };
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  if (tid == 0u) {
    const uint32_t one = 1u;
    asm volatile("s_nop 0\n\t"
                 "ds_write_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 :
                 : "v"(address), "v"(values[0])
                 : "memory");
    asm volatile("flat_atomic_add %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 :
                 : "v"(release_counter), "v"(one)
                 : "memory");
    *flag = 1u;
  }
  if (tid == 64u) {
    volatile uint32_t *volatile_flag = flag;
    while (*volatile_flag == 0u) {
    }
    const uint32_t one = 1u;
    uint32_t old = 0u;
    asm volatile("flat_atomic_add %0, %1, %2 sc0\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 : "=&v"(old)
                 : "v"(acquire_counter), "v"(one)
                 : "memory");
    asm volatile("s_nop 0\n\t"
                 "ds_read_b32 %0, %1\n\t"
                 ".rept 128\n\t"
                 "s_nop 0\n\t"
                 ".endr\n\t"
                 : "=&v"(old)
                 : "v"(address)
                 : "memory");
  }
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  const uint32_t base = tid * 8u;
  for (uint32_t i = 0; i < 8u; ++i)
    out[base + i] = values[i];
}

void expect_spill_values(const std::vector<uint32_t> &out, const uint32_t (&seeds)[8]) {
  for (uint32_t tid = 0; tid < 128u; ++tid) {
    for (uint32_t i = 0; i < 8u; ++i)
      EXPECT_EQ(out[tid * 8u + i], seeds[i] ^ tid) << "tid=" << tid << " i=" << i;
  }
}

void run_atomic_handoff(bool same_address) {
  uint32_t *device_out = nullptr;
  uint32_t *release_counter = nullptr;
  uint32_t *acquire_counter = nullptr;
  uint32_t *flag = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, 2u * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&release_counter, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&acquire_counter, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&flag, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, 2u * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(release_counter, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(acquire_counter, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(flag, 0, sizeof(uint32_t)));
  uint32_t *selected_acquire_counter = same_address ? release_counter : acquire_counter;
  gfx950_moi_atomic_handoff<<<1, 128>>>(device_out, release_counter, selected_acquire_counter,
                                        flag);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out[2] = {};
  HIP_ASSERT(hipMemcpy(out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(flag));
  HIP_ASSERT(hipFree(acquire_counter));
  HIP_ASSERT(hipFree(release_counter));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out[0], 0x95090001u);
  EXPECT_EQ(out[1], same_address ? 1u : 0u);
}

void run_forced_atomic_spill(bool same_address) {
  constexpr uint32_t kWords = 128u * 8u;
  uint32_t *device_out = nullptr;
  uint32_t *release_counter = nullptr;
  uint32_t *acquire_counter = nullptr;
  uint32_t *flag = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kWords * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&release_counter, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&acquire_counter, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&flag, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kWords * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(release_counter, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(acquire_counter, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(flag, 0, sizeof(uint32_t)));
  uint32_t *selected_acquire_counter = same_address ? release_counter : acquire_counter;
  gfx950_moi_forced_atomic_spill<<<1, 128>>>(device_out, release_counter, selected_acquire_counter,
                                             flag);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint32_t> out(kWords);
  HIP_ASSERT(hipMemcpy(out.data(), device_out, kWords * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(flag));
  HIP_ASSERT(hipFree(acquire_counter));
  HIP_ASSERT(hipFree(release_counter));
  HIP_ASSERT(hipFree(device_out));
  constexpr uint32_t seeds[8] = {
      0x14500000u, 0x25500000u, 0x36500000u, 0x47500000u,
      0x58500000u, 0x69500000u, 0x7a500000u, 0x8b500000u,
  };
  expect_spill_values(out, seeds);
}

TEST(ConSanMoiGfx950Test, SingleLdsStoreCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_single_lds_store<<<1, 64>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out & 0xffff0000u, 0x95020000u);
}

TEST(ConSanMoiGfx950Test, ForcedSpillPreservesLiveVgprs) {
  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kValuesPerThread = 4;
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kThreads * kValuesPerThread * sizeof(uint32_t)));
  gfx950_moi_forced_spill<<<1, kThreads>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint32_t> out(kThreads * kValuesPerThread);
  HIP_ASSERT(
      hipMemcpy(out.data(), device_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    const uint32_t base = tid * kValuesPerThread;
    EXPECT_EQ(out[base + 0u], 0x10500000u ^ tid) << "tid=" << tid;
    EXPECT_EQ(out[base + 1u], 0x20500000u ^ (tid * 3u)) << "tid=" << tid;
    EXPECT_EQ(out[base + 2u], 0x30500000u ^ (tid * 5u)) << "tid=" << tid;
    EXPECT_EQ(out[base + 3u], 0x40500000u ^ (tid * 7u)) << "tid=" << tid;
  }
}

TEST(ConSanMoiGfx950Test, ForcedSampledSpillPreservesEightLiveVgprs) {
  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kValuesPerThread = 8;
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kThreads * kValuesPerThread * sizeof(uint32_t)));
  gfx950_moi_forced_sampled_spill<<<1, kThreads>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint32_t> out(kThreads * kValuesPerThread);
  HIP_ASSERT(
      hipMemcpy(out.data(), device_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  constexpr uint32_t seeds[kValuesPerThread] = {
      0x11500000u, 0x22500000u, 0x33500000u, 0x44500000u,
      0x55500000u, 0x66500000u, 0x77500000u, 0x88500000u,
  };
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    for (uint32_t i = 0; i < kValuesPerThread; ++i)
      EXPECT_EQ(out[tid * kValuesPerThread + i], seeds[i] ^ tid) << "tid=" << tid << " i=" << i;
  }
}

TEST(ConSanMoiGfx950Test, ForcedInlineSpillPreservesEightLiveVgprs) {
  constexpr uint32_t kThreads = 128;
  constexpr uint32_t kValuesPerThread = 8;
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kThreads * kValuesPerThread * sizeof(uint32_t)));
  gfx950_moi_forced_inline_spill<<<1, kThreads>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint32_t> out(kThreads * kValuesPerThread);
  HIP_ASSERT(
      hipMemcpy(out.data(), device_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  constexpr uint32_t seeds[kValuesPerThread] = {
      0x12500000u, 0x23500000u, 0x34500000u, 0x45500000u,
      0x56500000u, 0x67500000u, 0x78500000u, 0x89500000u,
  };
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    for (uint32_t i = 0; i < kValuesPerThread; ++i)
      EXPECT_EQ(out[tid * kValuesPerThread + i], seeds[i] ^ tid) << "tid=" << tid << " i=" << i;
  }
}

TEST(ConSanMoiGfx950Test, StandaloneReportPublication) {
  constexpr uint32_t kThreads = 64;
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kThreads * sizeof(uint32_t)));
  gfx950_standalone_report_publication<<<1, kThreads>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint32_t> out(kThreads);
  HIP_ASSERT(
      hipMemcpy(out.data(), device_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  for (uint32_t tid = 0; tid < kThreads; ++tid)
    EXPECT_EQ(out[tid], 0x95040000u ^ tid) << "tid=" << tid;
}

TEST(ConSanMoiGfx950Test, StableThreeDimensionalIdentity) {
  constexpr uint32_t kGroups = 8;
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kGroups * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kGroups * sizeof(uint32_t)));
  gfx950_moi_identity_grid<<<dim3(2, 2, 2), dim3(8, 4, 4)>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint32_t> out(kGroups);
  HIP_ASSERT(
      hipMemcpy(out.data(), device_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  for (uint32_t group = 0; group < kGroups; ++group)
    EXPECT_EQ(out[group], 0x95050000u | group) << "group=" << group;
}

TEST(ConSanMoiGfx950Test, StableOneDimensionalIdentity) {
  constexpr uint32_t kGroups = 8;
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kGroups * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kGroups * sizeof(uint32_t)));
  gfx950_moi_1d_identity_grid<<<dim3(kGroups, 1, 1), 128>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint32_t> out(kGroups);
  HIP_ASSERT(
      hipMemcpy(out.data(), device_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  for (uint32_t group = 0; group < kGroups; ++group)
    EXPECT_EQ(out[group], 0x95100000u | group) << "group=" << group;
}

TEST(ConSanMoiGfx950Test, StableTwoDimensionalIdentity) {
  constexpr uint32_t kGroups = 8;
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kGroups * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kGroups * sizeof(uint32_t)));
  gfx950_moi_2d_identity_grid<<<dim3(4, 2, 1), 128>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint32_t> out(kGroups);
  HIP_ASSERT(
      hipMemcpy(out.data(), device_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  for (uint32_t group = 0; group < kGroups; ++group)
    EXPECT_EQ(out[group], 0x95200000u | group) << "group=" << group;
}

TEST(ConSanMoiGfx950Test, StableNoCoordinateIdentity) {
  uint32_t *device_completion_count = nullptr;
  HIP_ASSERT(hipMalloc(&device_completion_count, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_completion_count, 0, sizeof(uint32_t)));
  gfx950_moi_no_coordinate_identity_grid<<<dim3(2, 2, 2), 128>>>(device_completion_count);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t completion_count = 0;
  HIP_ASSERT(hipMemcpy(&completion_count, device_completion_count, sizeof(completion_count),
                       hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_completion_count));
  EXPECT_EQ(completion_count, 8u);
}

TEST(ConSanMoiGfx950Test, NoCoordinateCleanIsolationCompletes) {
  uint32_t *device_completion_count = nullptr;
  HIP_ASSERT(hipMalloc(&device_completion_count, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_completion_count, 0, sizeof(uint32_t)));
  gfx950_moi_no_coordinate_clean_grid<<<dim3(2, 2, 2), 128>>>(device_completion_count);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t completion_count = 0;
  HIP_ASSERT(hipMemcpy(&completion_count, device_completion_count, sizeof(completion_count),
                       hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_completion_count));
  EXPECT_EQ(completion_count, 8u);
}

TEST(ConSanMoiGfx950Test, TwoWaveRaceCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_two_wave_race<<<1, 128>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out, 0x95060001u);
}

TEST(ConSanMoiGfx950Test, TwoWaveGroupFlatRaceCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_two_wave_group_flat_race<<<1, 128>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out, 0x95070001u);
}

TEST(ConSanMoiGfx950Test, TwoWaveReadReadCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, 2u * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, 2u * sizeof(uint32_t)));
  gfx950_moi_two_wave_read_read<<<1, 128>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipFree(device_out));
}

TEST(ConSanMoiGfx950Test, TwoWaveB128RaceCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_two_wave_b128_race<<<1, 128>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipFree(device_out));
}

TEST(ConSanMoiGfx950Test, InlineDiagnosticOverflowCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_inline_diagnostic_overflow<<<1, 128>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out, 0x950a0001u);
}

TEST(ConSanMoiGfx950Test, BarrierOrderedTwoWaveCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_barrier_ordered_two_wave<<<1, 128>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipFree(device_out));
}

TEST(ConSanMoiGfx950Test, ForcedBarrierSpillPreservesEightLiveVgprs) {
  constexpr uint32_t kWords = 128u * 8u;
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kWords * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kWords * sizeof(uint32_t)));
  gfx950_moi_forced_barrier_spill<<<1, 128>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint32_t> out(kWords);
  HIP_ASSERT(hipMemcpy(out.data(), device_out, kWords * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  constexpr uint32_t seeds[8] = {
      0x13500000u, 0x24500000u, 0x35500000u, 0x46500000u,
      0x57500000u, 0x68500000u, 0x79500000u, 0x8a500000u,
  };
  expect_spill_values(out, seeds);
}

TEST(ConSanMoiGfx950Test, SameWaveProgramOrderCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_same_wave_program_order<<<1, 64>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out, 0x95070001u);
}

TEST(ConSanMoiGfx950Test, SameWaveV0ClobberCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_same_wave_v0_clobber<<<1, 64>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out, 0x950b0001u);
}

TEST(ConSanMoiGfx950Test, InlinePrivateTwoDimensionalIdentityIsClean) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_inline_multidim_same_wave<<<1, dim3(8, 8, 1)>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out, 0x950c0002u);
}

TEST(ConSanMoiGfx950Test, InlinePrivateThreeDimensionalIdentityIsClean) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_inline_multidim_same_wave<<<1, dim3(4, 4, 4)>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out, 0x950c0002u);
}

TEST(ConSanMoiGfx950Test, StaticSampledSelectionCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_static_sampled_selection<<<1, 128>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out, 0x95080001u);
}

TEST(ConSanMoiGfx950Test, AtomicHandoffSameAddressCompletes) {
  run_atomic_handoff(/*same_address=*/true);
}

TEST(ConSanMoiGfx950Test, AtomicHandoffWrongAddressCompletes) {
  run_atomic_handoff(/*same_address=*/false);
}

TEST(ConSanMoiGfx950Test, ForcedAtomicSpillSameAddressPreservesEightLiveVgprs) {
  run_forced_atomic_spill(/*same_address=*/true);
}

TEST(ConSanMoiGfx950Test, ForcedAtomicSpillWrongAddressPreservesEightLiveVgprs) {
  run_forced_atomic_spill(/*same_address=*/false);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}
